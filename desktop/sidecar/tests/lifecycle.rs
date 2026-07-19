//! Sidecar lifecycle against fake servers (sh scripts) and, ignore-gated,
//! the real Python server. The sh-based tests are Unix-only; Windows gets
//! exercised by the CI desktop job (stage 2) and the unit tests.
#![cfg(unix)]

use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;

use loraforge_sidecar::{
    resolve_command, tail_lines, LaunchError, LaunchOptions, ServerCommand, ShutdownOutcome,
    Sidecar,
};

fn sh(script: &str) -> ServerCommand {
    ServerCommand {
        argv: vec!["sh".into(), "-c".into(), script.into()],
        cwd: None,
    }
}

fn opts(data_root: &Path) -> LaunchOptions {
    LaunchOptions {
        handshake_timeout: Duration::from_secs(10),
        shutdown_wait: Duration::from_millis(500),
        data_root: Some(data_root.to_path_buf()),
    }
}

fn ready_echo(port: u16) -> String {
    format!(
        r#"echo 'LORAFORGE_READY {{"url": "http://127.0.0.1:{port}", "port": {port}, "pid": 1}}'"#
    )
}

#[test]
fn handshake_parses_ready_and_tees_output() {
    let dir = tempfile::tempdir().unwrap();
    let script = format!(
        "echo starting up; {}; echo after ready; sleep 30",
        ready_echo(4242)
    );
    let sidecar = Sidecar::launch(&sh(&script), &opts(dir.path())).expect("handshake");
    assert_eq!(sidecar.ready.port, 4242);
    assert_eq!(sidecar.ready.url, "http://127.0.0.1:4242");
    let log_path = sidecar.log_path.clone();

    // nothing listens on 4242 → POST fails → short wait → force-kill
    assert_eq!(sidecar.shutdown(), ShutdownOutcome::ForceKilled);

    let log = fs::read_to_string(&log_path).unwrap();
    assert!(log.contains("starting up"), "stdout tee'd: {log}");
    assert!(
        log.contains("after ready"),
        "tee continues past ready: {log}"
    );
    assert!(
        log.contains("[shell] "),
        "force-kill logged as abnormal: {log}"
    );
}

#[test]
fn graceful_shutdown_when_server_honors_the_request() {
    let dir = tempfile::tempdir().unwrap();
    let flag = dir.path().join("exit-flag");

    // A stand-in control plane: accept the shutdown POST, answer 202, and
    // drop the flag file the fake server polls for its exit.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let flag_for_listener = flag.clone();
    let accept = std::thread::spawn(move || {
        let (mut conn, _) = listener.accept().unwrap();
        let mut buf = [0u8; 1024];
        let n = conn.read(&mut buf).unwrap();
        assert!(String::from_utf8_lossy(&buf[..n]).starts_with("POST /control/shutdown"));
        fs::write(&flag_for_listener, "").unwrap();
        conn.write_all(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });

    let script = format!(
        "{}; while [ ! -f {} ]; do sleep 0.05; done",
        ready_echo(port),
        flag.display()
    );
    let mut options = opts(dir.path());
    options.shutdown_wait = Duration::from_secs(10);
    let sidecar = Sidecar::launch(&sh(&script), &options).expect("handshake");
    let log_path = sidecar.log_path.clone();

    let outcome = sidecar.shutdown();
    accept.join().unwrap();
    assert_eq!(outcome, ShutdownOutcome::Graceful { exit_code: Some(0) });
    let log = fs::read_to_string(&log_path).unwrap();
    assert!(
        !log.contains("force-killing"),
        "graceful path stays quiet: {log}"
    );
}

#[test]
fn early_exit_reports_status_and_log() {
    let dir = tempfile::tempdir().unwrap();
    let err = Sidecar::launch(&sh("echo boom >&2; exit 3"), &opts(dir.path()))
        .expect_err("no ready line");
    let LaunchError::ExitedEarly { status, log_path } = err else {
        panic!("expected ExitedEarly, got: {err}");
    };
    assert_eq!(status, Some(3));
    // the error page shows the tail of exactly this file
    assert!(tail_lines(&log_path, 50).iter().any(|l| l.contains("boom")));
}

#[test]
fn handshake_timeout_kills_the_child() {
    let dir = tempfile::tempdir().unwrap();
    let mut options = opts(dir.path());
    options.handshake_timeout = Duration::from_millis(300);
    let err = Sidecar::launch(&sh("sleep 30"), &options).expect_err("never ready");
    assert!(matches!(err, LaunchError::TimedOut { .. }), "got: {err}");
}

#[test]
fn force_kill_takes_the_grandchildren_too() {
    let dir = tempfile::tempdir().unwrap();
    let pid_file = dir.path().join("grandchild.pid");
    let script = format!(
        "sleep 60 & echo $! > {}; {}; sleep 60",
        pid_file.display(),
        ready_echo(1)
    );
    let sidecar = Sidecar::launch(&sh(&script), &opts(dir.path())).expect("handshake");
    let grandchild: i32 = fs::read_to_string(&pid_file)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(alive(grandchild), "grandchild running before shutdown");

    assert_eq!(sidecar.shutdown(), ShutdownOutcome::ForceKilled);

    // the process-group kill must have taken the detached sleep as well
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while alive(grandchild) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!alive(grandchild), "grandchild survived the tree kill");
}

fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

/// The full contract against the real server. Ignored by default: needs the
/// repo's Python dev env (`uv sync`). Run with `cargo test -- --ignored`.
#[test]
#[ignore = "spawns the real python server; needs the repo dev env (uv)"]
fn real_server_full_lifecycle() {
    let repo_root = repo_root();
    let dir = tempfile::tempdir().unwrap();
    let cmd = resolve_command(Some(&repo_root)).unwrap();
    assert_eq!(cmd.argv, ["uv", "run", "loraforge", "serve"]);

    let options = LaunchOptions {
        data_root: Some(dir.path().to_path_buf()),
        ..LaunchOptions::default()
    };
    let sidecar = Sidecar::launch(&cmd, &options).expect("real server handshake");
    assert!(sidecar.ready.url.starts_with("http://127.0.0.1:"));
    assert!(sidecar.ready.pid > 0);
    let log_path = sidecar.log_path.clone();

    let outcome = sidecar.shutdown();
    assert!(
        matches!(outcome, ShutdownOutcome::Graceful { .. }),
        "expected graceful exit, got {outcome:?} — log: {:?}",
        tail_lines(&log_path, 50)
    );
}

fn repo_root() -> PathBuf {
    // desktop/sidecar → desktop → repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}
