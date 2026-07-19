//! Sidecar lifecycle for the LoRAForge desktop shell.
//!
//! Implements the shell side of the contract in `docs/design/tauri-shell.md`
//! (decision 18): spawn `loraforge serve`, read stdout until the
//! `LORAFORGE_READY ` prefix, tee everything to `<data_root>/logs/server.log`
//! (keep last 3), and shut down via `POST /control/shutdown` with a 10s wait
//! before a logged force-kill of the process group (Unix) / Job Object
//! (Windows).
//!
//! Deliberately dumb: no business logic, no job-state interpretation. Every
//! decision that can live in Python lives in Python.

use std::env;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// The ready marker `loraforge serve` prints — matched with a cheap
/// starts-with, never by parsing uvicorn's log banner.
pub const READY_PREFIX: &str = "LORAFORGE_READY ";
/// No ready line within this window → error page, no silent blank window.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// After POST /control/shutdown, wait this long before force-killing.
pub const SHUTDOWN_WAIT: Duration = Duration::from_secs(10);
/// server.log rotation depth: server.log + .1 + .2.
pub const LOG_KEEP: usize = 3;

// ── Ready payload ────────────────────────────────────────────────────────────

/// Payload of the ready line: `LORAFORGE_READY {"url", "port", "pid"}`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ReadyInfo {
    pub url: String,
    pub port: u16,
    /// For watching a server that dies after announcing.
    pub pid: u32,
}

/// Parse a stdout line; `Some` only for a well-formed ready line.
pub fn parse_ready_line(line: &str) -> Option<ReadyInfo> {
    let payload = line.strip_prefix(READY_PREFIX)?;
    serde_json::from_str(payload).ok()
}

// ── Command resolution ───────────────────────────────────────────────────────

/// How to start the server: argv plus an optional working directory.
#[derive(Debug, Clone)]
pub struct ServerCommand {
    pub argv: Vec<String>,
    pub cwd: Option<PathBuf>,
}

#[derive(Debug)]
pub struct ResolveError(pub String);

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ResolveError {}

/// Resolve the server command in contract order:
/// 1. `LORAFORGE_SERVER_CMD` env var (dev/test override, split shell-style),
/// 2. packaged location (step B — stub below),
/// 3. dev fallback: `uv run loraforge serve` with cwd = repo root.
pub fn resolve_command(repo_root: Option<&Path>) -> Result<ServerCommand, ResolveError> {
    if let Ok(raw) = env::var("LORAFORGE_SERVER_CMD") {
        let argv = shlex::split(&raw)
            .ok_or_else(|| ResolveError("LORAFORGE_SERVER_CMD is not shell-splittable".into()))?;
        if argv.is_empty() {
            return Err(ResolveError("LORAFORGE_SERVER_CMD is set but empty".into()));
        }
        return Ok(ServerCommand { argv, cwd: None });
    }
    if let Some(packaged) = packaged_server_command() {
        return Ok(packaged);
    }
    let root = repo_root.ok_or_else(|| {
        ResolveError(
            "no LORAFORGE_SERVER_CMD, no packaged server, and no repo root for the \
             dev fallback (`uv run loraforge serve`)"
                .into(),
        )
    })?;
    Ok(ServerCommand {
        argv: ["uv", "run", "loraforge", "serve"]
            .map(String::from)
            .to_vec(),
        cwd: Some(root.to_path_buf()),
    })
}

/// STEP B STUB: the packaged install location (bundled uv + managed env) is
/// part of the first-run bootstrap phase. Until then there is no packaged
/// server, so resolution falls through to the dev fallback.
fn packaged_server_command() -> Option<ServerCommand> {
    None
}

// ── Data root and logs ───────────────────────────────────────────────────────

/// Mirror of the Python `default_data_root()` (engines/bootstrap.py): the log
/// file must land where `loraforge diagnose` bug reports expect it.
pub fn default_data_root() -> PathBuf {
    #[cfg(windows)]
    {
        let base = env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir().join("AppData").join("Local"));
        base.join("LoRAForge")
    }
    #[cfg(not(windows))]
    {
        let base = env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir().join(".local").join("share"));
        base.join("loraforge")
    }
}

fn home_dir() -> PathBuf {
    env::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// Shift server.log → server.log.1 → server.log.2 (dropping the oldest) and
/// return the fresh server.log path. One rotation per launch: "the last 3
/// runs" is what a bug report wants.
pub fn rotate_logs(logs_dir: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(logs_dir)?;
    let slot = |n: usize| -> PathBuf {
        if n == 0 {
            logs_dir.join("server.log")
        } else {
            logs_dir.join(format!("server.log.{n}"))
        }
    };
    for n in (0..LOG_KEEP - 1).rev() {
        let from = slot(n);
        if from.exists() {
            fs::rename(&from, slot(n + 1))?;
        }
    }
    Ok(slot(0))
}

/// Last `n` lines of a log — the error page's "what just happened" excerpt.
pub fn tail_lines(path: &Path, n: usize) -> Vec<String> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].iter().map(|s| s.to_string()).collect()
}

// ── Launch ───────────────────────────────────────────────────────────────────

/// Knobs with contract defaults; tests shrink the waits.
#[derive(Debug, Clone)]
pub struct LaunchOptions {
    pub handshake_timeout: Duration,
    pub shutdown_wait: Duration,
    /// Where `logs/server.log` lives. `None` → [`default_data_root`].
    pub data_root: Option<PathBuf>,
}

impl Default for LaunchOptions {
    fn default() -> Self {
        Self {
            handshake_timeout: HANDSHAKE_TIMEOUT,
            shutdown_wait: SHUTDOWN_WAIT,
            data_root: None,
        }
    }
}

#[derive(Debug)]
pub enum LaunchError {
    /// The process could not be started at all.
    Spawn { cmd: String, source: std::io::Error },
    /// Log directory/file setup failed.
    Logs(std::io::Error),
    /// The child exited before announcing readiness.
    ExitedEarly {
        status: Option<i32>,
        log_path: PathBuf,
    },
    /// No ready line within the handshake timeout (child was force-killed).
    TimedOut { log_path: PathBuf },
}

impl std::fmt::Display for LaunchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaunchError::Spawn { cmd, source } => {
                write!(f, "could not start the server ({cmd}): {source}")
            }
            LaunchError::Logs(source) => write!(f, "could not prepare server.log: {source}"),
            LaunchError::ExitedEarly { status, log_path } => write!(
                f,
                "the server exited (code {status:?}) before it was ready — see {}",
                log_path.display()
            ),
            LaunchError::TimedOut { log_path } => write!(
                f,
                "the server did not become ready within the timeout — see {}",
                log_path.display()
            ),
        }
    }
}

impl std::error::Error for LaunchError {}

/// How shutdown ended.
#[derive(Debug, PartialEq, Eq)]
pub enum ShutdownOutcome {
    /// The server exited on its own after `POST /control/shutdown`.
    Graceful { exit_code: Option<i32> },
    /// The 10s wait ran out and the process group / Job Object was killed.
    /// Logged as abnormal — this is a fallback, never routine.
    ForceKilled,
}

/// A running server child with its ready info and log plumbing.
pub struct Sidecar {
    child: Child,
    pub ready: ReadyInfo,
    pub log_path: PathBuf,
    log_tx: mpsc::Sender<String>,
    tee_threads: Vec<JoinHandle<()>>,
    writer_thread: Option<JoinHandle<()>>,
    shutdown_wait: Duration,
    #[cfg(windows)]
    job: windows_job::Job,
}

impl std::fmt::Debug for Sidecar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sidecar")
            .field("ready", &self.ready)
            .field("log_path", &self.log_path)
            .finish_non_exhaustive()
    }
}

impl Sidecar {
    /// Spawn the server, tee its output, and wait for the ready line.
    pub fn launch(cmd: &ServerCommand, opts: &LaunchOptions) -> Result<Sidecar, LaunchError> {
        let data_root = opts.data_root.clone().unwrap_or_else(default_data_root);
        let log_path = rotate_logs(&data_root.join("logs")).map_err(LaunchError::Logs)?;
        let log_file = fs::File::create(&log_path).map_err(LaunchError::Logs)?;

        let mut child = spawn_in_own_group(cmd).map_err(|source| LaunchError::Spawn {
            cmd: cmd.argv.join(" "),
            source,
        })?;
        #[cfg(windows)]
        let job = windows_job::Job::assign(&child);

        // One writer thread owns the file; stdout/stderr tees and shell-side
        // events all send lines through the same channel, flushed per line so
        // the log survives a crash.
        let (log_tx, log_rx) = mpsc::channel::<String>();
        let writer_thread = thread::spawn(move || {
            let mut file = log_file;
            for line in log_rx {
                let _ = writeln!(file, "{line}");
                let _ = file.flush();
            }
        });

        let (ready_tx, ready_rx) = mpsc::channel::<StdoutMsg>();
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        let tee_threads = vec![
            spawn_stdout_tee(stdout, log_tx.clone(), ready_tx),
            spawn_stderr_tee(stderr, log_tx.clone()),
        ];

        let mut sidecar = Sidecar {
            child,
            ready: ReadyInfo {
                url: String::new(),
                port: 0,
                pid: 0,
            },
            log_path,
            log_tx,
            tee_threads,
            writer_thread: Some(writer_thread),
            shutdown_wait: opts.shutdown_wait,
            #[cfg(windows)]
            job,
        };

        match ready_rx.recv_timeout(opts.handshake_timeout) {
            Ok(StdoutMsg::Ready(info)) => {
                sidecar.ready = info;
                Ok(sidecar)
            }
            Ok(StdoutMsg::Eof) => {
                // Child closed stdout without announcing: it exited (or is
                // dying) early. Reap it and report with the log location.
                let status = sidecar.child.wait().ok().and_then(|s| s.code());
                let log_path = sidecar.log_path.clone();
                sidecar.finalize_logs();
                Err(LaunchError::ExitedEarly { status, log_path })
            }
            Err(_) => {
                sidecar.log_event("no ready line within the handshake timeout — killing");
                sidecar.force_kill();
                let _ = sidecar.child.wait();
                let log_path = sidecar.log_path.clone();
                sidecar.finalize_logs();
                Err(LaunchError::TimedOut { log_path })
            }
        }
    }

    /// Ordered shutdown per decision 18: POST /control/shutdown, wait up to
    /// `shutdown_wait` for the child to exit, then force-kill the process
    /// group / Job Object as a logged, abnormal fallback.
    pub fn shutdown(mut self) -> ShutdownOutcome {
        match post_shutdown(&self.ready.url) {
            Ok(true) => {}
            Ok(false) => self.log_event("shutdown endpoint did not answer 202"),
            Err(err) => self.log_event(&format!("shutdown request failed: {err}")),
        }

        let deadline = Instant::now() + self.shutdown_wait;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    self.finalize_logs();
                    return ShutdownOutcome::Graceful {
                        exit_code: status.code(),
                    };
                }
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
                _ => break,
            }
        }

        // Abnormal path: also covers a truly stalled HF download thread.
        self.log_event("server did not exit after shutdown request — force-killing its tree");
        eprintln!(
            "loraforge shell: force-killing the server (abnormal — see {})",
            self.log_path.display()
        );
        self.force_kill();
        let _ = self.child.wait();
        self.finalize_logs();
        ShutdownOutcome::ForceKilled
    }

    /// Write a shell-side event into server.log, tagged apart from server output.
    fn log_event(&self, message: &str) {
        let _ = self.log_tx.send(format!("[shell] {message}"));
    }

    fn force_kill(&mut self) {
        #[cfg(unix)]
        {
            // setsid at spawn put the whole server tree in its own process
            // group (pgid == child pid); -pgid signals every member.
            unsafe { libc::kill(-(self.child.id() as i32), libc::SIGKILL) };
        }
        #[cfg(windows)]
        {
            self.job.terminate();
            let _ = self.child.kill(); // belt and braces if job assignment failed
        }
    }

    /// Join tees (they end at child EOF), then the writer (ends when the last
    /// sender drops) so every line is on disk before we return.
    fn finalize_logs(&mut self) {
        for handle in self.tee_threads.drain(..) {
            let _ = handle.join();
        }
        let (dead_tx, _) = mpsc::channel();
        drop(std::mem::replace(&mut self.log_tx, dead_tx));
        if let Some(writer) = self.writer_thread.take() {
            let _ = writer.join();
        }
    }
}

enum StdoutMsg {
    Ready(ReadyInfo),
    Eof,
}

fn spawn_stdout_tee(
    stdout: std::process::ChildStdout,
    log_tx: mpsc::Sender<String>,
    ready_tx: mpsc::Sender<StdoutMsg>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let _ = log_tx.send(line.clone());
            if let Some(info) = parse_ready_line(&line) {
                let _ = ready_tx.send(StdoutMsg::Ready(info));
            }
        }
        let _ = ready_tx.send(StdoutMsg::Eof);
    })
}

fn spawn_stderr_tee(
    stderr: std::process::ChildStderr,
    log_tx: mpsc::Sender<String>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            let _ = log_tx.send(line);
        }
    })
}

fn spawn_in_own_group(cmd: &ServerCommand) -> std::io::Result<Child> {
    let mut command = Command::new(&cmd.argv[0]);
    command
        .args(&cmd.argv[1..])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(cwd) = &cmd.cwd {
        command.current_dir(cwd);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Own session + process group (the doc's `setsid`): a force-kill of
        // -pgid takes uv, python, and any engine grandchildren together.
        unsafe {
            command.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    command.spawn()
}

// ── Shutdown request ─────────────────────────────────────────────────────────

/// Minimal loopback HTTP: POST /control/shutdown, expect 202. A hand-rolled
/// request keeps the crate dependency-free here — plain HTTP/1.1 to
/// 127.0.0.1, no TLS, no redirects, and the only thing read is the status
/// line.
fn post_shutdown(url: &str) -> std::io::Result<bool> {
    let authority = authority_of(url);
    // connect_timeout, not connect: some stacks (WSL2 mirrored networking)
    // let a dead loopback port hang instead of refusing, and an unbounded
    // connect here would eat the whole shutdown-wait budget.
    let addr = authority
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| std::io::Error::other(format!("unresolvable authority '{authority}'")))?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3))?;
    stream.set_read_timeout(Some(Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(Duration::from_secs(3)))?;
    write!(
        stream,
        "POST /control/shutdown HTTP/1.1\r\nHost: {authority}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    let _ = stream.read_to_string(&mut response); // server closes; timeout caps it
    Ok(response
        .lines()
        .next()
        .is_some_and(|status| status.contains(" 202 ") || status.ends_with(" 202")))
}

/// `http://127.0.0.1:8471/x` → `127.0.0.1:8471` (also `[::1]:9000`).
fn authority_of(url: &str) -> &str {
    let rest = url.strip_prefix("http://").unwrap_or(url);
    rest.split('/').next().unwrap_or(rest)
}

// ── Windows Job Object ───────────────────────────────────────────────────────

#[cfg(windows)]
mod windows_job {
    //! Windows has no SIGTERM and `TerminateProcess` alone orphans
    //! grandchildren (decision 13); a Job Object with KILL_ON_JOB_CLOSE
    //! takes the whole tree — including when the shell itself dies, since
    //! closing the handle kills the job.

    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct Job(HANDLE);

    // HANDLE is a raw pointer; the job handle is only touched from &mut self.
    unsafe impl Send for Job {}

    impl Job {
        /// Create a kill-on-close job and put the child in it. Assignment
        /// happens right after spawn — the window in which the child could
        /// outrace us with grandchildren is microscopic and accepted.
        pub fn assign(child: &Child) -> Job {
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if !job.is_null() {
                    let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                    SetInformationJobObject(
                        job,
                        JobObjectExtendedLimitInformation,
                        &info as *const _ as *const std::ffi::c_void,
                        std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                    );
                    AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE);
                }
                Job(job)
            }
        }

        pub fn terminate(&mut self) {
            if !self.0.is_null() {
                unsafe { TerminateJobObject(self.0, 1) };
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CloseHandle(self.0) };
            }
        }
    }
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env-var tests mutate process-global state; serialize them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn ready_line_parses_and_noise_does_not() {
        let info = parse_ready_line(
            r#"LORAFORGE_READY {"url": "http://127.0.0.1:8471", "port": 8471, "pid": 42}"#,
        )
        .expect("well-formed ready line");
        assert_eq!(info.url, "http://127.0.0.1:8471");
        assert_eq!(info.port, 8471);
        assert_eq!(info.pid, 42);

        assert!(parse_ready_line("INFO:     Uvicorn running on http://127.0.0.1:8471").is_none());
        assert!(parse_ready_line("LORAFORGE_READY not json").is_none());
        // Extra keys must not break older shells (forward compatibility).
        assert!(parse_ready_line(
            r#"LORAFORGE_READY {"url": "http://x:1", "port": 1, "pid": 2, "later": true}"#
        )
        .is_some());
    }

    #[test]
    fn command_resolution_order() {
        let _guard = ENV_LOCK.lock().unwrap();
        env::remove_var("LORAFORGE_SERVER_CMD");

        // env override wins and splits shell-style (quoted path survives)
        env::set_var(
            "LORAFORGE_SERVER_CMD",
            r#""/opt/my tools/uv" run loraforge serve"#,
        );
        let cmd = resolve_command(None).unwrap();
        assert_eq!(cmd.argv[0], "/opt/my tools/uv");
        assert_eq!(cmd.argv.last().unwrap(), "serve");
        assert!(cmd.cwd.is_none());

        env::set_var("LORAFORGE_SERVER_CMD", "");
        assert!(resolve_command(None).is_err());
        env::remove_var("LORAFORGE_SERVER_CMD");

        // no packaged server yet (step B stub) → dev fallback with repo cwd
        let cmd = resolve_command(Some(Path::new("/repo"))).unwrap();
        assert_eq!(cmd.argv, ["uv", "run", "loraforge", "serve"]);
        assert_eq!(cmd.cwd.as_deref(), Some(Path::new("/repo")));

        // …and a clear error when there is no repo root either
        assert!(resolve_command(None).is_err());
    }

    #[test]
    fn log_rotation_keeps_three() {
        let dir = tempfile::tempdir().unwrap();
        let logs = dir.path();
        for run in 0..5 {
            let path = rotate_logs(logs).unwrap();
            fs::write(&path, format!("run {run}\n")).unwrap();
        }
        let entries: Vec<String> = {
            let mut names: Vec<String> = fs::read_dir(logs)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            names.sort();
            names
        };
        assert_eq!(entries, ["server.log", "server.log.1", "server.log.2"]);
        // newest is the current file, oldest surviving is two runs back
        assert_eq!(
            fs::read_to_string(logs.join("server.log")).unwrap(),
            "run 4\n"
        );
        assert_eq!(
            fs::read_to_string(logs.join("server.log.2")).unwrap(),
            "run 2\n"
        );
    }

    #[test]
    fn tail_lines_returns_last_n() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("server.log");
        let body: Vec<String> = (0..80).map(|i| format!("line {i}")).collect();
        fs::write(&path, body.join("\n")).unwrap();
        let tail = tail_lines(&path, 50);
        assert_eq!(tail.len(), 50);
        assert_eq!(tail.first().unwrap(), "line 30");
        assert_eq!(tail.last().unwrap(), "line 79");
        assert!(tail_lines(&dir.path().join("missing.log"), 50).is_empty());
    }

    #[test]
    fn authority_extraction() {
        assert_eq!(authority_of("http://127.0.0.1:8471"), "127.0.0.1:8471");
        assert_eq!(authority_of("http://127.0.0.1:8471/x/y"), "127.0.0.1:8471");
        assert_eq!(authority_of("http://[::1]:9000"), "[::1]:9000");
    }
}
