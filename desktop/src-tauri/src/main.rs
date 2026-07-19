#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! LoRAForge desktop shell — a thin Tauri window over the local server.
//!
//! The shell orchestrates processes and windows, nothing else (contract:
//! docs/design/tauri-shell.md). It spawns `loraforge serve` via the sidecar
//! crate, waits for the LORAFORGE_READY handshake, points the webview at the
//! announced URL, and on window close runs the ordered shutdown — after a
//! native confirm dialog if training is running.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use loraforge_sidecar::{
    loopback_get, resolve_command, tail_lines, LaunchError, LaunchOptions, Sidecar,
};
use tauri::{AppHandle, Manager, RunEvent, WindowEvent};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

/// Job states the server considers finished (mirrors jobs/runner.py); a job
/// in any other state means training is queued or running. The /jobs list
/// keeps finished runs, so "list non-empty" alone would nag forever.
const TERMINAL_JOB_STATES: [&str; 4] = ["completed", "completed_early", "failed", "cancelled"];

struct ShellState {
    sidecar: Mutex<Option<Sidecar>>,
    /// Latched when a close is committed; further close clicks are ignored
    /// while the (up to ~28s) ordered shutdown runs. Reset on decline.
    closing: AtomicBool,
}

fn main() {
    tauri::Builder::default()
        // Registered first, per the plugin's docs: a second launch focuses
        // the existing window and exits — no port fights, no dup sidecars.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .manage(ShellState {
            sidecar: Mutex::new(None),
            closing: AtomicBool::new(false),
        })
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || start_server(&handle));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                // Never let the window die on its own — the server must be
                // shut down first, and the user asked if training is running.
                api.prevent_close();
                let state = window.state::<ShellState>();
                if state.closing.swap(true, Ordering::SeqCst) {
                    return; // a close is already in flight
                }
                let app = window.app_handle().clone();
                std::thread::spawn(move || confirm_then_shutdown(&app));
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to build the tauri app")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // Exit paths that skip the close flow (SIGTERM, exit(0)):
                // never leave the server running. No-op if already shut down.
                shutdown_sidecar(app);
            }
        });
}

/// Spawn the sidecar and navigate to it, or land on the error page. Runs on
/// its own thread so the splash renders immediately.
fn start_server(app: &AppHandle) {
    let cmd = match resolve_command(dev_repo_root().as_deref()) {
        Ok(cmd) => cmd,
        Err(err) => return show_error(app, &err.to_string(), None),
    };
    match Sidecar::launch(&cmd, &LaunchOptions::default()) {
        Ok(sidecar) => {
            let url = sidecar.ready.url.clone();
            *app.state::<ShellState>().sidecar.lock().unwrap() = Some(sidecar);
            navigate(app, &url);
        }
        Err(err) => {
            let log_path = match &err {
                LaunchError::ExitedEarly { log_path, .. } | LaunchError::TimedOut { log_path } => {
                    Some(log_path.clone())
                }
                _ => None,
            };
            show_error(app, &err.to_string(), log_path);
        }
    }
}

/// Dev fallback needs the repo root (`uv run loraforge serve` cwd): walk up
/// from the current dir looking for pyproject.toml. Packaged installs never
/// get here (LORAFORGE_SERVER_CMD or the step-B packaged location resolve
/// first).
fn dev_repo_root() -> Option<PathBuf> {
    let start = std::env::current_dir().ok()?;
    start
        .ancestors()
        .find(|dir| dir.join("pyproject.toml").is_file())
        .map(Into::into)
}

fn confirm_then_shutdown(app: &AppHandle) {
    let server_url = {
        let state = app.state::<ShellState>();
        let guard = state.sidecar.lock().unwrap();
        guard.as_ref().map(|sidecar| sidecar.ready.url.clone())
    };
    if server_url.as_deref().is_some_and(training_active) {
        let confirmed = app
            .dialog()
            .message(
                "Training is running — closing LoRAForge will stop it. \
                 Progress up to the last checkpoint is kept.",
            )
            .title("Stop training?")
            .kind(MessageDialogKind::Warning)
            .buttons(MessageDialogButtons::OkCancelCustom(
                "Stop training and close".into(),
                "Keep training".into(),
            ))
            .blocking_show();
        if !confirmed {
            let state = app.state::<ShellState>();
            state.closing.store(false, Ordering::SeqCst);
            return; // window stays, training continues
        }
    }
    shutdown_sidecar(app); // ordered shutdown; job cancel is inside it
    app.exit(0);
}

/// GET /jobs and answer "is anything still queued or running". Errors mean
/// "no" — a dead server has nothing to protect with a dialog.
fn training_active(base_url: &str) -> bool {
    let Ok(body) = loopback_get(base_url, "/jobs") else {
        return false;
    };
    let Ok(jobs) = serde_json::from_str::<serde_json::Value>(&body) else {
        return false;
    };
    jobs.as_array().is_some_and(|list| {
        list.iter().any(|job| {
            job.get("state")
                .and_then(|state| state.as_str())
                .is_some_and(|state| !TERMINAL_JOB_STATES.contains(&state))
        })
    })
}

fn shutdown_sidecar(app: &AppHandle) {
    let taken = app.state::<ShellState>().sidecar.lock().unwrap().take();
    if let Some(sidecar) = taken {
        sidecar.shutdown();
    }
}

/// Point the webview at the running server (same origin the browser uses).
fn navigate(app: &AppHandle, url: &str) {
    eval_in_main(app, &format!("window.location.replace({})", js_string(url)));
}

/// Swap the splash for the bundled error page: message + log path + the last
/// ~50 log lines, all in the query string so the page stays static.
fn show_error(app: &AppHandle, message: &str, log_path: Option<PathBuf>) {
    let tail = log_path
        .as_deref()
        .map(|path| tail_lines(path, 50).join("\n"))
        .unwrap_or_default();
    let log = log_path
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let query = format!(
        "error.html?message={}&log={}&tail={}",
        urlencode(message),
        urlencode(&log),
        urlencode(&tail)
    );
    eval_in_main(
        app,
        &format!("window.location.replace({})", js_string(&query)),
    );
}

fn eval_in_main(app: &AppHandle, script: &str) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.eval(script);
    }
}

/// A JS string literal (JSON string is valid JS).
fn js_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".into())
}

/// Percent-encode for a query value (RFC 3986 unreserved kept).
fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}
