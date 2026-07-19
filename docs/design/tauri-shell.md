# Design brief — Phase: Tauri shell (step A)

Status: agreed 2026-07-19 (design session); updated same day, twice, to
as-built after the Python side landed and was reviewed. THIS FILE IS THE
CONTRACT — replace any older copy wholesale; an earlier draft briefly lived
in the repo and caused a reconciliation round. Scope: desktop shell + sidecar
lifecycle + packaging config. **Not** in this step: first-run uv/engine
bootstrap UX (that is step B, next session). Assume a working dev checkout
(env exists, `uv run loraforge serve` works, `ui/dist` is built).

## Shape

One new top-level directory, `desktop/`, containing a Tauri v2 app. The
shell is deliberately dumb — the analog of the server rule "routes
translate, never decide": **the shell orchestrates processes and windows;
every decision that can live in Python lives in Python**, where it is
testable by the existing fast suite. Rust code should stay small enough to
review in one sitting.

The webview does NOT bundle the React app. It points at
`http://127.0.0.1:<port>` and the FastAPI server serves `ui/dist` exactly as
it does today. One origin, one code path, WS works unchanged, and browser
use on Linux stays byte-identical to the desktop experience (decision 11).
Tauri's `frontendDist` holds only a tiny static splash/error page.

## Sidecar lifecycle

Spawn. The shell resolves the server command in order:
1. `LORAFORGE_SERVER_CMD` env var (dev/test override, split shell-style),
2. packaged location (step B — leave a clearly marked stub),
3. dev fallback: `uv run loraforge serve` with cwd = repo root.
On Linux spawn in its own process group (`setsid`); on Windows put the child
in a Job Object (or equivalent) so a force-kill can take the whole tree —
Windows has no SIGTERM (decision 13) and TerminateProcess alone orphans
grandchildren.

Ready handshake (as built). `loraforge serve` unconditionally prints one
line to stdout once its socket is listening (harmless in a terminal, and
the shell depends on it — no flag to forget):
`LORAFORGE_READY {"url": "http://127.0.0.1:8471", "port": 8471, "pid": 12345}`
The shell reads child stdout line-by-line until a line starts with
`LORAFORGE_READY`, parses the JSON after the prefix, and navigates the
webview to `url` (pid is for watching a server that dies after announcing).
The prefix is the ready marker — cheap starts_with in Rust, no need to
attempt JSON parsing on every line. Do not parse uvicorn's log banner —
that's someone else's format. Everything else on stdout/stderr is tee'd to
`<data_root>/logs/server.log` (rotate: keep last 3), because that file is
tomorrow's bug report attachment alongside `loraforge diagnose`.

Port policy (as built). `pick_port(preferred=8471)` in `server/run.py`
binds and listens on the preferred port, falling back to an OS-assigned
ephemeral port if taken, and returns the *listening socket*, which uvicorn
adopts via `sockets=[sock]`. No close-and-rebind, no TOCTOU race, and
connections queue in the backlog from the moment the announce prints.
SO_REUSEADDR is deliberately skipped on Windows (port-stealing semantics).
The announce line always carries the real port, so the shell never assumes.

Timeout. If no ready line within 30s or the child exits early, show the
bundled error page with the last ~50 lines of server.log and a "copy log
path" affordance. No silent blank window.

Shutdown (as built). `POST /control/shutdown` (loopback-only like
everything else — LocalRequestsOnly already blocks cross-origin browser
writes) returns 202 `{"stopping": true}` immediately and runs the ordered
sequence in a background task: `runner.cancel_all(keep=True)` (checkpointed
stop for a running job — matching the close dialog's promise — plain cancel
for queued ones) → `downloads.stop_all()` (cancels the wrapping task and
pushes a terminal event; the HF hub thread can't be interrupted mid-file,
but partials resume from the cache next launch) → flip uvicorn's
`should_exit` via the `ServerDeps.request_shutdown` hook wired in `serve()`.
The ordering is load-bearing, not tidiness: uvicorn's graceful shutdown
waits for open connections, and job/download WebSocket streams only close
on a terminal event — flipping the exit flag first can deadlock exit with
a WS attached. With no hook wired (tests/embedding) the endpoint 503s with
instructions. The shell's exit sequence is: call shutdown → wait up to 10s
for child exit → force-kill the process group / Job Object as fallback.
The fallback should be logged as abnormal, not routine (it also covers a
truly stalled HF thread).

Close-while-training (product decision, settled): intercept the window
close event; ask the server whether a job is running (existing `/jobs`
list). If yes, native confirm dialog: "Training is running — closing
LoRAForge will stop it. Progress up to the last checkpoint is kept."
Confirm → normal shutdown sequence (job cancel is inside it). Decline →
window stays. If no job is running, close immediately through the same
shutdown sequence. No tray, no background mode in this step.

Single instance. Use Tauri's single-instance plugin: a second launch
focuses the existing window and exits. This sidesteps port fights and
duplicate sidecars entirely.

## What changes in Python (all testable, keep the suite <1s)

- `server/run.py`: `pick_port()`, ready-announce line, and wiring so
  `serve()` prints the announce after startup (uvicorn lifespan or server
  startup hook — CC's choice, but the announce must fire only once the
  socket accepts).
- `server/app.py`: `POST /control/shutdown` as above. Test: submitted fake
  job gets cancelled, response 202, server's exit flag set. Reuse the
  existing ServerDeps fakes.
- `cli.py`: no new flags needed — announce is unconditional (harmless in a
  terminal, one line).

## What lives in Rust (keep it thin)

Spawn/handshake/tee, close interception + confirm dialog, shutdown-then-
force-kill, single instance, splash/error pages. No business logic, no
job state interpretation beyond "is the jobs list non-empty".

## Packaging & CI

- `tauri build` producing an `.exe`/NSIS installer on Windows and `.deb` +
  AppImage on Linux — as CI artifacts only, no release process yet.
- CI: add a `desktop` job to the existing matrix (ubuntu + windows):
  `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build`, and
  `tauri build` on tags/main only if build minutes are a concern.
- Linux build deps (webkit2gtk 4.1, libappindicator, etc.) documented in
  CLAUDE.md dev commands; they're needed in WSL for `tauri dev` too.
- `.gitignore`: `desktop/target/`, `desktop/gen/`.
- Pin the Tauri crate minor version; single-maintainer-adjacent ecosystem
  caution from decision 14 applies in spirit.

## Acceptance (review checklist)

1. `uv run pytest` still <1s, new tests for pick_port + shutdown + announce.
2. `tauri dev` in WSL: window opens, splash → UI, jobs page works over WS.
3. Close during a (fake or real) running job → dialog → confirm → process
   tree fully gone (verify with `ps`/Task Manager: no orphaned python).
4. Kill -9 the shell mid-run → orphaned server is the known limitation of
   step A; note it in the doc (revisit with tray/reattach later if wanted).
5. Second launch focuses the first window.
6. Port 8471 occupied by `python -m http.server 8471` → app still starts,
   webview lands on the ephemeral port.
7. CI green on both OSes including the new desktop job.

## decisions.md — append these entries

16. **Desktop shell is a dumb orchestrator; the webview points at the
    local server.** The Tauri app bundles no UI and holds no logic beyond
    process/window lifecycle; it navigates to the same localhost origin the
    browser uses. One origin, one code path, testable in Python. Rejected:
    bundling the React app into Tauri and calling the API cross-origin —
    two serving paths to keep honest for zero user benefit.

17. **Closing the window stops training, with confirmation.** Close during
    a run asks first, then cancels through the runner's checkpointed-stop
    path and shuts the server down. Rejected for now: tray/background mode
    (adds Windows background-process expectations and reattach complexity
    before anyone has asked for it) and silent kill (losing a 3-hour run to
    a misclick contradicts the graceful-degradation brand). Revisit tray
    when real users ask for close-and-keep-training.

18. **Sidecar contract: stdout-announced readiness, endpoint-driven
    ordered shutdown.** `loraforge serve` unconditionally prints a single
    prefix-tagged ready line — `LORAFORGE_READY {"url", "port", "pid"}` —
    once its socket is listening; the shell matches the prefix, never
    parses uvicorn logs, and never assumes the port. `pick_port()` returns
    a listening socket that uvicorn adopts (`sockets=[sock]`), so the
    handshake is race-free. `POST /control/shutdown` → 202, then cancel
    all jobs (keep=True) → stop downloads → flip the exit flag, in that
    order, because WS streams only close on terminal events and uvicorn's
    graceful shutdown waits for open connections. Shell-side force-kill of
    the process group/Job Object is only a logged fallback. Preferred port
    8471, ephemeral fallback.
