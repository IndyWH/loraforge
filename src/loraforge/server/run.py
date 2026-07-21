"""Server entrypoint: loopback-only binding policy and real dependency wiring.

Also the desktop-shell contract (docs/design/tauri-shell.md): the Tauri shell
spawns ``loraforge serve``, reads stdout line-by-line until it sees the
``LORAFORGE_READY`` prefix (a cheap starts-with — no JSON parsing of every
line), then parses the JSON payload after it and points its webview at the
announced URL. The socket is bound and listening *before* the line is
printed, so the shell can connect the moment it sees it.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

from loraforge.datasets.library import DatasetLibrary
from loraforge.downloader import ModelDownloader, adapter_paths
from loraforge.engines.bootstrap import KOHYA, default_data_root, engine_paths
from loraforge.engines.kohya import KohyaAdapter
from loraforge.jobs.runner import JobRunner
from loraforge.probe import probe
from loraforge.server.app import ServerDeps, create_app, ui_build_stamp
from loraforge.server.downloads import DownloadManager

if TYPE_CHECKING:
    from loraforge.downloader import DownloadedModel


def ensure_local_bind(host: str, allow_remote: bool = False) -> None:
    """Refuse to bind beyond loopback unless explicitly overridden.

    This server is an unauthenticated control plane for GPU training jobs:
    exposed on a network interface, anyone on the LAN could start jobs, read
    local paths, and burn the GPU. ``--allow-remote`` is the deliberate,
    typed-out escape hatch for people who put their own auth in front.
    """
    if allow_remote:
        return
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host == "localhost"
    if not is_loopback:
        raise SystemExit(
            f"refusing to bind to '{host}': LoRAForge's server has no authentication "
            "and is meant for this machine only. Use 127.0.0.1 (the default), or pass "
            "--allow-remote if you really intend to expose it."
        )


READY_PREFIX = "LORAFORGE_READY "


def pick_port(host: str = "127.0.0.1", preferred: int = 8471) -> socket.socket:
    """Bind and listen on the preferred port, or an OS-assigned free one.

    Returns the listening socket (hand it to uvicorn via ``sockets=[...]``).
    Binding up front — instead of probing and binding later — means there is
    no window where the announced port can be stolen, and the shell's first
    connection simply queues in the backlog until uvicorn starts accepting.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    if os.name != "nt":  # on Windows SO_REUSEADDR lets another bind steal a live port
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, preferred))
    except OSError:  # in use (or a port Windows reserves) — fall back to any free port
        sock.close()
        sock = socket.socket(family, socket.SOCK_STREAM)
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
    sock.listen(128)
    return sock


def ready_line(host: str, port: int) -> str:
    """The single stdout line the desktop shell waits for.

    ``LORAFORGE_READY`` prefix as the ready marker (the shell keys on a
    starts-with, never uvicorn's log banner), JSON payload for structure.
    The pid lets it watch for a server that dies after announcing; the url
    always carries the real (possibly fallback) port.
    """
    display_host = f"[{host}]" if ":" in host else host
    payload = {"url": f"http://{display_host}:{port}", "port": port, "pid": os.getpid()}
    return READY_PREFIX + json.dumps(payload)


def build_default_deps() -> ServerDeps:
    """Wire the real probe/downloader/adapter/runner under the user data dir."""
    data_root = default_data_root()
    engine = engine_paths(KOHYA, data_root / "engines")
    downloader = ModelDownloader()
    already_local = [d for d in map(downloader.peek, downloader.sources) if d is not None]
    model_paths, asset_paths = adapter_paths(already_local)
    adapter = KohyaAdapter(
        sd_scripts_dir=engine.checkout,
        env_dir=engine.env,
        model_paths=model_paths,
        asset_paths=asset_paths,
    )

    def wire_download(result: DownloadedModel) -> None:  # new weights → visible to compile()
        adapter.model_paths[result.model_key] = result.model_path
        if result.asset_paths:
            adapter.asset_paths[result.model_key] = dict(result.asset_paths)

    # Dev checkout: serve the built web UI if it exists (repo_root/ui/dist).
    # The packaged desktop build will ship the bundle differently (Tauri).
    ui_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    return ServerDeps(
        runner=JobRunner(adapter, data_root / "jobs"),
        downloads=DownloadManager(downloader, on_complete=wire_download),
        datasets=DatasetLibrary(data_root / "datasets"),
        recipes_dir=data_root / "recipes",
        jobs_root=data_root / "jobs",
        probe=probe,
        ui_dist=ui_dist if ui_dist.is_dir() else None,
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 8471,
    allow_remote: bool = False,
    deps: ServerDeps | None = None,
    force_presets: dict[str, str] | None = None,
) -> None:
    ensure_local_bind(host, allow_remote)
    import uvicorn

    sock = pick_port(host, port)
    actual_port = sock.getsockname()[1]
    if actual_port != port:
        print(f"port {port} is in use — using {actual_port} instead", flush=True)

    # Under --allow-remote the Host/Origin checks come off too: the operator
    # fronts their own auth/proxy, and remote Hosts are then legitimate.
    deps = deps or build_default_deps()
    if (stamp := ui_build_stamp(deps.ui_dist)) is not None:
        print(f"ui bundle: {stamp.git} built {stamp.built_at}", flush=True)
    elif deps.ui_dist is not None:
        print("ui bundle: no build stamp (predates stamping — rebuild with `npm run build`)",
              flush=True)
    if force_presets:
        deps.force_presets = force_presets
        for model_key, preset in force_presets.items():
            print(f"MEASUREMENT MODE: {model_key} preset forced to '{preset}'", flush=True)
    app = create_app(deps, local_only=not allow_remote)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=actual_port))

    def request_shutdown() -> None:  # POST /control/shutdown → graceful uvicorn exit
        server.should_exit = True

    deps.request_shutdown = request_shutdown
    # Unconditional: one extra line in a terminal, and the desktop shell
    # depends on it. The socket already accepts, so announcing here is safe.
    print(ready_line(host, actual_port), flush=True)
    server.run(sockets=[sock])
