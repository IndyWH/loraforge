"""Server entrypoint: loopback-only binding policy and real dependency wiring."""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING

from loraforge.datasets.library import DatasetLibrary
from loraforge.downloader import ModelDownloader, adapter_paths
from loraforge.engines.bootstrap import KOHYA, default_data_root, engine_paths
from loraforge.engines.kohya import KohyaAdapter
from loraforge.jobs.runner import JobRunner
from loraforge.probe import probe
from loraforge.server.app import ServerDeps, create_app
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

    return ServerDeps(
        runner=JobRunner(adapter, data_root / "jobs"),
        downloads=DownloadManager(downloader, on_complete=wire_download),
        datasets=DatasetLibrary(data_root / "datasets"),
        recipes_dir=data_root / "recipes",
        jobs_root=data_root / "jobs",
        probe=probe,
    )


def serve(
    host: str = "127.0.0.1",
    port: int = 8471,
    allow_remote: bool = False,
    deps: ServerDeps | None = None,
) -> None:
    ensure_local_bind(host, allow_remote)
    import uvicorn

    # Under --allow-remote the Host/Origin checks come off too: the operator
    # fronts their own auth/proxy, and remote Hosts are then legitimate.
    app = create_app(deps or build_default_deps(), local_only=not allow_remote)
    uvicorn.run(app, host=host, port=port)
