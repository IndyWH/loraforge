"""Downloader tests: cache reuse, disk preflight, gated-model UX, adapter wiring.

The hub is faked in-memory (no network): snapshot_download's cache contract —
local_files_only raises LocalEntryNotFoundError on a miss, a network fetch
materializes the requested patterns — is reproduced against tmp_path.
"""

from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

from loraforge.downloader import (
    DownloadError,
    DownloadEvent,
    DownloadState,
    ModelDownloader,
    adapter_paths,
    model_sources,
)
from loraforge.engines.kohya import MODEL_SPECS, KohyaAdapter
from loraforge.recipes.schema import Recipe

# ── Fake hub ─────────────────────────────────────────────────────────────────


class FakeHub:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.remote: dict[str, dict[str, str]] = {}  # repo → {filename: content}
        self.gated_refusals: set[str] = set()  # repos that 401 (no accepted license/token)
        self.network_calls: list[str] = []

    def _snap(self, repo_id: str) -> Path:
        return self.root / repo_id.replace("/", "--")

    def snapshot_download(
        self,
        *,
        repo_id: str,
        allow_patterns: list[str] | None = None,
        local_files_only: bool = False,
        cache_dir: Path | None = None,
        **_: object,
    ) -> str:
        snap = self._snap(repo_id)
        patterns = list(allow_patterns or [])
        if local_files_only:
            if snap.is_dir() and all((snap / p).exists() for p in patterns):
                return str(snap)
            raise LocalEntryNotFoundError(f"{repo_id} not found in local cache")
        self.network_calls.append(repo_id)
        if repo_id in self.gated_refusals:
            raise GatedRepoError(
                f"401 Client Error: Unauthorized for url .../{repo_id}",
                response=httpx.Response(
                    401, request=httpx.Request("GET", f"https://huggingface.co/{repo_id}")
                ),
            )
        snap.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            (snap / pattern).write_text(self.remote[repo_id][pattern])
        return str(snap)


def make_downloader(tmp_path: Path, free_gb: float = 999.0) -> tuple[ModelDownloader, FakeHub]:
    hub = FakeHub(tmp_path / "hub")
    for source in model_sources().values():  # seed the fake remote from the real matrix data
        hub.remote.setdefault(source.repo, {})[source.file] = source.model_key
        for name, asset in source.assets.items():
            hub.remote.setdefault(asset.repo, {})[asset.file] = name
    downloader = ModelDownloader(
        snapshot=hub.snapshot_download, free_gb=lambda path: free_gb, cache_dir=tmp_path / "hub"
    )
    return downloader, hub


def make_recipe(tmp_path: Path, model: str = "sdxl") -> Recipe:
    return Recipe.model_validate(
        {
            "name": "test-run",
            "model": model,
            "dataset": {"path": str(tmp_path / "images")},
            "train": {"sample_every_steps": 0},
        }
    )


# ── Adapter wiring ───────────────────────────────────────────────────────────


def test_download_produces_paths_the_adapter_accepts(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path)
    result = downloader.download("sdxl")
    assert result.model_path.exists()
    assert result.asset_paths == {}  # sdxl is a single checkpoint

    model_paths, asset_paths = adapter_paths([result])
    adapter = KohyaAdapter(
        sd_scripts_dir=tmp_path / "sd-scripts",
        env_dir=tmp_path / "env",
        model_paths=model_paths,
        asset_paths=asset_paths,
    )
    plan = adapter.compile(make_recipe(tmp_path), tmp_path)  # would raise if paths were missing
    assert str(result.model_path) in " ".join(plan.argv)


def test_flux_assets_downloaded_and_wired(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path)
    result = downloader.download("flux_dev")
    assert set(result.asset_paths) == {"clip_l", "t5xxl", "ae"}
    assert all(path.exists() for path in result.asset_paths.values())

    model_paths, asset_paths = adapter_paths([result])
    adapter = KohyaAdapter(
        sd_scripts_dir=tmp_path / "sd-scripts",
        env_dir=tmp_path / "env",
        model_paths=model_paths,
        asset_paths=asset_paths,
    )
    argv = " ".join(adapter.compile(make_recipe(tmp_path, model="flux_dev"), tmp_path).argv)
    assert f"--clip_l={result.asset_paths['clip_l']}" in argv
    assert f"--ae={result.asset_paths['ae']}" in argv


def test_matrix_assets_cover_kohya_requirements() -> None:
    # data drift guard: every asset the kohya adapter demands has a source
    sources = model_sources()
    for key, spec in MODEL_SPECS.items():
        if spec.required_assets:
            assert set(spec.required_assets) <= set(sources[key].assets), key


# ── Cache reuse ──────────────────────────────────────────────────────────────


def test_second_download_reuses_cache_without_network(tmp_path: Path) -> None:
    downloader, hub = make_downloader(tmp_path)
    first = downloader.download("flux_dev")
    calls_after_first = len(hub.network_calls)
    second = downloader.download("flux_dev")
    assert len(hub.network_calls) == calls_after_first  # nothing re-downloaded
    assert second.model_path == first.model_path
    assert second.asset_paths == first.asset_paths


def test_files_cached_by_other_tools_are_reused(tmp_path: Path) -> None:
    downloader, hub = make_downloader(tmp_path)
    # simulate ComfyUI having already pulled the flux text encoders into the shared cache
    snap = hub.root / "comfyanonymous--flux_text_encoders"
    snap.mkdir(parents=True)
    (snap / "clip_l.safetensors").write_text("comfyui")
    (snap / "t5xxl_fp16.safetensors").write_text("comfyui")

    downloader.download("flux_dev")
    assert "comfyanonymous/flux_text_encoders" not in hub.network_calls


# ── Disk preflight ───────────────────────────────────────────────────────────


def test_disk_preflight_refuses_before_any_network_call(tmp_path: Path) -> None:
    downloader, hub = make_downloader(tmp_path, free_gb=3.0)
    events: list[DownloadEvent] = []
    with pytest.raises(DownloadError, match="disk space"):
        downloader.download("flux_dev", on_event=events.append)
    assert hub.network_calls == []  # refused before downloading a single byte
    assert events[-1].state is DownloadState.FAILED  # stream still ends terminally
    assert "35GB" in events[-1].message and "3GB" in events[-1].message
    assert "Free up space" in events[-1].message


# ── Gated models ─────────────────────────────────────────────────────────────


def test_gated_401_names_the_license_url_and_the_token_step(tmp_path: Path) -> None:
    downloader, hub = make_downloader(tmp_path)
    hub.gated_refusals.add("black-forest-labs/FLUX.1-dev")
    events: list[DownloadEvent] = []
    with pytest.raises(DownloadError) as excinfo:
        downloader.download("flux_dev", on_event=events.append)

    message = str(excinfo.value)
    assert "https://huggingface.co/black-forest-labs/FLUX.1-dev" in message  # exact license URL
    assert "hf auth login" in message  # the one-line token step
    assert "never sees or saves" in message  # we do not store tokens
    assert events[-1].state is DownloadState.FAILED
    assert events[-1].message == message


# ── Events and errors ────────────────────────────────────────────────────────


def test_events_check_then_download_then_terminal(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path)
    events: list[DownloadEvent] = []
    result = downloader.download("flux_dev", on_event=events.append)

    assert events[0].state is DownloadState.CHECKING
    downloading = [e.item for e in events if e.state is DownloadState.DOWNLOADING]
    assert downloading[0] == "base"
    assert set(downloading) == {"base", "clip_l", "t5xxl", "ae"}
    assert events[-1].is_terminal and events[-1].state is DownloadState.COMPLETED
    assert events[-1].result == result


def test_unknown_model_speaks_human(tmp_path: Path) -> None:
    downloader, _ = make_downloader(tmp_path)
    with pytest.raises(DownloadError, match="known models"):
        downloader.download("wan_video")
