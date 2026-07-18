"""HF model downloader — get the weights a recipe needs, reuse what exists.

Downloads go through ``huggingface_hub.snapshot_download`` into the standard
shared HF cache (``HF_HOME`` / ``HF_HUB_CACHE`` respected), so models already
pulled by ComfyUI or any other hub-aware tool are reused, not re-downloaded.
An override directory is supported but the ecosystem cache is the default.

What to download is DATA: each model's ``source:`` block in the capability
matrix names its base repo/file and any component assets (FLUX's clip_l,
t5xxl, ae) with their own repos and filenames. On success the downloader
hands back exactly the ``model_paths`` / ``asset_paths`` dicts the
``KohyaAdapter`` constructor expects — see ``adapter_paths()``.

Rules honored here:
- Disk preflight before any network call: the probe's free-disk measure of
  the cache directory is compared against the model's ``download_gb``; a
  refusal is a human message, not an exception 20GB in.
- Gated models: LoRAForge never stores tokens. Auth is delegated entirely to
  huggingface_hub's own login/keystore (``hf auth login``); a 401/403 fails
  with the exact license-acceptance URL and a one-line token explanation.
- Progress is the same transport-free event pattern as the job runner: typed
  events, terminal event guaranteed (``completed`` or ``failed``). The
  downloader is synchronous (hub calls block); the server layer wraps it in
  a thread and forwards events to its own streams.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError

from loraforge.capability.resolver import load_matrix

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# ── Sources (data, loaded from the capability matrix) ────────────────────────


@dataclass(frozen=True)
class AssetSource:
    repo: str
    file: str


@dataclass(frozen=True)
class ModelSource:
    model_key: str
    display_name: str
    repo: str
    file: str  # single-file checkpoint within the repo
    download_gb: float
    gated: bool = False
    assets: dict[str, AssetSource] = field(default_factory=dict)


def model_sources(matrix: dict[str, Any] | None = None) -> dict[str, ModelSource]:
    """Parse every model's ``source:`` block out of the capability matrix."""
    matrix = matrix or load_matrix()
    sources: dict[str, ModelSource] = {}
    for key, spec in matrix["models"].items():
        src = spec.get("source")
        if src is None:
            continue
        sources[key] = ModelSource(
            model_key=key,
            display_name=spec["display_name"],
            repo=src["repo"],
            file=src["file"],
            download_gb=float(src["download_gb"]),
            gated=bool(src.get("gated", False)),
            assets={
                name: AssetSource(repo=a["repo"], file=a["file"])
                for name, a in (src.get("assets") or {}).items()
            },
        )
    return sources


# ── Events and results ───────────────────────────────────────────────────────


class DownloadState(StrEnum):
    CHECKING = "checking"  # cache probe + disk preflight
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATES = frozenset({DownloadState.COMPLETED, DownloadState.FAILED})


@dataclass(frozen=True)
class DownloadedModel:
    """Everything the engine adapter needs to reference these weights."""

    model_key: str
    model_path: Path
    asset_paths: dict[str, Path]


@dataclass(frozen=True)
class DownloadEvent:
    model_key: str
    state: DownloadState
    item: str | None = None  # "base" or an asset name, while downloading
    message: str | None = None
    result: DownloadedModel | None = None  # set on the completed event

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES


class DownloadError(RuntimeError):
    """Download failed; the message is already human-readable."""


def adapter_paths(
    downloads: Iterable[DownloadedModel],
) -> tuple[dict[str, Path], dict[str, dict[str, Path]]]:
    """Fold results into the (model_paths, asset_paths) KohyaAdapter takes."""
    downloads = list(downloads)
    model_paths = {d.model_key: d.model_path for d in downloads}
    asset_paths = {d.model_key: dict(d.asset_paths) for d in downloads if d.asset_paths}
    return model_paths, asset_paths


# ── Downloader ───────────────────────────────────────────────────────────────


def _disk_free_gb(path: Path) -> float:
    """Free space on the filesystem holding ``path`` (same measure the probe uses)."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free / (1024**3)


class _Emitter:
    def __init__(self, model_key: str, on_event: Callable[[DownloadEvent], None] | None) -> None:
        self.model_key = model_key
        self.on_event = on_event

    def __call__(self, state: DownloadState, **kwargs: Any) -> None:
        if self.on_event is not None:
            self.on_event(DownloadEvent(model_key=self.model_key, state=state, **kwargs))

    def fail(self, message: str) -> DownloadError:
        """Emit the terminal failed event and hand back the error to raise."""
        self(DownloadState.FAILED, message=message)
        return DownloadError(message)


class ModelDownloader:
    """Fetch a model's base weights + component assets. See module docstring."""

    def __init__(
        self,
        sources: dict[str, ModelSource] | None = None,
        cache_dir: Path | None = None,  # None → the shared HF cache (the default on purpose)
        snapshot: Callable[..., str] | None = None,
        free_gb: Callable[[Path], float] | None = None,
    ) -> None:
        self.sources = sources or model_sources()
        self.cache_dir = cache_dir
        self._snapshot = snapshot or snapshot_download
        self._free_gb = free_gb or _disk_free_gb

    def download(
        self, model_key: str, on_event: Callable[[DownloadEvent], None] | None = None
    ) -> DownloadedModel:
        """Ensure all files for ``model_key`` exist locally; return their paths.

        Idempotent: anything already in the cache is reused untouched.
        Raises DownloadError (after emitting a terminal failed event) with a
        human message on unknown model, full disk, auth, or network problems.
        """
        emit = _Emitter(model_key, on_event)
        source = self.sources.get(model_key)
        if source is None:
            raise emit.fail(
                f"unknown model '{model_key}' — known models: {', '.join(sorted(self.sources))}"
            )

        emit(DownloadState.CHECKING, message=f"Checking local caches for {source.display_name}…")
        paths: dict[str, Path] = {}
        missing: list[tuple[str, str, str]] = []
        for name, repo, filename in self._items(source):
            cached = self._cached(repo, filename)
            if cached is None:
                missing.append((name, repo, filename))
            else:
                paths[name] = cached

        if missing:
            target = self.cache_dir or Path(HF_HUB_CACHE)
            free = self._free_gb(target)
            if free < source.download_gb:
                raise emit.fail(
                    f"Not enough disk space for {source.display_name}: the download needs "
                    f"~{source.download_gb:g}GB but only {free:.0f}GB is free at {target}. "
                    "Free up space, or point LoRAForge at a bigger drive in settings."
                )
            for name, repo, filename in missing:
                emit(
                    DownloadState.DOWNLOADING,
                    item=name,
                    message=f"Downloading {source.display_name} [{name}] from {repo}…",
                )
                paths[name] = self._fetch(source, repo, filename, emit)

        result = DownloadedModel(
            model_key=model_key, model_path=paths.pop("base"), asset_paths=paths
        )
        emit(
            DownloadState.COMPLETED,
            message=f"{source.display_name} is ready.",
            result=result,
        )
        return result

    def peek(self, model_key: str) -> DownloadedModel | None:
        """Already-local paths for a model, without downloading anything.

        None if the model is unknown or any of its files is not yet cached.
        """
        source = self.sources.get(model_key)
        if source is None:
            return None
        paths: dict[str, Path] = {}
        for name, repo, filename in self._items(source):
            cached = self._cached(repo, filename)
            if cached is None:
                return None
            paths[name] = cached
        return DownloadedModel(
            model_key=model_key, model_path=paths.pop("base"), asset_paths=paths
        )

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _items(source: ModelSource) -> list[tuple[str, str, str]]:
        """(name, repo, filename) for the base checkpoint and every asset."""
        return [
            ("base", source.repo, source.file),
            *((name, asset.repo, asset.file) for name, asset in source.assets.items()),
        ]

    def _cached(self, repo: str, filename: str) -> Path | None:
        """The file's path if it is already in the cache (ours or ComfyUI's)."""
        try:
            root = Path(
                self._snapshot(
                    repo_id=repo,
                    allow_patterns=[filename],
                    local_files_only=True,
                    cache_dir=self.cache_dir,
                )
            )
        except LocalEntryNotFoundError:
            return None
        path = root / filename
        return path if path.exists() else None

    def _fetch(self, source: ModelSource, repo: str, filename: str, emit: _Emitter) -> Path:
        try:
            root = Path(
                self._snapshot(
                    repo_id=repo, allow_patterns=[filename], cache_dir=self.cache_dir
                )
            )
        except GatedRepoError as exc:
            raise emit.fail(self._gated_message(source, repo)) from exc
        except HfHubHTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise emit.fail(self._gated_message(source, repo)) from exc
            raise emit.fail(f"Download of {repo} failed: {exc}") from exc
        path = root / filename
        if not path.exists():
            raise emit.fail(
                f"{repo} no longer contains '{filename}' — the capability matrix entry "
                f"for {source.model_key} needs updating (please report this)."
            )
        return path

    @staticmethod
    def _gated_message(source: ModelSource, repo: str) -> str:
        # Never store or handle tokens here: huggingface_hub's own keystore does.
        return (
            f"{source.display_name} is a gated model and Hugging Face refused access. "
            f"First accept the license at https://huggingface.co/{repo} (while signed in), "
            "then run `hf auth login` once so your token lives in Hugging Face's own "
            "credential store — LoRAForge never sees or saves it."
        )
