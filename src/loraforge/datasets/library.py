"""Dataset preparation — transport-free, torch-free.

Ingestion COPIES images into a managed dataset directory under the data
root; the user's originals are never moved or modified. Every verdict is
data (pydantic models) for the UI to render as stats and chips: counts,
warnings, and exclusions with human reasons.

Duplicate policy:
- exact duplicates (identical bytes, sha256) are skipped at ingest and
  excluded at status time if they slipped in some other way;
- near duplicates (perceptual dHash, horizontal+vertical, 128 bits) are
  flagged with a warning, never auto-excluded — burst shots may be wanted.

Quality policy:
- unreadable files are excluded with a reason;
- images under 256px on the short side are excluded (no preset buckets
  that low); under 512px they train but carry a warning.

Captions are ``.txt`` sidecars next to each image (kohya's default layout).
Trigger-word injection prepends the word as the first tag without ever
duplicating an already-present tag. Auto-captioning is NOT here: it needs
torch, so it will run engine-style as a subprocess — see ``captioner.py``.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pillow_heif
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable

pillow_heif.register_heif_opener()  # iPhone HEIC/HEIF opens like any other format

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"})
MIN_SIDE_HARD = 256  # below: excluded — no preset trains this small
MIN_SIDE_SOFT = 512  # below: warning — buckets will upscale, detail suffers
NEAR_DUP_MAX_DISTANCE = 8  # of 128 dHash bits

_UNREADABLE = "cannot be read as an image — the file may be corrupted"

# ── UI-facing data ───────────────────────────────────────────────────────────


class ImageStatus(BaseModel):
    filename: str
    width: int | None = None
    height: int | None = None
    included: bool = True
    reason: str | None = None  # human, set when excluded
    warnings: list[str] = Field(default_factory=list)
    has_caption: bool = False


class DatasetSummary(BaseModel):
    """What a recipe's ``dataset.path`` points at, plus the UI's stats."""

    name: str
    path: Path
    total: int
    included: int
    excluded: int
    captioned: int  # included images that already have a caption sidecar
    images: list[ImageStatus]


class IngestSkip(BaseModel):
    source: str
    reason: str


class IngestResult(BaseModel):
    added: list[str] = Field(default_factory=list)
    skipped: list[IngestSkip] = Field(default_factory=list)


# ── Hashing ──────────────────────────────────────────────────────────────────


def _content_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(image: Image.Image, size: int = 8) -> int:
    """Perceptual hash: horizontal + vertical gradient signs, 2*size² bits."""
    gray = image.convert("L")
    horizontal = gray.resize((size + 1, size), Image.Resampling.LANCZOS).tobytes()
    vertical = gray.resize((size, size + 1), Image.Resampling.LANCZOS).tobytes()
    bits = 0
    for row in range(size):
        for col in range(size):
            index = row * (size + 1) + col
            bits = (bits << 1) | (horizontal[index] > horizontal[index + 1])
    for row in range(size):
        for col in range(size):
            bits = (bits << 1) | (vertical[row * size + col] > vertical[(row + 1) * size + col])
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ── The library ──────────────────────────────────────────────────────────────


class DatasetLibrary:
    """Manages dataset directories under ``root`` (data_root/datasets)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def list_names(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def delete(self, name: str) -> None:
        shutil.rmtree(self._require(name))

    # ── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, name: str, sources: Iterable[Path]) -> IngestResult:
        """Copy images (and their caption sidecars) in; never move originals."""
        directory = self._require(name)
        known = {_content_digest(p): p.name for p in self._image_files(directory)}
        result = IngestResult()
        for source in map(Path, sources):
            if not source.is_file():
                result.skipped.append(IngestSkip(source=str(source), reason="file not found"))
                continue
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                supported = "/".join(sorted(s.lstrip(".") for s in IMAGE_SUFFIXES))
                result.skipped.append(
                    IngestSkip(source=str(source), reason=f"not a supported image ({supported})")
                )
                continue
            heic = source.suffix.lower() in (".heic", ".heif")
            try:
                with Image.open(source) as img:
                    img.verify()
                # engines read the dataset dir directly and kohya cannot decode
                # HEIC, so iPhone photos are transcoded to JPEG at the door
                payload = self._transcode_to_jpeg(source) if heic else source.read_bytes()
            except Exception:  # Pillow raises assorted types for broken files
                result.skipped.append(IngestSkip(source=str(source), reason=_UNREADABLE))
                continue
            digest = hashlib.sha256(payload).hexdigest()
            if digest in known:
                result.skipped.append(
                    IngestSkip(
                        source=str(source),
                        reason=f"exact duplicate of '{known[digest]}' already in the dataset",
                    )
                )
                continue
            filename = f"{Path(source.name).stem}.jpg" if heic else source.name
            dest = self._unique_dest(directory, filename)
            dest.write_bytes(payload)  # copy (or transcode) — never move the original
            sidecar = source.with_suffix(".txt")
            if sidecar.is_file():  # bring the user's existing caption along
                shutil.copy2(sidecar, dest.with_suffix(".txt"))
            known[digest] = dest.name
            result.added.append(dest.name)
        return result

    @staticmethod
    def _transcode_to_jpeg(source: Path) -> bytes:
        from io import BytesIO

        with Image.open(source) as img:
            upright = ImageOps.exif_transpose(img)
            buffer = BytesIO()
            upright.convert("RGB").save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    # ── Status / quality checks ──────────────────────────────────────────────

    def status(self, name: str) -> DatasetSummary:
        directory = self._require(name)
        images: list[ImageStatus] = []
        content_seen: dict[str, str] = {}
        perceptual_seen: list[tuple[int, str]] = []
        for path in self._image_files(directory):
            entry = ImageStatus(
                filename=path.name, has_caption=path.with_suffix(".txt").is_file()
            )
            images.append(entry)
            try:
                with Image.open(path) as img:
                    img.load()
                    entry.width, entry.height = img.size
                    perceptual = dhash(img)
            except Exception:  # Pillow raises assorted types for broken files
                entry.included = False
                entry.reason = _UNREADABLE
                continue

            digest = _content_digest(path)
            if digest in content_seen:
                entry.included = False
                entry.reason = f"exact duplicate of '{content_seen[digest]}'"
                continue
            content_seen[digest] = path.name

            short_side = min(entry.width, entry.height)
            if short_side < MIN_SIDE_HARD:
                entry.included = False
                entry.reason = (
                    f"too small to train on ({entry.width}x{entry.height}; "
                    f"needs at least {MIN_SIDE_HARD}px on the short side)"
                )
                continue
            if short_side < MIN_SIDE_SOFT:
                entry.warnings.append(
                    f"small image ({entry.width}x{entry.height}) — it will be upscaled "
                    f"into the training buckets; a copy over {MIN_SIDE_SOFT}px would look better"
                )

            near = next(
                (
                    other
                    for other_hash, other in perceptual_seen
                    if hamming(perceptual, other_hash) <= NEAR_DUP_MAX_DISTANCE
                ),
                None,
            )
            if near is not None:
                entry.warnings.append(
                    f"looks nearly identical to '{near}' — near-duplicates make the "
                    "model overfit; keep only the best one unless that is intended"
                )
            else:
                perceptual_seen.append((perceptual, path.name))

        included = sum(1 for i in images if i.included)
        return DatasetSummary(
            name=name,
            path=directory,
            total=len(images),
            included=included,
            excluded=len(images) - included,
            captioned=sum(1 for i in images if i.included and i.has_caption),
            images=images,
        )

    # ── Captions ─────────────────────────────────────────────────────────────

    def get_caption(self, name: str, filename: str) -> str | None:
        sidecar = self._image_path(name, filename).with_suffix(".txt")
        return sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else None

    def set_caption(self, name: str, filename: str, caption: str) -> None:
        sidecar = self._image_path(name, filename).with_suffix(".txt")
        sidecar.write_text(caption.strip() + "\n", encoding="utf-8")

    def inject_trigger_word(self, name: str, trigger: str) -> int:
        """Prepend the trigger as the first tag, without ever duplicating it.

        A caption already containing the trigger as a whole word — as a tag
        or mid-sentence ("photo of sks-cat, outdoors") — is left untouched.
        Images without a caption get one holding just the trigger. Returns
        how many captions were created or changed. Idempotent.
        """
        directory = self._require(name)
        trigger = trigger.strip()
        present = re.compile(rf"(?<![\w-]){re.escape(trigger)}(?![\w-])")
        updated = 0
        for image in self._image_files(directory):
            sidecar = image.with_suffix(".txt")
            caption = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
            if present.search(caption):
                continue
            sidecar.write_text(
                (f"{trigger}, {caption}" if caption else trigger) + "\n", encoding="utf-8"
            )
            updated += 1
        return updated

    # ── Internals ────────────────────────────────────────────────────────────

    def _require(self, name: str) -> Path:
        directory = self.root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"no dataset named '{name}'")
        return directory

    def _image_path(self, name: str, filename: str) -> Path:
        directory = self._require(name)
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError(f"invalid image filename '{filename}'")
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"no image named '{filename}' in dataset '{name}'")
        return path

    @staticmethod
    def _image_files(directory: Path) -> list[Path]:
        return sorted(
            p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )

    @staticmethod
    def _unique_dest(directory: Path, filename: str) -> Path:
        dest = directory / filename
        counter = 1
        while dest.exists():
            dest = directory / f"{Path(filename).stem}-{counter}{Path(filename).suffix}"
            counter += 1
        return dest
