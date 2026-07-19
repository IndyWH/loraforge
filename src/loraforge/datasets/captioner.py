"""Auto-captioner interface — designed now, implemented later.

Captioning models (Florence-2, WD14) need torch, and the app layer stays
torch-free. So a captioner runs exactly like a training engine: its own
uv-managed environment, launched as a subprocess — a ``LaunchPlan`` in,
stdout lines out, sidecar files as the result. Same shapes, same isolation,
and the engine bootstrap machinery can set its environment up.

This pass ships only the protocol and a stub; manual captions (``.txt``
sidecars, edited through the caption routes) are the supported path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from loraforge.engines.base import LaunchPlan


@dataclass(frozen=True)
class CaptionEvent:
    """One parsed line of captioner output."""

    filename: str | None = None  # image just captioned
    caption: str | None = None
    done: int | None = None
    total: int | None = None
    message: str | None = None


class Captioner(Protocol):
    """Implement one per captioning model. Mirrors EngineAdapter on purpose."""

    name: str

    def check_environment(self, env_dir: Path) -> list[str]:
        """Problems standing between this machine and captioning (empty = go)."""
        ...

    def compile(self, image_dir: Path, workdir: Path) -> LaunchPlan:
        """Render a subprocess launch that writes .txt sidecars into image_dir."""
        ...

    def parse_line(self, line: str) -> CaptionEvent | None: ...

    def collect(self, image_dir: Path) -> dict[str, str]:
        """Read back the sidecars the run produced: filename → caption."""
        ...


class StubCaptioner:
    """Placeholder until the first real captioner adapter lands."""

    name = "none"

    _MESSAGE = (
        "auto-captioning is not available yet — write captions by hand (a .txt file "
        "next to each image, or the caption editor), or wait for the Florence-2/WD14 "
        "captioner which will install into its own environment like a training engine"
    )

    def check_environment(self, env_dir: Path) -> list[str]:
        return [self._MESSAGE]

    def compile(self, image_dir: Path, workdir: Path) -> LaunchPlan:
        raise NotImplementedError(self._MESSAGE)

    def parse_line(self, line: str) -> CaptionEvent | None:
        return None

    def collect(self, image_dir: Path) -> dict[str, str]:
        return {}
