"""Engine adapter protocol.

An engine (kohya sd-scripts, musubi-tuner, SimpleTuner, ...) lives in its own
uv-managed environment with its own pins. The app talks to it only through
this interface: compile a Recipe into a launchable command, parse its stdout
into progress events, collect the artifact at the end.

Keeping this surface tiny is what makes "NeMo AutoModel as a future backend"
a weekend job instead of a rewrite.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from loraforge.recipes.schema import Recipe


def venv_bin(env_dir: Path, name: str) -> Path:
    """Cross-platform path to an executable inside a venv (Scripts/ vs bin/)."""
    if sys.platform == "win32":
        return env_dir / "Scripts" / f"{name}.exe"
    return env_dir / "bin" / name


@dataclass(frozen=True)
class LaunchPlan:
    """Everything needed to start a training subprocess."""

    argv: list[str]  # e.g. [python, "-m", "accelerate", ...] inside the engine env
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    config_files: dict[Path, str] = field(default_factory=dict)  # path → rendered content


@dataclass(frozen=True)
class ProgressEvent:
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    message: str | None = None
    sample_image: Path | None = None  # engine wrote a preview image
    is_oom: bool = False  # triggers the auto-step-down retry path
    # Human-worded diagnosis of a line that predicts a fatal exit. The runner
    # uses the last hint as the failure message instead of a bare exit code
    # (rule 5); the raw line stays in job.log.
    fatal_hint: str | None = None


@dataclass(frozen=True)
class TrainResult:
    artifact: Path  # the LoRA .safetensors
    format: str  # "kohya" | "peft" | ... (for the converter layer)
    logs: Path


class EngineAdapter(Protocol):
    """Implement one of these per engine. Keep them stateless."""

    name: str

    def check_environment(self, env_dir: Path | None = None) -> list[str]:
        """Return problems (empty list = ready); None → the adapter's own env.

        Used by the diagnostics page and the runner's submit preflight."""
        ...

    def compile(self, recipe: Recipe, workdir: Path) -> LaunchPlan:
        """Render the recipe into the engine's native config + argv.

        Must raise pydantic/ValueError with a human message on anything
        unrepresentable — never let the engine discover it at step 40.
        """
        ...

    def parse_line(self, line: str) -> ProgressEvent | None:
        """Translate one stdout/stderr line into a progress event (or None)."""
        ...

    def collect(self, workdir: Path) -> TrainResult:
        """Locate and describe the produced artifact after a clean exit."""
        ...
