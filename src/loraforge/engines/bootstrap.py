"""Engine environment bootstrap.

Builds the isolated world a training engine runs in:

1. a git checkout of the engine repo at a PINNED tag — never a moving branch,
2. its own venv on a uv-managed standalone CPython (the user's system Python
   is never touched, never assumed to exist),
3. the torch wheel line matching the probed GPU generation (cu126 for older
   cards; cu128+ REQUIRED for Blackwell / RTX 50, sm_120),
4. the engine's own pinned requirements.

Everything is planned as data first (``plan()`` → list of ``Step``) and only
then executed, so tests can assert exact commands without network or GPU, and
``loraforge setup --dry-run`` can show the user what will happen.

Idempotent by design: a completed install plans zero steps. A state file
records what was installed, so a changed pin plans only the re-checkout +
requirements steps, and a new GPU (say the user upgrades to an RTX 50 card
whose old cu126 wheels can't drive it) plans only the torch reinstall. This
is what powers ``check_environment`` repair.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loraforge.engines.base import venv_bin
from loraforge.probe import GpuArch

if TYPE_CHECKING:
    from collections.abc import Callable

    from loraforge.probe import HardwareReport

# ── Engine pins (data, not code) ─────────────────────────────────────────────


@dataclass(frozen=True)
class EngineSpec:
    key: str
    display_name: str
    repo_url: str
    pinned_ref: str  # a release tag; bump deliberately, we test against this tag only
    python_version: str  # uv downloads this as a standalone build


KOHYA = EngineSpec(
    key="kohya",
    display_name="kohya sd-scripts",
    repo_url="https://github.com/kohya-ss/sd-scripts",
    pinned_ref="v0.9.1",
    python_version="3.10",
)

ENGINE_SPECS: dict[str, EngineSpec] = {KOHYA.key: KOHYA}

# ── Torch wheel selection ────────────────────────────────────────────────────

# One torch version, two CUDA wheel lines. Blackwell silicon (sm_120) does not
# exist in pre-cu128 wheels; everything Turing→Ada runs fine on cu126.
_TORCH_PACKAGES = ("torch==2.7.1", "torchvision==0.22.1")


@dataclass(frozen=True)
class TorchPlan:
    cuda: str  # "cu126" | "cu128"
    packages: tuple[str, ...] = _TORCH_PACKAGES
    note: str | None = None

    @property
    def index_url(self) -> str:
        return f"https://download.pytorch.org/whl/{self.cuda}"


def select_torch(report: HardwareReport) -> TorchPlan:
    gpu = report.primary_gpu
    if gpu is None:
        return TorchPlan(
            cuda="cu126",
            note=(
                "no NVIDIA GPU detected — installing default cu126 wheels; "
                "re-run `loraforge setup` after fixing drivers and the right "
                "wheels will be swapped in"
            ),
        )
    if gpu.arch is GpuArch.BLACKWELL:
        return TorchPlan(cuda="cu128")
    return TorchPlan(cuda="cu126")


# ── Managed directories ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class EnginePaths:
    root: Path  # engines_root/<engine key>
    checkout: Path  # pinned repo clone (KohyaAdapter's sd_scripts_dir)
    env: Path  # the engine's own venv (KohyaAdapter's env_dir)
    state_file: Path  # what was installed, for idempotence + repair


def engine_paths(spec: EngineSpec, engines_root: Path) -> EnginePaths:
    root = engines_root / spec.key
    return EnginePaths(
        root=root, checkout=root / "repo", env=root / "env", state_file=root / "engine.json"
    )


def default_engines_root() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "LoRAForge" / "engines"
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "loraforge" / "engines"


# ── Plan / execute ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Step:
    description: str  # human-readable, shown live in the UI/CLI
    argv: tuple[str, ...]
    cwd: Path | None = None


class BootstrapError(RuntimeError):
    """A bootstrap step failed; the message is already human-readable."""


def _run_subprocess(step: Step) -> None:
    result = subprocess.run(step.argv, cwd=step.cwd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-15:])
        raise BootstrapError(
            f"step failed: {step.description}\n"
            f"  command: {' '.join(step.argv)}\n"
            f"{tail}"
        )


class EngineBootstrapper:
    """Plans and runs the setup of one engine's checkout + environment."""

    def __init__(
        self,
        spec: EngineSpec,
        report: HardwareReport,
        engines_root: Path | None = None,
        uv: str = "uv",
        git: str = "git",
    ) -> None:
        self.spec = spec
        self.uv = uv
        self.git = git
        self.paths = engine_paths(spec, engines_root or default_engines_root())
        self.torch = select_torch(report)

    # ── Inspection ───────────────────────────────────────────────────────────

    def preflight(self) -> list[str]:
        """Tool availability problems, in human words. Empty list = go."""
        problems: list[str] = []
        if shutil.which(self.git) is None:
            problems.append(
                "git is not installed or not on PATH — install it from "
                "https://git-scm.com/downloads, then re-run setup"
            )
        if shutil.which(self.uv) is None:
            problems.append(
                "uv is not installed or not on PATH — install it from "
                "https://docs.astral.sh/uv/getting-started/installation/, then re-run setup"
            )
        return problems

    def plan(self) -> list[Step]:
        """Only the steps this machine still needs. Empty list = ready."""
        spec, paths = self.spec, self.paths
        state = self._read_state()
        env_python = venv_bin(paths.env, "python")
        steps: list[Step] = []

        if not (paths.checkout / ".git").is_dir():
            steps.append(
                Step(
                    f"Download {spec.display_name} {spec.pinned_ref}",
                    (
                        self.git, "clone", "--depth", "1",
                        "--branch", spec.pinned_ref,
                        spec.repo_url, str(paths.checkout),
                    ),
                )
            )
        elif state.get("ref") != spec.pinned_ref:
            steps.append(
                Step(
                    f"Fetch {spec.display_name} {spec.pinned_ref}",
                    (self.git, "fetch", "--depth", "1", "origin", "tag", spec.pinned_ref),
                    cwd=paths.checkout,
                )
            )
            steps.append(
                Step(
                    f"Switch checkout to {spec.pinned_ref}",
                    (self.git, "checkout", spec.pinned_ref),
                    cwd=paths.checkout,
                )
            )

        fresh_env = not env_python.exists()
        if fresh_env:
            steps.append(
                Step(
                    f"Create engine environment (standalone Python {spec.python_version})",
                    (
                        self.uv, "venv",
                        "--python", spec.python_version,
                        "--python-preference", "only-managed",
                        str(paths.env),
                    ),
                )
            )

        if fresh_env or state.get("torch_cuda") != self.torch.cuda:
            steps.append(
                Step(
                    f"Install PyTorch ({self.torch.cuda} wheels)",
                    (
                        self.uv, "pip", "install",
                        "--python", str(env_python),
                        "--index-url", self.torch.index_url,
                        *self.torch.packages,
                    ),
                )
            )

        if fresh_env or state.get("ref") != spec.pinned_ref:
            steps.append(
                Step(
                    f"Install {spec.display_name} requirements",
                    (
                        self.uv, "pip", "install",
                        "--python", str(env_python),
                        "-r", "requirements.txt",
                    ),
                    cwd=paths.checkout,
                )
            )
        return steps

    def problems(self) -> list[str]:
        """Everything standing between this machine and a ready engine.

        Feeds the diagnostics page: preflight issues plus the description of
        each pending step, so 'blocked' capability states can say exactly what
        the repair will do.
        """
        return self.preflight() + [step.description for step in self.plan()]

    # ── Execution ────────────────────────────────────────────────────────────

    def run(
        self,
        runner: Callable[[Step], None] | None = None,
        on_step: Callable[[Step], None] | None = None,
    ) -> EnginePaths:
        """Execute the plan. Safe to re-run; raises BootstrapError on failure.

        The state file is written only after every step succeeds, so an
        interrupted install re-plans its remaining work next time.
        """
        problems = self.preflight()
        if problems:
            raise BootstrapError("\n".join(problems))
        runner = runner or _run_subprocess
        self.paths.root.mkdir(parents=True, exist_ok=True)
        for step in self.plan():
            if on_step is not None:
                on_step(step)
            runner(step)
        self._write_state()
        return self.paths

    # ── State file ───────────────────────────────────────────────────────────

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.paths.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self) -> None:
        state = {
            "engine": self.spec.key,
            "ref": self.spec.pinned_ref,
            "python": self.spec.python_version,
            "torch_cuda": self.torch.cuda,
        }
        self.paths.state_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
