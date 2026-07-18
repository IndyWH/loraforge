"""Resolve a HardwareReport against the capability matrix.

Output philosophy: the UI never hides an option silently. Every model
resolves to either an available preset or a *reason* a human can act on
("needs ~11GB free VRAM — your card has 8GB; SDXL is your best option").
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from loraforge.probe import GpuArch, HardwareReport

_MATRIX_PATH = Path(__file__).with_name("matrix.yaml")

# Order matters: newer archs satisfy "min_arch" of older ones.
_ARCH_ORDER = [
    GpuArch.PASCAL_OR_OLDER,
    GpuArch.TURING,
    GpuArch.AMPERE,
    GpuArch.ADA,
    GpuArch.HOPPER,
    GpuArch.BLACKWELL,
]


class Availability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"  # shown greyed-out, with reason
    BLOCKED = "blocked"  # environment problem (e.g. wrong torch wheel)


class ModelCapability(BaseModel):
    model_key: str
    display_name: str
    status: Availability
    preset_name: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    min_free_vram_mb: int | None = None  # the chosen preset's VRAM requirement
    reason: str | None = None  # set when not AVAILABLE, human-actionable


class CapabilityReport(BaseModel):
    models: list[ModelCapability]
    warnings: list[str] = Field(default_factory=list)

    def get(self, key: str) -> ModelCapability:
        return next(m for m in self.models if m.model_key == key)


def _arch_at_least(arch: GpuArch, floor: str) -> bool:
    try:
        return _ARCH_ORDER.index(arch) >= _ARCH_ORDER.index(GpuArch(floor))
    except ValueError:
        return False


def load_matrix(path: Path | None = None) -> dict[str, Any]:
    with (path or _MATRIX_PATH).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(report: HardwareReport, matrix: dict[str, Any] | None = None) -> CapabilityReport:
    matrix = matrix or load_matrix()
    gpu = report.primary_gpu
    warnings: list[str] = []
    models: list[ModelCapability] = []

    if gpu is None:
        return CapabilityReport(
            models=[
                ModelCapability(
                    model_key=key,
                    display_name=spec["display_name"],
                    status=Availability.BLOCKED,
                    reason="No NVIDIA GPU detected. Check drivers, or see the diagnostics page.",
                )
                for key, spec in matrix["models"].items()
            ],
            warnings=[
                "No NVIDIA GPU detected."
                + (" Notes: " + "; ".join(report.notes) if report.notes else "")
            ],
        )

    # Environment gate: Blackwell silicon on a pre-cu128 torch wheel can't run at all.
    blocked_reason: str | None = None
    bw = matrix.get("features", {}).get("blackwell_wheels", {})
    if (
        gpu.arch is GpuArch.BLACKWELL
        and report.torch is not None
        and report.torch.cuda_version is not None
        and _cuda_lt(report.torch.cuda_version, bw.get("min_torch_cuda", "12.8"))
    ):
        blocked_reason = (
            f"Your {gpu.name} (Blackwell, sm_{gpu.sm_major}{gpu.sm_minor}) needs a PyTorch build "
            f"with CUDA {bw.get('min_torch_cuda', '12.8')}+ (found {report.torch.cuda_version}). "
            "Run the installer's repair step to fetch the correct wheel."
        )

    if gpu.is_laptop:
        warnings.append(
            f"{gpu.name} is a laptop GPU: usable VRAM and sustained speed are lower than the "
            "desktop card of the same name. Conservative presets selected."
        )

    fp8_floor = matrix.get("features", {}).get("fp8_base", {}).get("min_arch", "ada")
    fp8_ok = _arch_at_least(gpu.arch, fp8_floor)
    # Laptop thermal/VRAM headroom: demand ~10% extra margin before green-lighting a preset.
    vram_budget = int(gpu.vram_free_mb * (0.9 if gpu.is_laptop else 1.0))

    for key, spec in matrix["models"].items():
        if blocked_reason:
            models.append(
                ModelCapability(
                    model_key=key,
                    display_name=spec["display_name"],
                    status=Availability.BLOCKED,
                    reason=blocked_reason,
                )
            )
            continue

        chosen: dict[str, Any] | None = None
        for preset in spec["presets"]:  # ordered best → tightest
            if vram_budget < preset["min_free_vram_mb"]:
                continue
            if (min_ram := preset.get("min_ram_mb")) and (report.ram_total_mb or 0) < min_ram:
                continue
            if preset["settings"].get("fp8_base") and not fp8_ok:
                continue  # pre-Ada card: this preset's memory math doesn't hold
            chosen = preset
            break

        if chosen is not None:
            models.append(
                ModelCapability(
                    model_key=key,
                    display_name=spec["display_name"],
                    status=Availability.AVAILABLE,
                    preset_name=chosen["name"],
                    settings=dict(chosen["settings"]),
                    min_free_vram_mb=chosen["min_free_vram_mb"],
                )
            )
        else:
            need_gb = min(p["min_free_vram_mb"] for p in spec["presets"]) / 1024
            models.append(
                ModelCapability(
                    model_key=key,
                    display_name=spec["display_name"],
                    status=Availability.UNAVAILABLE,
                    reason=(
                        f"{spec['not_available_reason']} "
                        f"(needs ≥{need_gb:.1f}GB free; {gpu.name} has "
                        f"{gpu.vram_free_mb / 1024:.1f}GB free)"
                    ),
                )
            )

    return CapabilityReport(models=models, warnings=warnings)


def _cuda_lt(found: str, required: str) -> bool:
    """True if CUDA version string ``found`` < ``required`` (e.g. '12.6' < '12.8')."""

    def parts(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split(".")[:2])

    return parts(found) < parts(required)
