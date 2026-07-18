"""Hardware diagnostic probe.

Collects everything the capability resolver (and a bug report) needs:
GPU model, VRAM (total *and* free — the desktop eats VRAM), compute
capability, driver, torch/CUDA build, system RAM, and free disk.

Design rules:
- Never crash on a machine with no GPU / no torch / no NVML. Every field
  that can't be probed is None, and ``notes`` says why. The UI renders
  what it has; the resolver treats unknowns conservatively.
- Pure data out (pydantic model). JSON-serializable so it can be attached
  verbatim to GitHub issues.
"""

from __future__ import annotations

import platform
import shutil
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

# ── Report models ────────────────────────────────────────────────────────────


class GpuArch(StrEnum):
    """NVIDIA architecture generations we gate features on."""

    PASCAL_OR_OLDER = "pascal_or_older"  # < sm_75
    TURING = "turing"  # sm_75          (RTX 20)
    AMPERE = "ampere"  # sm_80/86       (RTX 30)
    ADA = "ada"  # sm_89                (RTX 40)  → FP8 capable
    HOPPER = "hopper"  # sm_90          (H100, not consumer but be honest)
    BLACKWELL = "blackwell"  # sm_100/120 (RTX 50) → needs cu128+ wheels
    UNKNOWN = "unknown"


_SM_TO_ARCH: list[tuple[int, GpuArch]] = [
    (100, GpuArch.BLACKWELL),
    (90, GpuArch.HOPPER),
    (89, GpuArch.ADA),
    (80, GpuArch.AMPERE),
    (75, GpuArch.TURING),
]


def arch_from_sm(major: int, minor: int) -> GpuArch:
    sm = major * 10 + minor
    for floor, arch in _SM_TO_ARCH:
        if sm >= floor:
            return arch
    return GpuArch.PASCAL_OR_OLDER


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int
    sm_major: int | None = None
    sm_minor: int | None = None
    arch: GpuArch = GpuArch.UNKNOWN
    is_laptop: bool = False

    @property
    def supports_fp8(self) -> bool:
        return self.arch in (GpuArch.ADA, GpuArch.HOPPER, GpuArch.BLACKWELL)


class TorchInfo(BaseModel):
    version: str
    cuda_version: str | None = None  # CUDA runtime torch was built with
    cuda_available: bool = False
    device_capability: tuple[int, int] | None = None
    sdpa_available: bool = True  # torch>=2.0 always has SDPA


class HardwareReport(BaseModel):
    """Everything the resolver and a bug report need, in one JSON blob."""

    os: str = Field(default_factory=platform.system)
    os_version: str = Field(default_factory=platform.version)
    python_version: str = Field(default_factory=platform.python_version)
    driver_version: str | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    torch: TorchInfo | None = None
    ram_total_mb: int | None = None
    disk_free_gb: float | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return max(self.gpus, key=lambda g: g.vram_total_mb) if self.gpus else None


# ── Probes (each degrades gracefully) ────────────────────────────────────────


def _probe_nvml(report: HardwareReport) -> None:
    try:
        import pynvml  # nvidia-ml-py
    except ImportError:
        report.notes.append("nvidia-ml-py not installed; GPU probe skipped")
        return
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        report.notes.append(f"NVML init failed (no NVIDIA driver?): {exc}")
        return
    try:
        report.driver_version = _nvml_str(pynvml.nvmlSystemGetDriverVersion())
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = _nvml_str(pynvml.nvmlDeviceGetName(handle))
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            except Exception:
                major = minor = None
            gpu = GpuInfo(
                index=i,
                name=name,
                vram_total_mb=mem.total // (1024**2),
                vram_free_mb=mem.free // (1024**2),
                sm_major=major,
                sm_minor=minor,
                arch=arch_from_sm(major, minor) if major is not None else GpuArch.UNKNOWN,
                is_laptop="laptop" in name.lower() or "mobile" in name.lower(),
            )
            report.gpus.append(gpu)
    finally:
        pynvml.nvmlShutdown()


def _nvml_str(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _probe_torch(report: HardwareReport) -> None:
    try:
        import torch
    except ImportError:
        report.notes.append("torch not importable from app env (expected: engines own torch)")
        return
    info = TorchInfo(
        version=torch.__version__,
        cuda_version=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
    )
    if info.cuda_available:
        info.device_capability = torch.cuda.get_device_capability(0)
    report.torch = info


def _probe_system(report: HardwareReport, workdir: Path) -> None:
    try:
        import psutil

        report.ram_total_mb = psutil.virtual_memory().total // (1024**2)
    except ImportError:
        report.notes.append("psutil not installed; RAM unknown")
    try:
        report.disk_free_gb = round(shutil.disk_usage(workdir).free / (1024**3), 1)
    except OSError as exc:
        report.notes.append(f"disk probe failed: {exc}")


def probe(workdir: Path | None = None) -> HardwareReport:
    """Run all probes. Never raises; consult ``report.notes`` for gaps."""
    report = HardwareReport()
    _probe_nvml(report)
    _probe_torch(report)
    _probe_system(report, workdir or Path.cwd())
    return report
