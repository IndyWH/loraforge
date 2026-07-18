"""Minimal CLI: `loraforge diagnose` prints the hardware + capability report.

The server and train commands land next; diagnose ships first because it is
also the bug-report generator.
"""

from __future__ import annotations

import argparse
import json
import sys

from loraforge.capability.resolver import Availability, resolve
from loraforge.probe import probe

_ICONS = {
    Availability.AVAILABLE: "[ok]",
    Availability.UNAVAILABLE: "[--]",
    Availability.BLOCKED: "[!!]",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loraforge")
    sub = parser.add_subparsers(dest="command", required=True)
    diag = sub.add_parser("diagnose", help="probe hardware and show what you can train")
    diag.add_argument("--json", action="store_true", help="machine-readable output (bug reports)")
    args = parser.parse_args(argv)

    report = probe()
    caps = resolve(report)

    if args.json:
        print(json.dumps({"hardware": report.model_dump(mode="json"),
                          "capabilities": caps.model_dump(mode="json")}, indent=2))
        return 0

    gpu = report.primary_gpu
    print("LoRAForge diagnostics")
    print("=" * 60)
    if gpu:
        print(f"GPU     : {gpu.name} ({gpu.arch.value}, sm_{gpu.sm_major}{gpu.sm_minor})")
        print(f"VRAM    : {gpu.vram_free_mb / 1024:.1f}GB free of {gpu.vram_total_mb / 1024:.1f}GB")
        print(f"FP8     : {'yes' if gpu.supports_fp8 else 'no (needs RTX 40/50)'}")
    else:
        print("GPU     : none detected")
    print(f"Driver  : {report.driver_version or 'unknown'}")
    if report.torch:
        print(f"Torch   : {report.torch.version} (CUDA {report.torch.cuda_version or 'cpu-only'})")
    if report.ram_total_mb:
        print(f"RAM     : {report.ram_total_mb / 1024:.0f}GB")
    if report.disk_free_gb is not None:
        print(f"Disk    : {report.disk_free_gb:.0f}GB free")
    for note in report.notes:
        print(f"note    : {note}")

    print("\nWhat you can train")
    print("-" * 60)
    for m in caps.models:
        line = f"{_ICONS[m.status]} {m.display_name}"
        if m.status is Availability.AVAILABLE:
            line += f"  → preset '{m.preset_name}': {m.settings}"
        elif m.reason:
            line += f"\n     {m.reason}"
        print(line)
    for w in caps.warnings:
        print(f"\nwarning: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
