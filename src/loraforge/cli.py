"""Minimal CLI.

`loraforge diagnose` prints the hardware + capability report (also the
bug-report generator). `loraforge setup` bootstraps the training engine:
pinned checkout + uv-managed env + the torch wheels this GPU needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loraforge.capability.resolver import Availability, resolve
from loraforge.engines.bootstrap import ENGINE_SPECS, BootstrapError, EngineBootstrapper
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
    setup = sub.add_parser("setup", help="install the training engine for this GPU (idempotent)")
    setup.add_argument("--engine", default="kohya", choices=sorted(ENGINE_SPECS))
    setup.add_argument(
        "--root", type=Path, default=None, help="engines directory (default: per-user data dir)"
    )
    setup.add_argument(
        "--dry-run", action="store_true", help="show what setup would do without doing it"
    )
    server = sub.add_parser("serve", help="run the local API server (loopback only by default)")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8471)
    server.add_argument(
        "--allow-remote",
        action="store_true",
        help="bind beyond loopback — exposes an unauthenticated training server; "
        "only with your own auth in front",
    )
    args = parser.parse_args(argv)

    if args.command == "setup":
        return _cmd_setup(args)
    if args.command == "serve":
        from loraforge.server.run import serve  # server deps stay off diagnose's import path

        serve(host=args.host, port=args.port, allow_remote=args.allow_remote)
        return 0

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


def _cmd_setup(args: argparse.Namespace) -> int:
    report = probe()
    boot = EngineBootstrapper(ENGINE_SPECS[args.engine], report, engines_root=args.root)

    gpu = report.primary_gpu
    if gpu:
        print(f"GPU     : {gpu.name} → PyTorch {boot.torch.cuda} wheels")
    if boot.torch.note:
        print(f"note    : {boot.torch.note}")

    problems = boot.preflight()
    if problems:
        for p in problems:
            print(f"problem : {p}", file=sys.stderr)
        return 1

    steps = boot.plan()
    if not steps:
        print(f"Engine '{args.engine}' is ready at {boot.paths.root} — nothing to do.")
        return 0
    if args.dry_run:
        print(f"Would run {len(steps)} step(s):")
        for step in steps:
            print(f"  - {step.description}")
        return 0

    try:
        boot.run(on_step=lambda s: print(f"→ {s.description}"))
    except BootstrapError as exc:
        print(f"setup failed:\n{exc}", file=sys.stderr)
        return 1
    print("Engine ready.")
    print(f"  checkout: {boot.paths.checkout}")
    print(f"  env     : {boot.paths.env}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
