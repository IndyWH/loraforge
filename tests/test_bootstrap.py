"""Engine bootstrap tests: planned commands, torch wheel selection, idempotence.

No network, no GPU, no subprocesses — plans are data, and execution effects
are faked by a runner that creates the files a real run would create.
"""

import json
from pathlib import Path

import pytest
from test_capability import fake_report

from loraforge.cli import main as cli_main
from loraforge.engines import bootstrap
from loraforge.engines.base import venv_bin
from loraforge.engines.bootstrap import (
    KOHYA,
    BootstrapError,
    EngineBootstrapper,
    Step,
    select_torch,
)

RTX_3060 = fake_report("NVIDIA GeForce RTX 3060", 12288, 11400, (8, 6))
RTX_5090 = fake_report("NVIDIA GeForce RTX 5090", 32607, 31000, (12, 0))


def faking_runner(boot: EngineBootstrapper):
    """Simulate each step's filesystem effect without running anything."""

    def run(step: Step) -> None:
        if step.argv[1] == "clone":
            (boot.paths.checkout / ".git").mkdir(parents=True)
        elif step.argv[1] == "venv":
            python = venv_bin(boot.paths.env, "python")
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()

    return run


def installed(tmp_path: Path, report=RTX_3060) -> EngineBootstrapper:
    """A bootstrapper whose engine has been fully 'installed' under tmp_path."""
    boot = EngineBootstrapper(KOHYA, report, engines_root=tmp_path)
    boot.run(runner=faking_runner(boot))
    return boot


# ── Torch wheel selection ────────────────────────────────────────────────────


def test_torch_selection_by_generation() -> None:
    assert select_torch(RTX_3060).cuda == "cu126"
    assert select_torch(RTX_5090).cuda == "cu128"  # Blackwell needs cu128+ wheels


def test_no_gpu_defaults_to_cu126_with_a_note() -> None:
    from loraforge.probe import HardwareReport

    plan = select_torch(HardwareReport())
    assert plan.cuda == "cu126"
    assert plan.note is not None and "GPU" in plan.note


# ── Planning ─────────────────────────────────────────────────────────────────


def test_fresh_machine_plans_all_five_steps(tmp_path: Path) -> None:
    boot = EngineBootstrapper(KOHYA, RTX_3060, engines_root=tmp_path)
    steps = boot.plan()
    assert len(steps) == 5

    clone, venv, torch, reqs, pins = steps
    assert "--branch" in clone.argv and KOHYA.pinned_ref in clone.argv  # pinned, not a branch tip
    assert "--python-preference" in venv.argv  # standalone CPython, never system Python
    assert "only-managed" in venv.argv
    assert "https://download.pytorch.org/whl/cu126" in torch.argv
    assert reqs.cwd == boot.paths.checkout  # requirements.txt paths resolve in the checkout
    # compat pins install last, so requirements.txt can't re-loosen them
    assert "numpy<2" in pins.argv


def test_blackwell_plans_cu128_wheels(tmp_path: Path) -> None:
    boot = EngineBootstrapper(KOHYA, RTX_5090, engines_root=tmp_path)
    torch_step = next(s for s in boot.plan() if "PyTorch" in s.description)
    assert "https://download.pytorch.org/whl/cu128" in torch_step.argv


# ── Idempotence and repair ───────────────────────────────────────────────────


def test_completed_install_plans_nothing(tmp_path: Path) -> None:
    boot = installed(tmp_path)
    assert boot.plan() == []
    assert boot.problems() == []


def test_gpu_upgrade_repairs_only_torch(tmp_path: Path) -> None:
    installed(tmp_path, report=RTX_3060)  # cu126 on disk
    boot = EngineBootstrapper(KOHYA, RTX_5090, engines_root=tmp_path)  # now a Blackwell card
    steps = boot.plan()
    assert len(steps) == 1
    assert "https://download.pytorch.org/whl/cu128" in steps[0].argv

    boot.run(runner=lambda step: None)  # repair completes → state updated
    assert boot.plan() == []


def test_pin_bump_refetches_and_reinstalls_requirements(tmp_path: Path) -> None:
    import dataclasses

    installed(tmp_path)
    bumped = dataclasses.replace(KOHYA, pinned_ref="v9.9.9")
    steps = EngineBootstrapper(bumped, RTX_3060, engines_root=tmp_path).plan()
    descriptions = " | ".join(s.description for s in steps)
    assert len(steps) == 4  # fetch tag, checkout, requirements, re-pin — venv/torch untouched
    assert "v9.9.9" in descriptions
    assert "requirements" in descriptions
    assert not any("PyTorch" in s.description for s in steps)


def test_interrupted_install_is_replanned(tmp_path: Path) -> None:
    boot = EngineBootstrapper(KOHYA, RTX_3060, engines_root=tmp_path)

    def failing(step: Step) -> None:
        raise BootstrapError("boom")

    with pytest.raises(BootstrapError):
        boot.run(runner=failing)
    assert not boot.paths.state_file.exists()  # no success recorded
    assert len(boot.plan()) == 5  # everything still pending


def test_new_pin_repairs_an_existing_install(tmp_path: Path) -> None:
    # An env installed before `numpy<2` existed (state has no pins recorded):
    # the next `loraforge setup` must plan exactly the pin fix, nothing else.
    boot = installed(tmp_path)
    state = json.loads(boot.paths.state_file.read_text(encoding="utf-8"))
    del state["pins"]
    boot.paths.state_file.write_text(json.dumps(state), encoding="utf-8")

    fresh = EngineBootstrapper(KOHYA, RTX_3060, engines_root=tmp_path)
    steps = fresh.plan()
    assert len(steps) == 1
    assert "numpy<2" in steps[0].argv
    assert "Pin" in steps[0].description

    fresh.run(runner=lambda step: None)  # repair completes → state updated
    assert fresh.plan() == []


# ── Human-facing failure modes ───────────────────────────────────────────────


def test_missing_tools_speak_human(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: None)
    boot = EngineBootstrapper(KOHYA, RTX_3060, engines_root=tmp_path)
    problems = boot.preflight()
    assert len(problems) == 2
    assert all("install" in p for p in problems)
    with pytest.raises(BootstrapError, match="install"):
        boot.run(runner=lambda step: None)


def test_cli_setup_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bootstrap.shutil, "which", lambda name: f"/usr/bin/{name}")
    rc = cli_main(["setup", "--dry-run", "--root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Would run 5 step(s):" in out
    assert "sd-scripts" in out
