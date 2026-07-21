"""Capability resolver tests across the gaming-card spectrum."""

from pathlib import Path

import pytest

from loraforge.capability.resolver import Availability, resolve
from loraforge.probe import GpuArch, GpuInfo, HardwareReport, TorchInfo, arch_from_sm
from loraforge.recipes.schema import Recipe


def fake_report(
    name: str,
    vram_total: int,
    vram_free: int,
    sm: tuple[int, int],
    ram_mb: int = 32000,
    torch_cuda: str = "12.8",
    laptop: bool = False,
) -> HardwareReport:
    return HardwareReport(
        gpus=[
            GpuInfo(
                index=0,
                name=name,
                vram_total_mb=vram_total,
                vram_free_mb=vram_free,
                sm_major=sm[0],
                sm_minor=sm[1],
                arch=arch_from_sm(*sm),
                is_laptop=laptop,
            )
        ],
        torch=TorchInfo(version="2.12.0", cuda_version=torch_cuda, cuda_available=True),
        ram_total_mb=ram_mb,
    )


RTX_4060 = fake_report("NVIDIA GeForce RTX 4060", 8188, 7400, (8, 9))
RTX_3060 = fake_report("NVIDIA GeForce RTX 3060", 12288, 11400, (8, 6))
RTX_4090 = fake_report("NVIDIA GeForce RTX 4090", 24564, 23000, (8, 9))
RTX_5090_OLD_WHEEL = fake_report(
    "NVIDIA GeForce RTX 5090", 32607, 31000, (12, 0), torch_cuda="12.6"
)


def test_arch_mapping() -> None:
    assert arch_from_sm(8, 9) is GpuArch.ADA
    assert arch_from_sm(12, 0) is GpuArch.BLACKWELL
    assert arch_from_sm(6, 1) is GpuArch.PASCAL_OR_OLDER


def test_4060_gets_sdxl_tight_but_not_flux() -> None:
    caps = resolve(RTX_4060)
    sdxl = caps.get("sdxl")
    assert sdxl.status is Availability.AVAILABLE
    assert sdxl.preset_name == "tight"
    flux = caps.get("flux_dev")
    assert flux.status is Availability.UNAVAILABLE
    assert "SDXL" in (flux.reason or "")  # points the user somewhere useful


def test_3060_ampere_cannot_use_fp8_flux_presets() -> None:
    # 11.4GB free clears flux "tight" VRAM bar, but every flux preset needs
    # fp8_base and Ampere has no FP8 silicon → must be unavailable, not OOM-later.
    caps = resolve(RTX_3060)
    assert caps.get("flux_dev").status is Availability.UNAVAILABLE
    # standard's threshold is 11400: 10.4GB measured appetite x ~10% headroom
    # (decision 20). A 3060 12GB with 11.4GB free clears it — record a real
    # 3060 run in matrix.yaml when one exists.
    assert caps.get("sdxl").preset_name == "standard"


def test_force_preset_bypasses_fit_checks_with_a_warning() -> None:
    # measurement mode (decision 20): a 4090 deliberately runs tight so its
    # real appetite can be recorded — settings come verbatim from the matrix
    caps = resolve(RTX_4090, force_presets={"sdxl": "tight"})
    sdxl = caps.get("sdxl")
    assert sdxl.preset_name == "tight"
    assert sdxl.settings["cache_text_encoder_outputs"] is True  # real tight, no drift
    assert any("forced" in w and "sdxl" in w for w in caps.warnings)
    # other models resolve normally
    assert caps.get("flux_dev").preset_name == "comfortable"


def test_force_preset_unknown_name_falls_back_to_normal_resolution() -> None:
    caps = resolve(RTX_4090, force_presets={"sdxl": "cosy"})
    assert caps.get("sdxl").preset_name == "comfortable"
    warning = next(w for w in caps.warnings if "cosy" in w)
    assert "tight" in warning  # names the presets that do exist


def test_4090_gets_comfortable_everything() -> None:
    caps = resolve(RTX_4090)
    assert caps.get("sdxl").preset_name == "comfortable"
    flux = caps.get("flux_dev")
    assert flux.status is Availability.AVAILABLE
    assert flux.preset_name == "comfortable"
    assert flux.settings["fp8_base"] is True


def test_blackwell_on_old_wheel_is_blocked_with_repair_hint() -> None:
    caps = resolve(RTX_5090_OLD_WHEEL)
    for m in caps.models:
        assert m.status is Availability.BLOCKED
        assert "12.8" in (m.reason or "")


def test_no_gpu_is_blocked_not_crashed() -> None:
    caps = resolve(HardwareReport())
    assert all(m.status is Availability.BLOCKED for m in caps.models)


def test_laptop_gets_margin_and_warning() -> None:
    # Desktop 4060: 7400MB free → sdxl tight (needs 7000). Laptop same numbers:
    # 10% margin → 6660MB budget → nothing fits.
    laptop = fake_report("NVIDIA GeForce RTX 4060 Laptop GPU", 8188, 7400, (8, 9), laptop=True)
    caps = resolve(laptop)
    assert caps.get("sdxl").status is Availability.UNAVAILABLE
    assert caps.warnings


def test_recipe_roundtrip_and_overrides(tmp_path: Path) -> None:
    recipe = Recipe.from_yaml(Path("examples/recipes/sdxl-character-12gb.yaml"))
    assert recipe.peft.rank == 16
    bumped = recipe.with_overrides({"peft.rank": 32, "train.max_steps": 2000})
    assert (bumped.peft.rank, bumped.train.max_steps) == (32, 2000)
    out = tmp_path / "roundtrip.yaml"
    bumped.to_yaml(out)
    assert Recipe.from_yaml(out) == bumped


def test_recipe_validation_speaks_human() -> None:
    recipe = Recipe.from_yaml(Path("examples/recipes/sdxl-character-12gb.yaml"))
    with pytest.raises(ValueError, match="multiple of 64"):
        recipe.with_overrides({"dataset.resolution": 1000})
    with pytest.raises(ValueError, match="sample_prompts"):
        recipe.with_overrides({"train.sample_prompts": []})
