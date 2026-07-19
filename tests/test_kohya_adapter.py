"""kohya adapter tests: recipe → launch plan, progress parsing, artifact collection."""

import os
from pathlib import Path

import pytest

from loraforge.engines.kohya import KohyaAdapter
from loraforge.recipes.schema import Recipe

RECIPE_PATH = Path("examples/recipes/sdxl-character-12gb.yaml")


@pytest.fixture()
def adapter(tmp_path: Path) -> KohyaAdapter:
    return KohyaAdapter(
        sd_scripts_dir=tmp_path / "sd-scripts",
        env_dir=tmp_path / "env",
        model_paths={"sdxl": tmp_path / "models" / "sdxl.safetensors"},
    )


def test_compile_produces_expected_argv(adapter: KohyaAdapter, tmp_path: Path) -> None:
    recipe = Recipe.from_yaml(RECIPE_PATH)
    plan = adapter.compile(recipe, tmp_path)

    argv = " ".join(plan.argv)
    assert "sdxl_train_network.py" in argv
    assert "--network_dim=16" in argv
    assert "--network_alpha=16" in argv
    assert "--optimizer_type=AdamW8bit" in argv
    assert "--max_train_steps=1500" in argv
    assert "--mixed_precision=bf16" in argv
    assert "--gradient_checkpointing" in argv
    assert "--sdpa" in argv  # happy path never requires flash-attn
    assert "--sample_every_n_steps=200" in argv
    # low-VRAM flags absent when the recipe doesn't ask for them
    assert "--fp8_base" not in argv
    assert "--blocks_to_swap" not in argv


def test_compile_renders_dataset_toml(adapter: KohyaAdapter, tmp_path: Path) -> None:
    recipe = Recipe.from_yaml(RECIPE_PATH)
    plan = adapter.compile(recipe, tmp_path)
    toml = plan.config_files[tmp_path / "dataset.toml"]
    assert "enable_bucket = true" in toml
    assert "resolution = 1024" in toml
    assert "batch_size = 2" in toml
    assert "num_repeats = 10" in toml
    assert "\\" not in toml  # forward slashes only — Windows paths must not leak escapes


def test_compile_flux_low_vram_flags(adapter: KohyaAdapter, tmp_path: Path) -> None:
    adapter.model_paths["flux_dev"] = tmp_path / "flux"
    adapter.asset_paths["flux_dev"] = {
        "clip_l": tmp_path / "clip_l.safetensors",
        "t5xxl": tmp_path / "t5xxl.safetensors",
        "ae": tmp_path / "ae.safetensors",
    }
    recipe = Recipe.from_yaml(RECIPE_PATH).with_overrides(
        {
            "model": "flux_dev",
            "train.fp8_base": True,
            "train.blocks_to_swap": 18,
            "train.batch_size": 1,
        }
    )
    argv = " ".join(adapter.compile(recipe, tmp_path).argv)
    assert "flux_train_network.py" in argv
    assert "--network_module=networks.lora_flux" in argv
    assert "--fp8_base" in argv
    assert "--blocks_to_swap=18" in argv
    assert "--clip_l=" in argv and "--t5xxl=" in argv and "--ae=" in argv


def test_compile_errors_speak_human(adapter: KohyaAdapter, tmp_path: Path) -> None:
    recipe = Recipe.from_yaml(RECIPE_PATH)

    with pytest.raises(ValueError, match="not supported by the kohya engine"):
        adapter.compile(recipe.model_copy(update={"model": "wan_video"}), tmp_path)

    with pytest.raises(ValueError, match="download"):
        adapter.compile(recipe.model_copy(update={"model": "sd15"}), tmp_path)

    adapter.model_paths["flux_dev"] = tmp_path / "flux"
    with pytest.raises(ValueError, match="clip_l"):
        adapter.compile(recipe.model_copy(update={"model": "flux_dev"}), tmp_path)


def test_parse_line_progress_and_loss(adapter: KohyaAdapter) -> None:
    line = "steps:  12%|█▏        | 240/2000 [02:04<15:12,  1.93it/s, avr_loss=0.0821]"
    event = adapter.parse_line(line)
    assert event is not None
    assert (event.step, event.total_steps, event.loss) == (240, 2000, 0.0821)
    assert not event.is_oom

    assert adapter.parse_line("prepare optimizer, data loader etc.") is None


def test_parse_line_detects_oom(adapter: KohyaAdapter) -> None:
    event = adapter.parse_line(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.50 GiB"
    )
    assert event is not None and event.is_oom


def test_compile_resolves_model_symlinks(tmp_path: Path) -> None:
    # The HF cache hands out snapshot *symlinks*; kohya readlink()s them to a
    # relative blob path and dies ("neither a valid local path nor a valid
    # repo id"). Rendered argv must always carry the real file.
    if os.name == "nt":
        pytest.skip("symlink creation needs privileges on Windows runners")
    blob = tmp_path / "blobs" / "31e35c80fc4829"
    blob.parent.mkdir()
    blob.write_bytes(b"weights")
    snapshot = tmp_path / "snapshots" / "462165"
    snapshot.mkdir(parents=True)
    link = snapshot / "sd_xl_base_1.0.safetensors"
    link.symlink_to(Path("..") / ".." / "blobs" / "31e35c80fc4829")  # relative, like HF

    adapter = KohyaAdapter(
        sd_scripts_dir=tmp_path / "sd-scripts",
        env_dir=tmp_path / "env",
        model_paths={"sdxl": link},
    )
    plan = adapter.compile(Recipe.from_yaml(RECIPE_PATH), tmp_path / "job")
    model_arg = next(a for a in plan.argv if a.startswith("--pretrained_model_name_or_path="))
    assert model_arg.endswith(str(blob))  # the resolved target, not the symlink
    assert "sd_xl_base_1.0.safetensors" not in model_arg


def test_parse_line_diagnoses_broken_model_path(adapter: KohyaAdapter) -> None:
    event = adapter.parse_line(
        'ValueError: The provided pretrained_model_name_or_path "../../blobs/31e3" '
        "is neither a valid local path nor a valid repo id. Please check the parameter."
    )
    assert event is not None and event.fatal_hint is not None
    assert "base model" in event.fatal_hint
    assert "Download the model again" in event.fatal_hint


def test_parse_line_diagnoses_numpy_mismatch(adapter: KohyaAdapter) -> None:
    # Real lines from a run that died at import time (QA, engine env with
    # NumPy 2): the diagnosis must name the problem and the fix, so the
    # runner can show it instead of a bare exit code 1.
    for line in (
        "ImportError: numpy.core.multiarray failed to import",
        "AttributeError: _ARRAY_API not found",
    ):
        event = adapter.parse_line(line)
        assert event is not None and not event.is_oom
        assert event.fatal_hint is not None
        assert "NumPy" in event.fatal_hint
        assert "loraforge setup" in event.fatal_hint


def test_collect_finds_newest_artifact(adapter: KohyaAdapter, tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir()
    older = out / "run-step500.safetensors"
    newer = out / "run.safetensors"
    older.write_bytes(b"0")
    newer.write_bytes(b"0")
    import os
    import time

    past = time.time() - 100
    os.utime(older, (past, past))

    result = adapter.collect(tmp_path)
    assert result.artifact == newer
    assert result.format == "kohya"


def test_collect_empty_is_a_clear_error(adapter: KohyaAdapter, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"no \.safetensors artifact"):
        adapter.collect(tmp_path)


def test_check_environment_reports_missing_pieces(adapter: KohyaAdapter) -> None:
    problems = adapter.check_environment()
    assert len(problems) == 2  # no sd-scripts checkout, no engine env
    assert any("sd-scripts" in p for p in problems)
