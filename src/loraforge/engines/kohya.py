"""kohya sd-scripts adapter — the default engine.

Compiles a LoRAForge Recipe into kohya's native launch format: a dataset
config TOML, a sample-prompts file, and an ``accelerate launch`` argv that
runs inside the engine's own uv-managed environment (``env_dir``), against a
checkout of sd-scripts (``sd_scripts_dir``).

Everything model-specific lives in ``MODEL_SPECS`` — adding a model family is
a data edit, mirroring the capability-matrix philosophy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loraforge.engines.base import LaunchPlan, ProgressEvent, TrainResult, venv_bin

if TYPE_CHECKING:
    from pathlib import Path

    from loraforge.recipes.schema import Recipe

# ── Model specifics (data, not code paths) ───────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    script: str
    network_module: str
    base_model_arg: str = "--pretrained_model_name_or_path"
    # HF-hub component assets some scripts need on top of the base checkpoint
    # (filled in by the model downloader): name → CLI flag.
    required_assets: dict[str, str] | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "sd15": ModelSpec(script="train_network.py", network_module="networks.lora"),
    "sdxl": ModelSpec(script="sdxl_train_network.py", network_module="networks.lora"),
    "flux_dev": ModelSpec(
        script="flux_train_network.py",
        network_module="networks.lora_flux",
        required_assets={"clip_l": "--clip_l", "t5xxl": "--t5xxl", "ae": "--ae"},
    ),
}

_OPTIMIZER_MAP = {
    "adamw": "AdamW",
    "adamw8bit": "AdamW8bit",
    "prodigy": "Prodigy",
    "lion": "Lion",
}

# ── Progress parsing ─────────────────────────────────────────────────────────

# kohya/tqdm lines look like:
#   steps:  12%|█▏        | 240/2000 [02:04<15:12, 1.93it/s, avr_loss=0.0821]
_STEP_RE = re.compile(r"(\d+)/(\d+)\s*\[")
_LOSS_RE = re.compile(r"avr_loss=([0-9.eE+-]+)")
_OOM_MARKERS = ("CUDA out of memory", "torch.OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")

# Log lines that predict a fatal exit, mapped to human-worded diagnoses the
# runner shows instead of a bare exit code (rule 5). Markers are data: new
# failure modes get a row + test, not new code paths.
_FATAL_MARKERS: tuple[tuple[str, str], ...] = (
    (
        "numpy.core.multiarray failed to import",
        "The training engine's Python packages are mismatched (NumPy 2 is "
        "installed where NumPy 1 is required). Run `loraforge setup` to repair "
        "the engine environment, then start the job again.",
    ),
    (
        "_ARRAY_API not found",
        "The training engine's Python packages are mismatched (NumPy 2 is "
        "installed where NumPy 1 is required). Run `loraforge setup` to repair "
        "the engine environment, then start the job again.",
    ),
    (
        "is neither a valid local path nor a valid repo id",
        "The training engine couldn't load the base model file — its download "
        "looks broken or incomplete. Download the model again in the Models "
        "step, then start the job again.",
    ),
)


class KohyaAdapter:
    """EngineAdapter implementation for kohya-ss/sd-scripts."""

    name = "kohya"

    def __init__(
        self,
        sd_scripts_dir: Path,
        env_dir: Path,
        model_paths: dict[str, Path] | None = None,
        asset_paths: dict[str, dict[str, Path]] | None = None,
    ) -> None:
        self.sd_scripts_dir = sd_scripts_dir
        self.env_dir = env_dir
        self.model_paths = model_paths or {}  # model key → base checkpoint/repo path
        self.asset_paths = asset_paths or {}  # model key → {asset name → path}

    # ── EngineAdapter protocol ───────────────────────────────────────────────

    def check_environment(self, env_dir: Path | None = None) -> list[str]:
        env_dir = env_dir or self.env_dir
        problems: list[str] = []
        if not self.sd_scripts_dir.is_dir():
            problems.append(f"sd-scripts checkout not found at {self.sd_scripts_dir}")
        accelerate = venv_bin(env_dir, "accelerate")
        if not accelerate.exists():
            problems.append(
                f"engine environment incomplete: {accelerate} missing "
                "(run the installer's engine-setup step)"
            )
        return problems

    def compile(self, recipe: Recipe, workdir: Path) -> LaunchPlan:
        spec = MODEL_SPECS.get(recipe.model)
        if spec is None:
            raise ValueError(
                f"model '{recipe.model}' is not supported by the kohya engine "
                f"(supported: {', '.join(sorted(MODEL_SPECS))})"
            )
        base_model = self.model_paths.get(recipe.model)
        if base_model is None:
            raise ValueError(
                f"base model for '{recipe.model}' has not been downloaded yet — "
                "run the model download step first"
            )
        # The HF cache serves files as snapshot symlinks; kohya readlink()s
        # those to a *relative* blob path and then can't find it. Engines get
        # fully resolved paths, never symlinks.
        base_model = base_model.resolve()

        config_files: dict[Path, str] = {
            workdir / "dataset.toml": self._render_dataset_toml(recipe)
        }

        argv: list[str] = [
            str(venv_bin(self.env_dir, "accelerate")),
            "launch",
            "--num_cpu_threads_per_process", "2",
            str(self.sd_scripts_dir / spec.script),
            f"{spec.base_model_arg}={base_model}",
            f"--dataset_config={workdir / 'dataset.toml'}",
            f"--output_dir={recipe.output_dir}",
            f"--output_name={recipe.name}",
            "--save_model_as=safetensors",
            # PEFT
            f"--network_module={spec.network_module}",
            f"--network_dim={recipe.peft.rank}",
            f"--network_alpha={recipe.peft.alpha:g}",
            # Optimization
            f"--learning_rate={recipe.optim.learning_rate:g}",
            f"--optimizer_type={_OPTIMIZER_MAP[recipe.optim.optimizer]}",
            f"--lr_scheduler={recipe.optim.lr_scheduler}",
            f"--lr_warmup_steps={recipe.optim.warmup_steps}",
            # Train loop
            f"--max_train_steps={recipe.train.max_steps}",
            f"--mixed_precision={recipe.train.mixed_precision}",
            f"--seed={recipe.train.seed}",
        ]

        if recipe.peft.dropout > 0:
            argv.append(f"--network_dropout={recipe.peft.dropout:g}")
        if recipe.train.gradient_checkpointing:
            argv.append("--gradient_checkpointing")
        if recipe.dataset.cache_latents:
            argv += ["--cache_latents", "--cache_latents_to_disk"]
        if recipe.train.fp8_base:
            argv.append("--fp8_base")
        if recipe.train.blocks_to_swap > 0:
            argv.append(f"--blocks_to_swap={recipe.train.blocks_to_swap}")
        if recipe.train.save_every_steps > 0:  # intermediate saves power stop-and-keep
            argv.append(f"--save_every_n_steps={recipe.train.save_every_steps}")
        argv.append("--sdpa")  # cross-platform attention: never require flash/xformers

        if recipe.train.sample_every_steps > 0:
            prompts_path = workdir / "sample_prompts.txt"
            config_files[prompts_path] = "\n".join(recipe.train.sample_prompts) + "\n"
            argv += [
                f"--sample_every_n_steps={recipe.train.sample_every_steps}",
                f"--sample_prompts={prompts_path}",
                "--sample_sampler=euler_a",
            ]

        # Component assets (e.g. FLUX needs clip_l / t5xxl / ae paths)
        if spec.required_assets:
            assets = self.asset_paths.get(recipe.model, {})
            missing = sorted(set(spec.required_assets) - set(assets))
            if missing:
                raise ValueError(
                    f"'{recipe.model}' needs component files not yet downloaded: "
                    f"{', '.join(missing)} — run the model download step first"
                )
            for asset_name, flag in spec.required_assets.items():
                argv.append(f"{flag}={assets[asset_name].resolve()}")  # same symlink rule

        return LaunchPlan(argv=argv, cwd=self.sd_scripts_dir, config_files=config_files)

    def parse_line(self, line: str) -> ProgressEvent | None:
        if any(marker in line for marker in _OOM_MARKERS):
            return ProgressEvent(is_oom=True, message=line.strip())
        for marker, diagnosis in _FATAL_MARKERS:
            if marker in line:
                return ProgressEvent(message=line.strip(), fatal_hint=diagnosis)
        if (m := _STEP_RE.search(line)) is None:
            return None
        loss = _LOSS_RE.search(line)
        return ProgressEvent(
            step=int(m.group(1)),
            total_steps=int(m.group(2)),
            loss=float(loss.group(1)) if loss else None,
        )

    def collect(self, workdir: Path) -> TrainResult:
        candidates = sorted(
            workdir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(
                f"training exited cleanly but no .safetensors artifact found under {workdir}"
            )
        return TrainResult(artifact=candidates[0], format="kohya", logs=workdir / "logs")

    # ── Config rendering ─────────────────────────────────────────────────────

    @staticmethod
    def _render_dataset_toml(recipe: Recipe) -> str:
        ds = recipe.dataset
        caption_hint = (
            'keep_tokens = 1\n' if ds.trigger_word else ""
        )
        # Forward slashes work on every OS in kohya's TOML and avoid escaping.
        image_dir = ds.path.resolve().as_posix()
        return (
            "[general]\n"
            "enable_bucket = true\n"
            f'caption_extension = "{ds.caption_extension}"\n'
            "shuffle_caption = false\n"
            f"{caption_hint}"
            "\n"
            "[[datasets]]\n"
            f"resolution = {ds.resolution}\n"
            f"batch_size = {recipe.train.batch_size}\n"
            "\n"
            "  [[datasets.subsets]]\n"
            f'  image_dir = "{image_dir}"\n'
            f"  num_repeats = {ds.repeats}\n"
        )
