"""The recipe — LoRAForge's single source of truth for a training run.

Borrowed straight from NeMo AutoModel's design: one validated document that
the UI edits, the CLI overrides, users share, and the engine adapter compiles
into its native format (kohya TOML+args, SimpleTuner config, ...).

Validation philosophy: errors must read like "peft.alpha must be a number",
not a stack trace 40 seconds into a run — hence pydantic with tight types
and cross-field checks up front.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


def validation_messages(exc: ValidationError) -> list[str]:
    """Flatten a ValidationError into our human-worded messages, verbatim.

    Strips pydantic's "Value error, " framing so what the validators said is
    exactly what the user reads, prefixed with the dotted field path.
    """
    messages: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        msg = error["msg"].removeprefix("Value error, ").removeprefix("Assertion failed, ")
        messages.append(f"{loc}: {msg}" if loc else msg)
    return messages


class PeftSection(BaseModel):
    rank: int = Field(16, ge=1, le=1024, description="LoRA rank (dim)")
    alpha: float = Field(16, gt=0, description="LoRA alpha; commonly == rank")
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    target: Literal["attn", "attn+mlp", "all-linear"] = "attn+mlp"


class OptimSection(BaseModel):
    learning_rate: float = Field(1e-4, gt=0, lt=1)
    optimizer: Literal["adamw", "adamw8bit", "prodigy", "lion"] = "adamw8bit"
    lr_scheduler: Literal["cosine", "constant", "constant_with_warmup"] = "cosine"
    warmup_steps: int = Field(0, ge=0)


class DatasetSection(BaseModel):
    path: Path
    caption_extension: str = ".txt"
    trigger_word: str | None = None
    repeats: int = Field(1, ge=1, le=100)
    resolution: int = Field(1024, description="Bucket base resolution")
    cache_latents: bool = True  # the AutoModel lesson: preprocess offline, always

    @field_validator("resolution")
    @classmethod
    def _sane_resolution(cls, v: int) -> int:
        if v % 64 != 0 or not 256 <= v <= 2048:
            raise ValueError("resolution must be a multiple of 64 between 256 and 2048")
        return v


class TrainSection(BaseModel):
    max_steps: int = Field(1500, ge=1)
    batch_size: int = Field(1, ge=1, le=64)
    gradient_checkpointing: bool = True
    mixed_precision: Literal["bf16", "fp16"] = "bf16"
    fp8_base: bool = False  # resolver flips this on for Ada/Blackwell presets
    blocks_to_swap: int = Field(0, ge=0, le=57, description="kohya block swap (VRAM↓, RAM↑)")
    cache_text_encoder_outputs: bool = Field(
        False,
        description="Cache text-encoder outputs to disk (~1.6GB VRAM freed); "
        "forces unet-only training — the text encoder is not fine-tuned",
    )
    seed: int = 42
    save_every_steps: int = Field(
        200, ge=0, description="Intermediate checkpoint cadence; powers stop-and-keep. 0 disables"
    )
    sample_every_steps: int = Field(200, ge=0, description="0 disables preview samples")
    sample_prompts: list[str] = Field(default_factory=list)
    max_seconds_per_step: float = Field(
        0,
        ge=0,
        description="Spill guard: steady-state s/step ceiling from the capability "
        "matrix; a breach is treated as OOM's sneaky sibling. 0 disables",
    )


class Recipe(BaseModel):
    """A complete, shareable training run definition."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=120)
    model: str = Field(description="Capability-matrix key, e.g. 'sdxl', 'flux_dev'")
    engine: str = "kohya"
    peft: PeftSection = Field(default_factory=PeftSection)
    optim: OptimSection = Field(default_factory=OptimSection)
    dataset: DatasetSection
    train: TrainSection = Field(default_factory=TrainSection)
    output_dir: Path = Path("outputs")
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Filled by the app: card, preset, app version — for shareability",
    )

    @model_validator(mode="after")
    def _cross_checks(self) -> Recipe:
        if self.train.sample_every_steps and not self.train.sample_prompts:
            raise ValueError(
                "train.sample_prompts must not be empty when sample_every_steps > 0 "
                "(or set sample_every_steps: 0 to disable previews)"
            )
        if self.train.blocks_to_swap > 0 and self.train.batch_size > 2:
            raise ValueError(
                "blocks_to_swap is a low-VRAM measure; batch_size > 2 defeats it — "
                "lower the batch size or disable block swap"
            )
        return self

    # ── I/O ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path) -> Recipe:
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: Path) -> None:
        path.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def with_overrides(self, overrides: dict[str, Any]) -> Recipe:
        """Apply dotted-path overrides, AutoModel-style: {'peft.rank': 32}."""
        data = self.model_dump(mode="python")
        for dotted, value in overrides.items():
            node = data
            *parents, leaf = dotted.split(".")
            for part in parents:
                node = node[part]
            node[leaf] = value
        return Recipe.model_validate(data)
