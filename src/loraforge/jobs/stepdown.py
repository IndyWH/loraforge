"""OOM step-down policy — rule 7 made concrete.

When training hits CUDA OOM, the runner asks this module for the next tighter
recipe. One rung per OOM event, in this order:

1. swap more model blocks to system RAM, if the model supports it (18 → 34):
   big VRAM win, costs only speed, never quality;
2. enable gradient checkpointing, if it is off: also costs only speed, so it
   must come before anything visible in results;
3. drop resolution one notch (1024 → 768 → 512): visible in results, so it
   comes after every speed-only lever;
4. halve the batch size: last, because presets already run tight batches and
   halving 1 is impossible.

Every rung returns a *validated* recipe (``with_overrides`` re-runs pydantic)
plus a plain-language message the runner shows the user. ``None`` means the
ladder is exhausted and the job should fail with an honest explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from loraforge.recipes.schema import Recipe

# Models whose kohya script accepts --blocks_to_swap (data, not code paths).
SUPPORTS_BLOCK_SWAP = frozenset({"flux_dev"})

_BLOCK_SWAP_RUNGS = (18, 34)
_RESOLUTION_NOTCHES = (1024, 768, 512)  # drop to the next notch below current


@dataclass(frozen=True)
class StepDown:
    recipe: Recipe  # the tighter, re-validated recipe to retry with
    message: str  # plain language, shown to the user before the retry


def vram_knobs(recipe: Recipe) -> dict[str, Any]:
    """The VRAM-relevant settings, for job records and matrix feedback."""
    return {
        "resolution": recipe.dataset.resolution,
        "batch_size": recipe.train.batch_size,
        "blocks_to_swap": recipe.train.blocks_to_swap,
        "gradient_checkpointing": recipe.train.gradient_checkpointing,
    }


def step_down(recipe: Recipe) -> StepDown | None:
    """The next tighter recipe after an OOM, or None if nothing is left."""
    train, dataset = recipe.train, recipe.dataset

    if recipe.model in SUPPORTS_BLOCK_SWAP:
        rung = next((b for b in _BLOCK_SWAP_RUNGS if b > train.blocks_to_swap), None)
        if rung is not None:
            overrides: dict[str, Any] = {"train.blocks_to_swap": rung}
            message = (
                f"Keeping {rung} model blocks in your computer's RAM instead of VRAM "
                "(training gets slower, not worse)."
            )
            if train.batch_size > 2:  # schema forbids block swap with batch_size > 2
                overrides["train.batch_size"] = 2
                message += " Batch size lowered to 2 to go with it."
            return StepDown(recipe.with_overrides(overrides), message)

    if not train.gradient_checkpointing:
        return StepDown(
            recipe.with_overrides({"train.gradient_checkpointing": True}),
            "Turning on gradient checkpointing "
            "(slower steps, much less VRAM, identical results).",
        )

    notch = next((n for n in _RESOLUTION_NOTCHES if n < dataset.resolution), None)
    if notch is not None:
        return StepDown(
            recipe.with_overrides({"dataset.resolution": notch}),
            f"Lowering training resolution from {dataset.resolution} to {notch}. "
            "The LoRA will still work at full resolution when you generate images.",
        )

    if train.batch_size > 1:
        half = train.batch_size // 2
        return StepDown(
            recipe.with_overrides({"train.batch_size": half}),
            f"Lowering batch size from {train.batch_size} to {half} "
            "(same training quality, smaller VRAM bites).",
        )

    return None
