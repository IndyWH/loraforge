# LoRAForge

> Train LoRAs on the GPU you actually own.

*(Working title — rename freely.)*

A hardware-aware, recipe-driven LoRA training studio for NVIDIA gaming cards.
One-click install on Windows and Linux, no WSL required. Inspired by the recipe
architecture of [NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel),
sized for a single consumer GPU.

## Why another trainer?

Every existing tool fails users the same three ways: brutal installation,
walls of unexplained parameters, and configs silently copied from a 4090
tutorial onto an 8GB laptop card. LoRAForge attacks exactly those three:

1. **Diagnose first.** On startup we probe your GPU, VRAM, driver, torch/CUDA
   build, RAM, and disk — then tell you *what you can train* before you waste
   a minute. Unavailable options are disabled **with a reason**, never hidden.
2. **Recipes, not knob soup.** Every run is a validated YAML recipe resolved
   against a community-maintained capability matrix (`model × VRAM tier ×
   GPU architecture`). Known-good presets per card; an advanced pane for
   tinkerers, pre-filled with safe values.
3. **Install that just works.** A small desktop shell bootstraps a pinned
   Python environment with [uv](https://github.com/astral-sh/uv) — including
   the correct torch wheel line for your GPU generation (cu126 / cu128+ for
   Blackwell). First run in about a minute, not an afternoon.

## Architecture

```
┌────────────────────────────────────────────────┐
│  Desktop shell (Tauri) / browser → localhost   │
├────────────────────────────────────────────────┤
│  FastAPI job server  (this package)            │
│    probe.py        hardware diagnostic         │
│    capability/     matrix.yaml + resolver      │
│    recipes/        pydantic recipe schema      │
│    engines/        adapter protocol            │
├────────────────────────────────────────────────┤
│  Engine environments (isolated, uv-managed)    │
│    kohya sd-scripts   ← default (Apache-2.0)   │
│    musubi-tuner       ← video (later)          │
│    SimpleTuner        ← optional ext. service  │
└────────────────────────────────────────────────┘
```

Principles borrowed from NeMo AutoModel:

- **The recipe is the single source of truth.** Shareable, reproducible,
  overridable from the CLI/UI. Imagine `4070-flux-character.yaml` shared on
  Civitai the way presets are shared today.
- **Preprocessing is a separate stage.** VAE latents and text embeddings are
  cached to disk before training; the training step loads only the
  transformer. This is the single biggest VRAM lever available.
- **Capability data, not capability code.** What fits on which card is a
  versioned YAML file the community can extend with PRs — not logic buried
  in UI code.
- **Typed configs.** Validation errors read like "peft.alpha must be a
  number", not a stack trace 40 seconds into a run.

## Status

Pre-alpha skeleton. The probe, capability resolver, recipe schema, and engine
adapter protocol are real and tested; the server, UI, and kohya adapter are
next.

## Development

```bash
uv venv && uv sync --extra dev
uv run pytest
uv run loraforge diagnose        # prints your hardware capability report
```

Cross-platform rules from the first commit: `pathlib` everywhere, no
Unix-only shell-outs, CI on Linux **and** Windows, and the happy path never
requires flash-attn or Triton (SDPA works everywhere; compile-based speedups
are detected extras).

## License

Apache-2.0.
