# CLAUDE.md — LoRAForge

Project context for Claude Code sessions. Read this before touching anything.

## What this project is

A hardware-aware, recipe-driven LoRA training studio for NVIDIA **gaming**
cards (8–32GB VRAM), Windows and Linux, no WSL required. Web UI + local job
server wrapping proven community training engines. Architecturally inspired by
NVIDIA NeMo AutoModel (YAML recipes, typed configs, preprocessing as a
separate stage, model-specifics as data) but sized for a single consumer GPU.
Public repo: https://github.com/IndyWH/loraforge — Apache-2.0.

**Target user:** someone with an RTX 3060/4070/4090/5080 on Windows who wants
to train a LoRA without reading a 40-step tutorial. The three problems we
exist to solve: brutal installation, unexplained parameter walls, and configs
that OOM because they were written for someone else's GPU.

## How this project is developed

LoRAForge is built by a three-way team — Wajira (product owner / QA Oracle:
decides what to build, judges results on real hardware, relays messages
between the AIs), Claude Cowork (design and review: writes design briefs
into docs/design/, reviews diffs, maintains docs/decisions.md wording), and
Claude Code (implementation: writes the code and tests, commits, flags
deviations from the brief for review). Design briefs in docs/design/ are
the contract; code follows the brief, and approved as-built deviations get
folded back into the brief. Wajira prefers minimal reading — reports to him
should be short verdicts, not essays.

## Architecture

```
Tauri desktop shell (Windows/Linux) — renders the web UI, ~few MB
  └─ FastAPI job server (this package, "app layer")
       ├─ probe.py               hardware diagnostics (GPU/VRAM/arch/driver/torch/RAM/disk)
       ├─ capability/            matrix.yaml (DATA) + resolver → what fits on this card
       ├─ recipes/               pydantic Recipe schema — single source of truth for a run
       └─ engines/               EngineAdapter protocol + one adapter per engine
            └─ kohya (default)   compile() → dataset TOML + accelerate argv
  └─ Engine environments — each engine gets its OWN uv-managed venv with its
     own pins; the app layer NEVER imports torch or engine code directly.
     App ↔ engine boundary is: subprocess argv in, stdout lines out.
```

Bootstrap story: the installer/first run uses **uv** to create environments,
including downloading a standalone CPython — never touch the user's system
Python. Detect GPU generation first, then install the matching torch wheel
line (cu126 for older cards; cu128+ REQUIRED for Blackwell/RTX 50, sm_120).

## Design rules (violating these is a bug, not a style choice)

1. **The recipe is the single source of truth.** Every run is a validated
   YAML Recipe. UI edits it, CLI overrides it (`--peft.rank 32`, dotted
   paths), users share it. Provenance (card, preset, app version) gets
   stamped into the recipe.
2. **Capability is data, not code.** What fits on which card lives in
   `capability/matrix.yaml` so the community can extend it via PR. Resolver
   logic stays generic. VRAM numbers in the matrix are FREE-VRAM-required,
   measured conservatively (desktop running, latents precached).
3. **Never hide, disable with a reason.** UI options the hardware can't do
   are greyed out with a human-actionable explanation ("FLUX needs ~11GB
   free; SDXL is your best option on this card"). Three availability states:
   available / unavailable (hardware) / blocked (fixable environment problem,
   e.g. Blackwell card on a pre-cu128 wheel → offer repair).
4. **The happy path never requires flash-attn, xformers, or Triton.** SDPA
   (`--sdpa`) everywhere. Compile/Triton-based speedups are detected extras,
   opt-in, never load-bearing. This is what keeps native Windows first-class.
5. **Errors speak human, before the run starts.** Pydantic validation with
   messages like "resolution must be a multiple of 64". compile() must catch
   everything unrepresentable — the engine must never discover a config
   problem at step 40.
6. **Preprocessing is a separate stage.** Cache VAE latents + text embeddings
   to disk before training (kohya: `--cache_latents --cache_latents_to_disk`).
   Biggest VRAM lever we have.
7. **OOM is a recoverable event.** parse_line() flags OOM lines; the job
   runner (when built) catches them, steps down to the next tighter preset
   (lower resolution / more blocks_to_swap), retells the user in plain
   language, and retries. A trainer that degrades gracefully is the brand.
8. **Cross-platform from every commit.** pathlib everywhere; no Unix-only
   shell-outs; forward slashes in rendered TOML (Windows backslashes must
   not leak into configs); `_venv_bin()` for Scripts/ vs bin/. CI runs
   Ubuntu + Windows and must stay green on both.
9. **Model-specifics are data.** Engine adapters keep per-model details in
   tables (see `MODEL_SPECS` in engines/kohya.py). Adding a model = data
   edit + matrix entry + test, not new code paths.
10. **Laptop GPUs get margin.** Probe flags them; resolver applies ~10% VRAM
    headroom and warns. Multi-GPU: pick one device, don't build distributed.

## Licensing constraints (hard)

- This repo: Apache-2.0. Everything vendored/imported in-process must be
  Apache/MIT/BSD-compatible.
- kohya sd-scripts (Apache-2.0): fine to drive as the default engine.
  musubi-tuner (same author, video): fine, planned later.
- **SimpleTuner is AGPL-3.0**: NEVER import it, vendor it, or modify it.
  If/when supported, it runs as an unmodified external service reached over
  its own REST API, optional, user-installed. Get this wrong and the whole
  project has a licensing problem.
- Don't use "NeMo"/"AutoModel" in naming — NVIDIA trademarks. Crediting the
  architectural inspiration in the README is fine and already done.
- HF gated models (FLUX.1-dev): user must accept the license on HF and
  provide their own token. Token is stored locally, never committed, never
  logged. `.env` is gitignored — keep it that way.

## Current state (as of this file)

Done and tested (pytest + vitest + cargo suites, ruff clean, CI on
Ubuntu+Windows incl. frontend and desktop jobs):
- `probe.py` — never crashes, degrades to notes; arch mapping sm→generation
- `capability/matrix.yaml` + `resolver.py` — sd15/sdxl/flux_dev presets.
  SDXL thresholds are MEASURED (decision 20): comments in matrix.yaml record
  the runs; standard=11400 (10.4GB appetite × ~10% headroom); tight enables
  cache_text_encoder_outputs (decision 21). Each preset carries a
  max_seconds_per_step spill-guard ceiling.
- `recipes/schema.py` — Recipe with YAML round-trip + dotted overrides;
  train.cache_text_encoder_outputs (unet-only trade, decision 21)
- `engines/base.py` — EngineAdapter protocol (compile/parse_line/collect/
  check_environment); LaunchPlan/ProgressEvent/TrainResult dataclasses
- `engines/kohya.py` — compile to accelerate argv + dataset TOML; tqdm/OOM
  parsing; artifact collection; _FATAL_MARKERS table mapping engine log
  lines to human diagnoses (rule 5 — grows with each diagnosed crash);
  _engine_loadable() materializes HF-cache symlinks into real *.safetensors
  paths kohya can load; TE-cache flags emitted as an inseparable trio
- `cli.py` — `loraforge diagnose` (also the bug-report generator, `--json`);
  `loraforge setup [--dry-run]` (engine bootstrap)
- `engines/bootstrap.py` — plan-then-execute engine setup: pinned sd-scripts
  clone (v0.9.1), uv venv on managed CPython, GPU-matched torch wheels
  (cu126/cu128), state file for idempotence + repair; extra_pins repair
  upstream drift as data (numpy<2, bitsandbytes==0.49.2 — decision 19)
- `jobs/` — transport-free async job runner: FIFO queue (one job at a time),
  explicit state machine (queued→preparing→running→terminal, oom_stepdown
  loops back), per-job event streams ending in a terminal event, job.json
  record + job.log in each job's workdir, psutil process-tree cancellation,
  submit preflight (refuse before a job exists), stale-record sweep at
  startup, carriage-return splitting for live tqdm steps/ETA. Spill guard:
  median s/step of steps 3–10 over the preset ceiling → same ladder as OOM,
  message names the real cause (decision 20).
  `jobs/stepdown.py` — OOM ladder: blocks_to_swap (18→34) → gradient
  checkpointing (if off) → cache text-encoder outputs (sdxl; decision 21) →
  resolution notch (1024→768→512) → halve batch; max 2 retries,
  human-worded failures.
- `downloader.py` — snapshot_download into the shared HF cache (ComfyUI
  reuse), model/asset sources as `source:` blocks in matrix.yaml, disk
  preflight, gated-401 message with license URL + `hf auth login` step
  (tokens live only in HF's own store), typed events with terminal
  guarantee, adapter_paths() → KohyaAdapter wiring. huggingface_hub 1.x.
- `server/` — thin FastAPI wrapper (routes translate, never decide):
  /diagnose, /models (+download POST, WS events), /recipes CRUD+validate
  (human errors verbatim), /jobs (submit/list/get/cancel, WS events
  wrapping runner.events()). Loopback-only bind unless --allow-remote
  (run.ensure_local_bind). DI via ServerDeps; real wiring in
  run.build_default_deps(). CLI: `loraforge serve`. Hardening:
  security.LocalRequestsOnly rejects non-loopback Host (DNS rebinding)
  and cross-origin writes/WS (loopback + Tauri origins allowed; no-Origin
  passes); GET /jobs/{id}/artifact serves the trained LoRA.
- `datasets/` — torch-free dataset prep (pillow): ingest copies into
  data_root/datasets (originals never moved), sha256 exact-dup skip,
  128-bit dHash near-dup warnings, quality checks (<256px excluded,
  <512px warned, unreadable excluded — human reasons), .txt caption
  sidecars with whole-word trigger injection, DatasetSummary whose path
  a recipe references. `captioner.py` — Captioner protocol (engine-style
  subprocess design) + stub; auto-captioning itself NOT implemented yet.
  Server routes: /datasets CRUD, images ingest, captions get/put,
  trigger-word.

- `ui/` — web UI phase A (Vite + React + TS, hand-rolled CSS, no component
  lib): six scroll-unlock sections (hardware → model → photos → configure →
  train → finish), receipt chips, disable-with-reason model cards, zero-knob
  configure + advanced accordion, multipart drag-drop upload, caption editor,
  trigger word, validate-then-submit, jobs WS with seq-deduped
  reconnect-and-replay (refresh mid-training resumes), step-down banners,
  stop-and-keep, artifact download, copy-paste prompt. Built bundle served
  by FastAPI at / (ServerDeps.ui_dist); `npm run dev` (5173) hits the API
  cross-origin (CORS grants loopback origins only). Pure logic lives in
  jobView.ts/recipe.ts with vitest tests. CI has a frontend job.
- `desktop/` — Tauri v2 shell, stages 1+2: sidecar crate owning the server
  lifecycle (spawn, LORAFORGE_READY handshake per decision 18, ordered
  shutdown via /control/shutdown, 25s wait then process-group/Job-Object
  kill), window→UI wiring, close-during-run confirmation dialog,
  single-instance focus, occupied-port fallback, desktop CI job. Real GPU
  training reached end-to-end 2026-07-19. Packaged-install/first-run
  bootstrap UX (stage B) NOT started.

## Roadmap (in order — don't skip ahead)

1. **Tauri shell stage B** — packaged install: installer story (bundle uv,
   first-run engine setup flow); walk the acceptance checklist in
   docs/design/tauri-shell.md.
2. **Captioner engine** — Florence-2 / WD14 adapters implementing
   datasets/captioner.Captioner, bootstrapped like training engines; then
   wire preview sample images into the training filmstrip.
3. Later: musubi-tuner adapter (video), LoRA format converters, sample
   gallery, community recipe sharing.

Nightly ambition once hardware CI exists (self-hosted runner on a real GPU):
"download SDXL, train 50 steps, produce a loadable LoRA" as a smoke test.

## Dev commands

```bash
uv venv && uv sync --extra dev
uv run pytest -q            # all tests, fast, no GPU needed
uv run ruff check src tests
uv run loraforge diagnose   # hardware + capability report
uv run loraforge serve      # API + web UI at http://127.0.0.1:8471

cd ui && npm install
npm run dev                 # UI dev server on 5173 (API must be running)
npm run typecheck && npm test && npm run build

cd desktop && cargo test    # desktop shell: sidecar lifecycle tests (no webkit needed)
cargo test -- --ignored     # + the full contract against the real python server
cargo fmt --check && cargo clippy --all-targets -- -D warnings
cargo install tauri-cli --version '^2' --locked
cargo tauri dev             # run the desktop shell (needs the webkit deps below)
```

Desktop shell (Tauri v2) Linux/WSL build deps — needed for `tauri dev`/`tauri
build` (the app crate), NOT for the sidecar tests above:

```bash
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libxdo-dev libssl-dev libayatana-appindicator3-dev librsvg2-dev
```

## Testing conventions

- Tests assert BEHAVIOR users care about: which preset a 4060 gets, that
  error messages contain the actionable word ("download", "clip_l"), that
  Windows paths can't corrupt TOML, that OOM is detected from real lines.
- Fake hardware via `fake_report()` in tests/test_capability.py — add new
  cards there. Never require a real GPU for the unit suite.
- Keep the suite fast (<1s aspirational; ~1.1s as of the server control
  endpoints). GPU/integration tests live elsewhere (later, behind a
  marker), so `uv run pytest` stays instant on CI.
