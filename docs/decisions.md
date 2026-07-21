# Design Decisions

Why LoRAForge is built the way it is. Each entry records what we chose, why,
and what we rejected — so future contributors (human or Claude) don't
relitigate settled questions without new evidence. Dates are decision dates.

All decided 2026-07-18 during the project's founding design session unless
noted otherwise.

## 1. NeMo AutoModel is the inspiration, not the engine

We evaluated NVIDIA NeMo AutoModel as the training backend and rejected it
for the mass market: its own docs set a hardware floor of 4× A100 40GB and
128GB RAM; its diffusion PEFT path supports only FLUX/FLUX.2/Qwen-Image/Wan/
Hunyuan (no SD1.5/SDXL — the models 8–16GB cards can actually train); its
QLoRA exists only for LLMs, and it lacks the consumer memory levers (fp8
frozen base for diffusion, block swap); CUDA wheels are Linux-gated. What we
kept from it is architecture: YAML recipes as the single source of truth with
CLI-style dotted overrides, typed/validated configs, offline preprocessing as
a separate stage, and model-specifics stored as data. An AutoModel *adapter*
remains possible later as a pro/cloud backend — the adapter protocol exists
partly for that.

## 2. kohya sd-scripts is the default engine

Chosen over SimpleTuner on the three axes that define our market:
- **License:** kohya is Apache-2.0 (compatible with shipping inside a
  commercial-friendly app); SimpleTuner is AGPL-3.0.
- **VRAM floor:** kohya's block swap + fp8 base reaches FLUX LoRA at ~8–12GB
  and SDXL on 8GB cards — the actual installed base (4060/4070 class).
  SimpleTuner's practical floor is ~16–24GB and it has no SD1.5.
- **Windows:** kohya documents native Windows; SimpleTuner is Linux-first.
Also: kohya's output safetensors are the de-facto community format
(ComfyUI/A1111/Civitai), and community preset knowledge maps onto its knobs.
Cost accepted: we build the job/API layer ourselves (kohya is CLI+TOML only).
Video LoRAs will come via musubi-tuner (same author, same license).

## 3. SimpleTuner, if ever supported, runs as an unmodified external service

AGPL-3.0 means we never import, vendor, or fork it. The only acceptable
integration is an optional adapter that talks to a user-installed, unmodified
SimpleTuner instance over its own REST API. This is a licensing firewall, not
a technical preference. (Not legal advice; a proper legal read is required
before shipping that adapter commercially.)

## 4. Engine adapters are a hard boundary

The app layer never imports torch or engine code. Engines live in their own
uv-managed venvs with their own pins; the boundary is subprocess argv in,
stdout lines out (see `engines/base.py`). Rationale: every engine pins
conflicting library versions (kohya pins older transformers/diffusers than
the app would want); isolation means the app layer rides current Python and
libraries while engines stay frozen and reproducible. It also makes engines
swappable and keeps "add AutoModel later" a weekend job.

## 5. The happy path never requires flash-attn, xformers, or Triton

SDPA (`--sdpa`) everywhere. flash-attn is painful to build on Windows;
Triton (and therefore torch.compile paths) is not officially shipped for
Windows. Since native Windows without WSL is the core product promise,
anything hard to install there cannot be load-bearing. Compile/Triton-based
speedups are detected extras, opt-in, never defaults.

## 6. Capability is data; unavailable options are disabled with a reason

What fits on which card lives in `capability/matrix.yaml`, keyed on
model × free-VRAM × GPU architecture, so the community can extend it with
PRs and CI can test it. The resolver never hides an option: it returns
available (with preset), unavailable (hardware — with a human reason and a
better alternative), or blocked (fixable environment problem — with a repair
hint). Rationale: silently missing options read as bugs; explained
limitations build trust and teach. VRAM thresholds are *free* VRAM,
measured conservatively — the desktop already eats 1–3GB, a fact confirmed
on the founding machine (4090 with 3.1GB gone at idle).

## 7. Torch wheel policy: cu126 default, cu128+ required for Blackwell

RTX 50-series (sm_120) simply does not run on older wheel lines — this is
the one place where "latest CUDA" is a correctness requirement, not polish.
Ada and older run the cu126 line fine; there is no need to force everyone
onto the newest line. Users never install the CUDA toolkit: modern torch
wheels bundle the runtime; only a current NVIDIA driver is required.

## 8. uv everywhere; interpreter speed is a non-goal

uv bootstraps everything, including standalone CPython downloads, so the
user's system Python is never touched or assumed. Install speed is a
first-class product feature (the incumbents lose most users at setup).
We explicitly do not chase "fastest Python": training is GPU-bound and the
interpreter is not the bottleneck; energy goes into cached preprocessing and
fp8 support instead. `uv.lock` is committed because LoRAForge is an
application, not a library — reproducibility beats flexibility.

## 9. Preprocessing is a separate, cached stage

VAE latents and text embeddings are encoded to disk before training
(kohya: `--cache_latents --cache_latents_to_disk`), so the training step
holds only the transformer. Taken directly from NeMo AutoModel's two-stage
design; it is the single biggest VRAM lever available and also speeds up
every subsequent run over the same dataset.

## 10. OOM is a recoverable event with an ordered step-down ladder

The job runner catches OOM, kills the process tree, derives a tighter
recipe, and retries (max 2). Ladder order is by *user-visible cost*,
cheapest first: more block swap (speed only) → enable gradient
checkpointing if off (speed only) → drop resolution a notch (visible in
results) → halve batch (last; presets already run tight batches). Every
rung re-validates through the Recipe schema and recompiles through the
adapter — argv is never hand-mutated. Step-downs are recorded in the job
record so real-world OOMs can tighten the capability matrix over time.

## 11. Desktop app = Tauri shell + managed Python, never a frozen binary

PyInstaller-freezing the ML stack was rejected: multi-GB, breaks torch's
dynamic loading, unmaintainable. The shipped .exe/.deb is a small Tauri
shell that renders the same web UI the server serves at localhost and
bootstraps the Python side with uv on first run. Same pattern as ComfyUI
Desktop / LM Studio. Linux users can skip the shell and use the browser.

## 12. HF tokens are delegated, models download to the shared cache

We never store HF credentials in LoRAForge files or logs; token handling is
delegated to huggingface_hub's own login/token store, so users who already
authenticated for other tools just work. Downloads go to the standard HF
cache so models already pulled by ComfyUI etc. are reused, not re-downloaded.
Gated models (FLUX.1-dev) get an explicit UX: link to the license-acceptance
page, one-line token explanation, clear 401 path. Disk preflight (probe's
free-disk vs download size) runs before any download.

## 13. Cross-platform is enforced, not aspired to

Dev happens on Linux/WSL2 (upstream ecosystems are Linux-first), but native
Windows is a first-class *test* target because that's where users are and
WSL exercises none of the Windows-specific surface (installer, wheels, path
handling, process termination — Windows has no SIGTERM, hence psutil
process-tree cancellation). CI runs Ubuntu + Windows on every push. Rendered
configs use forward slashes; `pathlib` everywhere; no Unix-only shell-outs.
Long-term ambition: a self-hosted GPU runner doing a nightly "download SDXL,
train 50 steps, load the LoRA" smoke test.

## 14. Licensing and naming hygiene

The project is Apache-2.0; in-process dependencies must be Apache/MIT/BSD-
compatible. "NeMo" and "AutoModel" are NVIDIA trademarks — credited as
inspiration in the README, never used in naming. Both training engines we
depend on are effectively single-maintainer projects: engines are pinned to
exact tags (bootstrap clones a pinned ref, never a branch) and upgrades are
treated as tested releases, not tracked continuously.

## 15. Errors speak human, before the run starts

Everything unrepresentable fails at validation or compile time with a
message a non-expert can act on ("resolution must be a multiple of 64",
"'flux_dev' needs component files not yet downloaded — run the model
download step first"). The engine must never discover a config problem at
step 40. Tests pin the actionable words in error messages, not just the
error types.

## 16. Desktop shell is a dumb orchestrator; the webview points at the local server

The Tauri app bundles no UI and holds no logic beyond process/window
lifecycle; it navigates to the same localhost origin the browser uses. One
origin, one code path, testable in Python. Rejected: bundling the React app
into Tauri and calling the API cross-origin — two serving paths to keep
honest for zero user benefit.

## 17. Closing the window stops training, with confirmation

Close during a run asks first, then cancels through the runner's
checkpointed-stop path and shuts the server down. Rejected for now:
tray/background mode (adds Windows background-process expectations and
reattach complexity before anyone has asked for it) and silent kill (losing
a 3-hour run to a misclick contradicts the graceful-degradation brand).
Revisit tray when real users ask for close-and-keep-training.

## 18. Sidecar contract: stdout-announced readiness, endpoint-driven ordered shutdown

`loraforge serve` unconditionally prints a single prefix-tagged ready line —
`LORAFORGE_READY {"url", "port", "pid"}` — once its socket is listening; the
shell matches the prefix, never parses uvicorn logs, and never assumes the
port. `pick_port()` returns a listening socket that uvicorn adopts
(`sockets=[sock]`), so the handshake is race-free. `POST /control/shutdown`
→ 202, then cancel all jobs (keep=True) → stop downloads → flip the exit
flag, in that order, because WS streams only close on terminal events and
uvicorn's graceful shutdown waits for open connections. Shell-side
force-kill of the process group/Job Object comes only after a 25s wait
(the Python cancel path has its own 10s terminate→kill grace before the
exit flag flips) and only as a logged fallback. Preferred port 8471,
ephemeral fallback.

## 19. Engine breakage is repaired as data, and every real-world crash buys a named error

`extra_pins` on the EngineSpec corrects upstream requirement drift
(numpy<2, bitsandbytes==0.49.2 — exact pins per decision 14), recorded in
the state file so existing installs self-repair on the next `setup`. The
kohya adapter owns engine-specific path materialization
(extension-sensitive loader); the downloader stays engine-agnostic. Every
failure reaching the generic exit-code message means a fatal marker is
missing — the marker table grows with each diagnosed crash, tests pinning
the actionable words.

## 20. Capability thresholds come from measured runs; spill is OOM's sneaky sibling

Preset VRAM thresholds are set from measured appetite on real hardware
(runs recorded in matrix.yaml comments), not estimates — the founding
example: "comfortable" passed its own 15GB check yet silently spilled a
24GB card into system RAM at ~46 s/step, because the driver spills
instead of OOMing and the step-down ladder never fired. Silent sysmem
spill is therefore treated as OOM's sneaky sibling: each preset carries a
generous max_seconds_per_step ceiling as matrix data (spill is ~40x, not
~20% — ceilings sit far above any healthy card), the runner takes the
median of steps 3–10 (warmup excluded), and a breach feeds the same
step-down ladder as a pseudo-OOM with a message naming the real cause.

## 21. Text-encoder outputs cache to disk on the tight path; unet-only is the trade

Decision 9 promised cached text embeddings; kohya delivers it via
--cache_text_encoder_outputs_to_disk, which frees ~1.6GB but forces
unet-only training (cached embeddings cannot backprop a text-encoder LoRA)
and requires caption shuffling/dropout off (shuffle is already off). Because
it changes what gets trained, it is preset/recipe data, never a global
default: SDXL tight enables it — an 8GB card cares more about fitting than
text-encoder finetuning, and trigger words still work through
cross-attention — while comfortable/standard keep text-encoder training. It
is also a new rung on the OOM step-down ladder, between gradient
checkpointing and the resolution notch, matching decision 10's
cheapest-visible-cost ordering. The three kohya flags travel together in
compile(); thresholds move only after a measured run per decision 20. FLUX
presets likely want the same treatment (t5xxl) — unmeasured, later.
