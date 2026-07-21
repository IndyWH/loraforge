// Assemble the recipe document from the chosen preset + dataset + the few
// advanced knobs. The preset's settings dict comes straight from the
// capability matrix; this only routes each key to its schema section —
// the server validates everything.

import type { ModelCapability, RecipeDoc } from "./api";

export interface AdvancedKnobs {
  rank: number;
  learningRate: number;
  maxSteps: number;
  resolution: number | null; // null → the preset's choice stands
}

export const defaultKnobs: AdvancedKnobs = {
  rank: 16,
  learningRate: 1e-4,
  maxSteps: 1500,
  resolution: null,
};

const TRAIN_KEYS = [
  "batch_size",
  "gradient_checkpointing",
  "fp8_base",
  "blocks_to_swap",
  "max_seconds_per_step", // spill guard: matrix data the runner enforces
  "cache_text_encoder_outputs", // tight-preset rescue: unet-only trade (decision 21)
];

export function buildRecipe(
  name: string,
  preset: ModelCapability,
  datasetPath: string,
  triggerWord: string,
  knobs: AdvancedKnobs,
  previewPrompt: string,
): RecipeDoc {
  const train: Record<string, unknown> = {
    max_steps: knobs.maxSteps,
    sample_every_steps: 0,
  };
  const dataset: Record<string, unknown> = { path: datasetPath };
  for (const [key, value] of Object.entries(preset.settings)) {
    if (key === "resolution") dataset.resolution = value;
    else if (TRAIN_KEYS.includes(key)) train[key] = value;
  }
  if (knobs.resolution !== null) dataset.resolution = knobs.resolution;
  if (triggerWord.trim()) dataset.trigger_word = triggerWord.trim();
  if (previewPrompt.trim()) {
    train.sample_every_steps = 200;
    train.sample_prompts = [previewPrompt.trim()];
  }
  return {
    name,
    model: preset.model_key,
    dataset,
    peft: { rank: knobs.rank, alpha: knobs.rank }, // alpha == rank, the sane default
    optim: { learning_rate: knobs.learningRate },
    train,
    provenance: { preset: preset.preset_name, source: "loraforge-ui" },
  };
}

// "ohwx person" + "sdxl" → "ohwx-person-sdxl", the artifact's file name.
export function defaultLoraName(triggerWord: string, modelKey: string): string {
  const slug = triggerWord.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return slug ? `${slug}-${modelKey}` : "my-first-lora";
}
