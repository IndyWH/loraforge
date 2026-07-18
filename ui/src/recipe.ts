// Assemble the recipe document from the chosen preset + dataset + the few
// advanced knobs. The preset's settings dict comes straight from the
// capability matrix; this only routes each key to its schema section —
// the server validates everything.

import type { ModelCapability, RecipeDoc } from "./api";

export interface AdvancedKnobs {
  rank: number;
  learningRate: number;
  maxSteps: number;
}

export const defaultKnobs: AdvancedKnobs = { rank: 16, learningRate: 1e-4, maxSteps: 1500 };

const TRAIN_KEYS = ["batch_size", "gradient_checkpointing", "fp8_base", "blocks_to_swap"];

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
