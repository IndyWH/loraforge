import { describe, expect, it } from "vitest";

import type { ModelCapability } from "./api";
import { buildRecipe, defaultKnobs, defaultLoraName } from "./recipe";

const sdxlPreset: ModelCapability = {
  model_key: "sdxl",
  display_name: "Stable Diffusion XL",
  status: "available",
  preset_name: "tight",
  settings: { resolution: 768, batch_size: 1, gradient_checkpointing: true },
  min_free_vram_mb: 7000,
  reason: null,
};

describe("buildRecipe", () => {
  it("routes preset settings into their schema sections", () => {
    const doc = buildRecipe("my-lora", sdxlPreset, "/data/cats", "sks-cat", defaultKnobs, "");
    expect(doc.model).toBe("sdxl");
    expect(doc.dataset).toMatchObject({ path: "/data/cats", resolution: 768, trigger_word: "sks-cat" });
    expect(doc.train).toMatchObject({ batch_size: 1, gradient_checkpointing: true, max_steps: 1500 });
    expect(doc.peft).toEqual({ rank: 16, alpha: 16 });
  });

  it("only enables preview sampling when a prompt is given (schema demands prompts)", () => {
    const silent = buildRecipe("x", sdxlPreset, "/d", "", defaultKnobs, "");
    expect(silent.train).toMatchObject({ sample_every_steps: 0 });
    expect(silent.dataset).not.toHaveProperty("trigger_word");

    const sampling = buildRecipe("x", sdxlPreset, "/d", "", defaultKnobs, "a sks-cat photo");
    expect(sampling.train).toMatchObject({
      sample_every_steps: 200,
      sample_prompts: ["a sks-cat photo"],
    });
  });

  it("lets the advanced resolution knob override the preset", () => {
    const doc = buildRecipe("x", sdxlPreset, "/d", "", { ...defaultKnobs, resolution: 1024 }, "");
    expect(doc.dataset).toMatchObject({ resolution: 1024 });
  });
});

describe("defaultLoraName", () => {
  it("slugs the trigger word with the model key", () => {
    expect(defaultLoraName("ohwx person", "sdxl")).toBe("ohwx-person-sdxl");
    expect(defaultLoraName("", "sdxl")).toBe("my-first-lora");
  });
});
