import { describe, expect, it } from "vitest";

import type { JobEventMsg } from "./api";
import { emptyJobView, foldEvents } from "./jobView";

let seq = 0;
const state = (s: string, message: string | null = null): JobEventMsg => ({
  job_id: "j1", seq: seq++, kind: "state", state: s, progress: null, message,
});
const progress = (step: number, total: number, loss: number | null = null): JobEventMsg => ({
  job_id: "j1", seq: seq++, kind: "progress", state: "running",
  progress: { step, total_steps: total, loss, message: null, sample_image: null, is_oom: false },
  message: null,
});

describe("foldEvents", () => {
  it("tracks progress, loss series and terminal state", () => {
    seq = 0;
    const events = [
      state("queued"), state("preparing"), state("running"),
      progress(1, 100, 0.3), progress(2, 100, 0.25),
      state("completed", "Training complete — LoRA saved to /x/lora.safetensors"),
    ];
    const view = foldEvents(emptyJobView, events);
    expect(view.step).toBe(2);
    expect(view.totalSteps).toBe(100);
    expect(view.losses).toEqual([{ step: 1, loss: 0.3 }, { step: 2, loss: 0.25 }]);
    expect(view.terminal).toBe(true);
    expect(view.state).toBe("completed");
    expect(view.finalMessage).toContain("lora.safetensors");
  });

  it("collects step-down banners with their friendly messages", () => {
    seq = 0;
    const view = foldEvents(emptyJobView, [
      state("running"),
      state("oom_stepdown", "Your GPU ran out of memory. Lowering resolution. Retrying (1/2)."),
      state("running"),
    ]);
    expect(view.banners).toHaveLength(1);
    expect(view.banners[0].text).toContain("ran out of memory");
    expect(view.terminal).toBe(false);
  });

  it("deduplicates replayed events by seq across reconnects", () => {
    seq = 0;
    const first = [state("queued"), state("running"), progress(5, 100, 0.2)];
    const view1 = foldEvents(emptyJobView, first);
    // reconnect: server replays everything, then continues
    seq = 0;
    const replayPlusLive = [
      state("queued"), state("running"), progress(5, 100, 0.2),
      progress(6, 100, 0.19), state("completed", "done"),
    ];
    const view2 = foldEvents(view1, replayPlusLive);
    expect(view2.losses).toHaveLength(2); // no double-counted points
    expect(view2.step).toBe(6);
    expect(view2.terminal).toBe(true);
  });

  it("treats completed_early as terminal (stop-and-keep)", () => {
    seq = 0;
    const view = foldEvents(emptyJobView, [
      state("running"),
      state("completed_early", "Training stopped early — kept the latest saved LoRA"),
    ]);
    expect(view.terminal).toBe(true);
    expect(view.state).toBe("completed_early");
  });
});
