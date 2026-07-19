import { describe, expect, it } from "vitest";

import type { JobEventMsg } from "./api";
import {
  emptyJobView,
  etaMinutes,
  foldEvents,
  reconnectDecision,
  viewFromRecord,
} from "./jobView";

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

  it("estimates ETA from live samples but not replay bursts", () => {
    // replay: 100 steps "arrive" in 200ms — no evidence of real speed
    const burst = [
      { t: 1000, step: 0 },
      { t: 1200, step: 100 },
    ];
    expect(etaMinutes(burst, 1500)).toBeNull();
    // live: 100 steps in 100s → 1 step/s → 1400s ≈ 24 min left
    const live = [
      { t: 0, step: 0 },
      { t: 100_000, step: 100 },
    ];
    expect(etaMinutes(live, 1500)).toBe(24);
    expect(etaMinutes(live, null)).toBeNull();
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

describe("viewFromRecord", () => {
  const history = (states: [string, string | null][]) =>
    states.map(([state, message]) => ({ state, message }));

  it("resumes a swept/failed record as terminal — no ghost 'running forever'", () => {
    const view = viewFromRecord({
      state: "failed",
      error:
        "LoRAForge was closed while this job was running; it did not finish. " +
        "Start the job again when you're ready.",
      state_history: history([["queued", null], ["running", null], ["failed", "…"]]),
    });
    expect(view.terminal).toBe(true); // activeJob = job && !terminal → Start unlocks
    expect(view.state).toBe("failed");
    expect(view.finalMessage).toContain("did not finish");
    expect(view.finalMessage).toContain("Start the job again");
  });

  it("resumes a completed record with the completion message from history", () => {
    const view = viewFromRecord({
      state: "completed",
      error: null,
      state_history: history([
        ["running", null],
        ["completed", "Training complete — LoRA saved to /out/lora.safetensors"],
      ]),
    });
    expect(view.terminal).toBe(true);
    expect(view.finalMessage).toContain("Training complete");
  });

  it("leaves a live job empty so the WS replay fills it in", () => {
    const view = viewFromRecord({
      state: "running",
      error: null,
      state_history: history([["queued", null], ["running", null]]),
    });
    expect(view).toEqual(emptyJobView);
  });
});

describe("reconnectDecision", () => {
  const failedRecord = {
    state: "failed",
    error: "LoRAForge was closed while this job was running; it did not finish.",
    state_history: [{ state: "failed", message: null }],
  };
  const liveRecord = { state: "running", error: null, state_history: [] };

  it("refused socket + terminal record → adopt terminal view, stop retrying", () => {
    const decision = reconnectDecision(emptyJobView, failedRecord);
    expect(decision.retry).toBe(false); // never retries forever on a dead stream
    expect(decision.view.terminal).toBe(true); // Start unghosts
    expect(decision.view.finalMessage).toContain("did not finish");
  });

  it("dropped socket + live record → keep the reconnect loop", () => {
    const decision = reconnectDecision(emptyJobView, liveRecord);
    expect(decision.retry).toBe(true);
    expect(decision.view).toBe(emptyJobView); // untouched; WS replay will fill it
  });

  it("record fetch failed → retry (server may just be coming back)", () => {
    expect(reconnectDecision(emptyJobView, null).retry).toBe(true);
  });

  it("already-terminal view → no socket business at all", () => {
    const terminal = viewFromRecord(failedRecord);
    const decision = reconnectDecision(terminal, liveRecord);
    expect(decision.retry).toBe(false);
    expect(decision.view).toBe(terminal);
  });
});
