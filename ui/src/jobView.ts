// Fold the job event stream into what the training screen renders.
// Pure and seq-deduplicating, so WS reconnects can replay the full history
// over an existing view without double-counting anything.

import type { JobEventMsg } from "./api";

export const TERMINAL_STATES = ["completed", "completed_early", "failed", "cancelled"];

export interface LossPoint {
  step: number;
  loss: number;
}

export interface Banner {
  seq: number;
  kind: "stepdown" | "final";
  text: string;
}

export interface JobView {
  state: string;
  step: number | null;
  totalSteps: number | null;
  losses: LossPoint[];
  banners: Banner[];
  finalMessage: string | null;
  terminal: boolean;
  maxSeq: number;
}

export const emptyJobView: JobView = {
  state: "queued",
  step: null,
  totalSteps: null,
  losses: [],
  banners: [],
  finalMessage: null,
  terminal: false,
  maxSeq: -1,
};

// ETA from live progress samples. Replayed history arrives in a burst, so a
// span under 5 seconds is not evidence of speed — report "unknown" instead.
export interface RateSample {
  t: number; // wall-clock ms
  step: number;
}

export function etaMinutes(samples: RateSample[], totalSteps: number | null): number | null {
  if (!totalSteps || samples.length < 2) return null;
  const first = samples[0];
  const last = samples[samples.length - 1];
  const seconds = (last.t - first.t) / 1000;
  const steps = last.step - first.step;
  if (seconds < 5 || steps <= 0) return null;
  return Math.ceil((totalSteps - last.step) / (steps / seconds) / 60);
}

export function foldEvents(view: JobView, events: JobEventMsg[]): JobView {
  let next = view;
  for (const event of events) {
    if (event.seq <= next.maxSeq) continue; // replay overlap: already seen
    next = next === view ? { ...view, losses: [...view.losses], banners: [...view.banners] } : next;
    next.maxSeq = event.seq;
    if (event.kind === "progress" && event.progress) {
      const { step, total_steps, loss } = event.progress;
      if (step !== null) next.step = step;
      if (total_steps !== null) next.totalSteps = total_steps;
      if (loss !== null && step !== null) next.losses.push({ step, loss });
    } else if (event.kind === "state") {
      next.state = event.state;
      if (event.state === "oom_stepdown" && event.message) {
        next.banners.push({ seq: event.seq, kind: "stepdown", text: event.message });
      }
      if (TERMINAL_STATES.includes(event.state)) {
        next.terminal = true;
        next.finalMessage = event.message;
      }
    }
  }
  return next;
}
