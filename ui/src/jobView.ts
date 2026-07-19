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

// A REST-fetched record is the source of truth for a finished job: after a
// server restart there is no event stream left to replay, so a resumed view
// must not wait for one — that freeze is exactly the "training running
// forever" ghost. Live (non-terminal) jobs still start empty and let the
// WS replay fill in history.
export function viewFromRecord(record: {
  state: string;
  error: string | null;
  state_history: { state: string; message: string | null }[];
}): JobView {
  if (!TERMINAL_STATES.includes(record.state)) return emptyJobView;
  const last = record.state_history[record.state_history.length - 1];
  return {
    ...emptyJobView,
    state: record.state,
    terminal: true,
    // failures carry the message in `error`; completions in the last
    // state_history entry ("Training complete — LoRA saved to …")
    finalMessage: record.error ?? last?.message ?? null,
  };
}

// One decision per socket close, from the freshly re-fetched record (null if
// that fetch itself failed): adopt a terminal record's view and stop, or
// keep the reconnect loop alive for a genuinely live job. Pure so the ghost
// can be pinned by tests — a socket the server refuses (job gone from the
// runner after a restart, close 4004) must never retry forever.
export function reconnectDecision(
  view: JobView,
  record: {
    state: string;
    error: string | null;
    state_history: { state: string; message: string | null }[];
  } | null,
): { view: JobView; retry: boolean } {
  if (view.terminal) return { view, retry: false };
  if (record && TERMINAL_STATES.includes(record.state)) {
    return { view: viewFromRecord(record), retry: false };
  }
  return { view, retry: true };
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
