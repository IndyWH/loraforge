import { useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { JobRecord } from "../api";
import type { JobView, LossPoint } from "../jobView";

export function TrainSection(props: { unlocked: boolean; job: JobRecord | null; view: JobView }) {
  const [stopping, setStopping] = useState(false);
  const { view, job } = props;
  const pct =
    view.step !== null && view.totalSteps ? Math.min(100, (view.step / view.totalSteps) * 100) : 0;

  const stop = async (keep: boolean) => {
    if (!job) return;
    setStopping(true);
    try {
      await api.cancel(job.id, keep);
    } finally {
      setStopping(false);
    }
  };

  const phase =
    view.state === "queued"
      ? "Waiting for the GPU…"
      : view.state === "preparing"
        ? "Preparing configs and caches…"
        : view.state === "running"
          ? "Training"
          : view.state === "oom_stepdown"
            ? "Adjusting settings…"
            : "";

  return (
    <Step
      num={5}
      title="Training"
      lede="Watch it learn. You can close this page — training continues, and this screen picks back up where it was."
      unlocked={props.unlocked}
      lockHint="Start a training run above."
    >
      {view.banners.map((banner) => (
        <div key={banner.seq} className="banner">
          {banner.text}
        </div>
      ))}

      <div className="progresslabel">
        <span>{phase}</span>
        <span>
          {view.step ?? 0} / {view.totalSteps ?? "…"} steps
        </span>
      </div>
      <div className="progressbar">
        <div style={{ width: `${pct}%` }} />
      </div>

      <div className="filmstrip">
        {[1, 2, 3, 4].map((slot) => (
          <div key={slot} className="filmframe">
            {view.losses.length === 0
              ? "previews appear here as training progresses"
              : "preview rendering lands with the sample gallery"}
          </div>
        ))}
      </div>

      <LossChart losses={view.losses} />

      {!view.terminal && (
        <div className="row" style={{ marginTop: 14 }}>
          <button className="primary" disabled={stopping} onClick={() => void stop(true)}>
            Stop &amp; keep what it learned
          </button>
          <button className="danger" disabled={stopping} onClick={() => void stop(false)}>
            Cancel, keep nothing
          </button>
          <span className="smallnote">
            “Stop &amp; keep” grabs the newest saved checkpoint — you lose minutes, not the LoRA.
          </span>
        </div>
      )}
    </Step>
  );
}

function LossChart({ losses }: { losses: LossPoint[] }) {
  if (losses.length < 2) {
    return <p className="smallnote">The loss curve draws here once numbers start flowing.</p>;
  }
  const w = 800;
  const h = 90;
  const min = Math.min(...losses.map((p) => p.loss));
  const max = Math.max(...losses.map((p) => p.loss));
  const span = max - min || 1;
  const x = (i: number) => (i / (losses.length - 1)) * w;
  const y = (loss: number) => h - 6 - ((loss - min) / span) * (h - 12);
  const path = losses.map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p.loss).toFixed(1)}`).join(" ");
  return (
    <div className="losschart">
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" />
      </svg>
      <div className="caption">
        avg loss {losses[losses.length - 1].loss.toFixed(4)} — trending is what matters, not the exact number
      </div>
    </div>
  );
}
