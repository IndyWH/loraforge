import { useEffect, useRef, useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { JobRecord } from "../api";
import { etaMinutes } from "../jobView";
import type { JobView, LossPoint, RateSample } from "../jobView";

export function TrainSection(props: { unlocked: boolean; job: JobRecord | null; view: JobView }) {
  const [stopping, setStopping] = useState(false);
  const [vram, setVram] = useState<{ used: number; total: number } | null>(null);
  const samples = useRef<RateSample[]>([]);
  const { view, job } = props;

  // Live rate samples for the ETA; replay bursts are filtered in etaMinutes.
  useEffect(() => {
    if (view.step === null) return;
    const now = Date.now();
    samples.current = [...samples.current, { t: now, step: view.step }].filter(
      (s) => now - s.t < 90_000,
    );
  }, [view.step]);

  // VRAM panel: the probe is the source of truth; poll it while training.
  useEffect(() => {
    if (!props.unlocked || view.terminal) return;
    let cancelled = false;
    const poll = () =>
      api
        .diagnose()
        .then((d) => {
          const gpu = d.hardware.gpus[0];
          if (gpu && !cancelled)
            setVram({
              used: (gpu.vram_total_mb - gpu.vram_free_mb) / 1024,
              total: gpu.vram_total_mb / 1024,
            });
        })
        .catch(() => undefined);
    poll();
    const timer = setInterval(poll, 15_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [props.unlocked, view.terminal]);

  const pct = view.step !== null && view.totalSteps ? Math.min(100, (view.step / view.totalSteps) * 100) : 0;
  const eta = etaMinutes(samples.current, view.totalSteps);
  const saveEvery = Number(job?.recipe.train?.save_every_steps ?? 200);
  const canKeep = !view.terminal && view.step !== null && saveEvery > 0 && view.step >= saveEvery;
  const latestStepdown = view.banners[view.banners.length - 1];

  const stop = async (keep: boolean) => {
    if (!job) return;
    setStopping(true);
    try {
      await api.cancel(job.id, keep);
    } finally {
      setStopping(false);
    }
  };

  return (
    <Step
      num={4}
      tag="Training"
      title="Watch it learn"
      sub="The previews below are the real signal — when they look right, you can stop early and keep the best version."
      unlocked={props.unlocked}
    >
      <div className="mon-top">
        <div className="fact">
          <div className="k">Progress</div>
          <div className="v">{Math.round(pct)}%</div>
        </div>
        <div className="fact">
          <div className="k">Step</div>
          <div className="v">
            {view.step ?? 0} / {view.totalSteps ?? "—"}
          </div>
        </div>
        <div className="fact">
          <div className="k">Time left</div>
          <div className="v">{view.terminal ? "done" : eta !== null ? `${eta} min` : "—"}</div>
        </div>
        <div style={{ flex: 1, minWidth: 200 }}>
          <div className="bar">
            <i style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>

      {latestStepdown && view.state === "oom_stepdown" && (
        <div className="stepdown">
          <b>Adjusted automatically:</b> {latestStepdown.text} Slower, <em>not</em> worse.
        </div>
      )}
      {view.banners
        .filter((b) => view.state !== "oom_stepdown" || b !== latestStepdown)
        .map((banner) => (
          <div key={banner.seq} className="stepdown" style={{ opacity: 0.7 }}>
            <b>Adjusted automatically:</b> {banner.text}
          </div>
        ))}

      <Filmstrip job={job} />

      <div className="keep">
        {canKeep && (
          <>
            <button className="primary" disabled={stopping} onClick={() => void stop(true)}>
              ✋ Looks good — stop &amp; keep this version
            </button>
            <span className="hint" style={{ marginLeft: 10 }}>
              Stopping early is often the <em>right</em> call.
            </span>
          </>
        )}
        {!view.terminal && (
          <button
            className="danger"
            style={{ marginLeft: canKeep ? 10 : 0 }}
            disabled={stopping}
            onClick={() => void stop(false)}
          >
            Cancel, keep nothing
          </button>
        )}
      </div>

      <div className="charts">
        <div className="panel" style={{ position: "relative" }}>
          <h4>Training loss (smoothed) — for the curious, previews matter more</h4>
          <LossChart losses={view.losses} />
        </div>
        <div className="panel">
          <h4>VRAM in use</h4>
          <div className="meter-num">{vram ? `${vram.used.toFixed(1)} GB` : "— GB"}</div>
          <div className="meter">
            <i style={{ width: vram ? `${(vram.used / vram.total) * 100}%` : "0%" }} />
          </div>
          <div className="meter-lab">
            {vram ? `of ${Math.round(vram.total)} GB` : "measuring…"}
          </div>
        </div>
      </div>
    </Step>
  );
}

function Filmstrip({ job }: { job: JobRecord | null }) {
  const sampleEvery = Number(job?.recipe.train?.sample_every_steps ?? 0);
  if (sampleEvery === 0) {
    return (
      <div className="film-hint">
        Previews are off for this run — add a preview prompt in Configure next time to watch it
        learn frame by frame. The loss curve below still tells the story.
      </div>
    );
  }
  // Sample images render here once the gallery route lands (captioner phase).
  return (
    <>
      <h4 style={{ fontSize: 13, color: "var(--ink-2)", marginBottom: 8 }}>Sample previews</h4>
      <div className="film">
        <div className="frame">
          <div className="img g2" style={{ filter: "blur(8px)" }} />
          <div className="cap">rendering every {sampleEvery} steps…</div>
        </div>
      </div>
      <div className="film-hint">
        New preview every {sampleEvery} steps. Sharpening likeness = learning. Weird artifacts =
        overtraining.
      </div>
    </>
  );
}

function LossChart({ losses }: { losses: LossPoint[] }) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  if (losses.length < 2) {
    return <p className="hint">The loss curve draws here once numbers start flowing.</p>;
  }
  const W = 520;
  const H = 170;
  const P = 28;
  // light exponential smoothing — raw kohya loss is noisy
  const smoothed: LossPoint[] = [];
  let ema = losses[0].loss;
  for (const point of losses) {
    ema = 0.8 * ema + 0.2 * point.loss;
    smoothed.push({ step: point.step, loss: ema });
  }
  const min = Math.min(...smoothed.map((p) => p.loss));
  const max = Math.max(...smoothed.map((p) => p.loss));
  const span = max - min || 1;
  const pts = smoothed.map((p, i) => ({
    x: P + (W - P - 8) * (i / (smoothed.length - 1)),
    y: H - P - (H - P - 14) * ((p.loss - min) / span),
    ...p,
  }));
  const path = pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
  return (
    <>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        height="170"
        aria-label="Training loss chart"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const mx = ((e.clientX - rect.left) * W) / rect.width;
          const nearest = pts.reduce((a, b) => (Math.abs(b.x - mx) < Math.abs(a.x - mx) ? b : a));
          setTip({
            x: e.clientX - rect.left + 14,
            y: e.clientY - rect.top - 10,
            text: `step ${nearest.step} · loss ${nearest.loss.toFixed(3)}`,
          });
        }}
        onMouseLeave={() => setTip(null)}
      >
        {[0, 1, 2, 3].map((g) => {
          const y = 14 + ((H - P - 14) * g) / 3;
          return <line key={g} x1={P} y1={y} x2={W - 8} y2={y} stroke="var(--grid)" strokeWidth={1} />;
        })}
        <line x1={P} y1={H - P} x2={W - 8} y2={H - P} stroke="var(--baseline)" />
        <text x={P - 6} y={H - P + 4} fill="var(--muted)" fontSize={10} textAnchor="end">
          {min.toFixed(2)}
        </text>
        <text x={P - 6} y={18} fill="var(--muted)" fontSize={10} textAnchor="end">
          {max.toFixed(2)}
        </text>
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" />
      </svg>
      {tip && (
        <div className="loss-tip" style={{ display: "block", left: tip.x, top: tip.y }}>
          {tip.text}
        </div>
      )}
    </>
  );
}
