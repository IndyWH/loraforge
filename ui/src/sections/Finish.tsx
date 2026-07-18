import { useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { JobRecord } from "../api";
import type { JobView } from "../jobView";

export function FinishSection(props: {
  unlocked: boolean;
  job: JobRecord | null;
  view: JobView;
  trigger: string;
  photos: number | null;
  onReset: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const { view, job } = props;
  const kept = view.state === "completed_early";
  const success = view.state === "completed" || kept;
  const trigger = props.trigger.trim() || "your-trigger-word";
  const loraName = job?.name ?? "my-lora";
  const prompt = `photo of ${trigger} as an astronaut, detailed, studio lighting <lora:${loraName}:0.8>`;
  const artifactFile = job?.artifact?.split(/[\\/]/).pop() ?? `${loraName}.safetensors`;

  const subline = [
    props.photos !== null ? `Trained on ${props.photos} photos` : null,
    view.step !== null && view.totalSteps
      ? kept
        ? `stopped at step ${view.step} of ${view.totalSteps}`
        : `${view.totalSteps} steps`
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  const copy = () => {
    void navigator.clipboard.writeText(prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <Step
      num={5}
      tag="Finished"
      title={
        success
          ? kept
            ? "Your LoRA is ready 🎉 (stopped early — smart move)"
            : "Your LoRA is ready 🎉"
          : "How it ended"
      }
      sub={
        success
          ? kept
            ? `${subline || "You kept the version that looked right"} · quality over quantity.`
            : subline || "Load it in your image tool of choice and start prompting."
          : "This run didn't finish — here's the honest story."
      }
      unlocked={props.unlocked}
    >
      {success && job && (
        <>
          <div className="artifact">
            <div className="icon">🧬</div>
            <div style={{ flex: 1 }}>
              <div className="name">{artifactFile}</div>
              {job.artifact && <div className="path">{job.artifact}</div>}
            </div>
            <a className="dlbtn" href={api.artifactUrl(job.id)}>
              ⬇ Download LoRA
            </a>
          </div>

          <h4 style={{ fontSize: 14, marginBottom: 2 }}>How to use it</h4>
          <p className="hint">
            Drop the file into ComfyUI's <code>models/loras</code> (or A1111's{" "}
            <code>models/Lora</code>), then prompt:
          </p>
          <div className="howto">
            photo of <b>{trigger}</b> as an astronaut, detailed, studio lighting &lt;lora:
            {loraName}:0.8&gt;
            <button className="copy" onClick={copy}>
              {copied ? "Copied ✓" : "Copy"}
            </button>
          </div>
        </>
      )}

      {view.state === "failed" && (
        <div className="banner-err">{view.finalMessage ?? job?.error ?? "Training failed."}</div>
      )}
      {view.state === "cancelled" && (
        <div className="stepdown" style={{ borderLeftColor: "var(--baseline)", background: "var(--surface-2)" }}>
          {view.finalMessage ?? "Training was cancelled."}
        </div>
      )}

      <div className="end-actions">
        <button className="primary" onClick={props.onReset}>
          Train another
        </button>
        <button
          onClick={() => document.getElementById("s3")?.scrollIntoView({ behavior: "smooth" })}
        >
          Tweak &amp; retrain this one
        </button>
      </div>
    </Step>
  );
}
