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
  onReset: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const { view, job } = props;
  const kept = view.state === "completed_early";
  const success = view.state === "completed" || kept;
  const trigger = props.trigger.trim() || "your-trigger-word";
  const prompt = `${trigger}, portrait photo, soft window light, 85mm, shallow depth of field`;

  const copy = () => {
    void navigator.clipboard.writeText(prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  return (
    <Step
      num={6}
      title={success ? (kept ? "Stopped early — and it's yours" : "It's done") : "How it ended"}
      lede={
        success
          ? "Your LoRA is ready. Load it in your image tool of choice and start prompting."
          : "This run didn't finish — here's the honest story."
      }
      unlocked={props.unlocked}
      lockHint="This unlocks when training finishes."
    >
      {success && job && (
        <>
          {kept && (
            <div className="banner ok">
              You stopped early, so this is the most recent checkpoint — trained a bit less than planned,
              but very much usable.
            </div>
          )}
          <div className="row">
            <a className="download" href={api.artifactUrl(job.id)}>
              Download {job.name}.safetensors
            </a>
          </div>
          <p style={{ marginBottom: 4 }}>Try this prompt to see what it learned:</p>
          <div className="promptbox">{prompt}</div>
          <div className="row">
            <button onClick={copy}>{copied ? "copied ✓" : "copy prompt"}</button>
            <span className="smallnote">
              The first word is your trigger — that's how you summon your subject.
            </span>
          </div>
        </>
      )}

      {view.state === "failed" && (
        <div className="banner err">{view.finalMessage ?? job?.error ?? "Training failed."}</div>
      )}
      {view.state === "cancelled" && (
        <div className="banner">{view.finalMessage ?? "Training was cancelled."}</div>
      )}

      <div className="row" style={{ marginTop: 16 }}>
        <button onClick={props.onReset}>Train another</button>
      </div>
    </Step>
  );
}
