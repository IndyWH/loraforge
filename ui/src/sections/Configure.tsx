import { useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { DatasetSummary, JobRecord, ModelCapability } from "../api";
import { buildRecipe, defaultKnobs } from "../recipe";

export function ConfigureSection(props: {
  unlocked: boolean;
  preset: ModelCapability | null;
  summary: DatasetSummary | null;
  trigger: string;
  activeJob: boolean;
  onSubmitted: (record: JobRecord) => void;
}) {
  const [name, setName] = useState("my-first-lora");
  const [previewPrompt, setPreviewPrompt] = useState("");
  const [knobs, setKnobs] = useState(defaultKnobs);
  const [errors, setErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  const submit = async () => {
    if (!props.preset || !props.summary) return;
    setSubmitting(true);
    setErrors([]);
    try {
      const doc = buildRecipe(name, props.preset, props.summary.path, props.trigger, knobs, previewPrompt);
      const verdict = await api.validate(doc);
      if (!verdict.valid) {
        setErrors(verdict.errors);
        return;
      }
      props.onSubmitted(await api.submitJob(doc));
    } catch (error) {
      setErrors([String(error)]);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Step
      num={4}
      title="Configure"
      lede="There's nothing you have to configure — your card already picked the settings. Peek under the hood if you like."
      unlocked={props.unlocked}
      lockHint="Add at least one usable photo above."
    >
      <div className="row">
        <label className="field">
          name your LoRA
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} />
        </label>
      </div>

      {props.preset && (
        <div className="chips">
          <span className="chip ok">
            <span className="k">model</span> <span className="v">{props.preset.display_name}</span>
          </span>
          <span className="chip ok">
            <span className="k">preset</span> <span className="v">{props.preset.preset_name}</span>
          </span>
          {props.summary && (
            <span className="chip">
              <span className="k">photos</span> <span className="v">{props.summary.included}</span>
            </span>
          )}
          {props.trigger.trim() && (
            <span className="chip">
              <span className="k">trigger</span> <span className="v">{props.trigger.trim()}</span>
            </span>
          )}
        </div>
      )}

      <details className="advanced">
        <summary>Advanced — the settings the preset chose for you</summary>
        <div className="inner">
          <div className="row">
            {Object.entries(props.preset?.settings ?? {}).map(([k, v]) => (
              <span key={k} className="chip">
                <span className="k">{k}</span> <span className="v">{String(v)}</span>
              </span>
            ))}
          </div>
          <div className="row">
            <label className="field">
              LoRA rank
              <input
                type="number"
                value={knobs.rank}
                min={1}
                max={1024}
                onChange={(e) => setKnobs({ ...knobs, rank: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              learning rate
              <input
                type="number"
                step="0.00001"
                value={knobs.learningRate}
                onChange={(e) => setKnobs({ ...knobs, learningRate: Number(e.target.value) })}
              />
            </label>
            <label className="field">
              total steps
              <input
                type="number"
                value={knobs.maxSteps}
                min={1}
                onChange={(e) => setKnobs({ ...knobs, maxSteps: Number(e.target.value) })}
              />
            </label>
            <label className="field" style={{ flex: 1 }}>
              preview prompt (optional — renders samples during training)
              <input
                type="text"
                placeholder={props.trigger ? `a photo of ${props.trigger}` : "a photo of …"}
                value={previewPrompt}
                onChange={(e) => setPreviewPrompt(e.target.value)}
              />
            </label>
          </div>
        </div>
      </details>

      {errors.length > 0 && (
        <ul className="errors">
          {errors.map((error) => (
            <li key={error}>{error}</li>
          ))}
        </ul>
      )}

      <div className="row" style={{ marginTop: 16 }}>
        <button className="primary" disabled={submitting || props.activeJob} onClick={() => void submit()}>
          {props.activeJob ? "Training in progress below" : submitting ? "Checking…" : "Start training"}
        </button>
        <span className="smallnote">Everything is checked before the GPU lifts a finger.</span>
      </div>
    </Step>
  );
}
