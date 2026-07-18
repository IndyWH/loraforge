import { useEffect, useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { DatasetSummary, DiagnoseResponse, JobRecord, ModelCapability } from "../api";
import { buildRecipe, defaultKnobs, defaultLoraName } from "../recipe";

export function ConfigureSection(props: {
  unlocked: boolean;
  preset: ModelCapability | null;
  diagnose: DiagnoseResponse | null;
  summary: DatasetSummary | null;
  trigger: string;
  activeJob: boolean;
  onSubmitted: (record: JobRecord) => void;
}) {
  const [name, setName] = useState("my-first-lora");
  const [nameTouched, setNameTouched] = useState(false);
  const [previewPrompt, setPreviewPrompt] = useState("");
  const [knobs, setKnobs] = useState(defaultKnobs);
  const [errors, setErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  // The artifact name follows the trigger word until the user takes over.
  useEffect(() => {
    if (!nameTouched && props.preset)
      setName(defaultLoraName(props.trigger, props.preset.model_key));
  }, [props.trigger, props.preset, nameTouched]);

  const gpu = props.diagnose?.hardware.gpus[0] ?? null;
  const preset = props.preset;
  const vramNeeded = preset?.min_free_vram_mb ?? null;
  const presetResolution = (preset?.settings.resolution as number | undefined) ?? 1024;

  const submit = async () => {
    if (!preset || !props.summary) return;
    setSubmitting(true);
    setErrors([]);
    try {
      const doc = buildRecipe(name, preset, props.summary.path, props.trigger, knobs, previewPrompt);
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
      num={3}
      tag="Configure & go"
      title="We've set everything for your card"
      sub={
        <>
          Nothing to fill in. Open “Advanced” only if you want to tinker — every field is
          pre-set to safe values.
        </>
      }
      unlocked={props.unlocked}
    >
      <div className="contract">
        Training <b>{preset?.display_name ?? "—"}</b> · preset <b>{preset?.preset_name ?? "—"}</b>
        {gpu && <>, sized for your {gpu.name.replace("NVIDIA GeForce ", "")}</>}.
        <div className="facts">
          {vramNeeded !== null && (
            <div className="fact">
              <div className="k">VRAM needed</div>
              <div className="v">
                {(vramNeeded / 1024).toFixed(1)} GB{" "}
                {gpu && (
                  <span className="hint">of {(gpu.vram_free_mb / 1024).toFixed(1)} free ✓</span>
                )}
              </div>
            </div>
          )}
          <div className="fact">
            <div className="k">Photos</div>
            <div className="v">{props.summary?.included ?? "—"}</div>
          </div>
          <div className="fact">
            <div className="k">Steps</div>
            <div className="v">{knobs.maxSteps}</div>
          </div>
          <div className="fact">
            <div className="k">Previews</div>
            <div className="v">
              {previewPrompt.trim() ? "every 200 steps" : "off"}
              {!previewPrompt.trim() && <span className="hint"> — add a prompt in Advanced</span>}
            </div>
          </div>
        </div>
      </div>

      <details>
        <summary>Advanced settings (pre-filled with the preset)</summary>
        <div className="adv">
          <div>
            <label>
              Detail capacity (rank)
              <span className="why">How much detail the LoRA can hold. 16 suits most subjects.</span>
            </label>
            <input
              type="number"
              value={knobs.rank}
              min={1}
              max={1024}
              onChange={(e) => setKnobs({ ...knobs, rank: Number(e.target.value) })}
            />
          </div>
          <div>
            <label>
              Learning rate
              <span className="why">How boldly it learns. Higher = faster but riskier.</span>
            </label>
            <input
              type="number"
              step="0.00001"
              value={knobs.learningRate}
              onChange={(e) => setKnobs({ ...knobs, learningRate: Number(e.target.value) })}
            />
          </div>
          <div>
            <label>
              Steps
              <span className="why">How long it studies. More isn't always better — watch the previews.</span>
            </label>
            <input
              type="number"
              value={knobs.maxSteps}
              min={1}
              onChange={(e) => setKnobs({ ...knobs, maxSteps: Number(e.target.value) })}
            />
          </div>
          <div>
            <label>
              Resolution
              <span className="why">Training image size. The preset matches your VRAM.</span>
            </label>
            <input
              type="number"
              step={64}
              value={knobs.resolution ?? presetResolution}
              onChange={(e) => setKnobs({ ...knobs, resolution: Number(e.target.value) })}
            />
          </div>
          <div>
            <label>
              LoRA name
              <span className="why">The output file's name. Follows your trigger word by default.</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => {
                setNameTouched(true);
                setName(e.target.value);
              }}
            />
          </div>
          <div>
            <label>
              Preview prompt
              <span className="why">Renders a sample image every 200 steps so you can watch it learn.</span>
            </label>
            <input
              type="text"
              placeholder={props.trigger ? `photo of ${props.trigger}` : "photo of …"}
              value={previewPrompt}
              onChange={(e) => setPreviewPrompt(e.target.value)}
            />
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

      <button
        className="primary"
        style={{ fontSize: 16, padding: "13px 30px" }}
        disabled={submitting || props.activeJob || !preset || !props.summary}
        onClick={() => void submit()}
      >
        {props.activeJob ? "Training runs below" : submitting ? "Checking everything…" : "▶ Start training"}
      </button>
    </Step>
  );
}
