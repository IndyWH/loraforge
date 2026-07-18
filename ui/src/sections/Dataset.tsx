import { useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { DatasetSummary, ImageStatus } from "../api";

export function DatasetSection(props: {
  unlocked: boolean;
  summary: DatasetSummary | null;
  onSummary: (s: DatasetSummary) => void;
  trigger: string;
  onTrigger: (t: string) => void;
}) {
  const [name, setName] = useState("my-dataset");
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [triggerNote, setTriggerNote] = useState<string | null>(null);

  const refresh = async (datasetName: string) => props.onSummary(await api.datasetSummary(datasetName));

  const ingest = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setNote(null);
    try {
      await api.createDataset(name);
      const result = await api.upload(name, files);
      const skipped = result.skipped.length
        ? ` · ${result.skipped.length} skipped (${result.skipped.map((s) => s.reason)[0]}${result.skipped.length > 1 ? ", …" : ""})`
        : "";
      setNote(`${result.added.length} image${result.added.length === 1 ? "" : "s"} added${skipped}`);
      await refresh(name);
    } catch (error) {
      setNote(String(error));
    } finally {
      setBusy(false);
    }
  };

  const applyTrigger = async () => {
    const { updated } = await api.applyTrigger(name, props.trigger);
    setTriggerNote(
      updated === 0
        ? "Every caption already mentions it — nothing to change."
        : `Added to ${updated} caption${updated === 1 ? "" : "s"} (always as the first tag).`,
    );
    await refresh(name);
  };

  const s = props.summary;
  return (
    <Step
      num={3}
      title="Your photos"
      lede="Drop them in — we copy, never move. Duplicates, tiny files and problems get called out per image."
      unlocked={props.unlocked}
      lockHint="Pick a model first — it decides the resolution your photos train at."
    >
      <div className="row">
        <label className="field">
          dataset name
          <input type="text" value={name} onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9._-]/g, "-"))} />
        </label>
      </div>

      <div
        className={`dropzone ${over ? "over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          void ingest(Array.from(e.dataTransfer.files));
        }}
      >
        {busy ? "Ingesting…" : "Drag photos here (JPG, PNG, WebP — iPhone HEIC works too), or"}
        {!busy && (
          <>
            {" "}
            <label style={{ textDecoration: "underline", cursor: "pointer" }}>
              browse
              <input
                type="file"
                multiple
                accept=".jpg,.jpeg,.png,.webp,.bmp,.heic,.heif"
                style={{ display: "none" }}
                onChange={(e) => void ingest(Array.from(e.target.files ?? []))}
              />
            </label>
          </>
        )}
      </div>
      {note && <p className="smallnote">{note}</p>}

      {s && s.total > 0 && (
        <>
          <div className="chips">
            <span className="chip ok">
              <span className="k">training on</span> <span className="v">{s.included}</span>
            </span>
            {s.excluded > 0 && (
              <span className="chip err">
                <span className="k">excluded</span> <span className="v">{s.excluded}</span>
              </span>
            )}
            <span className="chip">
              <span className="k">captioned</span>{" "}
              <span className="v">
                {s.captioned}/{s.included}
              </span>
            </span>
          </div>

          <div className="row">
            <label className="field">
              trigger word (how you'll summon your subject in prompts)
              <input
                type="text"
                placeholder="e.g. sks-cat"
                value={props.trigger}
                onChange={(e) => props.onTrigger(e.target.value)}
              />
            </label>
            <button disabled={!props.trigger.trim()} onClick={() => void applyTrigger()}>
              Add to all captions
            </button>
          </div>
          {triggerNote && <p className="smallnote">{triggerNote}</p>}

          <div className="imagegrid">
            {s.images.map((image) => (
              <ImageCard key={image.filename} dataset={name} image={image} onChanged={() => void refresh(name)} />
            ))}
          </div>
        </>
      )}
    </Step>
  );
}

function ImageCard({
  dataset,
  image,
  onChanged,
}: {
  dataset: string;
  image: ImageStatus;
  onChanged: () => void;
}) {
  const [caption, setCaption] = useState<string | null>(null); // null = not loaded yet
  const [saved, setSaved] = useState(false);

  const openEditor = async () => {
    if (caption === null) {
      const current = await api.getCaption(dataset, image.filename);
      setCaption(current.caption ?? "");
    }
  };

  const save = async () => {
    await api.putCaption(dataset, image.filename, caption ?? "");
    setSaved(true);
    setTimeout(() => setSaved(false), 1200);
    onChanged();
  };

  return (
    <div className={`imagecard ${image.included ? "" : "excluded"}`}>
      <div className="name">{image.filename}</div>
      <div className="meta">
        {image.width && image.height ? `${image.width}×${image.height}` : "—"}
        {image.has_caption ? " · captioned" : " · no caption"}
      </div>
      {image.reason && <div style={{ color: "var(--err)" }}>{image.reason}</div>}
      {image.warnings.map((w) => (
        <div key={w} style={{ color: "var(--warn)" }}>
          {w}
        </div>
      ))}
      {image.included && (
        <details onToggle={(e) => (e.target as HTMLDetailsElement).open && void openEditor()}>
          <summary className="smallnote" style={{ cursor: "pointer" }}>
            edit caption
          </summary>
          {caption !== null && (
            <>
              <textarea value={caption} onChange={(e) => setCaption(e.target.value)} />
              <button onClick={() => void save()}>{saved ? "saved ✓" : "save"}</button>
            </>
          )}
        </details>
      )}
    </div>
  );
}
