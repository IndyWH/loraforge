import { useEffect, useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { DatasetSummary, ImageStatus } from "../api";

const THUMBS = ["g1", "g2", "g3", "g4"];

export function DatasetSection(props: {
  unlocked: boolean;
  summary: DatasetSummary | null;
  onSummary: (s: DatasetSummary) => void;
  trigger: string;
  onTrigger: (t: string) => void;
  onContinue: () => void;
}) {
  const [name, setName] = useState("my-photos");
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [triggerNote, setTriggerNote] = useState<string | null>(null);

  const refresh = async () => props.onSummary(await api.datasetSummary(name));

  const ingest = async (files: File[]) => {
    if (!files.length) return;
    setBusy(true);
    setNote(null);
    try {
      await api.createDataset(name);
      const result = await api.upload(name, files);
      if (result.skipped.length) {
        setNote(`${result.added.length} added · ${result.skipped.length} skipped — hover a photo for why`);
      }
      await refresh();
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
        ? "Every caption already mentions it — nothing changed."
        : `Added to ${updated} caption${updated === 1 ? "" : "s"}, always as the first word.`,
    );
    await refresh();
  };

  const s = props.summary;
  const nearDups = s?.images.filter((i) => i.warnings.some((w) => w.includes("nearly identical"))).length ?? 0;

  return (
    <Step
      num={2}
      tag="Prepare your photos"
      title="Show it what to learn"
      sub="This is the step that decides your LoRA's quality — more than any setting later."
      unlocked={props.unlocked}
    >
      <div className="coach">
        💡 15–30 varied photos beat 100 similar ones. Different angles, lighting, and
        backgrounds; the subject sharp and central.
      </div>

      <div className="name-row">
        <label>Dataset name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9._-]/g, "-"))}
        />
      </div>

      <label>
        <div
          className={`drop ${over ? "over" : ""}`}
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
          <div style={{ fontSize: 30 }}>📁</div>
          {busy ? "Ingesting…" : "Drop photos here, or click to browse"}
          <br />
          <span className="hint">JPG, PNG, WebP — iPhone HEIC works too</span>
        </div>
        <input
          type="file"
          multiple
          accept=".jpg,.jpeg,.png,.webp,.bmp,.heic,.heif"
          style={{ display: "none" }}
          onChange={(e) => void ingest(Array.from(e.target.files ?? []))}
        />
      </label>
      {note && <p className="hint" style={{ margin: "-8px 0 14px" }}>{note}</p>}

      {s && s.total > 0 && (
        <>
          <div className="ds-chips">
            <div className={`ds-chip ${s.included >= 10 ? "ok" : "warn"}`}>
              {s.included} photo{s.included === 1 ? "" : "s"}
              {s.included >= 15 && s.included <= 40
                ? " — a good amount"
                : s.included < 10
                  ? " — a few more would help"
                  : ""}
            </div>
            {nearDups > 0 && (
              <div className="ds-chip warn">
                {nearDups} near-duplicate{nearDups === 1 ? "" : "s"} flagged — keep the sharper one
              </div>
            )}
            {s.excluded > 0 && (
              <div className="ds-chip warn">
                {s.excluded} excluded — hover those photos for the reason
              </div>
            )}
            {s.captioned === s.included && s.included > 0 ? (
              <div className="ds-chip ok">Captions ready ✓ — edit any below</div>
            ) : (
              <div className="ds-chip">{s.captioned}/{s.included} captioned</div>
            )}
          </div>

          <div className="thumbs">
            {s.images.map((image, index) => (
              <div
                key={image.filename}
                className={`t ${THUMBS[index % THUMBS.length]} ${image.included ? "" : "excluded"}`}
                title={`${image.filename}${image.reason ? ` — ${image.reason}` : ""}${image.warnings.length ? ` — ${image.warnings.join("; ")}` : ""}`}
              />
            ))}
          </div>

          <div className="trigger-row">
            <label>
              Trigger word <span className="hint">(type this in prompts to summon your subject)</span>
            </label>
            <input
              type="text"
              placeholder="ohwx person"
              value={props.trigger}
              onChange={(e) => props.onTrigger(e.target.value)}
            />
            <button disabled={!props.trigger.trim()} onClick={() => void applyTrigger()}>
              Add to all captions
            </button>
          </div>
          {triggerNote && <p className="hint" style={{ margin: "-14px 0 16px" }}>{triggerNote}</p>}

          <div className="cap-rows">
            {s.images
              .filter((i) => i.included)
              .map((image, index) => (
                <CaptionRow key={image.filename} dataset={name} image={image} thumb={THUMBS[index % THUMBS.length]} />
              ))}
          </div>

          <button className="primary" disabled={s.included === 0} onClick={props.onContinue}>
            Dataset looks good →
          </button>
        </>
      )}
    </Step>
  );
}

function CaptionRow({ dataset, image, thumb }: { dataset: string; image: ImageStatus; thumb: string }) {
  const [caption, setCaption] = useState("");
  const [loadedFor, setLoadedFor] = useState<string | null>(null);

  useEffect(() => {
    if (loadedFor === image.filename) return;
    api.getCaption(dataset, image.filename).then((r) => {
      setCaption(r.caption ?? "");
      setLoadedFor(image.filename);
    });
  }, [dataset, image.filename, loadedFor]);

  return (
    <>
      <div className="cap-row">
        <div className={`t ${thumb}`} title={image.filename} />
        <input
          value={caption}
          placeholder={`describe this photo… (${image.filename})`}
          onChange={(e) => setCaption(e.target.value)}
          onBlur={() => void api.putCaption(dataset, image.filename, caption)}
        />
      </div>
      {image.warnings.map((w) => (
        <div key={w} className="cap-note warn">
          {w}
        </div>
      ))}
    </>
  );
}
