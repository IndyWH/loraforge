import { useEffect, useRef, useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { DownloadEventMsg, ModelStatus } from "../api";

// Card blurbs from the mockup; unknown models fall back to the display name.
const BLURBS: Record<string, string> = {
  sdxl: "The reliable all-rounder. Fast to train, huge community.",
  flux_dev: "Best image quality. Slower, bigger download.",
  sd15: "The lightweight classic. Quickest experiments.",
};
const THUMBS = ["g1", "g2", "g3", "g4"];

// Gated-model failures carry the license-acceptance URL (docs/decisions.md
// §12 promises a clickable link, not a string to retype).
function Linkified({ text }: { text: string }) {
  const parts = text.split(/(https?:\/\/[^\s)]+)/g);
  return (
    <>
      {parts.map((part, index) =>
        /^https?:\/\//.test(part) ? (
          <a key={index} href={part} target="_blank" rel="noreferrer">
            {part}
          </a>
        ) : (
          part
        ),
      )}
    </>
  );
}

export function ModelSection(props: {
  unlocked: boolean;
  models: ModelStatus[];
  selected: string | null;
  onSelect: (key: string) => void;
  onModelsChanged: () => void;
}) {
  const [dlLabel, setDlLabel] = useState<string | null>(null);
  const [dlFailed, setDlFailed] = useState(false);
  const [dlDone, setDlDone] = useState(false);
  const socket = useRef<WebSocket | null>(null);

  useEffect(() => () => socket.current?.close(), []);

  const pick = (entry: ModelStatus) => {
    const key = entry.capability.model_key;
    props.onSelect(key);
    if (entry.download_state === "downloaded" || entry.download_state === "downloading") return;
    void api.startDownload(key).then(() => {
      props.onModelsChanged();
      setDlFailed(false);
      setDlDone(false);
      setDlLabel(`Downloading ${entry.capability.display_name} weights…`);
      socket.current?.close();
      const ws = new WebSocket(api.downloadEventsUrl(key));
      socket.current = ws;
      ws.onmessage = (frame) => {
        const event: DownloadEventMsg = JSON.parse(frame.data);
        if (event.state === "completed") {
          setDlDone(true);
          setDlLabel(`✓ ${entry.capability.display_name} ready (cached for next time)`);
          props.onModelsChanged();
          ws.close();
        } else if (event.state === "failed") {
          setDlFailed(true);
          setDlLabel(event.message ?? "Download failed — see the server log.");
          props.onModelsChanged();
          ws.close();
        } else if (event.message) {
          setDlLabel(`${event.message} (reusing what your other tools already have)`);
        }
      };
    });
  };

  const firstAvailable = props.models.find((m) => m.capability.status === "available");

  return (
    <Step
      num={1}
      tag="Choose a model"
      title="What do you want to teach?"
      sub={
        <>
          Pick the base model your LoRA will be built on. Cards are matched to <em>your</em>{" "}
          hardware.
        </>
      }
      unlocked={props.unlocked}
    >
      <div className="models">
        {props.models.map((entry, index) => {
          const cap = entry.capability;
          const available = cap.status === "available";
          return (
            <div
              key={cap.model_key}
              className={`model ${props.selected === cap.model_key ? "selected" : ""} ${available ? "" : "disabled"}`}
              onClick={() => available && pick(entry)}
            >
              {available && entry === firstAvailable && (
                <div className="badge">Recommended for your GPU</div>
              )}
              <div className={`thumb ${THUMBS[index % THUMBS.length]}`} />
              <h3>{cap.display_name}</h3>
              <p>{BLURBS[cap.model_key] ?? ""}</p>
              {available ? (
                <div className="meta">
                  {entry.download_gb !== null && `${entry.download_gb} GB download · `}
                  {entry.download_state === "downloaded"
                    ? "weights on disk ✓"
                    : entry.gated
                      ? "needs free HF licence step"
                      : "free license"}
                </div>
              ) : (
                <div className="reason">{cap.reason}</div>
              )}
            </div>
          );
        })}
      </div>

      {dlLabel && (
        <div className="dl">
          <div className="bar">
            <i className={dlDone || dlFailed ? "" : "busy"} style={dlDone ? { width: "100%" } : undefined} />
          </div>
          <div className={`bar-label ${dlFailed ? "err" : ""}`}>
            <Linkified text={dlLabel} />
          </div>
        </div>
      )}
    </Step>
  );
}
