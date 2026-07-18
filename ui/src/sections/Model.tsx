import { useEffect, useRef, useState } from "react";

import { Step } from "../App";
import { api } from "../api";
import type { DownloadEventMsg, ModelStatus } from "../api";

export function ModelSection(props: {
  unlocked: boolean;
  models: ModelStatus[];
  selected: string | null;
  onSelect: (key: string) => void;
  onModelsChanged: () => void;
}) {
  const [downloadNote, setDownloadNote] = useState<Record<string, string>>({});
  const sockets = useRef<Record<string, WebSocket>>({});

  useEffect(() => () => Object.values(sockets.current).forEach((s) => s.close()), []);

  const startDownload = async (key: string) => {
    await api.startDownload(key);
    props.onModelsChanged();
    const socket = new WebSocket(api.downloadEventsUrl(key));
    sockets.current[key] = socket;
    socket.onmessage = (frame) => {
      const event: DownloadEventMsg = JSON.parse(frame.data);
      if (event.state === "completed" || event.state === "failed") {
        setDownloadNote((n) => ({ ...n, [key]: event.state === "failed" ? (event.message ?? "download failed") : "" }));
        props.onModelsChanged();
        socket.close();
      } else {
        setDownloadNote((n) => ({ ...n, [key]: event.message ?? "downloading…" }));
      }
    };
  };

  return (
    <Step
      num={2}
      title="Pick a model"
      lede="These verdicts come from your card, not from wishful thinking. Greyed-out options say why."
      unlocked={props.unlocked}
      lockHint="Waiting for the hardware check above."
    >
      <div className="cards">
        {props.models.map(({ capability: cap, download_state }) => {
          const available = cap.status === "available";
          const selected = props.selected === cap.model_key;
          return (
            <div
              key={cap.model_key}
              className={`card ${selected ? "selected" : ""} ${available ? "" : "disabled"}`}
              onClick={() => available && props.onSelect(cap.model_key)}
            >
              <h3>{cap.display_name}</h3>
              {available ? (
                <>
                  <div className="preset">preset: {cap.preset_name}</div>
                  <div className="chips">
                    {Object.entries(cap.settings).map(([k, v]) => (
                      <span key={k} className="chip">
                        <span className="k">{k}</span> <span className="v">{String(v)}</span>
                      </span>
                    ))}
                  </div>
                  {download_state === "downloaded" && (
                    <p className="why" style={{ color: "var(--ok)" }}>✓ weights on disk</p>
                  )}
                  {download_state === "not_downloaded" && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        void startDownload(cap.model_key);
                      }}
                    >
                      Download weights
                    </button>
                  )}
                  {download_state === "downloading" && (
                    <p className="why">{downloadNote[cap.model_key] || "downloading…"}</p>
                  )}
                  {download_state === "failed" && (
                    <p className="why" style={{ color: "var(--err)" }}>
                      {downloadNote[cap.model_key] || "download failed — see server log"}
                    </p>
                  )}
                </>
              ) : (
                <p className="why">{cap.reason}</p>
              )}
            </div>
          );
        })}
      </div>
      <p className="smallnote" style={{ marginTop: 10 }}>
        You can keep going while weights download — training waits for them, not you.
      </p>
    </Step>
  );
}
