import { useEffect, useRef, useState } from "react";

import { Step } from "../App";
import type { DiagnoseResponse } from "../api";

const PROBE_LINES = [
  "→ probing GPU via NVML…",
  "→ reading driver + CUDA runtime…",
  "→ checking torch build…",
  "→ measuring free VRAM, RAM, disk…",
  "✓ diagnostics complete",
];

const FP8_ARCHES = ["ada", "hopper", "blackwell"];

export function DiagnoseSection(props: {
  diagnose: DiagnoseResponse | null;
  revealed: boolean;
  onProbe: () => void;
  onContinue: () => void;
}) {
  const [logLines, setLogLines] = useState<string[]>([]);
  const [probing, setProbing] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval>>();

  const runProbe = () => {
    setProbing(true);
    props.onProbe();
    let i = 0;
    timer.current = setInterval(() => {
      setLogLines((lines) => [...lines, PROBE_LINES[i]]);
      if (++i === PROBE_LINES.length) clearInterval(timer.current);
    }, 380);
  };
  useEffect(() => () => clearInterval(timer.current), []);

  const d = props.diagnose;
  const gpu = d?.hardware.gpus[0] ?? null;
  const showVerdict = props.revealed && d !== null && (!probing || logLines.length === PROBE_LINES.length);
  const available = d?.capabilities.models.filter((m) => m.status === "available") ?? [];
  const allAvailable = d !== null && available.length === d.capabilities.models.length;

  return (
    <Step
      num={0}
      tag="Hardware check"
      title="Let's look at your machine"
      sub={
        <>
          LoRAForge checks what your GPU can do <em>before</em> you spend a minute on anything
          else. Nothing here leaves your computer.
        </>
      }
      unlocked
    >
      {!showVerdict && (
        <button className="primary" disabled={probing} onClick={runProbe}>
          Check my machine
        </button>
      )}
      {probing && logLines.length < PROBE_LINES.length && (
        <div className="probe-log">{logLines.join("\n")}</div>
      )}

      {showVerdict && d && (
        <div>
          <div className="hw-grid">
            <div className="hw-card">
              <div className="k">GPU</div>
              <div className="v">{gpu ? gpu.name.replace("NVIDIA GeForce ", "") : "none found"}</div>
              <div className="n">
                {gpu ? `${gpu.arch} · FP8 ${FP8_ARCHES.includes(gpu.arch) ? "✓" : "—"}` : "check drivers"}
              </div>
            </div>
            <div className="hw-card">
              <div className="k">VRAM free</div>
              <div className="v">{gpu ? `${(gpu.vram_free_mb / 1024).toFixed(1)} GB` : "—"}</div>
              <div className="n">
                {gpu &&
                  `of ${(gpu.vram_total_mb / 1024).toFixed(0)} GB (desktop uses ${(
                    (gpu.vram_total_mb - gpu.vram_free_mb) / 1024
                  ).toFixed(1)})`}
              </div>
            </div>
            <div className="hw-card">
              <div className="k">Driver</div>
              <div className="v">{d.hardware.driver_version ?? "—"}</div>
              <div className="n">{d.hardware.os}</div>
            </div>
            <div className="hw-card">
              <div className="k">RAM · Disk</div>
              <div className="v">
                {d.hardware.ram_total_mb ? `${Math.round(d.hardware.ram_total_mb / 1024)} GB` : "—"}
              </div>
              <div className="n">
                {d.hardware.disk_free_gb !== null && `${Math.round(d.hardware.disk_free_gb)} GB free`}
              </div>
            </div>
          </div>

          {gpu ? (
            <div className="verdict-line">
              ✓{" "}
              {allAvailable ? (
                <>
                  <b>Great news:</b> this machine can train every model LoRAForge offers.
                </>
              ) : (
                <>
                  <b>Ready:</b> {available.map((m) => m.display_name).join(" and ")}{" "}
                  {available.length === 1 ? "fits" : "fit"} on this card — the rest explain
                  themselves below.
                </>
              )}
            </div>
          ) : (
            <div className="verdict-line bad">
              No NVIDIA GPU detected — check your drivers, then run the check again.
            </div>
          )}

          {!d.engine.ready && (
            <p className="hint" style={{ marginTop: 10 }}>
              ⚠ The training engine isn't installed yet — run <code>loraforge setup</code> in a
              terminal, then run the check again. ({d.engine.problems[0]})
            </p>
          )}
          {d.capabilities.warnings.map((w) => (
            <p key={w} className="hint" style={{ marginTop: 10 }}>
              {w}
            </p>
          ))}
          {gpu && (
            <p style={{ marginTop: 18 }}>
              <button className="primary" onClick={props.onContinue}>
                Choose a model →
              </button>
            </p>
          )}
        </div>
      )}
    </Step>
  );
}
