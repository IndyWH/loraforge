import { Step } from "../App";
import type { DiagnoseResponse } from "../api";

export function DiagnoseSection({ diagnose }: { diagnose: DiagnoseResponse | null }) {
  const gpu = diagnose?.hardware.gpus[0] ?? null;
  return (
    <Step
      num={1}
      title="Your hardware"
      lede="First, let's see what you're working with. Everything below adapts to this card."
      unlocked
      lockHint=""
    >
      {!diagnose && <p className="smallnote">Probing your GPU…</p>}
      {diagnose && (
        <>
          <div className="chips">
            {gpu ? (
              <>
                <span className="chip ok">
                  <span className="k">GPU</span> <span className="v">{gpu.name}</span>
                </span>
                <span className="chip">
                  <span className="k">VRAM free</span>{" "}
                  <span className="v">
                    {(gpu.vram_free_mb / 1024).toFixed(1)} / {(gpu.vram_total_mb / 1024).toFixed(1)} GB
                  </span>
                </span>
                {gpu.is_laptop && (
                  <span className="chip warn">
                    <span className="v">laptop GPU — presets get extra headroom</span>
                  </span>
                )}
              </>
            ) : (
              <span className="chip err">
                <span className="v">no NVIDIA GPU detected</span>
              </span>
            )}
            {diagnose.hardware.driver_version && (
              <span className="chip">
                <span className="k">driver</span> <span className="v">{diagnose.hardware.driver_version}</span>
              </span>
            )}
            {diagnose.hardware.disk_free_gb !== null && (
              <span className="chip">
                <span className="k">disk free</span>{" "}
                <span className="v">{Math.round(diagnose.hardware.disk_free_gb)} GB</span>
              </span>
            )}
          </div>
          {diagnose.capabilities.warnings.map((w) => (
            <div key={w} className="banner">
              {w}
            </div>
          ))}
          {diagnose.hardware.notes.map((n) => (
            <p key={n} className="smallnote">
              note: {n}
            </p>
          ))}
        </>
      )}
    </Step>
  );
}
