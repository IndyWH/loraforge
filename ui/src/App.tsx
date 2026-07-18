// The six-section scroll spine. Each section unlocks when the previous one
// has what it needs; all state flows from the API — the UI never decides
// what fits, it renders the backend's verdicts.

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "./api";
import type { DatasetSummary, DiagnoseResponse, JobRecord, ModelStatus } from "./api";
import { emptyJobView, foldEvents } from "./jobView";
import type { JobView } from "./jobView";
import { ConfigureSection } from "./sections/Configure";
import { DatasetSection } from "./sections/Dataset";
import { DiagnoseSection } from "./sections/Diagnose";
import { FinishSection } from "./sections/Finish";
import { ModelSection } from "./sections/Model";
import { TrainSection } from "./sections/Train";

const JOB_KEY = "loraforge.jobId";

export function App() {
  const [diagnose, setDiagnose] = useState<DiagnoseResponse | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [modelKey, setModelKey] = useState<string | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [trigger, setTrigger] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [view, setView] = useState<JobView>(emptyJobView);
  const [fatal, setFatal] = useState<string | null>(null);

  const refreshModels = useCallback(() => {
    api.models().then(setModels).catch(() => undefined);
  }, []);

  // Boot: diagnose + models, then resume a stored job so a refresh
  // mid-training lands back on the training section with history intact.
  useEffect(() => {
    api.diagnose().then(setDiagnose).catch((e) => setFatal(String(e)));
    refreshModels();
    const stored = localStorage.getItem(JOB_KEY);
    if (stored) {
      api
        .job(stored)
        .then((record) => {
          setJob(record);
          setModelKey(record.model);
          const triggerWord = record.recipe.dataset?.trigger_word;
          if (typeof triggerWord === "string") setTrigger(triggerWord);
        })
        .catch(() => localStorage.removeItem(JOB_KEY)); // stale id: start fresh
    }
  }, [refreshModels]);

  // Job event stream: replay + live, reconnect with backoff until terminal.
  useEffect(() => {
    if (!job) return;
    let socket: WebSocket | null = null;
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;
    const connect = () => {
      socket = new WebSocket(api.jobEventsUrl(job.id));
      socket.onmessage = (frame) => {
        const event = JSON.parse(frame.data);
        setView((current) => foldEvents(current, [event]));
      };
      socket.onclose = () => {
        if (closed) return;
        setView((current) => {
          if (!current.terminal) retry = setTimeout(connect, 1500); // replay dedupes by seq
          return current;
        });
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(retry);
      socket?.close();
    };
  }, [job]);

  const startJob = (record: JobRecord) => {
    localStorage.setItem(JOB_KEY, record.id);
    setView(emptyJobView);
    setJob(record);
  };

  const reset = () => {
    localStorage.removeItem(JOB_KEY);
    setJob(null);
    setView(emptyJobView);
    setSummary(null);
    refreshModels();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const preset = models.find((m) => m.capability.model_key === modelKey)?.capability ?? null;
  const unlocked = {
    model: diagnose !== null,
    dataset: preset !== null,
    configure: (summary?.included ?? 0) > 0 || job !== null,
    train: job !== null,
    finish: view.terminal,
  };

  return (
    <div className="wrap">
      <header className="masthead">
        <h1>
          LoRA<span>Forge</span>
        </h1>
        <p>Train a LoRA on the GPU you actually own. One step at a time — no wall of knobs.</p>
        {fatal && <div className="banner err">Can't reach the LoRAForge server: {fatal}</div>}
      </header>

      <DiagnoseSection diagnose={diagnose} />
      <ModelSection
        unlocked={unlocked.model}
        models={models}
        selected={modelKey}
        onSelect={setModelKey}
        onModelsChanged={refreshModels}
      />
      <DatasetSection
        unlocked={unlocked.dataset}
        summary={summary}
        onSummary={setSummary}
        trigger={trigger}
        onTrigger={setTrigger}
      />
      <ConfigureSection
        unlocked={unlocked.configure}
        preset={preset}
        summary={summary}
        trigger={trigger}
        activeJob={job !== null && !view.terminal}
        onSubmitted={startJob}
      />
      <TrainSection unlocked={unlocked.train} job={job} view={view} />
      <FinishSection unlocked={unlocked.finish} job={job} view={view} trigger={trigger} onReset={reset} />
    </div>
  );
}

export function Step(props: {
  num: number;
  title: string;
  lede: string;
  unlocked: boolean;
  lockHint: string;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  const wasLocked = useRef(!props.unlocked);
  useEffect(() => {
    if (props.unlocked && wasLocked.current) {
      wasLocked.current = false;
      ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [props.unlocked]);
  return (
    <section ref={ref} className={`step ${props.unlocked ? "" : "locked"}`}>
      <h2>
        <span className="num">{props.num} ·</span> {props.title}
      </h2>
      <p className="lede">{props.lede}</p>
      {props.unlocked ? props.children : <p className="locknote">{props.lockHint}</p>}
    </section>
  );
}
