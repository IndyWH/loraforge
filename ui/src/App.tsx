// The scroll-unlock spine from design/mockup.html: sticky header receipt
// chips, steps 0–5, locked sections visible but blurred. All verdicts flow
// from the API — the UI renders decisions, it never makes them.

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
const STEP_NAMES = ["Hardware", "Model", "Dataset", "Configure", "Training", "Done"];

export function App() {
  const [diagnose, setDiagnose] = useState<DiagnoseResponse | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [modelKey, setModelKey] = useState<string | null>(null);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [trigger, setTrigger] = useState("");
  const [job, setJob] = useState<JobRecord | null>(null);
  const [view, setView] = useState<JobView>(emptyJobView);
  const [reached, setReached] = useState(0);
  const [resumed, setResumed] = useState(false);
  const [fatal, setFatal] = useState<string | null>(null);

  const advance = useCallback((n: number) => setReached((r) => Math.max(r, n)), []);
  const refreshModels = useCallback(() => {
    api.models().then(setModels).catch(() => undefined);
  }, []);

  // Boot: models list always; a stored job means a refresh mid-run — land
  // back on the training section with the replayed history.
  useEffect(() => {
    refreshModels();
    const stored = localStorage.getItem(JOB_KEY);
    if (!stored) return;
    api
      .job(stored)
      .then((record) => {
        setJob(record);
        setModelKey(record.model);
        const word = record.recipe.dataset?.trigger_word;
        if (typeof word === "string") setTrigger(word);
        setResumed(true);
        setReached(4);
        api.diagnose().then(setDiagnose).catch(() => undefined);
      })
      .catch(() => localStorage.removeItem(JOB_KEY));
  }, [refreshModels]);

  // Selected model's weights ready → the dataset step opens (mockup flow).
  useEffect(() => {
    if (!modelKey || reached < 1) return;
    const entry = models.find((m) => m.capability.model_key === modelKey);
    if (entry?.download_state === "downloaded") advance(2);
  }, [modelKey, models, reached, advance]);

  useEffect(() => {
    if (view.terminal) advance(5);
  }, [view.terminal, advance]);

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
    advance(4);
  };

  const reset = () => {
    localStorage.removeItem(JOB_KEY);
    setJob(null);
    setView(emptyJobView);
    setSummary(null);
    setReached(0);
    setResumed(false);
    refreshModels();
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const preset = models.find((m) => m.capability.model_key === modelKey)?.capability ?? null;

  return (
    <>
      <header className="top">
        <div className="logo">
          LoRA<span>Forge</span>
        </div>
        <div className="chips">
          {STEP_NAMES.map((name, index) => (
            <div
              key={name}
              className={`chip ${index < reached ? "done" : ""} ${index === reached ? "active" : ""}`}
              onClick={() => {
                if (index <= reached)
                  document.getElementById(`s${index}`)?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              {name}
            </div>
          ))}
        </div>
      </header>

      <main>
        {fatal && <div className="banner-err">Can't reach the LoRAForge server: {fatal}</div>}

        <DiagnoseSection
          diagnose={diagnose}
          revealed={resumed || diagnose !== null}
          onProbe={() => api.diagnose().then(setDiagnose).catch((e) => setFatal(String(e)))}
          onContinue={() => advance(1)}
        />
        <ModelSection
          unlocked={reached >= 1}
          models={models}
          selected={modelKey}
          onSelect={setModelKey}
          onModelsChanged={refreshModels}
        />
        <DatasetSection
          unlocked={reached >= 2}
          summary={summary}
          onSummary={setSummary}
          trigger={trigger}
          onTrigger={setTrigger}
          onContinue={() => advance(3)}
        />
        <ConfigureSection
          unlocked={reached >= 3}
          preset={preset}
          diagnose={diagnose}
          summary={summary}
          trigger={trigger}
          activeJob={job !== null && !view.terminal}
          onSubmitted={startJob}
        />
        <TrainSection unlocked={reached >= 4} job={job} view={view} />
        <FinishSection
          unlocked={reached >= 5}
          job={job}
          view={view}
          trigger={trigger}
          photos={summary?.included ?? null}
          onReset={reset}
        />
      </main>
    </>
  );
}

export function Step(props: {
  num: number;
  tag: string;
  title: string;
  sub: React.ReactNode;
  unlocked: boolean;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
  const wasLocked = useRef(!props.unlocked);
  useEffect(() => {
    if (props.unlocked && wasLocked.current) {
      wasLocked.current = false;
      setTimeout(() => ref.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
    }
  }, [props.unlocked]);
  return (
    <section ref={ref} id={`s${props.num}`} className={`step ${props.unlocked ? "" : "locked"}`}>
      <div className="step-tag">
        Step {props.num} · {props.tag}
      </div>
      <h2>{props.title}</h2>
      <p className="sub">{props.sub}</p>
      {props.children}
    </section>
  );
}
