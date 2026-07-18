// Typed client for the LoRAForge API. The backend owns every decision; this
// file only moves data. Types mirror the server's pydantic models.

export const API_BASE = import.meta.env.DEV ? "http://127.0.0.1:8471" : "";

export const WS_BASE = API_BASE
  ? API_BASE.replace(/^http/, "ws")
  : (typeof location !== "undefined" ? location.origin.replace(/^http/, "ws") : "");

// ── Types (mirroring the OpenAPI schema) ─────────────────────────────────────

export type Availability = "available" | "unavailable" | "blocked";
export type DownloadState = "not_downloaded" | "downloading" | "downloaded" | "failed";

export interface GpuInfo {
  name: string;
  vram_total_mb: number;
  vram_free_mb: number;
  arch: string;
  is_laptop: boolean;
}

export interface DiagnoseResponse {
  hardware: {
    os: string;
    driver_version: string | null;
    gpus: GpuInfo[];
    ram_total_mb: number | null;
    disk_free_gb: number | null;
    notes: string[];
  };
  capabilities: { models: ModelCapability[]; warnings: string[] };
}

export interface ModelCapability {
  model_key: string;
  display_name: string;
  status: Availability;
  preset_name: string | null;
  settings: Record<string, number | boolean>;
  reason: string | null;
}

export interface ModelStatus {
  capability: ModelCapability;
  download_state: DownloadState;
}

export interface ImageStatus {
  filename: string;
  width: number | null;
  height: number | null;
  included: boolean;
  reason: string | null;
  warnings: string[];
  has_caption: boolean;
}

export interface DatasetSummary {
  name: string;
  path: string;
  total: number;
  included: number;
  excluded: number;
  captioned: number;
  images: ImageStatus[];
}

export interface IngestResult {
  added: string[];
  skipped: { source: string; reason: string }[];
}

export interface JobRecord {
  id: string;
  name: string;
  model: string;
  state: string;
  recipe: RecipeDoc;
  state_history: { state: string; at: string; message: string | null }[];
  stepdowns: { at: string; message: string }[];
  artifact: string | null;
  error: string | null;
}

export interface JobEventMsg {
  job_id: string;
  seq: number;
  kind: "state" | "progress";
  state: string;
  progress: {
    step: number | null;
    total_steps: number | null;
    loss: number | null;
    message: string | null;
    sample_image: string | null;
    is_oom: boolean;
  } | null;
  message: string | null;
}

export interface DownloadEventMsg {
  model_key: string;
  state: "checking" | "downloading" | "completed" | "failed";
  item: string | null;
  message: string | null;
}

// A recipe document as the API accepts it (subset we build client-side).
export interface RecipeDoc {
  name: string;
  model: string;
  dataset: Record<string, unknown>;
  peft?: Record<string, unknown>;
  optim?: Record<string, unknown>;
  train?: Record<string, unknown>;
  output_dir?: string;
  [key: string]: unknown;
}

// ── HTTP helpers ─────────────────────────────────────────────────────────────

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  diagnose: () => request<DiagnoseResponse>("/diagnose"),
  models: () => request<ModelStatus[]>("/models"),
  startDownload: (key: string) => request(`/models/${key}/download`, { method: "POST" }),

  createDataset: (name: string) => request("/datasets", json({ name })),
  datasetSummary: (name: string) => request<DatasetSummary>(`/datasets/${name}`),
  upload: (name: string, files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    return request<IngestResult>(`/datasets/${name}/upload`, { method: "POST", body: form });
  },
  getCaption: (name: string, file: string) =>
    request<{ filename: string; caption: string | null }>(`/datasets/${name}/captions/${file}`),
  putCaption: (name: string, file: string, caption: string) =>
    request(`/datasets/${name}/captions/${file}`, { ...json({ caption }), method: "PUT" }),
  applyTrigger: (name: string, trigger_word: string) =>
    request<{ updated: number }>(`/datasets/${name}/trigger-word`, json({ trigger_word })),

  validate: (doc: RecipeDoc) =>
    request<{ valid: boolean; errors: string[] }>("/recipes/validate", json(doc)),
  submitJob: (doc: RecipeDoc) => request<JobRecord>("/jobs", json(doc)),
  job: (id: string) => request<JobRecord>(`/jobs/${id}`),
  cancel: (id: string, keep: boolean) =>
    request<JobRecord>(`/jobs/${id}/cancel?keep=${keep}`, { method: "POST" }),
  artifactUrl: (id: string) => `${API_BASE}/jobs/${id}/artifact`,

  jobEventsUrl: (id: string) => `${WS_BASE}/jobs/${id}/events`,
  downloadEventsUrl: (key: string) => `${WS_BASE}/models/${key}/events`,
};
