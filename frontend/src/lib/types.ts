/* ─────────────────────────────────────────────────────────────────────────
   The API contract, transcribed from src/pixiecad/web/app.py.

   These types describe what the server actually returns today, including the
   parts that are loose (`stage` is a free string, `parts[]` entries vary by
   which code path produced them). Where a field is optional it is because the
   server genuinely may omit it, not because it was convenient to type.
   ───────────────────────────────────────────────────────────────────────── */

export type JobStatus = "queued" | "running" | "done" | "failed" | "cancelled";

/** Terminal statuses stop all polling for a job. "cancelled" belongs here for
 *  the same reason as "failed": the job will produce nothing further, and a
 *  status left out of this set is polled forever. */
export const isTerminal = (s: JobStatus | string): boolean =>
  s === "done" || s === "failed" || s === "cancelled";

/** GET /api/jobs — deliberately thin; the list view needs nothing more. */
export interface JobSummary {
  job_id: string;
  name: string;
  status: JobStatus;
  stage: string;
  glb_url: string | null;
  created_at?: string;
  /** Rebuilt from disk after a restart, so its log and timings are gone even
   *  though its artifacts are intact. Worth saying rather than pretending. */
  restored?: boolean;
  /** How many models this job holds: the build's own, plus one per
   *  re-optimise. The only thing that tells you which jobs you have already
   *  re-optimised without opening each one. */
  models?: number;
}

/** One model file a job holds. The build writes model.glb; each re-optimise
 *  adds model-<faces>.glb beside it rather than overwriting anything. */
export interface JobModel {
  name: string;
  /** Variants carry it in their filename; the build's own comes from the
   *  session, and is null for jobs re-optimised before that was recorded. */
  faces: number | null;
  bytes: number;
  url: string;
  /** model.glb — what the build itself produced. */
  original: boolean;
  /** The newest, and what the viewer and the sidebar describe. */
  current: boolean;
  /** Stem a CAD conversion of this model lands under, so the panel can tell
   *  whether one already exists without duplicating the server's naming. */
  basename: string;
}

/** One pipeline stage as the pipeline itself recorded it (StageOutcome). */
export interface StageOutcome {
  name: string;
  status: "ok" | "skipped" | "failed" | "running" | "pending";
  detail: string;
  seconds: number;
}

export interface JobPart {
  name?: string;
  file?: string;
  faces?: number | null;
  url: string;
  [k: string]: unknown;
}

/** GET /api/jobs/{id} */
export interface JobDetail {
  job_id: string;
  name: string;
  status: JobStatus;
  stage: string;
  log: string[];
  report: unknown | null;
  glb_url: string | null;
  regime: string | null;
  faces: number | null;
  /** The budget the job was BUILT with. Survives a dashboard restart even
   *  when the achieved face count does not, so it is the fallback the
   *  re-optimise control seeds itself from. */
  target_faces?: number | null;
  parts: JobPart[];
  warnings: string[];
  session: string;
  dir: string;
  output_dir: string;
  /** Added server-side so the pipeline view stops regex-parsing the log. */
  stages?: StageOutcome[];
  /** The cursor the server actually honoured. 0 means it resynced us and the
   *  returned log is complete rather than a delta. */
  log_since?: number;
  /** Total log length, so a `log_since` client knows where its cursor is. */
  log_total?: number;
  web_url?: string | null;
  created_at?: string;
  restored?: boolean;
  /** Every model this job holds, oldest first. Read from disk server-side, so
   *  it survives a dashboard restart. */
  models?: JobModel[];
}

/** GET /api/jobs/{id}/files */
export interface JobFile {
  name: string;
  bytes: number;
  url: string;
}

export interface JobFiles {
  files: JobFile[];
  total_bytes: number;
  output_dir: string;
}

/** POST /api/jobs/{id}/convert */
export interface ConvertResult {
  file: string;
  bytes: number;
  url: string;
  printable: boolean;
  solid: string;
  actions: string[];
}

/** GET /api/cloud/vm — the cheap poll behind the header pill. */
export interface VmInstance {
  name: string;
  zone: string;
  status: string;
  gpu: string | null;
  spot: boolean;
  uptime_hours: number | null;
  host: string;
  /** Machine image the VM booted from, e.g. "pixiecad-worker-v4". Decides
   *  what tooling it has: a pre-COLMAP image cannot do photogrammetry. */
  image: string | null;
  /** --max-run-duration, in seconds. Null when the VM has no hard stop. */
  max_run_seconds: number | null;
  /** Seconds until Compute Engine DELETES the instance. */
  seconds_remaining: number | null;
}

export interface VmStatus {
  available: boolean;
  reason?: string;
  vms: VmInstance[];
  running?: number;
  /** A provision is in flight: a job submitted now would find no worker image. */
  building?: boolean;
  provision_status?: string;
  host?: string;
}

/** GET|POST /api/cloud/provision */
export interface ProvisionState {
  status?: "running" | "ready" | "failed";
  gpu?: string;
  name?: string;
  zone?: string;
  host?: string;
  baked?: string;
  log?: string[];
}

/** GET /api/gpu-options */
export interface GpuTiming {
  total_seconds: number;
  usd_spot: number;
  usd_ondemand: number;
}

export interface GpuOption {
  key: string;
  label: string;
  accelerator: string;
  machine_type: string;
  vram_gb: number;
  available: boolean;
  note: string;
  speed_basis: string;
  warm: GpuTiming;
  cold: GpuTiming;
}

/** GET /api/cloud — the richer, slower cloud snapshot.
 *
 *  Distinct from /api/cloud/vm, which is the cheap poll behind the header
 *  pill. This one carries the account you are signed in as, the project being
 *  billed, and EVERY instance rather than only the running one — which is how
 *  you notice a stopped VM still holding a disk you are paying for. */
export interface CloudInstance {
  name: string;
  zone: string;
  machine_type: string;
  status: string;
  accelerator: string | null;
  preemptible: boolean;
  uptime_hours: number | null;
  estimated_cost_usd: number | null;
}

export interface CloudSnapshot {
  available: boolean;
  reason?: string;
  gcloud?: {
    installed: boolean;
    version: string | null;
    account: string | null;
    project: string | null;
    billing_enabled?: boolean;
  };
  instances?: CloudInstance[];
  total_estimated_cost_usd?: number;
  console_billing_url?: string;
  checked_at?: string;
  has_running_gpu?: boolean;
}

/** GET /api/backends — which generative model a job may be pointed at. */
export interface BackendOption {
  key: string;
  label: string;
  /** What must exist for this backend to run at all, or null for none. */
  requires: "gpu_host" | null;
  /** False when the backend has never produced a model on this project. */
  validated: boolean;
  note: string;
}

export interface BackendList {
  backends: BackendOption[];
}

/** GET /api/cloud/inventory */
export interface BillableResource {
  name: string;
  kind: string;
  detail: string;
  advice: string;
  size_gb: number | null;
  est_usd_per_month: number | null;
  console_url: string;
}

export interface CloudInventory {
  available: boolean;
  reason?: string;
  project?: string;
  resources: BillableResource[];
  est_total_usd_per_month?: number;
  unmeasured?: number;
  console_billing_url?: string;
}

/** The view tags the generative backend understands. Order is the order the
 *  tag selector offers them in. */
export const VIEW_TAGS = ["front", "back", "left", "right"] as const;
export type ViewTag = (typeof VIEW_TAGS)[number];

/** Everything POST /api/jobs accepts. Mirrors the Form(...) signature so a
 *  field cannot be silently dropped on the way out. */
export interface NewJobParams {
  name: string;
  target_faces: number;
  length?: string;
  width?: string;
  height?: string;
  split: boolean;
  object_hint?: string;
  backend?: string;
  smooth_iterations: number;
  texture: boolean;
  segmentation: "auto" | "semantic";
  web_export: boolean;
  texture_size: number;
  max_parts: number;
  gpu_host?: string;
  octree_resolution: number;
  multiview: boolean;
  /** filename → tag. Applied by renaming on upload, server-side. */
  view_tags: Record<string, ViewTag>;
  /** Bake surface detail from the high-poly source into a normal map after
   *  decimation. On by default: cheaper than carrying the polygons. */
  bake_normals: boolean;
  /** Resolution of the baked normal map. Clamped server-side to [256, 2048]. */
  normal_res: number;
  /** Where the bake runs. "auto" resolves to the host when one is configured. */
  bake_location: "auto" | "local" | "host";
  /** Optional generation seed for reproducible mesh creation. Empty/omitted lets the server generate one. */
  seed?: number;
}

/** GET /api/stage-costs — what a job as configured will actually cost.
 *
 *  `basis` matters as much as the number: a measured figure and an estimate
 *  rendered identically would make a 16 GB machine's resource decision worse
 *  than showing nothing at all. */
export interface StageCost {
  key: string;
  label: string;
  where: "local" | "host";
  seconds: number | null;
  peak_ram_mb: number | null;
  basis: "measured" | "estimated";
  detail: string;
}

export interface StageCosts {
  stages: StageCost[];
  local_peak_ram_mb: number;
  total_known_seconds: number;
  has_unmeasured: boolean;
  bake_location_resolved: "local" | "host" | null;
}

/** GET /api/jobs/{id}/rerun-plan — what a rerun WOULD use. Read-only:
 *  fetching it starts nothing. */
export interface RerunPlan {
  job_id: string;
  name: string;
  status: JobStatus;
  target_faces: number;
  photos: string[];
  photo_count: number;
  can_rerun: boolean;
  settings: Record<string, unknown>;
  /** False for sessions written before settings were persisted; the form
   *  falls back to its own defaults and has to say so. */
  has_settings: boolean;
}

/** What a re-optimise actually produced, as opposed to what was requested.
 *  Decimation lands short of the target whenever the source mesh has fewer
 *  faces to give, so the UI reports these rather than echoing the input. */
export interface OptimizeResult {
  status: string;
  stage: string;
  log: string[];
  glb_url: string | null;
  faces: number | null;
  bytes: number | null;
  /** The file it landed in. The build's model is still there beside it, so
   *  saying "re-optimised" without saying which file appeared leaves the user
   *  hunting through the list. */
  model?: string | null;
  /** How many models the job holds now. */
  models?: number;
}
