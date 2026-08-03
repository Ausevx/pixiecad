import type {
  BackendList,
  CloudInventory,
  CloudSnapshot,
  ConvertResult,
  GpuOption,
  JobDetail,
  JobFiles,
  JobSummary,
  NewJobParams,
  ProvisionState,
  VmStatus,
} from "./types";

/* ─────────────────────────────────────────────────────────────────────────
   The only place in the app that knows a URL.

   Every function here maps to an endpoint that already exists in
   src/pixiecad/web/app.py. Nothing invents a route.
   ───────────────────────────────────────────────────────────────────────── */

/** Server-supplied detail, preserved. FastAPI's HTTPException detail is the
 *  only useful error text this API produces — 409 "a provision is already
 *  running" and 409 "dense.ply not found" are both actionable, and a generic
 *  "request failed" throws them away. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (cause) {
    // A dead server and a 500 are different problems with different fixes, so
    // they must not collapse into the same message.
    throw new ApiError(0, "cannot reach the dashboard server");
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* a non-JSON error body is not itself an error worth reporting */
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/* ── Jobs ─────────────────────────────────────────────────────────────── */

export const listJobs = (signal?: AbortSignal) =>
  request<JobSummary[]>("/api/jobs", { signal });

/** `logSince` asks the server for only the log lines after that index. A
 *  server that does not yet support it simply ignores the parameter and
 *  returns the whole log, which the caller handles. */
export const getJob = (id: string, logSince = 0, signal?: AbortSignal) =>
  request<JobDetail>(
    `/api/jobs/${encodeURIComponent(id)}?log_since=${logSince}`,
    { signal },
  );

export const getJobFiles = (id: string, signal?: AbortSignal) =>
  request<JobFiles>(`/api/jobs/${encodeURIComponent(id)}/files`, { signal });

export const deleteJob = (id: string) =>
  request<{ deleted: string; freed_bytes: number }>(
    `/api/jobs/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );

export const convertJob = (
  id: string,
  body: { format: string; source?: string; size_mm?: number | null; repair?: boolean },
) => request<ConvertResult>(`/api/jobs/${encodeURIComponent(id)}/convert`, json(body));

export const optimizeJob = (
  id: string,
  body: { target_faces: number; normal_res?: number },
) => request<Partial<JobDetail>>(`/api/jobs/${encodeURIComponent(id)}/optimize`, json(body));

/** POST /api/jobs is multipart, not JSON — the photos ride along with the
 *  settings in one request, and the server applies view tags by renaming the
 *  files as it writes them. */
export function createJob(
  files: File[],
  params: NewJobParams,
  onProgress?: (fraction: number) => void,
): Promise<{ job_id: string }> {
  const form = new FormData();
  for (const file of files) form.append("files", file);

  const { view_tags, ...rest } = params;
  for (const [key, value] of Object.entries(rest)) {
    if (value === undefined || value === null || value === "") continue;
    form.append(key, String(value));
  }
  form.append("view_tags", JSON.stringify(view_tags));

  // XHR rather than fetch: uploading a dozen phone photos is the one place in
  // this app where real progress exists, and fetch still cannot report it.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new ApiError(xhr.status, "malformed response from the server"));
        }
        return;
      }
      let detail = `${xhr.status} ${xhr.statusText}`;
      try {
        const parsed: unknown = JSON.parse(xhr.responseText);
        if (parsed && typeof parsed === "object" && "detail" in parsed) {
          const d = (parsed as { detail: unknown }).detail;
          if (typeof d === "string") detail = d;
        }
      } catch {
        /* keep the status line */
      }
      reject(new ApiError(xhr.status, detail));
    });
    xhr.addEventListener("error", () =>
      reject(new ApiError(0, "upload failed: cannot reach the dashboard server")),
    );
    xhr.addEventListener("abort", () => reject(new ApiError(0, "upload cancelled")));
    xhr.send(form);
  });
}

/** Which generative backends exist. Static server-side, so this is cheap and
 *  safe to fetch on the new-job screen. */
export const getBackends = (signal?: AbortSignal) =>
  request<BackendList>("/api/backends", { signal });

/* ── Compute ──────────────────────────────────────────────────────────── */

/** The cheap poll — one gcloud call, no SSH. Safe to run behind the header
 *  pill on a short interval. */
export const getVmStatus = (signal?: AbortSignal) =>
  request<VmStatus>("/api/cloud/vm", { signal });

export const getProvisionState = (signal?: AbortSignal) =>
  request<ProvisionState>("/api/cloud/provision", { signal });

export const provision = (body: {
  gpu: string;
  name?: string;
  zone?: string;
  texture?: boolean;
  semantic?: boolean;
}) => request<{ status: string }>("/api/cloud/provision", json(body));

export const teardown = (body: { name?: string; zone?: string } = {}) =>
  request<{ ok: boolean; output: string }>("/api/cloud/teardown", json(body));

export const getGpuOptions = (texture = true, semantic = true, signal?: AbortSignal) =>
  request<{ options: GpuOption[] }>(
    `/api/gpu-options?texture=${texture}&semantic=${semantic}`,
    { signal },
  );

/** The full cloud snapshot: who you are signed in as, which project is being
 *  billed, and every instance rather than only the running one. Slower than
 *  /api/cloud/vm, so it is only fetched while the compute drawer is open. */
export const getCloud = (signal?: AbortSignal) =>
  request<CloudSnapshot>("/api/cloud", { signal });

/** Deliberately manual, never polled: each call shells out to gcloud several
 *  times and takes seconds. */
export const getInventory = (signal?: AbortSignal) =>
  request<CloudInventory>("/api/cloud/inventory", { signal });

/* ── Formatting ───────────────────────────────────────────────────────── */

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatSeconds(s: number): string {
  if (s < 60) return `${s.toFixed(2)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

export const formatCount = (n: number) => n.toLocaleString("en-US");
