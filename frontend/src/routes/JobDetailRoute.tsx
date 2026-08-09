import { motion } from "motion/react";
import { useState } from "react";
import { Artifacts } from "@/components/Artifacts";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { LogPanel } from "@/components/LogPanel";
import { ModelViewerLazy } from "@/components/ModelViewerLazy";
import { PipelineTrack } from "@/components/PipelineTrack";
import { deleteJob, formatBytes, formatCount, optimizeJob } from "@/lib/api";
import { useReducedMotion } from "@/lib/hooks";
import { jobLayoutId, snap } from "@/lib/motion";
import { refreshJobs } from "@/lib/jobs";
import { go, href } from "@/lib/router";
import { useJob } from "@/lib/useJob";
import { isTerminal, type JobStatus } from "@/lib/types";
import { toast } from "@/shell/toast";

/* ─────────────────────────────────────────────────────────────────────────
   Job detail — the centre of gravity.

   Order on the page is the order of what the user came for: what did I get,
   how did it go, what does it look like, and only then the diagnostics. The
   old dashboard had exactly the opposite ordering, with finished output at
   the bottom of a long scroll.
   ───────────────────────────────────────────────────────────────────────── */

const STATUS_STYLE: Record<JobStatus, string> = {
  queued: "border-rule bg-panel text-ink-dim",
  running: "border-accent-dim bg-accent-wash text-accent",
  done: "border-ok-edge bg-ok-wash text-ok",
  failed: "border-fail-edge bg-fail-wash text-fail",
};

function DeleteJob({ jobId, running }: { jobId: string; running: boolean }) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  if (running) return null;

  const remove = async () => {
    setBusy(true);
    try {
      const res = await deleteJob(jobId);
      toast("ok", "Job deleted", `Freed ${formatBytes(res.freed_bytes)}.`);
      refreshJobs();
      go({ kind: "jobs" }, { replace: true });
    } catch (e) {
      toast("fail", "Delete failed", e instanceof Error ? e.message : "");
      setBusy(false);
    }
  };

  return confirming ? (
    <span className="flex items-center gap-2">
      <span className="font-mono text-[11px] text-warn">delete from disk?</span>
      <button
        type="button"
        disabled={busy}
        onClick={() => void remove()}
        className="rounded-sharp border border-fail-edge bg-fail-wash px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-fail hover:bg-fail hover:text-void disabled:opacity-50"
      >
        {busy ? "…" : "yes"}
      </button>
      <button
        type="button"
        onClick={() => setConfirming(false)}
        className="rounded-sharp border border-rule-bright px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-ink-dim hover:text-ink"
      >
        no
      </button>
    </span>
  ) : (
    <button
      type="button"
      onClick={() => setConfirming(true)}
      className="rounded-sharp border border-rule px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-ink-faint hover:border-fail-edge hover:text-fail"
    >
      delete
    </button>
  );
}

/** Re-optimise only exists for jobs that actually have a dense mesh. In the
 *  old dashboard this control sat in the centre of the screen and 409'd for
 *  every generative job — which is nearly all of them — so it is now shown
 *  only when the server has told us it can succeed. */
function Reoptimise({ jobId, faces }: { jobId: string; faces: number }) {
  const [target, setTarget] = useState(faces);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    try {
      await optimizeJob(jobId, { target_faces: target });
      toast("ok", "Re-optimised", `Rebuilt at ${formatCount(target)} triangles.`);
    } catch (e) {
      const detail = e instanceof Error ? e.message : String(e);
      toast("fail", "Re-optimise failed", detail);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-end gap-3 rounded-panel border border-rule bg-panel/70 p-3">
      <label className="flex flex-1 flex-col gap-1">
        <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">
          polygon budget · {formatCount(target)}
        </span>
        <input
          type="range"
          min={100}
          max={2000000}
          step={500}
          value={target}
          onChange={(e) => setTarget(Number(e.target.value))}
          className="accent-accent"
        />
      </label>
      <button
        type="button"
        disabled={busy}
        onClick={() => void run()}
        className="rounded-sharp border border-rule-bright px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:border-accent-dim hover:text-accent disabled:opacity-50"
      >
        {busy ? "optimising…" : "re-optimise"}
      </button>
    </div>
  );
}

/** Link to the rerun confirmation screen. Deliberately a link, not an action.
 *
 *  This used to POST on click and start the build immediately, which reused
 *  settings the user could neither see nor change. Reviewing them first is the
 *  point; nothing runs until the button on that screen is pressed.
 */
function RerunJob({
  jobId,
  status,
}: {
  jobId: string;
  status: JobStatus | undefined;
}) {
  const reduced = useReducedMotion();
  if (!status || !isTerminal(status)) return null;

  return (
    <motion.a
      href={`${href({ kind: "new" })}?from=${encodeURIComponent(jobId)}`}
      whileHover={reduced ? undefined : { scale: 1.03 }}
      whileTap={reduced ? undefined : { scale: 0.97 }}
      transition={snap}
      title="Review this job's settings, then start a new build from them"
      className="rounded-sharp border border-accent-edge bg-accent-wash px-2.5 py-1 font-mono text-[11px] text-accent"
    >
      ↻ rerun
    </motion.a>
  );
}

export function JobDetailRoute({ jobId }: { jobId: string }) {
  const { job, log, files, loading, notFound, error } = useJob(jobId);
  const reduced = useReducedMotion();

  if (notFound) {
    return (
      <div className="mx-auto max-w-[1400px] px-5 py-20">
        <p className="font-mono text-[11px] uppercase tracking-widest text-fail">
          job not found
        </p>
        <p className="mt-2 font-mono text-[12px] text-ink-dim">{jobId}</p>
        <p className="mt-1 font-mono text-[11px] text-ink-faint">
          It may have been deleted from disk.
        </p>
        <a
          href={href({ kind: "jobs" })}
          className="mt-6 inline-block rounded-panel border border-rule-bright px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:border-accent-dim hover:text-accent"
        >
          ← back to jobs
        </a>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-6">
      <a
        href={href({ kind: "jobs" })}
        className="font-mono text-[11px] uppercase tracking-widest text-ink-faint hover:text-accent"
      >
        ← jobs
      </a>

      {/* ── Header: the receiving end of the shared-element transition. */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-4">
        <motion.div layoutId={jobLayoutId(jobId)}>
          <h1 className="text-2xl text-ink">{job?.name ?? "…"}</h1>
          <p className="mt-0.5 font-mono text-[11px] text-ink-faint">{jobId}</p>
        </motion.div>

        <div className="flex items-center gap-2">
          {job && (
            <motion.span
              animate={
                reduced || job.status !== "running" ? undefined : { opacity: [1, 0.55, 1] }
              }
              transition={
                job.status === "running"
                  ? { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
                  : snap
              }
              className={`rounded-panel border px-2 py-1 font-mono text-[10px] uppercase tracking-widest ${STATUS_STYLE[job.status]}`}
            >
              {job.status}
              {!isTerminal(job.status) && job.stage ? ` · ${job.stage}` : ""}
            </motion.span>
          )}
          <RerunJob jobId={jobId} status={job?.status} />
          <DeleteJob jobId={jobId} running={job?.status === "running"} />
        </div>
      </div>

      {error && (
        <p className="mt-4 rounded-panel border border-fail-edge bg-fail-wash px-3 py-2 font-mono text-[12px] text-fail">
          {error}
        </p>
      )}

      {loading && !job && (
        <p className="mt-8 font-mono text-[12px] text-ink-faint">loading…</p>
      )}

      {job && (
        <div className="mt-6 space-y-6">
          {/* ── 1. What did I get. Top of the page, no scrolling. */}
          <Artifacts job={job} files={files} onChanged={() => undefined} />

          {job.warnings.length > 0 && (
            <ul className="list-none space-y-1 rounded-panel border border-warn-edge bg-warn-wash p-3">
              {job.warnings.map((w) => (
                <li key={w} className="font-mono text-[11px] leading-relaxed text-warn">
                  ⚠ {w}
                </li>
              ))}
            </ul>
          )}

          {/* ── 2. How did it go. */}
          <PipelineTrack job={job} />

          {/* ── 3. What does it look like. */}
          <div className="grid gap-6 lg:grid-cols-[1fr_20rem]">
            <div className="space-y-3">
              {job.glb_url ? (
                // The preview is the least important thing on this screen and
                // the most likely to fail, so it is fenced off from the rest.
                <ErrorBoundary
                  resetKey={job.glb_url}
                  fallback={() => (
                    <div className="flex h-[26rem] items-center justify-center rounded-panel border border-rule bg-void p-4 text-center">
                      <p className="font-mono text-[11px] leading-relaxed text-ink-faint">
                        3D preview unavailable on this machine.
                        <br />
                        The model and its parts are still downloadable above.
                      </p>
                    </div>
                  )}
                >
                  <ModelViewerLazy url={job.glb_url} className="h-[26rem]" />
                </ErrorBoundary>
              ) : (
                <div className="flex h-[26rem] items-center justify-center rounded-panel border border-rule bg-void">
                  <p className="font-mono text-[12px] text-ink-faint">
                    {job.status === "failed"
                      ? "no model was produced"
                      : "no model yet"}
                  </p>
                </div>
              )}

              {/* Gated on the MODEL, not on the face count. faces lived only
                  in memory, so every dashboard restart dropped it and this
                  control silently vanished for every past job -- with a
                  finished model sitting right there. It is persisted now, but
                  jobs built before that still have none, hence the fallback
                  to the budget the job was built with. */}
              {job.glb_url && (
                <Reoptimise
                  jobId={jobId}
                  faces={job.faces ?? job.target_faces ?? 20000}
                />
              )}
            </div>

            <aside className="space-y-4">
              <dl className="space-y-1 rounded-panel border border-rule bg-panel/70 p-3 font-mono text-[11px] tabular-nums">
                {[
                  ["regime", job.regime ?? "—"],
                  ["faces", job.faces != null ? formatCount(job.faces) : "—"],
                  ["parts", job.parts.length || "—"],
                  ["session", job.session],
                ].map(([k, v]) => (
                  <div key={String(k)} className="flex justify-between gap-3">
                    <dt className="text-ink-faint">{k}</dt>
                    <dd className="truncate text-ink-dim" title={String(v)}>
                      {String(v)}
                    </dd>
                  </div>
                ))}
              </dl>

              {job.parts.length > 0 && (
                <section aria-labelledby="parts-heading">
                  <h2
                    id="parts-heading"
                    className="mb-2 font-mono text-[11px] uppercase tracking-widest text-ink-dim"
                  >
                    Parts
                  </h2>
                  <ul className="max-h-64 list-none overflow-y-auto rounded-panel border border-rule bg-panel/70">
                    {job.parts.map((p, i) => (
                      <li key={p.file ?? p.name ?? i}>
                        <a
                          href={p.url}
                          download
                          className="group flex items-center gap-2 border-b border-rule px-2 py-1.5 last:border-0 hover:bg-hover"
                        >
                          <span aria-hidden="true" className="font-mono text-[11px] text-ink-faint group-hover:text-accent">
                            ⤓
                          </span>
                          <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-ink-dim group-hover:text-ink">
                            {p.name ?? p.file}
                          </span>
                          {p.faces != null && (
                            <span className="font-mono text-[10px] tabular-nums text-ink-faint">
                              {formatCount(p.faces)}
                            </span>
                          )}
                        </a>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
            </aside>
          </div>

          {/* ── 4. Diagnostics, collapsed. */}
          <LogPanel lines={log} live={!isTerminal(job.status)} />
        </div>
      )}
    </div>
  );
}
