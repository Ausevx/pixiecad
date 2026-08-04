import { AnimatePresence, motion } from "motion/react";
import type React from "react";
import { useMemo, useState } from "react";
import { go, href } from "@/lib/router";
import { jobLayoutId, listItemVariants, listVariants, signatures, snap } from "@/lib/motion";
import { refreshJobs, useJobs } from "@/lib/jobs";
import { useReducedMotion } from "@/lib/hooks";
import { rerunJob } from "@/lib/api";
import { useVm } from "@/lib/vm";
import { toast } from "@/shell/toast";
import { isTerminal, type JobStatus, type JobSummary } from "@/lib/types";

/* ─────────────────────────────────────────────────────────────────────────
   The jobs list: scannable, status at a glance, one click through.

   This is the landing screen because on a single-user local tool the thing
   you most often want is the work you already started, not a form.
   ───────────────────────────────────────────────────────────────────────── */

const STATUS: Record<JobStatus, { glyph: string; className: string }> = {
  queued: { glyph: "○", className: "text-ink-faint" },
  running: { glyph: "◐", className: "text-accent" },
  done: { glyph: "●", className: "text-ok" },
  failed: { glyph: "▲", className: "text-fail" },
};

type Filter = "all" | "active" | "done" | "failed";

/** ISO timestamp → something a human reads at a glance. Relative for today,
 *  absolute once it stops being "recent" — "3 days ago" is worse than a date
 *  when you are looking for a specific run. */
function when(iso: string | undefined): string {
  if (!iso) return "";
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return "";
  const mins = (Date.now() - then.getTime()) / 60000;
  if (mins < 1) return "just now";
  if (mins < 60) return `${Math.floor(mins)}m ago`;
  if (mins < 1440) return `${Math.floor(mins / 60)}h ago`;
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function JobRow({ job }: { job: JobSummary }) {
  const s = STATUS[job.status] ?? STATUS.queued;
  const reduced = useReducedMotion();
  const live = !isTerminal(job.status);

  return (
    <motion.li variants={listItemVariants} layout exit="exit">
      {/* The row is a link, so the rerun control has to be a SIBLING of the
          anchor rather than inside it: a button nested in an <a> is invalid
          markup and every click on it would also navigate. */}
      <div className="group flex items-center border-b border-rule transition-colors hover:bg-raised">
      <motion.a
        href={href({ kind: "job", id: job.job_id })}
        whileHover={reduced ? undefined : { x: 2 }}
        transition={snap}
        className="flex min-w-0 flex-1 items-center gap-4 px-4 py-3"
      >
        <motion.span
          aria-hidden="true"
          animate={reduced || !live ? undefined : signatures.running}
          className={`shrink-0 font-mono text-[12px] ${s.className}`}
        >
          {s.glyph}
        </motion.span>

        {/* The shared element: flies to the job detail header. */}
        <motion.div layoutId={jobLayoutId(job.job_id)} className="min-w-0 flex-1">
          <span className="block truncate text-ink">{job.name}</span>
          <span className="mt-0.5 block font-mono text-[11px] text-ink-faint">
            {job.job_id}
            {job.restored && " · restored"}
          </span>
        </motion.div>

        {live && job.stage && (
          <span className="hidden shrink-0 font-mono text-[11px] text-accent md:block">
            {job.stage}
          </span>
        )}

        <span className="hidden shrink-0 font-mono text-[11px] tabular-nums text-ink-faint sm:block">
          {when(job.created_at)}
        </span>

        <span
          className={`shrink-0 font-mono text-[11px] uppercase tracking-widest ${s.className}`}
        >
          {job.status}
        </span>

        <span
          aria-hidden="true"
          className="shrink-0 text-ink-faint transition-transform group-hover:translate-x-0.5"
        >
          ›
        </span>
      </motion.a>

      {isTerminal(job.status) && (
        <RerunRow jobId={job.job_id} name={job.name} />
      )}
      </div>
    </motion.li>
  );
}

/** Rerun straight from the list, without opening the job first.
 *
 *  Always rendered rather than revealed on hover: a hover-only control is
 *  invisible to keyboard and touch, and this is the fastest path back after a
 *  run that failed on infrastructure rather than on anything the user chose.
 */
function RerunRow({ jobId, name }: { jobId: string; name: string }) {
  const [busy, setBusy] = useState(false);
  const vm = useVm();

  async function run(e: React.MouseEvent) {
    e.preventDefault();
    e.stopPropagation();
    setBusy(true);
    try {
      const res = await rerunJob(jobId, vm.host);
      refreshJobs();
      toast("ok", "Rerun started", `${name} — same photos and settings.`);
      go({ kind: "job", id: res.job_id });
    } catch (err) {
      toast("fail", "Could not rerun", err instanceof Error ? err.message : "");
      setBusy(false);
    }
  }

  return (
    <button
      type="button"
      onClick={run}
      disabled={busy}
      aria-label={`Rerun ${name}`}
      title="Start a new job from this one's photos and settings"
      className="mr-3 shrink-0 rounded-sharp border border-rule px-2 py-1 font-mono text-[11px] text-ink-faint transition-colors hover:border-accent-edge hover:text-accent disabled:opacity-50"
    >
      {busy ? "…" : "↻"}
    </button>
  );
}

export function JobsRoute() {
  const { jobs, loaded, error } = useJobs();
  const [filter, setFilter] = useState<Filter>("all");

  const counts = useMemo(
    () => ({
      all: jobs.length,
      active: jobs.filter((j) => !isTerminal(j.status)).length,
      done: jobs.filter((j) => j.status === "done").length,
      failed: jobs.filter((j) => j.status === "failed").length,
    }),
    [jobs],
  );

  const shown = useMemo(
    () =>
      jobs.filter((j) => {
        if (filter === "all") return true;
        if (filter === "active") return !isTerminal(j.status);
        return j.status === filter;
      }),
    [jobs, filter],
  );

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-8">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-4">
        <h1 className="font-mono text-[12px] uppercase tracking-widest text-ink-dim">
          Jobs
        </h1>

        {loaded && jobs.length > 0 && (
          <div role="group" aria-label="Filter jobs" className="flex gap-1">
            {(["all", "active", "done", "failed"] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setFilter(f)}
                aria-pressed={filter === f}
                className={`relative rounded-sharp px-2 py-1 font-mono text-[10px] uppercase tracking-widest transition-colors ${
                  filter === f ? "text-accent" : "text-ink-faint hover:text-ink-dim"
                }`}
              >
                {f} {counts[f]}
                {filter === f && (
                  <motion.span
                    layoutId="filter-underline"
                    transition={snap}
                    aria-hidden="true"
                    className="absolute inset-x-1 -bottom-px h-px bg-accent"
                  />
                )}
              </button>
            ))}
          </div>
        )}
      </div>

      {error && (
        <p className="rounded-panel border border-fail-edge bg-fail-wash px-3 py-2 font-mono text-[12px] text-fail">
          {error}
        </p>
      )}

      {!loaded && !error && <p className="font-mono text-[12px] text-ink-faint">loading…</p>}

      {loaded && !error && jobs.length === 0 && (
        <div className="rounded-panel border border-dashed border-rule px-6 py-16 text-center">
          <p className="text-ink-dim">No jobs yet.</p>
          <p className="mt-1 font-mono text-[11px] text-ink-faint">
            Photos in, machinable geometry out.
          </p>
          <a
            href={href({ kind: "new" })}
            className="mt-5 inline-block rounded-panel border border-accent-dim bg-accent-wash px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-accent hover:bg-accent hover:text-void"
          >
            + start one
          </a>
        </div>
      )}

      {jobs.length > 0 && (
        <motion.ul
          variants={listVariants}
          initial="initial"
          animate="animate"
          className="list-none rounded-panel border border-rule bg-panel/60"
        >
          <AnimatePresence initial={false}>
            {shown.map((j) => (
              <JobRow key={j.job_id} job={j} />
            ))}
          </AnimatePresence>
        </motion.ul>
      )}

      {loaded && jobs.length > 0 && shown.length === 0 && (
        <p className="mt-4 font-mono text-[11px] text-ink-faint">
          No {filter} jobs.
        </p>
      )}
    </div>
  );
}
