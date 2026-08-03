import { motion } from "motion/react";
import { href } from "@/lib/router";
import { jobLayoutId, listItemVariants, listVariants } from "@/lib/motion";
import { useJobs } from "@/lib/jobs";
import type { JobStatus, JobSummary } from "@/lib/types";

/* Layer 2 renders this list well enough to prove routing and the shared
   element. Scanning affordances — filters, sizes, timestamps, thumbnails —
   arrive in layer 5. */

const STATUS: Record<JobStatus, { dot: string; text: string }> = {
  queued: { dot: "text-ink-faint", text: "text-ink-faint" },
  running: { dot: "text-accent", text: "text-accent" },
  done: { dot: "text-ok", text: "text-ok" },
  failed: { dot: "text-fail", text: "text-fail" },
};

function JobCard({ job }: { job: JobSummary }) {
  const s = STATUS[job.status] ?? STATUS.queued;
  return (
    <motion.li variants={listItemVariants} layout>
      <a
        href={href({ kind: "job", id: job.job_id })}
        className="group flex items-center gap-4 border-b border-rule px-4 py-3 transition-colors hover:bg-raised"
      >
        {/* The shared element: this block flies to the job detail header. */}
        <motion.div layoutId={jobLayoutId(job.job_id)} className="min-w-0 flex-1">
          <span className="block truncate text-ink">{job.name}</span>
          <span className="mt-0.5 block font-mono text-[11px] text-ink-faint">
            {job.job_id}
          </span>
        </motion.div>

        <span className="hidden font-mono text-[11px] text-ink-faint sm:block">
          {job.stage}
        </span>

        <span className={`flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-widest ${s.text}`}>
          <span aria-hidden="true" className={s.dot}>
            {job.status === "done" ? "●" : job.status === "failed" ? "▲" : "○"}
          </span>
          {job.status}
        </span>

        <span
          aria-hidden="true"
          className="text-ink-faint transition-transform group-hover:translate-x-0.5"
        >
          ›
        </span>
      </a>
    </motion.li>
  );
}

export function JobsRoute() {
  const { jobs, loaded, error } = useJobs();

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-8">
      <div className="mb-6 flex items-baseline gap-3">
        <h1 className="font-mono text-[12px] uppercase tracking-widest text-ink-dim">
          Jobs
        </h1>
        {loaded && (
          <span className="font-mono text-[11px] tabular-nums text-ink-faint">
            {jobs.length}
          </span>
        )}
      </div>

      {error && (
        <p className="rounded-panel border border-[#5c1d1d] bg-[#2a0d0d] px-3 py-2 font-mono text-[12px] text-fail">
          {error}
        </p>
      )}

      {!loaded && !error && (
        <p className="font-mono text-[12px] text-ink-faint">loading…</p>
      )}

      {loaded && !error && jobs.length === 0 && (
        <div className="rounded-panel border border-dashed border-rule px-6 py-12 text-center">
          <p className="text-ink-dim">No jobs yet.</p>
          <a
            href={href({ kind: "new" })}
            className="mt-4 inline-block rounded-panel border border-accent-dim bg-accent-wash px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-accent hover:bg-accent hover:text-void"
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
          {jobs.map((j) => (
            <JobCard key={j.job_id} job={j} />
          ))}
        </motion.ul>
      )}
    </div>
  );
}
