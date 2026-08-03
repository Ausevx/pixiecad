import { motion } from "motion/react";
import { href } from "@/lib/router";
import { jobLayoutId } from "@/lib/motion";
import { useJobs } from "@/lib/jobs";

/* Layer 3. Pipeline visualisation, artifacts panel, and preview land here;
   for now this is the receiving end of the shared-element transition, which
   is what layer 2 needs to prove. */
export function JobDetailRoute({ jobId }: { jobId: string }) {
  const { jobs } = useJobs();
  const job = jobs.find((j) => j.job_id === jobId);

  return (
    <div className="mx-auto max-w-[1400px] px-5 py-8">
      <a
        href={href({ kind: "jobs" })}
        className="font-mono text-[11px] uppercase tracking-widest text-ink-faint hover:text-accent"
      >
        ← jobs
      </a>

      <motion.div layoutId={jobLayoutId(jobId)} className="mt-4">
        <h1 className="text-2xl text-ink">{job?.name ?? "…"}</h1>
        <p className="mt-0.5 font-mono text-[11px] text-ink-faint">{jobId}</p>
      </motion.div>

      <p className="mt-8 font-mono text-[12px] text-ink-faint">
        Layer 3 — artifacts, pipeline track, preview, logs.
      </p>
    </div>
  );
}
