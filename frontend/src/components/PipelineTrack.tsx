import { motion } from "motion/react";
import { formatSeconds } from "@/lib/api";
import { useReducedMotion } from "@/lib/hooks";
import { move, signatures, snap } from "@/lib/motion";
import type { JobDetail } from "@/lib/types";

/* ─────────────────────────────────────────────────────────────────────────
   The pipeline as a horizontal track.

   Honest about what it can know. The pipeline reports its stages, with real
   elapsed times, only when the run finishes — so while a job is in flight the
   track shows the coarse stage the server is currently reporting, and once it
   completes it shows the measured timings. It never invents a duration for a
   stage still running, because a fabricated number on a screen full of real
   ones poisons all of them.
   ───────────────────────────────────────────────────────────────────────── */

/** Canonical order. The pipeline may skip stages — a generative run never
 *  does sparse or dense — and skipped stages stay visible rather than
 *  vanishing, so the shape of what ran is legible at a glance. */
const ORDER = [
  "ingest",
  "sparse",
  "dense",
  "generate",
  "scale",
  "clean",
  "decimate",
  "unwrap",
  "bake",
  "export",
] as const;

const LABEL: Record<string, string> = {
  ingest: "ingest",
  sparse: "sparse",
  dense: "dense",
  generate: "generate",
  scale: "scale",
  clean: "clean",
  decimate: "decimate",
  unwrap: "uv",
  bake: "bake",
  export: "export",
  finishing: "finish",
  build: "build",
  // Reported by the pipeline but not part of the canonical order: a stage can
  // fail here without failing the job, so it must be visible.
  sanity: "sanity",
  parts: "parts",
  regime: "regime",
};

type State = "ok" | "failed" | "skipped" | "running" | "pending";

const STYLE: Record<State, string> = {
  ok: "border-ok-edge bg-ok-wash text-ok",
  failed: "border-fail-edge bg-fail-wash text-fail",
  skipped: "border-rule bg-panel text-ink-faint",
  running: "border-accent-dim bg-accent-wash text-accent",
  pending: "border-rule bg-transparent text-ink-faint opacity-50",
};

interface Cell {
  key: string;
  label: string;
  state: State;
  seconds: number | null;
  detail: string;
}

/** Build the track.

 *  Once the pipeline has reported, ITS list is the truth and is shown in its
 *  own order — every stage, including ones this file has never heard of. An
 *  earlier version filtered against the hardcoded order above and silently
 *  dropped `sanity` and `parts`, which meant a FAILED sanity check simply did
 *  not appear. A pipeline view that hides a failure is worse than none. */
function buildCells(job: JobDetail): Cell[] {
  const reported = job.stages ?? [];

  if (reported.length > 0) {
    return reported.map((s): Cell => ({
      key: s.name,
      label: LABEL[s.name] ?? s.name,
      state:
        s.status === "ok" ? "ok" : s.status === "failed" ? "failed" : "skipped",
      seconds: s.seconds,
      detail: s.detail,
    }));
  }

  // Nothing reported yet.
  //
  // run_build accumulates its stages in a local list and only returns them
  // when the whole build is done, so there is genuinely no per-stage signal
  // while it runs. An earlier version filled that gap by marking the FIRST
  // lane running — which meant the track sat on "ingest" for the entire job,
  // including minutes of GPU texturing. That is not a cosmetic problem: it
  // states something false about where the work is.
  //
  // So no lane is marked running. The track shows the shape that is coming,
  // greyed, and the coarse phase is reported separately by the caller from
  // job.stage, which IS live.
  return ORDER.map((name): Cell => ({
    key: name,
    label: LABEL[name] ?? name,
    state: job.status === "failed" ? "skipped" : "pending",
    seconds: null,
    detail: "",
  }));
}

function StageCell({ cell }: { cell: Cell }) {
  const reduced = useReducedMotion();

  const animate =
    reduced || cell.state === "pending"
      ? undefined
      : cell.state === "running"
        ? signatures.running
        : cell.state === "failed"
          ? signatures.shake
          : cell.state === "ok"
            ? signatures.latch
            : undefined;

  return (
    <motion.li
      layout
      transition={move}
      className="min-w-0 flex-1"
      // The detail line the pipeline recorded is genuinely useful ("14 photos
      // accepted, 1 rejected") and there is nowhere else it is shown.
      title={cell.detail || undefined}
    >
      <motion.div
        animate={animate}
        transition={snap}
        className={`rounded-panel border px-2 py-1.5 ${STYLE[cell.state]} ${
          // A left rule is how a running stage reads under reduced motion,
          // where the pulse is gone.
          reduced && cell.state === "running" ? "border-l-2 border-l-accent" : ""
        }`}
      >
        <span className="block truncate font-mono text-[10px] uppercase tracking-widest">
          {cell.label}
        </span>
        <span className="mt-0.5 block font-mono text-[10px] tabular-nums text-ink-faint">
          {cell.seconds != null
            ? formatSeconds(cell.seconds)
            : cell.state === "running"
              ? "···"
              : cell.state === "skipped"
                ? "skip"
                : "—"}
        </span>
      </motion.div>
    </motion.li>
  );
}

/** What the server is actually doing right now, as opposed to which stages
 *  have finished. These are the only phases the worker reports live, and
 *  saying exactly this much is better than implying per-stage precision the
 *  pipeline does not emit while it runs. */
const PHASE: Record<string, string> = {
  queued: "queued",
  "waiting for GPU": "waiting for a GPU VM",
  build: "building — ingest through export, on the GPU",
  finishing: "finishing — smoothing, texturing, segmentation",
  clean: "cleaning mesh",
  decimate: "decimating",
  unwrap: "unwrapping UVs",
  bake: "baking normals",
  export: "exporting",
};

export function PipelineTrack({ job }: { job: JobDetail }) {
  const cells = buildCells(job);
  const total = (job.stages ?? []).reduce((sum, s) => sum + s.seconds, 0);
  const reduced = useReducedMotion();
  const live = job.status === "running" || job.status === "queued";
  const phase = PHASE[job.stage] ?? job.stage;

  // The whole track is one screen-reader sentence. Ten separate cells each
  // announcing "ingest 0.41 seconds" is unusable; the summary is what a
  // non-visual user actually wants from a progress display.
  const spoken = cells
    .map((c) => `${c.label}: ${c.state}${c.seconds != null ? `, ${formatSeconds(c.seconds)}` : ""}`)
    .join(". ");

  return (
    <section aria-labelledby="pipeline-heading">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2
          id="pipeline-heading"
          className="font-mono text-[11px] uppercase tracking-widest text-ink-dim"
        >
          Pipeline
        </h2>
        {total > 0 && (
          <span className="font-mono text-[11px] tabular-nums text-ink-faint">
            {formatSeconds(total)} total
          </span>
        )}
      </div>

      <p className="sr-only">{spoken}</p>

      {/* The live phase. Shown whenever the job is still moving, because the
          per-stage track below cannot update until the build reports. */}
      {live && (
        <div className="mb-2 flex items-center gap-2 rounded-panel border border-accent-dim bg-accent-wash px-3 py-1.5">
          <motion.span
            aria-hidden="true"
            animate={reduced ? undefined : signatures.running}
            className="text-[11px] text-accent"
          >
            ◐
          </motion.span>
          <span className="font-mono text-[11px] text-accent">{phase}</span>
          {cells.every((c) => c.seconds === null) && (
            <span className="ml-auto font-mono text-[10px] text-ink-faint">
              per-stage timings arrive when the build reports
            </span>
          )}
        </div>
      )}

      <ul aria-hidden="true" className="flex list-none gap-1 overflow-x-auto pb-1">
        {cells.map((c) => (
          <StageCell key={c.key} cell={c} />
        ))}
      </ul>

      {job.restored && (
        <p className="mt-2 font-mono text-[11px] text-ink-faint">
          Timings unavailable — this job was rebuilt from disk after a restart.
        </p>
      )}
    </section>
  );
}
