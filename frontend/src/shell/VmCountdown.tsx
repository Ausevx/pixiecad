import { useEffect, useState } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   How long until the GPU VM deletes itself.

   Every worker VM is launched with --max-run-duration and
   --instance-termination-action=DELETE, so it does not linger and quietly
   bill for a weekend. The cost of that is a hard deadline nobody could see,
   and it has bitten twice: once it took the VM down BETWEEN the normal-map
   bake and texturing, which surfaced as two unexplained "timed out after 30s"
   errors, and once it deleted a machine holding 43 GB of freshly downloaded
   model weights minutes before they were baked into an image.

   Neither was diagnosable from the dashboard, because uptime was shown and
   remaining time was not. Uptime tells you what you have spent; only this
   tells you whether the run you are about to start will survive.

   A ring rather than a bar: it reads as a clock, it costs one glance, and it
   keeps its size as the number changes so the header never reflows.
   ───────────────────────────────────────────────────────────────────────── */

/** Under this, a long job is a gamble -- a TRELLIS run plus texturing and a
 *  2048 bake has measured over 25 minutes end to end. */
const WARN_SECONDS = 30 * 60;
/** Under this, assume anything in flight will be cut off. */
const CRITICAL_SECONDS = 10 * 60;

function label(seconds: number): string {
  if (seconds <= 0) return "0m";
  const h = Math.floor(seconds / 3600);
  const m = Math.round((seconds % 3600) / 60);
  // Hours only past an hour: "5h 12m" is more than anyone needs at a glance,
  // and the extra characters widen the header for no gain.
  return h > 0 ? `${h}h${m > 0 ? ` ${m}m` : ""}` : `${m}m`;
}

export function VmCountdown({
  secondsRemaining,
  maxRunSeconds,
}: {
  secondsRemaining: number | null;
  maxRunSeconds: number | null;
}) {
  // The poll behind this is deliberately slow -- it shells out to gcloud --
  // so the countdown ticks locally and each poll corrects the drift, rather
  // than freezing between updates.
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    setElapsed(0);
    if (secondsRemaining == null) return;
    const id = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(id);
  }, [secondsRemaining]);

  if (secondsRemaining == null || !maxRunSeconds) return null;

  const left = Math.max(0, secondsRemaining - elapsed);
  const fraction = Math.max(0, Math.min(1, left / maxRunSeconds));

  const tone =
    left <= CRITICAL_SECONDS
      ? { stroke: "var(--px-fail)", text: "text-fail" }
      : left <= WARN_SECONDS
        ? { stroke: "var(--px-warn)", text: "text-warn" }
        : { stroke: "var(--px-ok)", text: "text-ok" };

  // r chosen so the 3px stroke sits fully inside the 18px box.
  const r = 6.5;
  const circumference = 2 * Math.PI * r;

  return (
    <span
      className="flex items-center gap-1.5"
      title={`GPU VM self-deletes in ${label(left)} (--max-run-duration ${Math.round(
        maxRunSeconds / 3600,
      )}h). Work in flight is lost when it does.`}
    >
      <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" className="shrink-0">
        <circle
          cx="9"
          cy="9"
          r={r}
          fill="none"
          stroke="var(--px-rule-bright)"
          strokeWidth="3"
        />
        <circle
          cx="9"
          cy="9"
          r={r}
          fill="none"
          stroke={tone.stroke}
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - fraction)}
          // Drain anticlockwise from 12 o'clock, the direction a clock hand
          // does not go, so it reads as depleting rather than elapsing.
          transform="rotate(-90 9 9)"
          style={{ transition: "stroke-dashoffset 1s linear" }}
        />
      </svg>
      <span className={`tabular-nums ${tone.text}`}>{label(left)} left</span>
    </span>
  );
}
