import { motion } from "motion/react";
import { signatures, snap } from "@/lib/motion";
import { useReducedMotion } from "@/lib/hooks";
import { useVm, type VmPresence, type VmState } from "@/lib/vm";
import { VmCountdown } from "./VmCountdown";

/* ─────────────────────────────────────────────────────────────────────────
   The VM pill: compute as persistent presence.

   It used to be a <label> inside the job form, which meant you could only see
   whether you had GPU capacity while you happened to be filling in a job. It
   now lives in the shell header on every route, and it is the only way into
   the compute drawer.

   Each state has a motion signature distinct enough to read peripherally,
   without looking directly at it or parsing the text.
   ───────────────────────────────────────────────────────────────────────── */

const PRESENTATION: Record<
  VmPresence,
  { glyph: string; label: string; className: string }
> = {
  unknown: {
    glyph: "◌",
    label: "gcloud unavailable",
    className: "bg-raised border-rule text-ink-faint",
  },
  absent: {
    glyph: "○",
    label: "no GPU",
    className: "bg-raised border-rule text-ink-dim",
  },
  building: {
    glyph: "◐",
    label: "building images",
    className: "bg-warn-wash border-warn-edge text-warn",
  },
  ready: {
    glyph: "●",
    label: "GPU ready",
    className: "bg-ok-wash border-ok-edge text-ok",
  },
  preempted: {
    glyph: "▲",
    label: "preempted",
    className: "bg-fail-wash border-fail-edge text-fail",
  },
};

/** Uptime is the number that costs money, so it is always shown when known,
 *  and always monospace so it does not jitter the pill width as it ticks. */
function uptime(vm: VmState): string | null {
  const h = vm.instance?.uptime_hours;
  if (h == null) return null;
  return h < 1 ? `${Math.round(h * 60)}m` : `${h.toFixed(1)}h`;
}

export function VmPill({ onOpen }: { onOpen: () => void }) {
  const vm = useVm();
  const reduced = useReducedMotion();
  const presence: VmPresence = vm.preemptedName ? "preempted" : vm.presence;
  const p = PRESENTATION[presence];
  const hours = uptime(vm);

  // Reduced motion gets the same information as a static difference: the
  // glyph and colour already distinguish every state, so nothing is lost by
  // removing the movement.
  const glyphAnimation = reduced
    ? undefined
    : presence === "building"
      ? signatures.breathe
      : presence === "preempted"
        ? signatures.strobe
        : undefined;

  const description =
    presence === "preempted"
      ? `GPU VM ${vm.preemptedName} was preempted. Open compute`
      : presence === "ready"
        ? `GPU ready${hours ? `, up ${hours}` : ""}${
            vm.instance?.seconds_remaining != null
              ? `, self-deletes in ${Math.round(vm.instance.seconds_remaining / 60)} minutes`
              : ""
          }. Open compute`
        : `${p.label}. Open compute`;

  return (
    <motion.button
      type="button"
      onClick={onOpen}
      aria-label={description}
      whileHover={reduced ? undefined : { scale: 1.03 }}
      whileTap={reduced ? undefined : { scale: 0.97 }}
      transition={snap}
      className={`group flex items-center gap-2 rounded-panel border-2 px-3 py-1.5 font-mono text-[12.5px] font-medium tracking-wide tabular-nums transition-[filter] hover:brightness-125 ${p.className}`}
    >
      <motion.span aria-hidden="true" animate={glyphAnimation} className="leading-none">
        {p.glyph}
      </motion.span>
      <span className="uppercase">{p.label}</span>
      {hours && presence === "ready" && <span className="text-ink-faint">· {hours}</span>}
      {/* Guarded on the data, not just on presence: a VM with no
          --max-run-duration has no deadline to show, and an older server does
          not send one at all. Rendering the separator outside this check left
          a dangling middot with nothing after it. */}
      {presence === "ready" &&
        vm.instance?.seconds_remaining != null &&
        vm.instance?.max_run_seconds != null && (
          <>
            <span aria-hidden="true" className="text-ink-faint">
              ·
            </span>
            <VmCountdown
              secondsRemaining={vm.instance.seconds_remaining}
              maxRunSeconds={vm.instance.max_run_seconds}
            />
          </>
        )}
      <span
        aria-hidden="true"
        className="text-ink-faint transition-transform group-hover:translate-x-0.5"
      >
        ›
      </span>
    </motion.button>
  );
}
