import { AnimatePresence, motion, MotionConfig } from "motion/react";
import { useEffect } from "react";
import { BackgroundField } from "./BackgroundField";
import { Header } from "./Header";
import { ToastHost, toast } from "./toast";
import { routeVariants } from "@/lib/motion";
import { useReducedMotion } from "@/lib/hooks";
import { href, useDrawer, useLinkInterception, useRoute } from "@/lib/router";
import { acknowledgePreemption, useVm } from "@/lib/vm";

/* ─────────────────────────────────────────────────────────────────────────
   The application frame: background field, header, routed content, drawer.

   The frame itself never unmounts, so the background canvas keeps its state
   and the header's shared-layout underline slides across navigations instead
   of being rebuilt.
   ───────────────────────────────────────────────────────────────────────── */

/** Preemption is the one event severe enough to interrupt whatever the user
 *  is doing: it costs a running job and, on this project, the VM's whole model
 *  cache along with the boot disk. Raised here rather than inside the poll so
 *  it fires exactly once per event regardless of how many components observe
 *  the store. */
function usePreemptionAlarm(): void {
  const vm = useVm();
  useEffect(() => {
    if (!vm.preemptedName) return;
    toast(
      "fail",
      "GPU VM preempted",
      `${vm.preemptedName} was reclaimed. Any job on it has stopped, and the boot disk — including the model cache — is gone. Provision again from Compute.`,
      null,
    );
    // Acknowledged immediately so the alarm does not re-fire on the next poll;
    // the toast itself persists until the user dismisses it.
    acknowledgePreemption();
  }, [vm.preemptedName]);
}

function NotFound({ path }: { path: string }) {
  return (
    <div className="mx-auto max-w-[1400px] px-5 py-20">
      <p className="font-mono text-[11px] uppercase tracking-widest text-fail">404</p>
      <h1 className="mt-2 text-xl">No such screen</h1>
      <p className="mt-1 font-mono text-[12px] text-ink-dim">{path}</p>
      <a
        href={href({ kind: "jobs" })}
        className="mt-6 inline-block rounded-panel border border-rule-bright px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:border-accent-dim hover:text-accent"
      >
        ← back to jobs
      </a>
    </div>
  );
}

export function Shell({
  children,
  /** 0..1 — how much of the queue is active. Drives the background field. */
  activity,
}: {
  children: React.ReactNode;
  activity: number;
}) {
  const route = useRoute();
  const drawer = useDrawer();
  const reduced = useReducedMotion();
  useLinkInterception();
  usePreemptionAlarm();

  // Escape closes the drawer from anywhere. Registered at the shell so it
  // works even when focus is inside the routed content.
  useEffect(() => {
    if (!drawer.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") drawer.setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawer.open, drawer.setOpen]);

  // Reduced motion is enforced at the root rather than per component, so a
  // component that forgets still cannot animate. Duration 0 makes layout
  // animations resolve instantly instead of merely faster.
  return (
    <MotionConfig
      reducedMotion={reduced ? "always" : "never"}
      transition={reduced ? { duration: 0 } : undefined}
    >
      <BackgroundField intensity={activity} />

      <div className="relative z-10 flex min-h-full flex-col">
        <Header onOpenCompute={() => drawer.setOpen(true)} />

        <main id="main" className="flex-1">
          {/* mode="wait" so the outgoing screen is gone before the next
              arrives; overlapping two full pages reads as a glitch, not a
              transition. */}
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={route.kind === "job" ? `job-${route.id}` : route.kind}
              variants={routeVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              {route.kind === "unknown" ? <NotFound path={route.path} /> : children}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>

      <ToastHost />
    </MotionConfig>
  );
}
