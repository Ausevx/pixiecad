import type { Transition, Variants } from "motion/react";

/* ─────────────────────────────────────────────────────────────────────────
   The motion vocabulary.

   Four springs, named for what they do rather than for their numbers. No
   component writes its own stiffness — if something needs to move and none
   of these fit, the right fix is to argue about which one it should be, not
   to add a fifth.

   There are no ease curves in this application. Every duration-based
   transition below is either a colour cross-fade (which has no physics to
   get wrong) or the background field, which drifts rather than travels.
   ───────────────────────────────────────────────────────────────────────── */

/** Buttons, checkboxes, tags, magnetic hover. Arrives before you notice. */
export const snap: Transition = {
  type: "spring",
  stiffness: 520,
  damping: 32,
  mass: 0.6,
};

/** Layout and shared-element transitions: job card → job detail header. */
export const move: Transition = {
  type: "spring",
  stiffness: 260,
  damping: 30,
  mass: 0.9,
};

/** Drawers, panel expansion, log collapse. Weighty; you watch it arrive. */
export const settle: Transition = {
  type: "spring",
  stiffness: 120,
  damping: 22,
  mass: 1.2,
};

/** Physical drop: uploaded photos landing. Overshoots, then bounces once. */
export const drop: Transition = {
  type: "spring",
  stiffness: 340,
  damping: 18,
  mass: 1.4,
};

/** Colour and opacity only. Nothing moves, so there is no spring to model. */
export const fade: Transition = { duration: 0.18 };

/* ── State signatures ─────────────────────────────────────────────────────
   Each pipeline/VM state gets a motion signature distinct enough to identify
   from peripheral vision, without reading the label. */

export const signatures = {
  /** Stage in progress: a slow swell, indefinitely. */
  running: {
    scale: [1, 1.035, 1],
    transition: { duration: 1.4, repeat: Infinity, ease: "easeInOut" as const },
  },
  /** Stage finished: one overshoot, then it latches and never moves again. */
  latch: {
    scale: [1, 1.12, 1],
    transition: { ...snap, times: [0, 0.4, 1] },
  },
  /** Stage failed: a decaying horizontal shake. The red border persists. */
  shake: {
    x: [0, -6, 5, -3, 2, 0],
    transition: { duration: 0.45 },
  },
  /** VM provisioning: a slow breath. Something is happening, slowly. */
  breathe: {
    opacity: [0.55, 1, 0.55],
    transition: { duration: 2.5, repeat: Infinity, ease: "easeInOut" as const },
  },
  /** VM preempted: three hard strobes. Unmissable, then it stops. */
  strobe: {
    opacity: [1, 0.15, 1, 0.15, 1, 0.15, 1],
    transition: { duration: 0.9, times: [0, 0.16, 0.33, 0.5, 0.66, 0.83, 1] },
  },
} satisfies Record<string, object>;

/* ── Enter/exit variants ──────────────────────────────────────────────── */

/** Route change. A 4px lift, never a slide — a whole page sliding on every
 *  navigation makes a tool feel like a slideshow. */
export const routeVariants: Variants = {
  initial: { opacity: 0, y: 4 },
  animate: { opacity: 1, y: 0, transition: { ...move, opacity: fade } },
  exit: { opacity: 0, y: -4, transition: fade },
};

/** Toasts spring up from the bottom-right and overshoot slightly. */
export const toastVariants: Variants = {
  initial: { opacity: 0, y: 16, scale: 0.94 },
  animate: { opacity: 1, y: 0, scale: 1, transition: drop },
  exit: { opacity: 0, y: 8, scale: 0.97, transition: fade },
};

/** The compute drawer, off the right edge. */
export const drawerVariants: Variants = {
  initial: { x: "100%" },
  animate: { x: 0, transition: settle },
  exit: { x: "100%", transition: { ...settle, damping: 30 } },
};

/** Staggered list entry — jobs, artifacts, photo cards. Capped so a long
 *  list never makes the last row wait on the first. */
export const listVariants: Variants = {
  animate: { transition: { staggerChildren: 0.03, delayChildren: 0.02 } },
};

export const listItemVariants: Variants = {
  initial: { opacity: 0, y: 6 },
  animate: { opacity: 1, y: 0, transition: move },
  exit: { opacity: 0, y: -4, transition: fade },
};

/** Shared-element id linking a job card to its detail header. Both ends must
 *  agree, so the id is generated here rather than written out twice. */
export const jobLayoutId = (jobId: string) => `job-${jobId}`;
