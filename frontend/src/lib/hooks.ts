import { useEffect, useRef, useState, useSyncExternalStore } from "react";

/* ── Page visibility ──────────────────────────────────────────────────────
   Every poll and every animation loop in this app is gated on this. A GPU job
   runs for many minutes and the user will switch away; continuing to hammer
   gcloud and repaint a canvas nobody is looking at is pure waste on a 16 GB
   laptop that is also doing the user's real work. */

const visibilityListeners = new Set<() => void>();

function subscribeVisibility(onChange: () => void): () => void {
  visibilityListeners.add(onChange);
  if (visibilityListeners.size === 1) {
    document.addEventListener("visibilitychange", notifyVisibility);
  }
  return () => {
    visibilityListeners.delete(onChange);
    if (visibilityListeners.size === 0) {
      document.removeEventListener("visibilitychange", notifyVisibility);
    }
  };
}

const notifyVisibility = () => visibilityListeners.forEach((l) => l());

export function usePageVisible(): boolean {
  return useSyncExternalStore(
    subscribeVisibility,
    () => document.visibilityState === "visible",
    () => true,
  );
}

/* ── Polling ──────────────────────────────────────────────────────────────
   The temporal shape of this app is: seconds of upload, then many minutes of
   GPU work during which the user leaves. So polling must be interval-aware
   (fast while running, slow while queued, stopped when finished), must pause
   when the tab is hidden, and must fire once immediately on return so the
   user never looks at a stale screen. */

export interface PollOptions {
  /** Milliseconds between polls. `null` stops polling entirely — used when a
   *  job reaches a terminal status and there is nothing left to learn. */
  interval: number | null;
  /** Poll even while the tab is hidden. Almost never right; defaults off. */
  whileHidden?: boolean;
}

export function usePoll(
  fn: (signal: AbortSignal) => void | Promise<void>,
  { interval, whileHidden = false }: PollOptions,
): void {
  const visible = usePageVisible();
  // The callback is captured in a ref so a caller can pass an inline closure
  // without restarting the timer on every render.
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (interval === null) return;
    if (!visible && !whileHidden) return;

    let cancelled = false;
    let timer: number | undefined;
    const controller = new AbortController();

    const tick = async () => {
      try {
        await fnRef.current(controller.signal);
      } catch {
        // A failed poll is not fatal; the next one may succeed. Callers that
        // care about the error surface it from inside `fn`.
      }
      // Scheduled after completion rather than on a fixed interval, so a slow
      // response cannot stack requests on top of each other.
      if (!cancelled) timer = window.setTimeout(tick, interval);
    };

    void tick();

    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [interval, visible, whileHidden]);
}

/* ── Ticking counter ──────────────────────────────────────────────────────
   Numbers that change while you watch them count up rather than cut. Used for
   byte totals, face counts and stage timings.

   Deliberately NOT a motion.dev spring: this drives text content, and a
   spring that overshoots would briefly render a face count higher than the
   real one. Machine-generated numbers must never lie, even for 200ms. */

export function useTicker(target: number, durationMs = 550): number {
  const [value, setValue] = useState(target);
  const fromRef = useRef(target);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const from = fromRef.current;
    if (reduced || from === target) {
      fromRef.current = target;
      setValue(target);
      return;
    }

    let raf = 0;
    const start = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      // easeOutCubic: fast then settling, monotonic, never overshoots.
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(from + (target - from) * eased);
      if (t < 1) raf = requestAnimationFrame(step);
      else fromRef.current = target;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);

  return value;
}

/* ── Reduced motion ───────────────────────────────────────────────────────
   motion.dev ships its own hook, but canvas code needs the raw answer outside
   a React tree, so the media query lives here and both use it. */

const REDUCED = "(prefers-reduced-motion: reduce)";

export const prefersReducedMotion = (): boolean =>
  typeof window !== "undefined" && window.matchMedia(REDUCED).matches;

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(prefersReducedMotion);
  useEffect(() => {
    const mq = window.matchMedia(REDUCED);
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}
