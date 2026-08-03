import { useEffect, useState } from "react";
import { listJobs } from "./api";
import { isTerminal, type JobSummary } from "./types";

/* ─────────────────────────────────────────────────────────────────────────
   The job list, as one shared store.

   Same reasoning as the VM store: the jobs screen, the header, and the
   background field's intensity all derive from this one list, and it should
   be fetched once.

   The cadence matters more here than anywhere else in the app. The old
   dashboard polled every 1.5 seconds unconditionally, forever, whether or not
   anything was running and whether or not the tab was even visible. This
   polls fast only while something is genuinely in flight, and stops dead
   otherwise.
   ───────────────────────────────────────────────────────────────────────── */

export interface JobsState {
  jobs: JobSummary[];
  /** False until the first response lands, so the list can distinguish "no
   *  jobs" from "not asked yet" — they deserve different screens. */
  loaded: boolean;
  error: string | null;
}

let state: JobsState = { jobs: [], loaded: false, error: null };
const listeners = new Set<(s: JobsState) => void>();

function set(next: Partial<JobsState>): void {
  state = { ...state, ...next };
  listeners.forEach((l) => l(state));
}

export const activeCount = (jobs: JobSummary[]): number =>
  jobs.filter((j) => !isTerminal(j.status)).length;

let subscribers = 0;
let timer: number | undefined;
let controller: AbortController | undefined;

/** 2s while work is in flight; 30s when everything has settled. A finished
 *  queue still gets checked, because a job can be created in another tab. */
const intervalFor = (jobs: JobSummary[]): number =>
  activeCount(jobs) > 0 ? 2_000 : 30_000;

async function tick(): Promise<void> {
  controller?.abort();
  controller = new AbortController();
  try {
    set({ jobs: await listJobs(controller.signal), loaded: true, error: null });
  } catch (e) {
    // Only surface an error once we have never succeeded. Losing one poll on
    // a list we already have is not worth telling the user about.
    if (!state.loaded) {
      set({ loaded: true, error: e instanceof Error ? e.message : "failed to load jobs" });
    }
  }
  if (subscribers > 0 && document.visibilityState === "visible") {
    timer = window.setTimeout(() => void tick(), intervalFor(state.jobs));
  }
}

function start(): void {
  if (timer !== undefined) return;
  void tick();
}

function stop(): void {
  window.clearTimeout(timer);
  timer = undefined;
  controller?.abort();
  controller = undefined;
}

function onVisibility(): void {
  // Refetch immediately on return rather than waiting out the interval: a
  // user coming back after ten minutes must never be shown a stale queue.
  if (document.visibilityState === "visible") {
    if (subscribers > 0) start();
  } else {
    stop();
  }
}

/** Force a refresh — after creating or deleting a job, so the list reflects
 *  the user's own action without waiting for the next tick. */
export function refreshJobs(): void {
  stop();
  if (subscribers > 0) start();
}

export function useJobs(): JobsState {
  const [local, setLocal] = useState(state);

  useEffect(() => {
    listeners.add(setLocal);
    subscribers += 1;
    if (subscribers === 1) {
      document.addEventListener("visibilitychange", onVisibility);
      start();
    }
    setLocal(state);

    return () => {
      listeners.delete(setLocal);
      subscribers -= 1;
      if (subscribers === 0) {
        document.removeEventListener("visibilitychange", onVisibility);
        stop();
      }
    };
  }, []);

  return local;
}
