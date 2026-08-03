import { useEffect, useState } from "react";
import { getVmStatus } from "./api";
import type { VmInstance, VmStatus } from "./types";

/* ─────────────────────────────────────────────────────────────────────────
   VM presence, as one shared store.

   Three separate parts of the UI need to know whether GPU capacity exists —
   the header pill, the compute drawer, and the new-job capacity strip — and
   each polling independently would triple the gcloud calls to tell three
   components the same thing. So there is one poll, reference-counted by how
   many components are mounted, and it stops when none are.
   ───────────────────────────────────────────────────────────────────────── */

export type VmPresence =
  /** gcloud missing or erroring — we genuinely do not know. */
  | "unknown"
  /** No VM. Jobs needing a GPU cannot run. */
  | "absent"
  /** A provision is in flight: the VM may answer RUNNING but has no worker
   *  images yet, so a job submitted now would die on a missing container. */
  | "building"
  /** Up and usable. */
  | "ready"
  /** Was running, then vanished without anyone asking it to. */
  | "preempted";

export interface VmState {
  presence: VmPresence;
  status: VmStatus | null;
  /** The SSH alias to hand the pipeline, when there is one. */
  host: string;
  /** The instance the pill is describing, when there is one. */
  instance: VmInstance | null;
  reason: string;
  /** Set once a preemption is detected; cleared by acknowledging it. Kept
   *  separate from `presence` so the warning survives the next poll, which
   *  will report a perfectly ordinary "absent". */
  preemptedName: string | null;
}

const INITIAL: VmState = {
  presence: "unknown",
  status: null,
  host: "",
  instance: null,
  reason: "",
  preemptedName: null,
};

let state: VmState = INITIAL;
const listeners = new Set<(s: VmState) => void>();

function set(next: Partial<VmState>): void {
  state = { ...state, ...next };
  listeners.forEach((l) => l(state));
}

/* ── Preemption detection ─────────────────────────────────────────────────
   Spot L4s do get preempted mid-job — and on this project preemption deletes
   the boot disk, losing the whole model cache, so it is the single most
   expensive event a user can fail to notice.

   It is detectable client-side with no backend work: a VM that was RUNNING
   and is now absent from the instance list, without the user having asked for
   it to go, was taken. `expectTeardown` is what distinguishes "I deleted it"
   from "it was deleted for me". */

let lastRunning = new Set<string>();
let teardownExpected = false;

/** Call immediately before a teardown request so its disappearance is not
 *  reported as a preemption. */
export function expectTeardown(): void {
  teardownExpected = true;
}

export function acknowledgePreemption(): void {
  set({ preemptedName: null });
}

function reduce(vm: VmStatus): void {
  if (!vm.available) {
    lastRunning = new Set();
    set({
      presence: "unknown",
      status: vm,
      reason: vm.reason ?? "gcloud unavailable",
      instance: null,
      host: "",
    });
    return;
  }

  const running = (vm.vms ?? []).filter(
    (v) => String(v.status).toUpperCase() === "RUNNING",
  );
  const names = new Set(running.map((v) => v.name));

  let preemptedName = state.preemptedName;
  for (const was of lastRunning) {
    if (!names.has(was) && !teardownExpected) preemptedName = was;
  }
  // Only consume the teardown grace once the VM has actually gone, so a slow
  // delete does not let a later genuine preemption slip through unreported.
  if (teardownExpected && lastRunning.size > 0 && names.size === 0) {
    teardownExpected = false;
  }
  lastRunning = names;

  const instance = running[0] ?? null;
  const presence: VmPresence = vm.building
    ? "building"
    : running.length > 0
      ? "ready"
      : preemptedName
        ? "preempted"
        : "absent";

  set({
    presence,
    status: vm,
    instance,
    host: vm.host || instance?.host || "",
    reason: "",
    preemptedName,
  });
}

/* ── Reference-counted polling ────────────────────────────────────────── */

let subscribers = 0;
let timer: number | undefined;
let controller: AbortController | undefined;

/** Fast while something is changing, slow when the answer is settled. A VM
 *  that has been up for an hour does not need checking every 5 seconds. */
const intervalFor = (p: VmPresence): number => (p === "building" ? 5_000 : 20_000);

async function tick(): Promise<void> {
  controller?.abort();
  controller = new AbortController();
  try {
    reduce(await getVmStatus(controller.signal));
  } catch {
    // Leave the last known state standing rather than flashing "unknown" on a
    // single dropped request — the pill flickering between states is worse
    // than it being a few seconds stale.
  }
  if (subscribers > 0 && document.visibilityState === "visible") {
    timer = window.setTimeout(() => void tick(), intervalFor(state.presence));
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
  if (document.visibilityState === "visible") {
    if (subscribers > 0) start();
  } else {
    stop();
  }
}

/** Force an immediate refresh — used right after provisioning or teardown, so
 *  the pill reacts to the user's own action without waiting for the interval. */
export function refreshVm(): void {
  stop();
  if (subscribers > 0) start();
}

export function useVm(): VmState {
  const [local, setLocal] = useState(state);

  useEffect(() => {
    listeners.add(setLocal);
    subscribers += 1;
    if (subscribers === 1) {
      document.addEventListener("visibilitychange", onVisibility);
      start();
    }
    // Adopt whatever the store already knows, so a component mounting midway
    // through does not sit on "unknown" until the next poll lands.
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

/** Whether a job that needs a GPU can start right now. The new-job form asks
 *  exactly this and nothing more — it should not have to reason about zones,
 *  images or provision state. */
export const hasGpuCapacity = (s: VmState): boolean => s.presence === "ready";
