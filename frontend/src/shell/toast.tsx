import { AnimatePresence, motion } from "motion/react";
import { useEffect, useState } from "react";
import { toastVariants } from "@/lib/motion";

/* ─────────────────────────────────────────────────────────────────────────
   Toasts.

   A module-level store rather than context, because the things that need to
   raise one — a failed convert, a preemption detected inside the VM poll, a
   provision that died — are not all inside the React tree.
   ───────────────────────────────────────────────────────────────────────── */

export type ToastTone = "info" | "ok" | "warn" | "fail";

export interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  body?: string;
  /** Milliseconds before auto-dismiss. `null` means it stays until dismissed
   *  — used for anything that costs money or loses work, which the user must
   *  actually acknowledge rather than miss while looking elsewhere. */
  ttl: number | null;
}

let nextId = 1;
let toasts: Toast[] = [];
const listeners = new Set<(t: Toast[]) => void>();

const emit = () => listeners.forEach((l) => l(toasts));

export function toast(
  tone: ToastTone,
  title: string,
  body?: string,
  ttl: number | null = tone === "fail" ? null : 5000,
): number {
  const id = nextId++;
  toasts = [...toasts, { id, tone, title, body, ttl }];
  emit();
  return id;
}

export function dismissToast(id: number): void {
  toasts = toasts.filter((t) => t.id !== id);
  emit();
}

const TONE: Record<ToastTone, string> = {
  info: "border-rule-bright bg-raised text-ink",
  ok: "border-[#17452b] bg-[#0a1d12] text-ok",
  warn: "border-[#4a3a10] bg-[#231a06] text-warn",
  fail: "border-[#5c1d1d] bg-[#2a0d0d] text-fail",
};

function ToastCard({ t }: { t: Toast }) {
  useEffect(() => {
    if (t.ttl === null) return;
    const timer = window.setTimeout(() => dismissToast(t.id), t.ttl);
    return () => window.clearTimeout(timer);
  }, [t.id, t.ttl]);

  return (
    <motion.li
      layout
      variants={toastVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className={`pointer-events-auto w-[min(24rem,calc(100vw-2rem))] rounded-panel border px-3 py-2 ${TONE[t.tone]}`}
    >
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="font-mono text-[11px] uppercase tracking-widest">{t.title}</p>
          {t.body && (
            <p className="mt-1 font-mono text-[11px] leading-relaxed break-words text-ink-dim">
              {t.body}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={() => dismissToast(t.id)}
          aria-label={`Dismiss: ${t.title}`}
          className="-mr-1 -mt-1 shrink-0 rounded-sharp px-1.5 py-0.5 font-mono text-[13px] leading-none text-ink-faint hover:text-ink"
        >
          ×
        </button>
      </div>
    </motion.li>
  );
}

export function ToastHost() {
  const [items, setItems] = useState<Toast[]>(toasts);
  useEffect(() => {
    listeners.add(setItems);
    setItems(toasts);
    return () => {
      listeners.delete(setItems);
    };
  }, []);

  return (
    // aria-live so a toast raised while focus is elsewhere is still announced;
    // "polite" rather than "assertive" because none of these interrupt a task
    // in progress, they report on one.
    <ul
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed bottom-4 right-4 z-50 flex list-none flex-col-reverse gap-2"
    >
      <AnimatePresence initial={false}>
        {items.map((t) => (
          <ToastCard key={t.id} t={t} />
        ))}
      </AnimatePresence>
    </ul>
  );
}
