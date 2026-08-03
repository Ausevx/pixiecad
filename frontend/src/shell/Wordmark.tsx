import { useEffect, useRef } from "react";
import { prefersReducedMotion } from "@/lib/hooks";

/* ─────────────────────────────────────────────────────────────────────────
   The ASCII wordmark.

   Resolves in from noise on first load, then settles and stops — permanently.
   An idle shimmer on a wordmark is the kind of motion that reads as charming
   once and as a distraction every time after, on a page people leave open for
   half an hour while a GPU job runs.
   ───────────────────────────────────────────────────────────────────────── */

// 5-row block font, one entry per letter of PIXIECAD.
const GLYPHS: Record<string, string[]> = {
  P: ["██▄", "█▀▀", "▀  "],
  I: ["█", "█", "█"],
  X: ["▀▄", " █", "▄▀"],
  E: ["█▀", "█▀", "▀▀"],
  C: ["▄▀", "█ ", "▀▄"],
  A: ["▄▀▄", "█▀█", "▀ ▀"],
  D: ["█▄ ", "█ █", "▀▀ "],
};

const WORD = "PIXIECAD";
const ROWS = 3;
const SCRAMBLE = "▁▂▃▄▅▆▇█▀▄▌▐░▒▓";

/** Rows of the settled wordmark, built once at module load. */
const SETTLED: string[] = (() => {
  const rows = Array.from({ length: ROWS }, () => "");
  for (const ch of WORD) {
    const g = GLYPHS[ch];
    if (!g) continue;
    for (let r = 0; r < ROWS; r++) rows[r] += (g[r] ?? "") + " ";
  }
  return rows;
})();

/* Module-level, not state: the resolve is a page-load event, not a mount
   event. Navigating between routes remounts the header, and re-running the
   animation on every navigation would be noise. */
let hasResolved = false;

export function Wordmark({ className }: { className?: string }) {
  const preRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    const node = preRef.current;
    if (!node) return;

    const settled = SETTLED.join("\n");
    if (hasResolved || prefersReducedMotion()) {
      node.textContent = settled;
      hasResolved = true;
      return;
    }

    // Each cell locks at a staggered time, left to right with jitter, so the
    // word reads as resolving rather than fading in as a block.
    const cols = Math.max(...SETTLED.map((r) => r.length));
    const lockAt: number[][] = SETTLED.map((row) =>
      Array.from(
        { length: row.length },
        (_, c) => (c / cols) * 620 + Math.random() * 280,
      ),
    );

    let raf = 0;
    const start = performance.now();

    const step = (now: number) => {
      const t = now - start;
      let done = true;
      let out = "";
      for (let r = 0; r < SETTLED.length; r++) {
        const row = SETTLED[r] ?? "";
        const locks = lockAt[r] ?? [];
        for (let c = 0; c < row.length; c++) {
          const target = row[c] ?? " ";
          // Whitespace never scrambles — filling the gaps between letters with
          // noise turns the wordmark into a rectangle of static.
          if (target === " " || t >= (locks[c] ?? 0)) {
            out += target;
          } else {
            out += SCRAMBLE[(Math.random() * SCRAMBLE.length) | 0] ?? " ";
            done = false;
          }
        }
        if (r < SETTLED.length - 1) out += "\n";
      }
      node.textContent = out;

      if (done) {
        hasResolved = true;
        return; // settled: no rAF rescheduled, ever
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);

    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <span className={`inline-flex items-center ${className ?? ""}`}>
      {/* The art carries no information a screen reader can use; the name
          travels in the sr-only span beside it. */}
      <pre
        ref={preRef}
        aria-hidden="true"
        className="m-0 font-mono text-[6px] leading-[1.05] tracking-[0.04em] text-accent select-none"
      >
        {SETTLED.join("\n")}
      </pre>
      <span className="sr-only">PixieCAD</span>
    </span>
  );
}
