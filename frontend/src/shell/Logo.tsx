import { useEffect, useRef } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   The mark: a real 3D object, not a picture of one.

   A wireframe solid inside its bounding box — the way a part looks in a CAD
   viewport before anyone has shaded it — projected in canvas 2D. It idles
   with a slow orbit and swings to face the pointer, so it reads as an object
   you are looking AT rather than a logo sitting on top of the page.

   Canvas 2D rather than WebGL on purpose: this is 20 edges. A WebGL context
   for that would cost more memory than the entire rest of the shell, and it
   would be a second context competing with the model viewer.
   ───────────────────────────────────────────────────────────────────────── */

type V3 = readonly [number, number, number];

// A cube with its corners cut — an octahedron-truncated form that stays
// legible at 28px, where a plain cube reads as a flat hexagon from most
// angles and an icosahedron turns to mush.
const S = 0.62;
const CORE: V3[] = [
  [-S, -S, S], [S, -S, S], [S, S, S], [-S, S, S],
  [-S, -S, -S], [S, -S, -S], [S, S, -S], [-S, S, -S],
];
const CORE_EDGES: readonly (readonly [number, number])[] = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

// The bounding box: the CAD tell. Drawn fainter and slightly larger.
const B = 1.0;
const BOX: V3[] = [
  [-B, -B, B], [B, -B, B], [B, B, B], [-B, B, B],
  [-B, -B, -B], [B, -B, -B], [B, B, -B], [-B, B, -B],
];

/** Diagonals through the solid — reads as construction geometry and gives the
 *  shape enough interior structure to show its rotation at small sizes. */
const DIAGONALS: readonly (readonly [number, number])[] = [
  [0, 6], [1, 7], [2, 4], [3, 5],
];

export function Logo({ className }: { className?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maybe = canvas.getContext("2d");
    if (!maybe) return;
    const ctx = maybe;

    // Scratch buffers, allocated once. The draw loop must not allocate.
    const px = new Float32Array(8);
    const py = new Float32Array(8);
    const bx = new Float32Array(8);
    const by = new Float32Array(8);
    const bd = new Float32Array(8);
    const pd = new Float32Array(8);

    let size = 0;
    let dpr = 1;

    const resize = () => {
      dpr = Math.min(2, window.devicePixelRatio || 1);
      size = canvas.clientWidth;
      canvas.width = Math.round(size * dpr);
      canvas.height = Math.round(size * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    const css = getComputedStyle(document.documentElement);
    let inkColor = "";
    let accentColor = "";
    let ruleColor = "";
    const readColors = () => {
      inkColor = css.getPropertyValue("--px-ink").trim() || "#e8ecf4";
      accentColor = css.getPropertyValue("--px-accent").trim() || "#ffb230";
      ruleColor = css.getPropertyValue("--px-rule-bright").trim() || "#2a3244";
    };
    readColors();

    let yaw = 0.6;
    let pitch = -0.35;
    let targetYaw = 0.6;
    let targetPitch = -0.35;
    let hover = 0;
    let targetHover = 0;
    let raf = 0;
    let running = false;

    const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function project(
      src: readonly V3[],
      outX: Float32Array,
      outY: Float32Array,
      outD: Float32Array,
      scale: number,
    ) {
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cp = Math.cos(pitch);
      const sp = Math.sin(pitch);
      const half = size / 2;

      for (let i = 0; i < src.length; i++) {
        const v = src[i]!;
        const x0 = v[0];
        const y0 = v[1];
        const z0 = v[2];
        // Yaw about Y, then pitch about X.
        const x1 = x0 * cy + z0 * sy;
        const z1 = -x0 * sy + z0 * cy;
        const y1 = y0 * cp - z1 * sp;
        const z2 = y0 * sp + z1 * cp;
        // Weak perspective: enough to sell depth, not enough to distort a
        // shape this small into something unrecognisable.
        const k = 1 / (1 + z2 * 0.22);
        outX[i] = half + x1 * scale * k;
        outY[i] = half + y1 * scale * k;
        outD[i] = z2;
      }
    }

    function strokeEdges(
      edges: readonly (readonly [number, number])[],
      xs: Float32Array,
      ys: Float32Array,
      ds: Float32Array,
      color: string,
      baseAlpha: number,
      width: number,
    ) {
      // Two passes — back half then front half — so the far edges read as
      // behind rather than the whole frame flattening into one tangle.
      for (let pass = 0; pass < 2; pass++) {
        ctx.beginPath();
        for (const e of edges) {
          const a = e[0];
          const b = e[1];
          const behind = (ds[a]! + ds[b]!) * 0.5 > 0;
          if (behind !== (pass === 0)) continue;
          ctx.moveTo(xs[a]!, ys[a]!);
          ctx.lineTo(xs[b]!, ys[b]!);
        }
        ctx.strokeStyle = color;
        ctx.globalAlpha = pass === 0 ? baseAlpha * 0.35 : baseAlpha;
        ctx.lineWidth = width;
        ctx.stroke();
      }
    }

    function draw() {
      const still = reduced();

      if (!still) {
        // Idle orbit, slowed to a crawl while the pointer is driving.
        targetYaw += 0.0022 * (1 - hover * 0.85);
        hover += (targetHover - hover) * 0.08;
        yaw += (targetYaw - yaw) * 0.08;
        pitch += (targetPitch - pitch) * 0.08;
      }

      ctx.clearRect(0, 0, size, size);

      const scale = size * 0.3;
      project(BOX, bx, by, bd, scale * 1.02);
      project(CORE, px, py, pd, scale);

      strokeEdges(CORE_EDGES, bx, by, bd, ruleColor, 0.85, 1);
      strokeEdges(DIAGONALS, px, py, pd, accentColor, 0.2 + hover * 0.5, 1);
      strokeEdges(CORE_EDGES, px, py, pd, hover > 0.5 ? accentColor : inkColor, 0.95, 1.4);

      // Vertex dots — the thing that makes it read as a CAD entity rather
      // than a doodle. One path, one fill.
      ctx.beginPath();
      for (let i = 0; i < 8; i++) {
        const r = pd[i]! > 0 ? 0.9 : 1.5;
        ctx.moveTo(px[i]! + r, py[i]!);
        ctx.arc(px[i]!, py[i]!, r, 0, Math.PI * 2);
      }
      ctx.fillStyle = accentColor;
      ctx.globalAlpha = 0.5 + hover * 0.5;
      ctx.fill();
      ctx.globalAlpha = 1;

      if (still) {
        running = false;
        return; // one frame, no loop
      }
      raf = requestAnimationFrame(draw);
    }

    function start() {
      if (running || document.visibilityState !== "visible") return;
      if (reduced()) {
        draw();
        return;
      }
      running = true;
      raf = requestAnimationFrame(draw);
    }

    function stop() {
      running = false;
      cancelAnimationFrame(raf);
    }

    // Pointer anywhere in the header region steers it; the mark is small, so
    // requiring a direct hover would make it feel dead.
    const onPointer = (e: PointerEvent) => {
      if (reduced()) return;
      const r = canvas.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const dx = (e.clientX - cx) / 260;
      const dy = (e.clientY - cy) / 260;
      const near = Math.abs(dx) < 1.6 && Math.abs(dy) < 1.6;
      targetHover = near ? 1 : 0;
      if (near) {
        targetYaw = 0.6 + dx * 1.1;
        targetPitch = -0.35 + dy * 0.8;
      } else {
        targetPitch = -0.35;
      }
      start();
    };

    const onVisibility = () =>
      document.visibilityState === "visible" ? start() : stop();

    // A theme change repaints the page but not this canvas, so the colours
    // must be re-read or the mark keeps the old theme's ink.
    const themeObserver = new MutationObserver(() => {
      readColors();
      start();
    });
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    const mqLight = window.matchMedia("(prefers-color-scheme: light)");
    const onScheme = () => {
      readColors();
      start();
    };
    mqLight.addEventListener("change", onScheme);

    const ro = new ResizeObserver(() => {
      resize();
      start();
    });
    ro.observe(canvas);

    window.addEventListener("pointermove", onPointer, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    start();

    return () => {
      stop();
      ro.disconnect();
      themeObserver.disconnect();
      mqLight.removeEventListener("change", onScheme);
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return (
    <span className={`inline-flex items-center gap-2.5 ${className ?? ""}`}>
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="size-8 shrink-0"
        style={{ width: 32, height: 32 }}
      />
      <span className="font-mono text-[13px] font-semibold tracking-[0.18em] text-ink">
        PIXIECAD
      </span>
      {/* The canvas carries no information; the name beside it is real text,
          so nothing here needs an alternative. */}
    </span>
  );
}
