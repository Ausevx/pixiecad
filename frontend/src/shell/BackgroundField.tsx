import { useEffect, useRef } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   The background field: a drifting technical grid and point cloud.

   Budgeted deliberately. This is decoration on a 16 GB laptop that is also
   doing the user's real work, so it earns its place only by being close to
   free: point positions are allocated once into flat Float32Arrays, the whole
   frame is three batched paths, and the loop stops outright whenever nobody
   is looking at it or nothing is happening.
   ───────────────────────────────────────────────────────────────────────── */

const POINTS = 620;
const GRID_SPACING = 68;
/** Idle for this long with no pointer movement and no running job, and the
 *  loop shuts down entirely rather than drifting on forever. */
const IDLE_TIMEOUT_MS = 4000;
const MAX_DPR = 2;

export function BackgroundField({
  intensity = 0,
  className,
}: {
  intensity?: number;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // The live target, read by the loop without re-running the effect — an
  // intensity change must not tear down and rebuild the point cloud.
  const targetRef = useRef(intensity);
  targetRef.current = intensity;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const maybeCtx = canvas.getContext("2d", { alpha: true });
    if (!maybeCtx) return;
    // Aliased after the guard because `draw` is a hoisted function
    // declaration: TypeScript considers the closure created before the null
    // check ran, so the narrowing on the original binding does not reach it.
    const ctx = maybeCtx;

    // Flat parallel arrays rather than an array of objects: the render loop
    // must not chase pointers or allocate, and this is the cheap way to get
    // that in JS.
    const px = new Float32Array(POINTS);
    const py = new Float32Array(POINTS);
    const pz = new Float32Array(POINTS); // 0 = far, 1 = near
    const pv = new Float32Array(POINTS); // vertical drift velocity
    for (let i = 0; i < POINTS; i++) {
      px[i] = Math.random();
      py[i] = Math.random();
      pz[i] = Math.random();
      pv[i] = 0.004 + Math.random() * 0.012;
    }

    const css = getComputedStyle(document.documentElement);
    const read = (name: string, fallback: string) =>
      css.getPropertyValue(name).trim() || fallback;
    let ruleColor = read("--field-line", "#1c2231");
    let faintColor = read("--field-point", "#5b6577");
    let accentColor = read("--px-accent", "#ffb230");

    let width = 0;
    let height = 0;
    let dpr = 1;

    const resize = () => {
      dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    const reducedQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

    let pointerX = 0;
    let pointerY = 0;
    let smoothX = 0;
    let smoothY = 0;
    let current = targetRef.current;
    let drift = 0;
    let lastActivity = performance.now();
    let raf = 0;
    let running = false;

    function draw(now: number) {
      const reduced = reducedQuery.matches;

      // Ease toward the target rather than stepping, so a job starting does
      // not make the whole background snap.
      current += (targetRef.current - current) * 0.02;
      if (!reduced) drift += 0.00018 + current * 0.0007;

      const damp = reduced ? 0 : 0.06;
      smoothX += (pointerX - smoothX) * damp;
      smoothY += (pointerY - smoothY) * damp;

      ctx.clearRect(0, 0, width, height);

      // ── Grid: one path, one stroke.
      const gx = smoothX * 6;
      const gy = smoothY * 6;
      ctx.beginPath();
      const offset = ((drift * 140) % GRID_SPACING) - GRID_SPACING;
      for (let x = offset; x < width + GRID_SPACING; x += GRID_SPACING) {
        ctx.moveTo(Math.round(x + gx) + 0.5, 0);
        ctx.lineTo(Math.round(x + gx) + 0.5, height);
      }
      for (let y = offset; y < height + GRID_SPACING; y += GRID_SPACING) {
        ctx.moveTo(0, Math.round(y + gy) + 0.5);
        ctx.lineTo(width, Math.round(y + gy) + 0.5);
      }
      ctx.strokeStyle = ruleColor;
      // Kept low deliberately: this sits behind body text on every screen, and
      // a background that competes with the content it is behind is wrong no
      // matter how good it looks in isolation.
      ctx.globalAlpha = 0.3 + current * 0.2;
      ctx.lineWidth = 1;
      ctx.stroke();

      // ── Points, in two batched passes so fillStyle is set twice per frame
      // rather than 620 times. The accent pass is the small "hot" fraction,
      // and it only exists once something is actually running.
      const hotCutoff = 1 - current * 0.12;
      const parX = smoothX * 22;
      const parY = smoothY * 22;

      for (let pass = 0; pass < 2; pass++) {
        const hot = pass === 1;
        if (hot && current < 0.05) break;
        ctx.beginPath();
        for (let i = 0; i < POINTS; i++) {
          const isHot = pz[i]! > hotCutoff;
          if (isHot !== hot) continue;
          const depth = pz[i]!;
          // Wrap rather than respawn: no branching on lifetime, no allocation.
          let y = py[i]! + (reduced ? 0 : drift * pv[i]! * 90);
          y -= Math.floor(y);
          const sx = px[i]! * width + parX * depth;
          const sy = y * height + parY * depth;
          const r = 0.5 + depth * 1.1;
          ctx.moveTo(sx + r, sy);
          ctx.arc(sx, sy, r, 0, Math.PI * 2);
        }
        ctx.fillStyle = hot ? accentColor : faintColor;
        ctx.globalAlpha = hot ? 0.25 + current * 0.35 : 0.16 + current * 0.14;
        ctx.fill();
      }

      ctx.globalAlpha = 1;

      // Shut down when there is genuinely nothing to show: no running job and
      // no recent pointer movement. A background that animates forever is the
      // single easiest way to cost a laptop user battery for nothing.
      const idle =
        targetRef.current < 0.02 && now - lastActivity > IDLE_TIMEOUT_MS;
      if (idle || reducedQuery.matches) {
        running = false;
        return;
      }
      raf = requestAnimationFrame(draw);
    }

    function start() {
      if (running || document.visibilityState !== "visible") return;
      if (reducedQuery.matches) {
        // One static frame, and no loop at all — not a slower loop.
        draw(performance.now());
        return;
      }
      running = true;
      raf = requestAnimationFrame(draw);
    }

    function stop() {
      running = false;
      cancelAnimationFrame(raf);
    }

    const onPointer = (e: PointerEvent) => {
      if (reducedQuery.matches) return;
      pointerX = (e.clientX / window.innerWidth) * 2 - 1;
      pointerY = (e.clientY / window.innerHeight) * 2 - 1;
      lastActivity = performance.now();
      start();
    };

    const onVisibility = () => (document.visibilityState === "visible" ? start() : stop());

    let resizeTimer = 0;
    const observer = new ResizeObserver(() => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        resize();
        lastActivity = performance.now();
        start();
      }, 120);
    });
    observer.observe(canvas);

    // The canvas paints its own colours, so a theme change has to be pushed
    // into it — CSS alone cannot repaint pixels already drawn.
    const rereadPalette = () => {
      ruleColor = read("--field-line", "#1c2231");
      faintColor = read("--field-point", "#5b6577");
      accentColor = read("--px-accent", "#ffb230");
      lastActivity = performance.now();
      stop();
      start();
    };
    const themeObserver = new MutationObserver(rereadPalette);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    const schemeQuery = window.matchMedia("(prefers-color-scheme: light)");
    schemeQuery.addEventListener("change", rereadPalette);

    const onReducedChange = () => {
      stop();
      lastActivity = performance.now();
      start();
    };

    window.addEventListener("pointermove", onPointer, { passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    reducedQuery.addEventListener("change", onReducedChange);
    start();

    // Nudge the loop awake whenever intensity rises from idle, since the
    // effect itself does not re-run on an intensity change.
    const wake = window.setInterval(() => {
      if (targetRef.current > 0.02) {
        lastActivity = performance.now();
        start();
      }
    }, 2000);

    return () => {
      stop();
      window.clearInterval(wake);
      window.clearTimeout(resizeTimer);
      observer.disconnect();
      themeObserver.disconnect();
      schemeQuery.removeEventListener("change", rereadPalette);
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVisibility);
      reducedQuery.removeEventListener("change", onReducedChange);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className={`pointer-events-none fixed inset-0 z-0 h-full w-full ${className ?? ""}`}
    />
  );
}
