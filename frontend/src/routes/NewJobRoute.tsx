import { motion } from "motion/react";
import { useState } from "react";
import { PhotoDrop, type PhotoEntry } from "@/components/PhotoDrop";
import { createJob } from "@/lib/api";
import { useReducedMotion } from "@/lib/hooks";
import { snap } from "@/lib/motion";
import { refreshJobs } from "@/lib/jobs";
import { go, useDrawer } from "@/lib/router";
import { useVm } from "@/lib/vm";
import { toast } from "@/shell/toast";
import type { NewJobParams, ViewTag } from "@/lib/types";

/* ─────────────────────────────────────────────────────────────────────────
   New job — one screen, one intent.

   Everything about compute has been removed from this form. It asks one
   question about infrastructure, and only when the answer matters: does GPU
   capacity exist, and would you like some. Choosing hardware, watching an
   image build, and paying for it all live in the compute drawer.
   ───────────────────────────────────────────────────────────────────────── */

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">
        {label}
      </span>
      {children}
      {hint && (
        <span className="font-mono text-[10px] leading-relaxed text-ink-faint">{hint}</span>
      )}
    </label>
  );
}

const inputClass =
  "rounded-sharp border border-rule bg-void px-2 py-1.5 font-mono text-[12px] text-ink placeholder:text-ink-faint";

function Check({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: string;
  hint?: string;
}) {
  return (
    <div>
      <label className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="size-3.5 accent-[#ffb230]"
        />
        <span className="font-mono text-[12px] text-ink-dim">{label}</span>
      </label>
      {hint && (
        <p className="mt-0.5 pl-[1.375rem] font-mono text-[10px] leading-relaxed text-ink-faint">
          {hint}
        </p>
      )}
    </div>
  );
}

/** The only compute question this screen asks. It reports capacity and offers
 *  to get some, inline, without leaving the form. */
function CapacityStrip({ needsGpu }: { needsGpu: boolean }) {
  const vm = useVm();
  const drawer = useDrawer();

  if (!needsGpu) return null;

  if (vm.presence === "ready") {
    return (
      <p className="rounded-panel border border-[#17452b] bg-[#0a1d12] px-3 py-2 font-mono text-[11px] text-ok">
        ● GPU ready — this job will use {vm.host || "the running VM"}.
      </p>
    );
  }

  if (vm.presence === "building") {
    return (
      <p className="rounded-panel border border-[#4a3a10] bg-[#231a06] px-3 py-2 font-mono text-[11px] leading-relaxed text-warn">
        ◐ A GPU VM is still building. You can start this job anyway — it will
        wait for the VM instead of failing.
      </p>
    );
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-panel border border-rule bg-panel/70 px-3 py-2">
      <p className="font-mono text-[11px] leading-relaxed text-ink-dim">
        ○ No GPU capacity. Texturing and semantic segmentation need one.
      </p>
      <button
        type="button"
        onClick={() => drawer.setOpen(true)}
        className="shrink-0 rounded-sharp border border-accent-dim bg-accent-wash px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-accent hover:bg-accent hover:text-void"
      >
        provision one →
      </button>
    </div>
  );
}

export function NewJobRoute() {
  const reduced = useReducedMotion();
  const vm = useVm();
  const [photos, setPhotos] = useState<PhotoEntry[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [name, setName] = useState("my_model");
  const [targetFaces, setTargetFaces] = useState(20000);
  const [length, setLength] = useState("");
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [hint, setHint] = useState("");
  const [octree, setOctree] = useState(256);
  const [split, setSplit] = useState(true);
  const [multiview, setMultiview] = useState(false);
  const [smooth, setSmooth] = useState(5);
  const [maxParts, setMaxParts] = useState(8);
  const [segmentation, setSegmentation] = useState<"auto" | "semantic">("auto");
  const [texture, setTexture] = useState(false);
  const [webExport, setWebExport] = useState(true);
  const [textureSize, setTextureSize] = useState(1024);

  const taggedCount = photos.filter((p) => p.tag).length;
  const needsGpu = texture || segmentation === "semantic";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (photos.length === 0) {
      toast("warn", "No photos", "Add at least one photo to start a job.");
      return;
    }
    setBusy(true);
    setProgress(0);

    const view_tags: Record<string, ViewTag> = {};
    for (const p of photos) if (p.tag) view_tags[p.file.name] = p.tag;

    const params: NewJobParams = {
      name: name.trim() || "object",
      target_faces: targetFaces,
      length: length.trim(),
      width: width.trim(),
      height: height.trim(),
      split,
      object_hint: hint.trim(),
      smooth_iterations: smooth,
      texture,
      segmentation,
      web_export: webExport,
      texture_size: textureSize,
      max_parts: maxParts,
      octree_resolution: octree,
      multiview,
      // The host the VM store already knows about, so the user never types an
      // SSH alias by hand. Blank when there is none, which is exactly what the
      // server expects for a local run.
      gpu_host: vm.host,
      view_tags,
    };

    try {
      const res = await createJob(
        photos.map((p) => p.file),
        params,
        setProgress,
      );
      refreshJobs();
      go({ kind: "job", id: res.job_id });
    } catch (err) {
      toast("fail", "Could not start the job", err instanceof Error ? err.message : "");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto max-w-[1400px] px-5 py-6">
      <h1 className="font-mono text-[12px] uppercase tracking-widest text-ink-dim">
        New job
      </h1>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_22rem]">
        <div className="space-y-4">
          <PhotoDrop photos={photos} onChange={setPhotos} />

          {multiview && taggedCount < 2 && (
            <p className="font-mono text-[11px] leading-relaxed text-warn">
              Multi-view needs at least two tagged views; with fewer, the
              backend silently falls back to a single view.
            </p>
          )}
        </div>

        <div className="space-y-4">
          <Field label="object name">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className={inputClass}
            />
          </Field>

          <Field
            label="target triangles"
            hint="Simplification budget. Independent of generation detail."
          >
            <input
              type="number"
              min={100}
              max={200000}
              value={targetFaces}
              onChange={(e) => setTargetFaces(Number(e.target.value))}
              className={`${inputClass} tabular-nums`}
            />
          </Field>

          <div className="grid grid-cols-3 gap-2">
            {(
              [
                // Distinct examples, because three identical placeholders read
                // as one repeated field rather than three different measurements.
                ["length", "4.5m", length, setLength],
                ["width", "2m", width, setWidth],
                ["height", "1.2m", height, setHeight],
              ] as const
            ).map(([label, example, value, set]) => (
              <Field key={label} label={label}>
                <input
                  value={value}
                  onChange={(e) => set(e.target.value)}
                  placeholder={example}
                  className={inputClass}
                />
              </Field>
            ))}
          </div>

          <Field label="what is it?">
            <input
              value={hint}
              onChange={(e) => setHint(e.target.value)}
              placeholder="chair, mug, engine part"
              className={inputClass}
            />
          </Field>

          <Field
            label="generation detail"
            hint="Raise this when parts of the object merge together. Independent of the polygon budget."
          >
            <select
              value={octree}
              onChange={(e) => setOctree(Number(e.target.value))}
              className={inputClass}
            >
              <option value={192}>draft — fastest</option>
              <option value={256}>standard</option>
              <option value={384}>high — separates close parts better</option>
              <option value={512}>maximum — slowest, most VRAM</option>
            </select>
          </Field>

          <Check checked={split} onChange={setSplit} label="split into parts" />
          <Check
            checked={texture}
            onChange={setTexture}
            label="generate texture"
            hint="Needs a GPU host, ~2 min."
          />
          <Check
            checked={multiview}
            onChange={setMultiview}
            label="use multiple views"
            hint="Measured worse than single-view on a car: the multi-view checkpoint uses an older architecture. Try both."
          />

          <CapacityStrip needsGpu={needsGpu} />

          <button
            type="button"
            onClick={() => setShowAdvanced((v) => !v)}
            aria-expanded={showAdvanced}
            className="font-mono text-[11px] text-ink-faint hover:text-accent"
          >
            {showAdvanced ? "▾" : "▸"} finishing options
          </button>

          {showAdvanced && (
            <div className="space-y-4 border-l border-rule pl-3">
              <Field
                label="smoothing passes"
                hint="Taubin smoothing. Face count is unchanged, so this is free in your budget."
              >
                <input
                  type="number"
                  min={0}
                  max={20}
                  value={smooth}
                  onChange={(e) => setSmooth(Number(e.target.value))}
                  className={`${inputClass} tabular-nums`}
                />
              </Field>

              <Field label="max parts">
                <input
                  type="number"
                  min={1}
                  max={32}
                  value={maxParts}
                  onChange={(e) => setMaxParts(Number(e.target.value))}
                  className={`${inputClass} tabular-nums`}
                />
              </Field>

              <Field
                label="part segmentation"
                hint="Generative meshes are one fused shell; geometric splitting falls back to spatial blobs on them."
              >
                <select
                  value={segmentation}
                  onChange={(e) =>
                    setSegmentation(e.target.value as "auto" | "semantic")
                  }
                  className={inputClass}
                >
                  <option value="auto">geometric — local, instant</option>
                  <option value="semantic">semantic via SAM — needs GPU, ~2 min</option>
                </select>
              </Field>

              <Check
                checked={webExport}
                onChange={setWebExport}
                label="also export a web-sized copy"
              />

              <Field
                label="web texture size"
                hint="Texture, not polygons, dominates download size: 2048 gives 7.7 MB, 1024 gives 1.8 MB."
              >
                <select
                  value={textureSize}
                  onChange={(e) => setTextureSize(Number(e.target.value))}
                  className={inputClass}
                >
                  <option value={2048}>2048 — full quality</option>
                  <option value={1024}>1024 — recommended</option>
                  <option value={512}>512 — small</option>
                  <option value={256}>256 — thumbnail</option>
                </select>
              </Field>
            </div>
          )}

          <motion.button
            type="submit"
            disabled={busy || photos.length === 0}
            whileHover={reduced || busy ? undefined : { scale: 1.02 }}
            whileTap={reduced || busy ? undefined : { scale: 0.98 }}
            transition={snap}
            className="w-full rounded-panel border border-accent-dim bg-accent-wash px-3 py-2.5 font-mono text-[12px] uppercase tracking-widest text-accent hover:bg-accent hover:text-void disabled:opacity-40"
          >
            {busy
              ? progress < 1
                ? `uploading ${Math.round(progress * 100)}%`
                : "starting…"
              : "start pipeline"}
          </motion.button>

          {busy && (
            <div
              role="progressbar"
              aria-valuenow={Math.round(progress * 100)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Upload progress"
              className="h-0.5 w-full overflow-hidden bg-rule"
            >
              <motion.div
                className="h-full bg-accent"
                animate={{ scaleX: progress }}
                style={{ transformOrigin: "left" }}
                transition={snap}
              />
            </div>
          )}
        </div>
      </div>
    </form>
  );
}
