import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { PhotoDrop, type PhotoEntry } from "@/components/PhotoDrop";
import { createJob, getBackends, getStageCosts } from "@/lib/api";
import { useReducedMotion } from "@/lib/hooks";
import { snap } from "@/lib/motion";
import { refreshJobs } from "@/lib/jobs";
import { go, useDrawer } from "@/lib/router";
import { useVm } from "@/lib/vm";
import { toast } from "@/shell/toast";
import type {
  BackendOption,
  NewJobParams,
  StageCost,
  StageCosts,
  ViewTag,
} from "@/lib/types";

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

/** What this job will actually cost, split by where the work lands.
 *
 *  Local and host seconds are not interchangeable and are deliberately not
 *  summed into one number: host time is billed but costs this machine nothing,
 *  while local time and memory compete with whatever else the user is doing.
 *  Peak local RAM gets its own line because that is the figure that decides
 *  whether a background job is tolerable on a 16 GB laptop.
 */
function CostEstimate({
  bake,
  normalRes,
  bakeLocation,
  texture,
  semantic,
  hasHost,
}: {
  bake: boolean;
  normalRes: number;
  bakeLocation: string;
  texture: boolean;
  semantic: boolean;
  hasHost: boolean;
}) {
  const [costs, setCosts] = useState<StageCosts | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    getStageCosts(
      {
        bake,
        normal_res: normalRes,
        bake_location: bakeLocation,
        texture,
        semantic,
        has_host: hasHost,
      },
      ac.signal,
    )
      .then(setCosts)
      .catch(() => undefined);
    return () => ac.abort();
  }, [bake, normalRes, bakeLocation, texture, semantic, hasHost]);

  if (!costs) return null;

  const local = costs.stages.filter((s) => s.where === "local");
  const host = costs.stages.filter((s) => s.where === "host");
  const heavy = costs.local_peak_ram_mb >= 3000;

  const row = (s: StageCost) => (
    <div key={s.key} className="flex items-baseline justify-between gap-3 py-0.5">
      <span className="font-mono text-[11px] text-ink-dim">
        {s.label}
        {s.basis === "estimated" && (
          <span className="text-ink-faint"> (est.)</span>
        )}
      </span>
      <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-faint">
        {s.seconds === null ? "—" : `${s.seconds.toFixed(1)}s`}
        {s.where === "local" && s.peak_ram_mb ? ` · ${(s.peak_ram_mb / 1024).toFixed(1)} GB` : ""}
      </span>
    </div>
  );

  return (
    <div className="rounded-sharp border border-rule bg-panel/40 p-3">
      <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
        estimated cost
      </div>

      <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
        on this machine
      </div>
      {local.map(row)}
      <div className="mt-1 flex items-baseline justify-between gap-3 border-t border-rule pt-1">
        <span className="font-mono text-[11px] text-ink-dim">peak memory</span>
        <span
          className={`font-mono text-[11px] tabular-nums ${heavy ? "text-warn" : "text-ink-dim"}`}
        >
          {(costs.local_peak_ram_mb / 1024).toFixed(1)} GB
        </span>
      </div>

      {host.length > 0 && (
        <>
          <div className="mb-1 mt-3 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            on the GPU host (billed, free to this machine)
          </div>
          {host.map(row)}
        </>
      )}

      <p className="mt-2 font-mono text-[10px] leading-relaxed text-ink-faint">
        Measured on this project against a 1.3M-face mesh, except where marked
        (est.).
        {costs.has_unmeasured &&
          " Remote baking has no measured time yet — it has not run end to end."}
      </p>
    </div>
  );
}

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
          className="size-3.5 accent-accent"
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

/** Which generative model produces the geometry.
 *
 *  Defaults to automatic, which is what every job has actually used. The other
 *  entries are shown even when they cannot run, greyed and with the reason,
 *  because "why can't I pick TRELLIS" is a better question to be able to
 *  answer than to hide. Unvalidated backends say so plainly rather than
 *  letting someone spend a GPU hour discovering it. */
function BackendPicker({
  value,
  onChange,
  hasGpuHost,
}: {
  value: string;
  onChange: (key: string) => void;
  hasGpuHost: boolean;
}) {
  const [list, setList] = useState<BackendOption[]>([]);

  useEffect(() => {
    getBackends()
      .then((r) => setList(r.backends))
      .catch(() => undefined);
  }, []);

  const runnable = (b: BackendOption) =>
    b.requires === null || (b.requires === "gpu_host" && hasGpuHost);

  const selected = list.find((b) => b.key === value);

  if (list.length === 0) return null;

  return (
    <Field label="generative model">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={inputClass}
      >
        {list.map((b) => (
          <option key={b.key} value={b.key} disabled={!runnable(b)}>
            {b.label}
            {!runnable(b)
              ? " — needs a GPU host"
              : b.validated
                ? ""
                : " — unvalidated"}
          </option>
        ))}
      </select>
      {selected && (
        <>
          <span className="font-mono text-[10px] leading-relaxed text-ink-faint">
            {selected.note}
          </span>
          {!selected.validated && (
            <span className="font-mono text-[10px] leading-relaxed text-warn">
              ⚠ This backend has never produced a model on this project.
            </span>
          )}
        </>
      )}
    </Field>
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
      <p className="rounded-panel border border-ok-edge bg-ok-wash px-3 py-2 font-mono text-[11px] text-ok">
        ● GPU ready — this job will use {vm.host || "the running VM"}.
      </p>
    );
  }

  if (vm.presence === "building") {
    return (
      <p className="rounded-panel border border-warn-edge bg-warn-wash px-3 py-2 font-mono text-[11px] leading-relaxed text-warn">
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
  const [backend, setBackend] = useState("auto");
  const [split, setSplit] = useState(true);
  const [multiview, setMultiview] = useState(false);
  const [smooth, setSmooth] = useState(5);
  const [maxParts, setMaxParts] = useState(8);
  const [segmentation, setSegmentation] = useState<"auto" | "semantic">("auto");
  const [texture, setTexture] = useState(false);
  const [webExport, setWebExport] = useState(true);
  const [textureSize, setTextureSize] = useState(1024);
  const [bakeNormals, setBakeNormals] = useState(true);
  const [normalRes, setNormalRes] = useState(1024);
  const [bakeLocation, setBakeLocation] =
    useState<"auto" | "local" | "host">("auto");

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
      backend,
      // The host the VM store already knows about, so the user never types an
      // SSH alias by hand. Blank when there is none, which is exactly what the
      // server expects for a local run.
      gpu_host: vm.host,
      view_tags,
      bake_normals: bakeNormals,
      normal_res: normalRes,
      bake_location: bakeLocation,
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

          <BackendPicker
            value={backend}
            onChange={setBackend}
            hasGpuHost={Boolean(vm.host)}
          />

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
                checked={bakeNormals}
                onChange={setBakeNormals}
                label="bake normal map"
                hint="Records detail lost to decimation as a texture, so 20k triangles are lit like 1.3M. A rendering trick — it does nothing for 3D printing, which follows geometry."
              />

              {bakeNormals && (
                <Field
                  label="bake on"
                  hint={
                    bakeLocation === "local" || !vm.host
                      ? "This machine. The bake is the heaviest local stage by a wide margin."
                      : "The GPU host — which costs this machine only the ~12 MB upload. It is remote CPU, not GPU acceleration; the bake does no GPU work anywhere."
                  }
                >
                  <select
                    value={bakeLocation}
                    onChange={(e) =>
                      setBakeLocation(e.target.value as "auto" | "local" | "host")
                    }
                    className={inputClass}
                  >
                    <option value="auto">
                      auto — host when one is running{vm.host ? "" : " (none now)"}
                    </option>
                    <option value="local">this machine</option>
                    <option value="host" disabled={!vm.host}>
                      GPU host{vm.host ? "" : " — needs a host"}
                    </option>
                  </select>
                </Field>
              )}

              {bakeNormals && (
                <Field
                  label="normal map resolution"
                  hint="Higher is sharper. The cost of each choice is in the estimate below."
                >
                  <select
                    value={normalRes}
                    onChange={(e) => setNormalRes(Number(e.target.value))}
                    className={inputClass}
                  >
                    <option value={512}>512 — draft</option>
                    <option value={1024}>1024 — recommended</option>
                    <option value={2048}>2048 — full quality</option>
                  </select>
                </Field>
              )}

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

          <CostEstimate
            bake={bakeNormals}
            normalRes={normalRes}
            bakeLocation={bakeLocation}
            texture={texture}
            semantic={segmentation === "semantic"}
            hasHost={Boolean(vm.host)}
          />

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
