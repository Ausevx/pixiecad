import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { convertJob, formatBytes, formatCount } from "@/lib/api";
import { useReducedMotion } from "@/lib/hooks";
import { listItemVariants, listVariants, snap } from "@/lib/motion";
import { toast } from "@/shell/toast";
import type { JobDetail, JobFile, JobFiles, JobModel } from "@/lib/types";

/* ─────────────────────────────────────────────────────────────────────────
   The artifacts panel — the top of the job screen, not the bottom of it.

   The old dashboard buried finished output below a twenty-two-control form,
   and getting an STL out of a finished job took five clicks: select the job,
   open a format dropdown, pick STL, press Convert, then click the result.
   Here the primary artifact is one click, and if it does not exist yet that
   one click converts and downloads it as a single action.
   ───────────────────────────────────────────────────────────────────────── */

const CAD_FORMATS = ["stl", "obj", "ply"] as const;
type CadFormat = (typeof CAD_FORMATS)[number];

/** What the user came for, in order of preference. STL first: this is a tool
 *  for getting machinable geometry into a slicer or a CAD package, and that
 *  is what STL is for. GLB is the fallback because it is what the pipeline
 *  always produces.
 *
 *  Anchored on the CURRENT model rather than on whatever STL happens to be in
 *  the directory: once a job has been re-optimised there are several models
 *  and possibly several STLs, and offering the one belonging to superseded
 *  geometry is exactly the confusion this panel has to avoid. */
function pickPrimary(files: JobFile[], current: JobModel | null): JobFile | null {
  const byName = (n: string) =>
    files.find((f) => f.name.toLowerCase() === n.toLowerCase());
  return (
    (current ? byName(`${current.basename}.stl`) : null) ??
    (current ? byName(current.name) : null) ??
    byName("model.stl") ??
    files.find((f) => f.name.toLowerCase().endsWith(".stl")) ??
    byName("model.glb") ??
    files.find((f) => f.name.toLowerCase().endsWith(".glb")) ??
    files[0] ??
    null
  );
}

/** One model, as a download.
 *
 *  Every re-optimise adds a row here instead of replacing the one above it:
 *  the point of trying a smaller budget is being able to compare it with the
 *  one you had, and to go back if it is worse. */
function ModelRow({ model }: { model: JobModel }) {
  const reduced = useReducedMotion();
  return (
    <motion.li variants={listItemVariants} layout>
      <motion.a
        href={model.url}
        download
        whileHover={reduced ? undefined : { x: 2 }}
        transition={snap}
        title={
          model.original
            ? "The model this job's build produced"
            : "Produced by re-optimising, at this polygon budget"
        }
        className={`group flex items-center gap-2 rounded-sharp px-2 py-1.5 hover:bg-hover ${
          model.current ? "bg-accent-wash" : ""
        }`}
      >
        <span
          aria-hidden="true"
          className={`font-mono text-[11px] ${
            model.current ? "text-accent" : "text-ink-faint group-hover:text-accent"
          }`}
        >
          ⤓
        </span>
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-dim group-hover:text-ink">
          {model.faces != null ? `${formatCount(model.faces)} tris` : model.name}
        </span>
        {model.current && (
          <span className="shrink-0 font-mono text-[9px] uppercase tracking-widest text-accent">
            shown
          </span>
        )}
        {model.original && (
          <span className="shrink-0 font-mono text-[9px] uppercase tracking-widest text-ink-faint">
            built
          </span>
        )}
        <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-faint">
          {formatBytes(model.bytes)}
        </span>
      </motion.a>
    </motion.li>
  );
}

function DownloadRow({ file }: { file: JobFile }) {
  const reduced = useReducedMotion();
  return (
    <motion.li variants={listItemVariants} layout>
      {/* One click. The row IS the link — there is no expand-then-download
          step, which is the interaction this panel exists to delete. */}
      <motion.a
        href={file.url}
        download
        whileHover={reduced ? undefined : { x: 2 }}
        transition={snap}
        className="group flex items-center gap-3 rounded-sharp px-2 py-1.5 hover:bg-hover"
      >
        <span aria-hidden="true" className="font-mono text-[11px] text-ink-faint group-hover:text-accent">
          ⤓
        </span>
        <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink-dim group-hover:text-ink">
          {file.name}
        </span>
        <span className="font-mono text-[11px] tabular-nums text-ink-faint">
          {formatBytes(file.bytes)}
        </span>
      </motion.a>
    </motion.li>
  );
}

export function Artifacts({
  job,
  files,
  onChanged,
}: {
  job: JobDetail;
  files: JobFiles | null;
  onChanged: () => void;
}) {
  const reduced = useReducedMotion();
  const [busy, setBusy] = useState(false);
  const [showOther, setShowOther] = useState(false);
  const [sizeMm, setSizeMm] = useState("");
  const [format, setFormat] = useState<CadFormat>("stl");
  const [repair, setRepair] = useState(false);

  const list = files?.files ?? [];
  const models = job.models ?? [];
  const current = models.find((m) => m.current) ?? null;
  const primary = pickPrimary(list, current);
  // Whether an STL of the CURRENT model exists. Asking "is there any STL"
  // meant that after a re-optimise the button offered the previous model's
  // STL as though it were this one's.
  const hasStl = current
    ? list.some((f) => f.name.toLowerCase() === `${current.basename}.stl`.toLowerCase())
    : list.some((f) => f.name.toLowerCase().endsWith(".stl"));
  const modelNames = new Set(models.map((m) => m.name));
  const secondary = list.filter(
    (f) => f.name !== primary?.name && !modelNames.has(f.name),
  );

  /** Convert, then download, as one action. The old flow made the user press
   *  Convert, wait, then find and click the produced link; the conversion is
   *  an implementation detail of "give me an STL". */
  async function convertAndDownload(fmt: CadFormat, sizeInput: string, doRepair = repair) {
    setBusy(true);
    try {
      const mm = Number.parseFloat(sizeInput);
      const reqBody: { format: string; source?: string; size_mm?: number | null; repair?: boolean } = {
        format: fmt,
        size_mm: Number.isFinite(mm) ? mm : null,
        // Convert what is on screen. Left unset the server picks the current
        // model too, but being explicit means the request cannot drift from
        // what this panel is describing.
        source: current?.name,
      };
      if (doRepair) {
        reqBody.repair = true;
      }
      const res = await convertJob(job.job_id, reqBody);
      // Triggering the download rather than only rendering a link is what
      // makes this one click instead of two.
      const a = document.createElement("a");
      a.href = res.url;
      a.download = res.file;
      document.body.appendChild(a);
      a.click();
      a.remove();

      toast(
        res.printable ? "ok" : "warn",
        `${res.file} · ${formatBytes(res.bytes)}`,
        res.printable
          ? "Closed solid, slicer-ready."
          : `Not a closed solid. ${res.solid}`,
      );
      onChanged();
    } catch (e) {
      toast("fail", "Conversion failed", e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (job.status !== "done") return null;

  return (
    <section aria-labelledby="artifacts-heading" className="rounded-panel border border-rule bg-panel/70 p-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2
          id="artifacts-heading"
          className="font-mono text-[11px] uppercase tracking-widest text-ink-dim"
        >
          Artifacts
        </h2>
        {files && (
          <span className="font-mono text-[11px] tabular-nums text-ink-faint">
            {list.length} files · {formatBytes(files.total_bytes)}
          </span>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-[minmax(0,20rem)_1fr]">
        {/* ── The dominant action. */}
        <div>
          {hasStl && primary ? (
            <motion.a
              href={primary.url}
              download
              whileHover={reduced ? undefined : { scale: 1.02 }}
              whileTap={reduced ? undefined : { scale: 0.98 }}
              transition={snap}
              className="flex flex-col gap-1 rounded-panel border border-accent-dim bg-accent-wash px-4 py-3 text-accent hover:bg-accent hover:text-void"
            >
              <span className="font-mono text-[13px] uppercase tracking-widest">
                ⤓ download {primary.name.split(".").pop()}
              </span>
              <span className="font-mono text-[11px] tabular-nums opacity-80">
                {primary.name} · {formatBytes(primary.bytes)}
              </span>
            </motion.a>
          ) : (
            <motion.button
              type="button"
              disabled={busy}
              onClick={() => void convertAndDownload("stl", sizeMm, repair)}
              whileHover={reduced || busy ? undefined : { scale: 1.02 }}
              whileTap={reduced || busy ? undefined : { scale: 0.98 }}
              transition={snap}
              className="flex w-full flex-col gap-1 rounded-panel border border-accent-dim bg-accent-wash px-4 py-3 text-left text-accent hover:bg-accent hover:text-void disabled:opacity-50"
            >
              <span className="font-mono text-[13px] uppercase tracking-widest">
                {busy ? "converting…" : "⤓ download stl"}
              </span>
              <span className="font-mono text-[11px] opacity-80">
                converts from {current?.name ?? primary?.name ?? "model.glb"} on the way
              </span>
            </motion.button>
          )}

          <button
            type="button"
            onClick={() => setShowOther((v) => !v)}
            aria-expanded={showOther}
            aria-controls="other-formats"
            className="mt-2 font-mono text-[11px] text-ink-faint hover:text-accent"
          >
            {showOther ? "▾" : "▸"} other formats
          </button>

          <AnimatePresence initial={false}>
            {showOther && (
              <motion.div
                id="other-formats"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-2 flex flex-wrap items-end gap-2">
                  <label className="flex flex-col gap-1">
                    <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">
                      format
                    </span>
                    <select
                      value={format}
                      onChange={(e) => setFormat(e.target.value as CadFormat)}
                      className="rounded-sharp border border-rule bg-void px-2 py-1 font-mono text-[12px] text-ink"
                    >
                      {CAD_FORMATS.map((f) => (
                        <option key={f} value={f}>
                          {f.toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </label>

                  <label className="flex flex-col gap-1">
                    <span className="font-mono text-[10px] uppercase tracking-widest text-ink-faint">
                      longest edge (mm)
                    </span>
                    <input
                      type="number"
                      min={1}
                      value={sizeMm}
                      onChange={(e) => setSizeMm(e.target.value)}
                      placeholder="auto"
                      className="w-28 rounded-sharp border border-rule bg-void px-2 py-1 font-mono text-[12px] tabular-nums text-ink"
                    />
                  </label>

                  {/* Trimesh mesh repair qualifier: fills holes, fixes inverted normals,
                      and closes non-manifold edges during format conversion. */}
                  <label className="flex items-center gap-1.5 pb-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={repair}
                      onChange={(e) => setRepair(e.target.checked)}
                      className="size-3.5 accent-accent"
                    />
                    <span className="font-mono text-[11px] text-ink-dim">
                      repair mesh
                    </span>
                  </label>

                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => void convertAndDownload(format, sizeMm, repair)}
                    className="rounded-sharp border border-rule-bright px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:border-accent-dim hover:text-accent disabled:opacity-50"
                  >
                    {busy ? "…" : "convert & download"}
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* ── The models, then everything else, one click each. */}
        <div className="space-y-3">
          {models.length > 0 && (
            <div>
              <h3 className="mb-1 font-mono text-[10px] uppercase tracking-widest text-ink-faint">
                {models.length > 1
                  ? `Models · ${models.length}`
                  : "Model"}
              </h3>
              <motion.ul
                variants={listVariants}
                initial="initial"
                animate="animate"
                className="max-h-40 list-none overflow-y-auto rounded-sharp border border-rule"
              >
                {models.map((m) => (
                  <ModelRow key={m.name} model={m} />
                ))}
              </motion.ul>
              {models.length > 1 && (
                <p className="mt-1 font-mono text-[10px] leading-relaxed text-ink-faint">
                  Re-optimising adds a model; it never replaces one.
                </p>
              )}
            </div>
          )}

          {secondary.length === 0 ? (
            <p className="font-mono text-[11px] text-ink-faint">No other files.</p>
          ) : (
            <motion.ul
              variants={listVariants}
              initial="initial"
              animate="animate"
              className="max-h-56 list-none overflow-y-auto"
            >
              {secondary.map((f) => (
                <DownloadRow key={f.name} file={f} />
              ))}
            </motion.ul>
          )}
        </div>
      </div>
    </section>
  );
}
