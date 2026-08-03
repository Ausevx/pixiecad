import { AnimatePresence, motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  formatSeconds,
  getGpuOptions,
  getCloud,
  getInventory,
  getProvisionState,
  provision,
  teardown,
} from "@/lib/api";
import { usePoll, useReducedMotion } from "@/lib/hooks";
import { drawerVariants, fade, settle, signatures, snap } from "@/lib/motion";
import { expectTeardown, refreshVm, useVm } from "@/lib/vm";
import { toast } from "./toast";
import type {
  CloudInventory,
  CloudSnapshot,
  GpuOption,
  ProvisionState,
} from "@/lib/types";

/* ─────────────────────────────────────────────────────────────────────────
   Compute.

   Provisioning, hardware choice, billing inventory and teardown used to live
   inside the new-job form, which said that choosing a GPU is part of "make a
   model from these photos". It is not — it is infrastructure, it outlives any
   one job, and it costs money whether or not a job is running.

   So it lives here: one click from the header pill on every route, in a
   drawer whose open state is in the URL, and never in the way of the work.
   ───────────────────────────────────────────────────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-rule px-5 py-4">
      <h3 className="mb-3 font-mono text-[10px] uppercase tracking-widest text-ink-faint">
        {title}
      </h3>
      {children}
    </section>
  );
}

function GpuChooser({
  options,
  value,
  onChange,
}: {
  options: GpuOption[];
  value: string;
  onChange: (key: string) => void;
}) {
  const selected = options.find((o) => o.key === value);
  return (
    <>
      <label className="flex flex-col gap-1">
        <span className="sr-only">GPU type</span>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-sharp border border-rule bg-void px-2 py-1.5 font-mono text-[12px] text-ink"
        >
          {options.map((o) => (
            <option key={o.key} value={o.key} disabled={!o.available}>
              {o.label}
              {o.available ? "" : " — unavailable"}
            </option>
          ))}
        </select>
      </label>

      {selected && (
        <dl className="mt-2 space-y-1 font-mono text-[11px] tabular-nums text-ink-faint">
          <div className="flex justify-between gap-3">
            <dt>warm run</dt>
            <dd className="text-ink-dim">
              {formatSeconds(selected.warm.total_seconds)} · $
              {selected.warm.usd_spot.toFixed(3)} spot
            </dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt>first run, new VM</dt>
            <dd className="text-ink-dim">
              {formatSeconds(selected.cold.total_seconds)} · $
              {selected.cold.usd_spot.toFixed(2)} spot
            </dd>
          </div>
          {/* The basis is shown because some of these numbers are measured on
              this project and some are inferred, and passing an estimate off
              as a measurement is how people end up spending real money on a
              guess. */}
          <p className="pt-1 leading-relaxed text-ink-faint">{selected.speed_basis}.</p>
          <p className="leading-relaxed text-ink-faint">{selected.note}</p>
        </dl>
      )}
    </>
  );
}

/** Account, project, and every instance — not just the running one.
 *
 *  This is the old dashboard's "Google Cloud Status" card. It answers two
 *  questions the cheap VM poll cannot: which account and project is this
 *  actually billing, and is there a stopped instance still holding a disk I
 *  am paying for. Fetched only while the drawer is open, because it is the
 *  slower of the two cloud endpoints. */
function CloudIdentity() {
  const [snap, setSnap] = useState<CloudSnapshot | null>(null);

  usePoll(async (signal) => setSnap(await getCloud(signal)), { interval: 20000 });

  if (!snap) return <p className="font-mono text-[11px] text-ink-faint">checking…</p>;

  if (!snap.available || !snap.gcloud?.installed) {
    return (
      <p className="font-mono text-[11px] leading-relaxed text-warn">
        {snap.reason ?? "gcloud is not installed — cloud features are unavailable."}
      </p>
    );
  }

  const g = snap.gcloud;
  const instances = snap.instances ?? [];

  return (
    <>
      <dl className="space-y-0.5 font-mono text-[11px] text-ink-faint">
        {[
          ["account", g.account],
          ["project", g.project],
          ["gcloud", g.version],
        ].map(([k, v]) => (
          <div key={String(k)} className="flex justify-between gap-3">
            <dt>{k}</dt>
            <dd className="truncate text-ink-dim" title={String(v ?? "")}>
              {v || "—"}
            </dd>
          </div>
        ))}
      </dl>

      <h4 className="mt-3 mb-1 font-mono text-[10px] uppercase tracking-widest text-ink-faint">
        Instances ({instances.length})
      </h4>
      {instances.length === 0 ? (
        <p className="font-mono text-[11px] text-ok">No compute instances.</p>
      ) : (
        <ul className="list-none space-y-1.5">
          {instances.map((i) => {
            const running = i.status.toUpperCase() === "RUNNING";
            return (
              <li key={`${i.zone}-${i.name}`} className="border-b border-rule pb-1.5">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-[11px] text-ink">{i.name}</span>
                  <span
                    className={`shrink-0 font-mono text-[10px] uppercase ${
                      running ? "text-ok" : "text-ink-faint"
                    }`}
                  >
                    {i.status}
                  </span>
                </div>
                <p className="font-mono text-[10px] tabular-nums text-ink-faint">
                  {i.machine_type} · {i.zone}
                  {i.accelerator ? ` · ${i.accelerator}` : ""}
                  {i.preemptible ? " · spot" : ""}
                  {i.uptime_hours != null ? ` · ${i.uptime_hours.toFixed(1)}h` : ""}
                  {i.estimated_cost_usd
                    ? ` · $${i.estimated_cost_usd.toFixed(2)}`
                    : ""}
                </p>
                {/* A stopped instance still bills for its disk, which is
                    exactly the cost people forget they are carrying. */}
                {!running && (
                  <p className="font-mono text-[10px] text-warn">
                    stopped — its disk is still billed
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {snap.total_estimated_cost_usd != null && (
        <p className="mt-2 font-mono text-[11px] tabular-nums text-ink-dim">
          session estimate ${snap.total_estimated_cost_usd.toFixed(2)}
        </p>
      )}
    </>
  );
}

function Inventory() {
  const [data, setData] = useState<CloudInventory | null>(null);
  const [loading, setLoading] = useState(false);

  // Deliberately manual, never polled: each call shells out to gcloud several
  // times and takes seconds. This was true of the old dashboard too and the
  // reasoning still holds.
  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await getInventory());
    } catch (e) {
      toast("fail", "Could not list billable resources", e instanceof Error ? e.message : "");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <>
      <button
        type="button"
        onClick={() => void load()}
        disabled={loading}
        className="rounded-sharp border border-rule-bright px-2.5 py-1 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:border-accent-dim hover:text-accent disabled:opacity-50"
      >
        {loading ? "checking gcloud…" : data ? "refresh" : "what am I paying for?"}
      </button>

      {data && !data.available && (
        <p className="mt-2 font-mono text-[11px] text-warn">
          Unavailable: {data.reason ?? "unknown"}
        </p>
      )}

      {data?.available && (
        <div className="mt-3">
          {data.resources.length === 0 ? (
            <p className="font-mono text-[11px] text-ok">
              Nothing billable in {data.project}.
            </p>
          ) : (
            <>
              <ul className="list-none space-y-2">
                {data.resources.map((r) => (
                  <li key={`${r.kind}-${r.name}`} className="border-b border-rule pb-2">
                    <div className="flex items-baseline justify-between gap-2">
                      <a
                        href={r.console_url}
                        target="_blank"
                        rel="noopener external"
                        className="truncate font-mono text-[12px] text-ink hover:text-accent"
                      >
                        {r.name}
                      </a>
                      <span className="shrink-0 font-mono text-[11px] tabular-nums text-ink-dim">
                        {r.est_usd_per_month == null
                          ? "not measured"
                          : `$${r.est_usd_per_month.toFixed(2)}/mo`}
                      </span>
                    </div>
                    <p className="font-mono text-[10px] text-ink-faint">
                      {r.kind} · {r.detail}
                      {r.size_gb ? ` · ${r.size_gb} GB` : ""}
                    </p>
                    <p className="font-mono text-[10px] text-ink-faint">{r.advice}</p>
                  </li>
                ))}
              </ul>
              <p className="mt-2 font-mono text-[11px] tabular-nums text-warn">
                ${data.est_total_usd_per_month?.toFixed(2)}/month if left as-is
                {data.unmeasured ? ` (${data.unmeasured} not measured)` : ""}
              </p>
              <p className="mt-1 font-mono text-[10px] leading-relaxed text-ink-faint">
                Estimated from list prices, not your invoice.{" "}
                <a
                  href={data.console_billing_url}
                  target="_blank"
                  rel="noopener external"
                  className="text-ink-dim hover:text-accent"
                >
                  Open GCP billing ↗
                </a>
              </p>
            </>
          )}
        </div>
      )}
    </>
  );
}

function DrawerBody() {
  const vm = useVm();
  const reduced = useReducedMotion();
  const [options, setOptions] = useState<GpuOption[]>([]);
  const [gpu, setGpu] = useState("l4");
  const [prov, setProv] = useState<ProvisionState>({});
  const [confirmingTeardown, setConfirmingTeardown] = useState(false);

  useEffect(() => {
    getGpuOptions()
      .then((r) => {
        setOptions(r.options);
        // l4 is the measured baseline for everything on this project, so it
        // is the default unless it is genuinely unavailable.
        const preferred = r.options.find((o) => o.key === "l4" && o.available);
        setGpu(preferred?.key ?? r.options.find((o) => o.available)?.key ?? "l4");
      })
      .catch(() => undefined);
  }, []);

  const building = prov.status === "running" || vm.presence === "building";

  usePoll(
    async (signal) => setProv(await getProvisionState(signal)),
    { interval: building ? 5000 : 20000 },
  );

  const start = async () => {
    try {
      await provision({ gpu });
      setProv({ status: "running", log: [] });
      refreshVm();
      toast("info", "Provisioning started", "This drawer can be closed; it keeps going.");
    } catch (e) {
      toast("fail", "Could not start provisioning", e instanceof Error ? e.message : "");
    }
  };

  const destroy = async () => {
    setConfirmingTeardown(false);
    // Told before the request so the VM's disappearance is not misread as a
    // preemption by the detector watching the same instance list.
    expectTeardown();
    try {
      const res = await teardown({});
      refreshVm();
      toast(
        res.ok ? "ok" : "fail",
        res.ok ? "VM deleted — billing stopped" : "Delete failed",
        res.ok ? undefined : res.output.slice(-300),
      );
    } catch (e) {
      toast("fail", "Delete failed", e instanceof Error ? e.message : "");
    }
  };

  const lastLog = prov.log?.slice(-1)[0] ?? "";

  return (
    <>
      <Section title="Status">
        <div className="flex items-center gap-2">
          <motion.span
            aria-hidden="true"
            animate={reduced ? undefined : building ? signatures.breathe : undefined}
            className={`font-mono text-[13px] ${
              vm.presence === "ready" ? "text-ok" : building ? "text-warn" : "text-ink-faint"
            }`}
          >
            {vm.presence === "ready" ? "●" : building ? "◐" : "○"}
          </motion.span>
          <span className="font-mono text-[12px] text-ink">
            {vm.presence === "ready"
              ? "GPU ready"
              : building
                ? "building worker images"
                : vm.presence === "unknown"
                  ? vm.reason || "gcloud unavailable"
                  : "no GPU running"}
          </span>
        </div>

        {vm.host && (
          <p className="mt-2 font-mono text-[11px] break-all text-ink-faint">
            host {vm.host}
          </p>
        )}

        {vm.instance && (
          <dl className="mt-2 space-y-0.5 font-mono text-[11px] tabular-nums text-ink-faint">
            <div className="flex justify-between gap-3">
              <dt>zone</dt>
              <dd className="text-ink-dim">{vm.instance.zone}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt>gpu</dt>
              <dd className="text-ink-dim">{vm.instance.gpu ?? "—"}</dd>
            </div>
            <div className="flex justify-between gap-3">
              <dt>uptime</dt>
              <dd className="text-ink-dim">
                {vm.instance.uptime_hours?.toFixed(1) ?? "—"}h
                {vm.instance.spot ? " · spot" : ""}
              </dd>
            </div>
          </dl>
        )}

        {building && lastLog && (
          <p className="mt-3 max-h-24 overflow-y-auto rounded-sharp border border-rule bg-void px-2 py-1.5 font-mono text-[10px] leading-relaxed break-words text-ink-faint">
            {lastLog}
          </p>
        )}

        {/* Spot preemption on this project deletes the boot disk along with
            the model cache, so the warning belongs where the choice is made. */}
        {vm.instance?.spot && (
          <p className="mt-2 font-mono text-[10px] leading-relaxed text-warn">
            Spot instance: preemption deletes the boot disk and the cached
            model weights with it.
          </p>
        )}
      </Section>

      <Section title="Provision">
        <GpuChooser options={options} value={gpu} onChange={setGpu} />
        <motion.button
          type="button"
          onClick={() => void start()}
          disabled={building}
          whileHover={reduced || building ? undefined : { scale: 1.02 }}
          whileTap={reduced || building ? undefined : { scale: 0.98 }}
          transition={snap}
          className="mt-3 w-full rounded-panel border border-accent-dim bg-accent-wash px-3 py-2 font-mono text-[11px] uppercase tracking-widest text-accent hover:bg-accent hover:text-void disabled:opacity-40"
        >
          {building ? "building…" : "create GPU VM"}
        </motion.button>
        {building && (
          <p className="mt-2 font-mono text-[10px] leading-relaxed text-ink-faint">
            A cold build is ~30 minutes; from a baked machine image, ~2. You can
            close this and submit a job — it will wait for the VM rather than
            failing.
          </p>
        )}
      </Section>

      <Section title="Teardown">
        {/* Deleting billable infrastructure now asks first. In the old
            dashboard this was a single unconfirmed click sitting next to
            Create, while deleting a mere job asked for confirmation. */}
        <AnimatePresence mode="wait" initial={false}>
          {confirmingTeardown ? (
            <motion.div
              key="confirm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={fade}
              className="flex flex-col gap-2"
            >
              <p className="font-mono text-[11px] leading-relaxed text-warn">
                Delete {vm.instance?.name ?? "the VM"}? Any running job stops,
                and a cached image is lost with the disk.
              </p>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => void destroy()}
                  className="flex-1 rounded-sharp border border-fail-edge bg-fail-wash px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-fail hover:bg-fail hover:text-void"
                >
                  delete it
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmingTeardown(false)}
                  className="flex-1 rounded-sharp border border-rule-bright px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:text-ink"
                >
                  keep
                </button>
              </div>
            </motion.div>
          ) : (
            <motion.button
              key="ask"
              type="button"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={fade}
              onClick={() => setConfirmingTeardown(true)}
              disabled={vm.presence !== "ready"}
              className="w-full rounded-sharp border border-rule-bright px-3 py-1.5 font-mono text-[11px] uppercase tracking-widest text-ink-dim hover:border-fail-edge hover:text-fail disabled:opacity-40"
            >
              delete VM
            </motion.button>
          )}
        </AnimatePresence>
        <p className="mt-2 font-mono text-[10px] leading-relaxed text-ink-faint">
          Billing continues until this succeeds.
        </p>
      </Section>

      <Section title="Account">
        <CloudIdentity />
      </Section>

      <Section title="Billable resources">
        <Inventory />
      </Section>
    </>
  );
}

export function ComputeDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreFocus = useRef<HTMLElement | null>(null);

  // Focus moves into the drawer on open and returns to whatever opened it on
  // close. Without this a keyboard user tabs from the pill straight into the
  // page behind the drawer.
  useEffect(() => {
    if (!open) return;
    restoreFocus.current = document.activeElement as HTMLElement | null;
    const first = panelRef.current?.querySelector<HTMLElement>(
      'button, [href], select, input, [tabindex]:not([tabindex="-1"])',
    );
    first?.focus();
    return () => restoreFocus.current?.focus?.();
  }, [open]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={fade}
            onClick={onClose}
            aria-hidden="true"
            className="fixed inset-0 z-40 bg-void/70 backdrop-blur-sm"
          />
          <motion.div
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-label="Compute"
            variants={drawerVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={settle}
            className="fixed inset-y-0 right-0 z-50 flex w-[min(26rem,100vw)] flex-col overflow-y-auto border-l border-rule bg-panel"
          >
            <div className="flex items-center justify-between px-5 py-4">
              <h2 className="font-mono text-[12px] uppercase tracking-widest text-ink">
                Compute
              </h2>
              <button
                type="button"
                onClick={onClose}
                aria-label="Close compute panel"
                className="rounded-sharp px-2 py-1 font-mono text-[13px] text-ink-faint hover:text-ink"
              >
                ×
              </button>
            </div>
            <DrawerBody />
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
