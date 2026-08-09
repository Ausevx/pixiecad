from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import subprocess

SPOT_USD_PER_HOUR = {
    "nvidia-tesla-t4": 0.11,
    "nvidia-l4": 0.21,
    "nvidia-tesla-v100": 0.74,
    "nvidia-tesla-p4": 0.09,
}

ONDEMAND_USD_PER_HOUR = {
    "nvidia-tesla-t4": 0.35,
    "nvidia-l4": 0.70,
    "nvidia-tesla-v100": 2.48,
    "nvidia-tesla-p4": 0.60,
}

CONSOLE_BILLING_URL = "https://console.cloud.google.com/billing"


@dataclass
class GcloudStatus:
    installed: bool
    version: str | None
    account: str | None
    project: str | None
    billing_enabled: bool | None
    error: str | None


@dataclass
class InstanceInfo:
    name: str
    zone: str
    machine_type: str
    status: str
    accelerator: str | None
    preemptible: bool
    uptime_hours: float | None
    estimated_cost_usd: float | None
    #: The guardrail, in seconds, from --max-run-duration. None when the VM
    #: has no hard stop.
    max_run_seconds: int | None = None
    #: Seconds until Compute Engine DELETES this instance. Clamped at 0.
    #:
    #: Worth surfacing rather than leaving implicit: the guardrail has twice
    #: taken a VM down mid-job here -- once between the bake and texturing,
    #: which surfaced as two unexplained "timed out after 30s" errors, and
    #: once during a bake that then had nowhere to run. The countdown is the
    #: difference between planning a long run and losing one.
    seconds_remaining: float | None = None


@dataclass
class CloudSnapshot:
    gcloud: GcloudStatus
    instances: list[InstanceInfo]
    total_estimated_cost_usd: float
    console_billing_url: str
    checked_at: str  # ISO-8601 UTC

    @property
    def has_running_gpu(self) -> bool:
        return any(
            inst.status.upper() == "RUNNING" and inst.accelerator is not None
            for inst in self.instances
        )


def _run(cmd: list[str], timeout_s: float = 20) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=cmd, returncode=127, stdout="", stderr="File not found"
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd, returncode=124, stdout="", stderr="Command timed out"
        )


def gcloud_status() -> GcloudStatus:
    try:
        ver_res = _run(["gcloud", "version", "--format=json"])
        if ver_res.returncode == 127:
            return GcloudStatus(
                installed=False,
                version=None,
                account=None,
                project=None,
                billing_enabled=None,
                error=None,
            )

        installed = True
        version = None
        error = None

        if ver_res.returncode == 0:
            try:
                data = json.loads(ver_res.stdout)
                if isinstance(data, dict):
                    version = data.get("Google Cloud SDK")
            except Exception:
                version = None
        else:
            error = ver_res.stderr.strip() or f"gcloud exited with code {ver_res.returncode}"

        account = None
        acc_res = _run(["gcloud", "config", "get-value", "account"])
        if acc_res.returncode == 0:
            val = acc_res.stdout.strip()
            if val and val != "(unset)":
                account = val

        project = None
        proj_res = _run(["gcloud", "config", "get-value", "project"])
        if proj_res.returncode == 0:
            val = proj_res.stdout.strip()
            if val and val != "(unset)":
                project = val

        billing_enabled = None
        if project is not None:
            bill_res = _run(
                ["gcloud", "billing", "projects", "describe", project, "--format=value(billingEnabled)"]
            )
            if bill_res.returncode == 0:
                val = bill_res.stdout.strip().lower()
                if val == "true":
                    billing_enabled = True
                elif val == "false":
                    billing_enabled = False

        return GcloudStatus(
            installed=installed,
            version=version,
            account=account,
            project=project,
            billing_enabled=billing_enabled,
            error=error,
        )
    except Exception as exc:
        return GcloudStatus(
            installed=False,
            version=None,
            account=None,
            project=None,
            billing_enabled=None,
            error=str(exc),
        )


def list_instances(project: str | None = None) -> list[InstanceInfo]:
    cmd = ["gcloud", "compute", "instances", "list", "--format=json"]
    if project:
        cmd.extend(["--project", project])

    res = _run(cmd)
    if res.returncode != 0:
        return []

    try:
        data = json.loads(res.stdout)
        if not isinstance(data, list):
            return []
    except Exception:
        return []

    instances: list[InstanceInfo] = []
    now = datetime.now(timezone.utc)

    for item in data:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "")

        raw_zone = item.get("zone", "")
        zone = raw_zone.rstrip("/").split("/")[-1] if raw_zone else ""

        raw_machine_type = item.get("machineType", "")
        machine_type = raw_machine_type.rstrip("/").split("/")[-1] if raw_machine_type else ""

        status = item.get("status", "")

        accelerator = None
        guest_accs = item.get("guestAccelerators")
        if isinstance(guest_accs, list) and guest_accs:
            first_acc = guest_accs[0]
            if isinstance(first_acc, dict):
                raw_acc_type = first_acc.get("acceleratorType", "")
                if raw_acc_type:
                    accelerator = raw_acc_type.rstrip("/").split("/")[-1]

        scheduling = item.get("scheduling")
        preemptible = False
        if isinstance(scheduling, dict):
            if scheduling.get("provisioningModel") == "SPOT" or scheduling.get("preemptible") is True:
                preemptible = True

        uptime_hours = None
        ts_str = item.get("creationTimestamp")
        if ts_str and isinstance(ts_str, str):
            try:
                clean_ts = ts_str
                if clean_ts.endswith("Z") or clean_ts.endswith("z"):
                    clean_ts = clean_ts[:-1] + "+00:00"
                created_dt = datetime.fromisoformat(clean_ts)
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                delta = now - created_dt
                uptime_hours = max(0.0, delta.total_seconds() / 3600.0)
            except Exception:
                uptime_hours = None

        # maxRunDuration is nested and its seconds field arrives as a string
        # from gcloud's JSON, hence the defensive parse.
        max_run_seconds = None
        seconds_remaining = None
        if isinstance(scheduling, dict):
            mrd = scheduling.get("maxRunDuration")
            if isinstance(mrd, dict):
                try:
                    max_run_seconds = int(mrd.get("seconds") or 0) or None
                except (TypeError, ValueError):
                    max_run_seconds = None
        if max_run_seconds and uptime_hours is not None:
            seconds_remaining = max(0.0, max_run_seconds - uptime_hours * 3600.0)

        estimated_cost_usd = None
        if accelerator is not None and uptime_hours is not None:
            estimated_cost_usd = estimate_cost(
                accelerator=accelerator,
                hours=uptime_hours,
                preemptible=preemptible,
            )

        instances.append(
            InstanceInfo(
                name=name,
                zone=zone,
                machine_type=machine_type,
                status=status,
                accelerator=accelerator,
                preemptible=preemptible,
                uptime_hours=uptime_hours,
                estimated_cost_usd=estimated_cost_usd,
                max_run_seconds=max_run_seconds,
                seconds_remaining=seconds_remaining,
            )
        )

    return instances


def estimate_cost(
    accelerator: str, hours: float, preemptible: bool = True
) -> float | None:
    if hours < 0:
        raise ValueError("Hours cannot be negative")
    table = SPOT_USD_PER_HOUR if preemptible else ONDEMAND_USD_PER_HOUR
    rate = table.get(accelerator)
    if rate is None:
        return None
    return rate * hours


def snapshot() -> CloudSnapshot:
    try:
        g_status = gcloud_status()
        if g_status.installed:
            insts = list_instances(project=g_status.project)
        else:
            insts = []
        total_cost = sum(
            i.estimated_cost_usd
            for i in insts
            if i.estimated_cost_usd is not None
        )
        return CloudSnapshot(
            gcloud=g_status,
            instances=insts,
            total_estimated_cost_usd=total_cost,
            console_billing_url=CONSOLE_BILLING_URL,
            checked_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        now_iso = datetime.now(timezone.utc).isoformat()
        return CloudSnapshot(
            gcloud=GcloudStatus(
                installed=False,
                version=None,
                account=None,
                project=None,
                billing_enabled=None,
                error="Failed to compute snapshot",
            ),
            instances=[],
            total_estimated_cost_usd=0.0,
            console_billing_url=CONSOLE_BILLING_URL,
            checked_at=now_iso,
        )


# --- Billable resource inventory -------------------------------------------
#
# A running GPU VM is the obvious expense and the one the dashboard already
# tracked. The costs that actually surprise people are the quiet ones: a disk
# left behind by a deleted instance, a machine image nobody remembers baking,
# a bucket of old runs. Those bill every month whether or not anything runs.
#
# Rates are indicative GCP list prices and are labelled as estimates
# everywhere they surface. This is a run-rate model, not your bill: real
# per-resource cost needs Cloud Billing export to BigQuery, which is a
# heavier setup. The number here is for "what am I paying to leave this
# lying around", which is the question that actually gets asked.

DISK_USD_PER_GB_MONTH = {
    "pd-standard": 0.04,
    "pd-balanced": 0.10,
    "pd-ssd": 0.17,
}
MACHINE_IMAGE_USD_PER_GB_MONTH = 0.05
BUCKET_USD_PER_GB_MONTH = 0.020  # standard, single region

CONSOLE = "https://console.cloud.google.com"


@dataclass
class BillableResource:
    kind: str  # instance | disk | machine-image | bucket
    name: str
    location: str
    detail: str
    size_gb: float | None
    est_usd_per_month: float | None
    console_url: str
    # What to do about it, in the user's terms -- an inventory that does not
    # say "this is safe to delete" just creates anxiety.
    advice: str


def _json(cmd: list[str], timeout_s: float = 30) -> list:
    import json as _json_mod

    res = _run(cmd, timeout_s=timeout_s)
    if res.returncode != 0 or not (res.stdout or "").strip():
        return []
    try:
        data = _json_mod.loads(res.stdout)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def list_billable(project: str | None = None) -> list[BillableResource]:
    """Every GCP resource this project bills for, with a console deep link."""
    out: list[BillableResource] = []
    proj = project or ""
    suffix = f"?project={proj}" if proj else ""

    for inst in _json(["gcloud", "compute", "instances", "list", "--format=json"]):
        zone = str(inst.get("zone", "")).rsplit("/", 1)[-1]
        name = inst.get("name", "?")
        status = inst.get("status", "?")
        accel = (inst.get("guestAccelerators") or [{}])[0].get("acceleratorType", "")
        accel = str(accel).rsplit("/", 1)[-1]
        spot = (inst.get("scheduling") or {}).get("provisioningModel") == "SPOT"
        rate = (SPOT_USD_PER_HOUR if spot else ONDEMAND_USD_PER_HOUR).get(accel)
        # A stopped instance bills for its disk only, which is why "turned
        # off" is not the same as "free".
        monthly = round(rate * 730, 2) if (rate and status == "RUNNING") else None
        out.append(
            BillableResource(
                kind="instance",
                name=name,
                location=zone,
                detail=f"{str(inst.get('machineType','')).rsplit('/',1)[-1]}"
                + (f" + {accel}" if accel else "")
                + f" [{status}{', spot' if spot else ''}]",
                size_gb=None,
                est_usd_per_month=monthly,
                console_url=f"{CONSOLE}/compute/instancesDetail/zones/{zone}/instances/{name}{suffix}",
                advice=(
                    "RUNNING and billing by the hour. Delete when idle."
                    if status == "RUNNING"
                    else "Stopped: no GPU charge, but its disk still bills."
                ),
            )
        )

    for disk in _json(["gcloud", "compute", "disks", "list", "--format=json"]):
        zone = str(disk.get("zone") or disk.get("region", "")).rsplit("/", 1)[-1]
        name = disk.get("name", "?")
        size = float(disk.get("sizeGb", 0) or 0)
        dtype = str(disk.get("type", "")).rsplit("/", 1)[-1]
        attached = bool(disk.get("users"))
        rate = DISK_USD_PER_GB_MONTH.get(dtype, 0.10)
        out.append(
            BillableResource(
                kind="disk",
                name=name,
                location=zone,
                detail=f"{dtype}, {'attached' if attached else 'UNATTACHED'}",
                size_gb=size,
                est_usd_per_month=round(size * rate, 2),
                console_url=f"{CONSOLE}/compute/disksDetail/zones/{zone}/disks/{name}{suffix}",
                advice=(
                    "In use by an instance."
                    if attached
                    else "Not attached to anything and still billing. Usually safe to delete."
                ),
            )
        )

    for image in _json(["gcloud", "compute", "machine-images", "list", "--format=json"]):
        name = image.get("name", "?")
        size = float(image.get("totalStorageBytes", 0) or 0) / 1e9
        out.append(
            BillableResource(
                kind="machine-image",
                name=name,
                location="global",
                detail="prebuilt worker template",
                size_gb=round(size, 1),
                est_usd_per_month=round(size * MACHINE_IMAGE_USD_PER_GB_MONTH, 2),
                console_url=f"{CONSOLE}/compute/machineImages/details/{name}{suffix}",
                advice="Delete if you no longer want fast VM launches.",
            )
        )

    for bucket in _json(["gcloud", "storage", "buckets", "list", "--format=json"], timeout_s=40):
        name = str(bucket.get("name") or bucket.get("id", "?")).replace("gs://", "")
        out.append(
            BillableResource(
                kind="bucket",
                name=name,
                location=str(bucket.get("location", "")),
                detail=str(bucket.get("storageClass", "")),
                size_gb=None,  # needs an object scan; too slow for a page load
                est_usd_per_month=None,
                console_url=f"{CONSOLE}/storage/browser/{name}{suffix}",
                advice=f"Size not measured (needs a scan). ~${BUCKET_USD_PER_GB_MONTH}/GB-month.",
            )
        )

    return out
