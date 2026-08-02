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
