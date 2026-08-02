from datetime import datetime, timedelta, timezone
import subprocess
import pytest

from pixiecad.cloud import (
    CloudSnapshot,
    GcloudStatus,
    InstanceInfo,
    estimate_cost,
    gcloud_status,
    list_instances,
    snapshot,
)


def test_gcloud_missing(monkeypatch):
    def fake_run(cmd, timeout_s=20):
        return subprocess.CompletedProcess(args=cmd, returncode=127, stdout="", stderr="not found")

    monkeypatch.setattr("pixiecad.cloud._run", fake_run)

    st = gcloud_status()
    assert st == GcloudStatus(
        installed=False,
        version=None,
        account=None,
        project=None,
        billing_enabled=None,
        error=None,
    )

    sn = snapshot()
    assert sn.gcloud.installed is False
    assert sn.instances == []
    assert sn.total_estimated_cost_usd == 0.0
    assert sn.has_running_gpu is False


def test_gcloud_healthy_path(monkeypatch):
    def fake_run(cmd, timeout_s=20):
        if "version" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"Google Cloud SDK": "450.0.0"}',
                stderr="",
            )
        elif "account" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="user@example.com\n", stderr=""
            )
        elif "project" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="my-project\n", stderr=""
            )
        elif "describe" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="true\n", stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr("pixiecad.cloud._run", fake_run)

    st = gcloud_status()
    assert st.installed is True
    assert st.version == "450.0.0"
    assert st.account == "user@example.com"
    assert st.project == "my-project"
    assert st.billing_enabled is True
    assert st.error is None


def test_gcloud_unset_project(monkeypatch):
    def fake_run(cmd, timeout_s=20):
        if "version" in cmd:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='{"Google Cloud SDK": "450.0.0"}',
                stderr="",
            )
        elif "account" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="user@example.com\n", stderr=""
            )
        elif "project" in cmd:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="(unset)\n", stderr=""
            )
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr("pixiecad.cloud._run", fake_run)

    st = gcloud_status()
    assert st.installed is True
    assert st.project is None
    assert st.billing_enabled is None


def test_list_instances_realistic_payload(monkeypatch):
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    # format trailing Z or standard ISO
    payload = f"""[
      {{
        "name": "gpu-worker-1",
        "zone": "https://www.googleapis.com/compute/v1/projects/my-proj/zones/us-central1-a",
        "machineType": "https://www.googleapis.com/compute/v1/projects/my-proj/zones/us-central1-a/machineTypes/n1-standard-4",
        "status": "RUNNING",
        "guestAccelerators": [
          {{
            "acceleratorType": "https://www.googleapis.com/compute/v1/projects/my-proj/zones/us-central1-a/acceleratorTypes/nvidia-tesla-t4",
            "acceleratorCount": 1
          }}
        ],
        "scheduling": {{
          "provisioningModel": "SPOT"
        }},
        "creationTimestamp": "{two_hours_ago}"
      }}
    ]"""

    def fake_run(cmd, timeout_s=20):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr("pixiecad.cloud._run", fake_run)

    insts = list_instances(project="my-proj")
    assert len(insts) == 1
    inst = insts[0]
    assert inst.name == "gpu-worker-1"
    assert inst.zone == "us-central1-a"
    assert inst.machine_type == "n1-standard-4"
    assert inst.status == "RUNNING"
    assert inst.accelerator == "nvidia-tesla-t4"
    assert inst.preemptible is True
    assert inst.uptime_hours == pytest.approx(2.0, abs=0.1)
    assert inst.estimated_cost_usd == pytest.approx(0.22, abs=0.02)


def test_list_instances_no_accelerator(monkeypatch):
    payload = """[
      {
        "name": "cpu-worker",
        "zone": "us-central1-a",
        "machineType": "n1-standard-2",
        "status": "RUNNING",
        "creationTimestamp": "2026-08-02T10:00:00Z"
      }
    ]"""

    def fake_run(cmd, timeout_s=20):
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr("pixiecad.cloud._run", fake_run)

    insts = list_instances()
    assert len(insts) == 1
    assert insts[0].accelerator is None
    assert insts[0].estimated_cost_usd is None


def test_estimate_cost():
    assert estimate_cost("nvidia-tesla-t4", 2.0, preemptible=True) == pytest.approx(0.22)
    assert estimate_cost("nvidia-tesla-t4", 2.0, preemptible=False) == pytest.approx(0.70)
    assert estimate_cost("unknown-gpu", 2.0) is None

    with pytest.raises(ValueError):
        estimate_cost("nvidia-tesla-t4", -1.0)


def test_malformed_json_gcloud(monkeypatch):
    def fake_run(cmd, timeout_s=20):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="BAD JSON {{{", stderr=""
        )

    monkeypatch.setattr("pixiecad.cloud._run", fake_run)

    assert list_instances() == []
