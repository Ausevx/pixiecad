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


class TestTheGuardrailDeadlineIsVisible:
    """Every worker VM self-DELETES at --max-run-duration, and that deadline
    was invisible in the dashboard.

    It has bitten twice: once it took a VM down between the normal-map bake
    and texturing, which surfaced only as two unexplained "timed out after
    30s" errors, and once it deleted a machine holding 43 GB of freshly
    downloaded weights minutes before they were baked into an image. Uptime
    was shown throughout; remaining time was not. Uptime says what you have
    spent, this says whether the run you are about to start will survive.
    """

    @staticmethod
    def _instances(monkeypatch, item):
        import json as _json

        from pixiecad import cloud

        class _Res:
            returncode = 0
            stdout = _json.dumps([item])
            stderr = ""

        monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **k: _Res())
        return cloud.list_instances()

    def _item(self, seconds=None, age_hours=1.0):
        # Relative to NOW, not a fixed date: a hardcoded timestamp drifts past
        # any guardrail you pick and the test starts asserting the clamp
        # instead of the countdown.
        from datetime import datetime, timedelta, timezone

        created = (
            datetime.now(timezone.utc) - timedelta(hours=age_hours)
        ).isoformat()
        item = {
            "name": "vm", "zone": "z/asia-southeast1-b", "machineType": "m/g2-standard-16",
            "status": "RUNNING", "creationTimestamp": created,
            "scheduling": {"provisioningModel": "SPOT"},
        }
        if seconds is not None:
            # gcloud returns this nested, and its seconds field as a STRING.
            item["scheduling"]["maxRunDuration"] = {"seconds": str(seconds)}
        return item

    def test_the_deadline_is_parsed_and_counted_down(self, monkeypatch):
        inst = self._instances(monkeypatch, self._item(seconds=28800))[0]
        assert inst.max_run_seconds == 28800
        assert inst.seconds_remaining is not None
        # Remaining plus elapsed must reconstruct the guardrail.
        assert inst.seconds_remaining + inst.uptime_hours * 3600 == pytest.approx(28800, abs=2)

    def test_a_vm_past_its_deadline_reports_zero_not_negative(self, monkeypatch):
        """A UI dividing by max_run_seconds must not get a negative ring."""
        inst = self._instances(monkeypatch, self._item(seconds=60, age_hours=2.0))[0]
        assert inst.seconds_remaining == 0.0

    def test_a_vm_with_no_guardrail_reports_nothing(self, monkeypatch):
        """An on-demand VM without --max-run-duration has no deadline; the
        pill must show no countdown rather than a wrong one."""
        inst = self._instances(monkeypatch, self._item(seconds=None))[0]
        assert inst.max_run_seconds is None
        assert inst.seconds_remaining is None

    def test_the_api_passes_both_fields_through(self):
        """The pill reads these two keys; losing them silently removes the
        countdown with everything else still working."""
        import inspect

        from pixiecad.web import app as appmod

        src = inspect.getsource(appmod.create_app)
        assert '"max_run_seconds": inst.max_run_seconds' in src
        assert '"seconds_remaining": inst.seconds_remaining' in src

    def test_our_own_metadata_beats_gcps_field(self, monkeypatch):
        """GCP returns sourceMachineImage for a while after creation and then
        stops. An instance launched from pixiecad-worker-v5 reported it
        correctly, and had dropped it entirely from both `instances list` and
        `instances describe` a couple of hours later. A row that decays to
        "unknown" is worse than none, so launch_from_image.sh stamps the name
        into instance metadata and that wins."""
        item = self._item(seconds=3600)
        item["metadata"] = {"items": [{"key": "pixiecad-image", "value": "pixiecad-worker-v5"}]}
        item["sourceMachineImage"] = "projects/p/global/machineImages/stale-v1"
        inst = self._instances(monkeypatch, item)[0]
        assert inst.source_machine_image == "pixiecad-worker-v5"

    def test_gcps_field_is_still_the_fallback(self, monkeypatch):
        """A VM launched by hand carries no stamp of ours."""
        item = self._item(seconds=3600)
        item["sourceMachineImage"] = "projects/p/global/machineImages/pixiecad-worker-v5"
        inst = self._instances(monkeypatch, item)[0]
        assert inst.source_machine_image == "pixiecad-worker-v5"

    def test_the_launch_script_stamps_the_image(self):
        from pathlib import Path

        script = Path("scripts/launch_from_image.sh").read_text()
        assert '--metadata=pixiecad-image="$IMAGE"' in script

    def test_the_source_machine_image_is_reported(self, monkeypatch):
        """Which image a worker booted from decides what it can do -- a
        pre-COLMAP image cannot run photogrammetry however the job form is
        configured -- and a VM once booted from the WRONG image, whose missing
        docker binary only surfaced mid-run."""
        item = self._item(seconds=3600)
        item["sourceMachineImage"] = (
            "projects/p/global/machineImages/pixiecad-worker-v5"
        )
        inst = self._instances(monkeypatch, item)[0]
        assert inst.source_machine_image == "pixiecad-worker-v5"

    def test_a_vm_with_no_source_image_says_so(self, monkeypatch):
        """A VM built from a plain OS image has no machine image; the drawer
        must show 'unknown' rather than a blank row."""
        inst = self._instances(monkeypatch, self._item(seconds=3600))[0]
        assert inst.source_machine_image is None

    def test_the_api_exposes_the_image(self):
        import inspect

        from pixiecad.web import app as appmod

        assert '"image": inst.source_machine_image' in inspect.getsource(appmod.create_app)
