"""Tests for PixieCAD web dashboard FastAPI app."""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pathlib
import pytest
from fastapi.testclient import TestClient

from pixiecad.web.app import create_app


def _generate_test_jpeg(seed: int = 0) -> bytes:
    img = np.zeros((700, 700, 3), dtype=np.uint8)
    grid_size = 50
    for i in range(0, 700, grid_size):
        for j in range(0, 700, grid_size):
            if ((i + j) // grid_size + seed) % 2 == 0:
                img[i:i + grid_size, j:j + grid_size] = (200, 150, 100)
            else:
                img[i:i + grid_size, j:j + grid_size] = (50, 100, 200)
    cv2.line(img, (0, seed * 20), (700, 700 - seed * 20), (255, 255, 255), 5)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path)
    return TestClient(app)


def test_get_root(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "pixiecad" in res.text.lower()


def test_post_jobs_and_get_job(client):
    img1 = _generate_test_jpeg(1)
    img2 = _generate_test_jpeg(2)

    files = [
        ("files", ("photo1.jpg", img1, "image/jpeg")),
        ("files", ("photo2.jpg", img2, "image/jpeg")),
    ]
    data = {
        "name": "test_object",
        "target_faces": "15000",
        "length": "4.5m",
        "width": "2m",
        "height": "1m",
    }

    res = client.post("/api/jobs", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()
    assert "job_id" in res_data
    job_id = res_data["job_id"]

    list_res = client.get("/api/jobs")
    assert list_res.status_code == 200
    jobs_list = list_res.json()
    assert any(j["job_id"] == job_id for j in jobs_list)

    detail_res = client.get(f"/api/jobs/{job_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["status"] in ("queued", "running", "done", "failed")


def test_get_model_glb_missing(client):
    img1 = _generate_test_jpeg(1)
    res = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img1, "image/jpeg"))],
        data={"name": "test_obj"},
    )
    job_id = res.json()["job_id"]

    res_glb = client.get(f"/api/jobs/{job_id}/model.glb")
    assert res_glb.status_code == 404


def test_optimize_no_dense_mesh(client):
    img1 = _generate_test_jpeg(1)
    res = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img1, "image/jpeg"))],
        data={"name": "test_obj"},
    )
    job_id = res.json()["job_id"]

    res_opt = client.post(
        f"/api/jobs/{job_id}/optimize", json={"target_faces": 5000, "normal_res": 128}
    )
    assert res_opt.status_code == 409


def test_get_unknown_job(client):
    res = client.get("/api/jobs/unknown-id")
    assert res.status_code == 404


def test_get_cloud(client, monkeypatch):
    # The stub must be installed BEFORE any request: the unstubbed endpoint
    # shells out to real gcloud (~10s and environment-dependent).
    @dataclass
    class Gcloud:
        installed: bool = True
        version: str = "450.0.0"
        account: str = "user@example.com"
        project: str = "my-project"
        billing_enabled: bool = True

    @dataclass
    class Instance:
        name: str = "gpu-1"
        zone: str = "us-central1-a"
        machine_type: str = "n1-standard-8"
        status: str = "RUNNING"
        accelerator: str = "T4"
        preemptible: bool = True
        uptime_hours: float = 1.0
        estimated_cost_usd: float = 0.50

    @dataclass
    class Snapshot:
        gcloud: Gcloud = field(default_factory=Gcloud)
        instances: list = field(default_factory=lambda: [Instance()])
        total_estimated_cost_usd: float = 0.50
        console_billing_url: str = "https://console.cloud.google.com"
        checked_at: str = "2026-08-02T12:00:00Z"
        has_running_gpu: bool = True

    dummy_mod = types.ModuleType("pixiecad.cloud")
    dummy_mod.snapshot = lambda: Snapshot()
    monkeypatch.setitem(sys.modules, "pixiecad.cloud", dummy_mod)

    res_cloud = client.get("/api/cloud")
    assert res_cloud.status_code == 200
    cloud_data = res_cloud.json()
    assert cloud_data["available"] is True
    assert cloud_data["gcloud"]["installed"] is True
    assert cloud_data["total_estimated_cost_usd"] == 0.50


def test_optimize_with_dense_mesh(tmp_path: Path):
    import trimesh

    app = create_app(tmp_path)
    client = TestClient(app)

    img1 = _generate_test_jpeg(1)
    res = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img1, "image/jpeg"))],
        data={"name": "sphere"},
    )
    job_id = res.json()["job_id"]
    # The job lives in its own timestamped session folder, not one named after
    # the opaque job id; the API is what tells you where.
    job_dir = Path(client.get(f"/api/jobs/{job_id}").json()["dir"])
    assert job_dir.parent == tmp_path

    # Create a dummy dense.ply in the job directory
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.export(job_dir / "dense.ply")

    opt_res = client.post(
        f"/api/jobs/{job_id}/optimize", json={"target_faces": 50, "normal_res": 128}
    )
    assert opt_res.status_code == 200
    opt_data = opt_res.json()
    assert opt_data["status"] == "done"
    assert opt_data["glb_url"] == f"/api/jobs/{job_id}/model.glb"

    glb_res = client.get(f"/api/jobs/{job_id}/model.glb")
    assert glb_res.status_code == 200
    assert len(glb_res.content) > 0


def test_full_pipeline_fake_backend_and_parts(client):
    import time

    img1 = _generate_test_jpeg(1)
    img2 = _generate_test_jpeg(2)

    files = [
        ("files", ("photo1.jpg", img1, "image/jpeg")),
        ("files", ("photo2.jpg", img2, "image/jpeg")),
    ]
    data = {
        "name": "fake_test",
        "backend": "fake",
        "split": "true",
        "object_hint": "table",
    }

    res = client.post("/api/jobs", files=files, data=data)
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    # Poll until done (max 30s)
    deadline = time.time() + 30
    detail = None
    while time.time() < deadline:
        detail_res = client.get(f"/api/jobs/{job_id}")
        assert detail_res.status_code == 200
        detail = detail_res.json()
        if detail["status"] in ("done", "failed"):
            break
        time.sleep(0.1)

    assert detail is not None
    assert detail["status"] == "done", f"Job failed, logs: {detail.get('log')}"
    assert detail["regime"] == "sparse_views"
    assert detail["glb_url"] is not None
    assert "parts" in detail and len(detail["parts"]) > 0
    for part in detail["parts"]:
        assert "url" in part and part["url"]

    # Download a part via new endpoint
    part0 = detail["parts"][0]
    part_res = client.get(part0["url"])
    assert part_res.status_code == 200
    assert len(part_res.content) > 0

    # Path traversal rejection -> 400 or 404, never 200
    bad_res = client.get(f"/api/jobs/{job_id}/parts/../../etc/passwd")
    assert bad_res.status_code in (400, 404)
    assert bad_res.status_code != 200

    # Unknown job for parts endpoint -> 404
    unknown_res = client.get("/api/jobs/unknown-id/parts/part-0.glb")
    assert unknown_res.status_code == 404




def test_each_job_gets_its_own_session_folder(tmp_path: Path):
    """Two uploads must never share input/ or output/ directories."""
    client = TestClient(create_app(tmp_path))

    ids = []
    for i in (1, 2):
        res = client.post(
            "/api/jobs",
            files=[("files", (f"photo{i}.jpg", _generate_test_jpeg(i), "image/jpeg"))],
            data={"name": "car", "backend": "fake"},
        )
        ids.append(res.json()["job_id"])

    a, b = (client.get(f"/api/jobs/{j}").json() for j in ids)

    assert a["dir"] != b["dir"]
    assert Path(a["dir"]).parent == tmp_path
    # Timestamp-first naming keeps the folders sortable and human-readable.
    assert "car" in a["session"]
    # Each session holds only the photo that was uploaded to it.
    assert [p.name for p in (Path(a["dir"]) / "input").iterdir()] == ["photo1.jpg"]
    assert [p.name for p in (Path(b["dir"]) / "input").iterdir()] == ["photo2.jpg"]


class TestFinishingOptionsWiring:
    """The dashboard form must actually reach FinishOptions.

    These are wiring tests, not behaviour tests: a renamed form field fails
    silently otherwise, falling back to a default and quietly not doing what
    the user ticked.
    """

    def test_form_fields_match_the_api(self):
        """Every finishing input in the HTML must be a real API parameter."""
        import inspect
        import re

        from pixiecad.web import app as app_module

        html = (
            pathlib.Path(app_module.__file__).parent / "static" / "index.html"
        ).read_text()
        names = set(re.findall(r'name="([a-z_]+)"', html))
        source = inspect.getsource(app_module.create_app)
        for field in (
            "smooth_iterations",
            "texture",
            "segmentation",
            "web_export",
            "texture_size",
            "max_parts",
            "gpu_host",
        ):
            assert field in names, f"{field} missing from the dashboard form"
            assert f"{field}:" in source, f"{field} not accepted by /api/jobs"

    def test_defaults_do_not_require_a_gpu(self):
        """An untouched form must run entirely locally."""
        from pixiecad.web.finishing import FinishOptions

        assert not FinishOptions().needs_gpu


class TestGPUOptions:
    def test_lists_hardware_with_costs(self, client):
        data = client.get("/api/gpu-options").json()["options"]
        keys = {o["key"] for o in data}
        assert {"t4", "l4", "a100", "h100"} <= keys
        for o in data:
            assert o["warm"]["total_seconds"] > 0
            assert o["warm"]["usd_spot"] > 0
            assert o["cold"]["total_seconds"] > o["warm"]["total_seconds"]

    def test_unavailable_hardware_is_flagged_not_hidden(self):
        """Quota-zero options stay visible so the reason is legible."""
        from pixiecad.cloud_options import GPU_OPTIONS

        blocked = [o for o in GPU_OPTIONS if not o.available]
        assert blocked
        assert all(o.note for o in blocked)

    def test_faster_hardware_costs_more_per_run_despite_being_quicker(self):
        """The point the estimates exist to make."""
        from pixiecad.cloud_options import GPU_OPTIONS, estimate

        l4 = next(o for o in GPU_OPTIONS if o.key == "l4")
        h100 = next(o for o in GPU_OPTIONS if o.key == "h100")
        a = estimate(l4, texture=True, semantic=True, warm=True)
        b = estimate(h100, texture=True, semantic=True, warm=True)
        assert b["total_seconds"] < a["total_seconds"]
        assert b["usd_spot"] > a["usd_spot"]

    def test_provision_rejects_unavailable_hardware(self, client):
        r = client.post("/api/cloud/provision", json={"gpu": "h100"})
        assert r.status_code == 400
        assert "not available" in r.json()["detail"]

    def test_provision_rejects_unknown_hardware(self, client):
        assert client.post("/api/cloud/provision", json={"gpu": "rtx4090"}).status_code == 400

    def test_provision_status_starts_empty(self, client):
        assert client.get("/api/cloud/provision").json() == {}


class TestCloudInventory:
    def test_endpoint_never_raises(self, client):
        """Runs on machines with no gcloud; must degrade, not 500."""
        d = client.get("/api/cloud/inventory").json()
        assert "available" in d and "resources" in d

    def test_resources_carry_console_links_and_advice(self):
        """An inventory that cannot say what is safe to delete just worries people."""
        from pixiecad.cloud import BillableResource

        r = BillableResource(
            kind="disk", name="d", location="z", detail="pd-balanced, UNATTACHED",
            size_gb=200.0, est_usd_per_month=20.0,
            console_url="https://console.cloud.google.com/compute/disksDetail/zones/z/disks/d",
            advice="Not attached to anything and still billing.",
        )
        assert r.console_url.startswith("https://console.cloud.google.com/")
        assert r.advice

    def test_stopped_instance_is_not_billed_at_the_gpu_rate(self):
        """'Turned off' is not free, but it is not the hourly GPU rate either."""
        import pixiecad.cloud as cloud

        calls = {"n": 0}

        def fake_json(cmd, timeout_s=30):
            calls["n"] += 1
            if "instances" in cmd:
                return [{
                    "name": "vm", "zone": "z/z1", "status": "TERMINATED",
                    "machineType": "m/g2-standard-16",
                    "guestAccelerators": [{"acceleratorType": "a/nvidia-l4"}],
                    "scheduling": {"provisioningModel": "SPOT"},
                }]
            return []

        original = cloud._json
        cloud._json = fake_json
        try:
            resources = cloud.list_billable("p")
        finally:
            cloud._json = original
        assert resources[0].est_usd_per_month is None
        assert "disk still bills" in resources[0].advice
