"""Tests for PixieCAD web dashboard FastAPI app."""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
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

    res_opt = client.post(f"/api/jobs/{job_id}/optimize", json={"target_faces": 5000})
    assert res_opt.status_code == 409


def test_get_unknown_job(client):
    res = client.get("/api/jobs/unknown-id")
    assert res.status_code == 404


def test_get_cloud(client, monkeypatch):
    res = client.get("/api/cloud")
    assert res.status_code == 200
    data = res.json()
    assert "available" in data

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
    job_dir = tmp_path / job_id

    # Create a dummy dense.ply in the job directory
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.export(job_dir / "dense.ply")

    opt_res = client.post(f"/api/jobs/{job_id}/optimize", json={"target_faces": 50})
    assert opt_res.status_code == 200
    opt_data = opt_res.json()
    assert opt_data["status"] == "done"
    assert opt_data["glb_url"] == f"/api/jobs/{job_id}/model.glb"

    glb_res = client.get(f"/api/jobs/{job_id}/model.glb")
    assert glb_res.status_code == 200
    assert len(glb_res.content) > 0

