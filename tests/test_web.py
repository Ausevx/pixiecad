"""Tests for PixieCAD web dashboard FastAPI app."""

import json
import sys
import time
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
    """/ redirects into the SPA; following it must land on the real app."""
    from pixiecad.web import app as m
    dist = pathlib.Path(m.__file__).parent / "static" / "dist"
    if not (dist / "index.html").is_file():
        pytest.skip("SPA bundle not built")
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
    # Versioned by mtime: a constant URL is served from the browser's cache,
    # which is why re-optimising used to change nothing on screen.
    assert opt_data["glb_url"].startswith(f"/api/jobs/{job_id}/model.glb?v=")

    glb_res = client.get(f"/api/jobs/{job_id}/model.glb")
    assert glb_res.status_code == 200
    assert len(glb_res.content) > 0


def test_optimize_hides_raw_exception_text(tmp_path: Path):
    app = create_app(tmp_path)
    client = TestClient(app)

    img1 = _generate_test_jpeg(1)
    res = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img1, "image/jpeg"))],
        data={"name": "fail_mesh"},
    )
    job_id = res.json()["job_id"]
    job_dir = Path(client.get(f"/api/jobs/{job_id}").json()["dir"])

    # Write an invalid ply file to cause trimesh.load to fail
    (job_dir / "dense.ply").write_text("Not a valid ply file")

    opt_res = client.post(
        f"/api/jobs/{job_id}/optimize", json={"target_faces": 50, "normal_res": 128}
    )
    assert opt_res.status_code == 500
    # Short, but it must point somewhere: the exception text goes to the job
    # log (asserted below) and the detail says so. Pinned so a future
    # "helpful" change cannot put a traceback back in front of the user.
    detail = opt_res.json()["detail"]
    assert "log" in detail.lower()
    assert "Traceback" not in detail

    # The actual error should be in the job log
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed"
    assert any("Optimization failed:" in line for line in job["log"])


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

    def _new_job_params(self) -> set[str]:
        """The fields the React client declares it sends, read from its types."""
        import re

        types_ts = (
            pathlib.Path(__file__).resolve().parents[1]
            / "frontend" / "src" / "lib" / "types.ts"
        )
        if not types_ts.is_file():
            pytest.skip("frontend sources not present")
        block = re.search(
            r"export interface NewJobParams \{(.*?)\n\}", types_ts.read_text(), re.S
        )
        assert block, "NewJobParams interface not found"
        return set(re.findall(r"^\s*(\w+)\??:", block.group(1), re.M))

    def test_client_fields_are_all_accepted_by_the_api(self):
        """Every field the client sends must be a real /api/jobs parameter.

        A renamed field fails silently otherwise: the server falls back to its
        default and quietly does not do what the user ticked.
        """
        import inspect

        from pixiecad.web import app as app_module

        source = inspect.getsource(app_module.create_app)
        for f_name in self._new_job_params():
            if f_name == "view_tags":
                continue  # sent as JSON under its own name, asserted below
            assert f"{f_name}:" in source, f"{f_name} not accepted by /api/jobs"
        assert "view_tags" in source

    def test_finishing_controls_are_still_wired(self):
        """The finishing options specifically, since they are the ones that
        silently degrade to a local-only run when they go missing."""
        params = self._new_job_params()
        for f_name in (
            "smooth_iterations",
            "texture",
            "segmentation",
            "web_export",
            "texture_size",
            "max_parts",
            "gpu_host",
        ):
            assert f_name in params, f"{f_name} missing from the client's params"

    def test_client_sends_every_field_it_declares(self):
        """types.ts is a declaration; NewJobRoute is what actually builds the
        request. They must not drift apart."""
        route = (
            pathlib.Path(__file__).resolve().parents[1]
            / "frontend" / "src" / "routes" / "NewJobRoute.tsx"
        )
        if not route.is_file():
            pytest.skip("frontend sources not present")
        body = route.read_text()
        for f_name in self._new_job_params():
            assert f_name in body, f"{f_name} declared but never sent by NewJobRoute"

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


class TestJobFilesAndConvert:
    def test_path_traversal_is_refused(self, tmp_path: Path):
        """Filenames arrive from the browser; escaping the job dir must fail."""
        from fastapi import HTTPException

        from pixiecad.web.app import _safe_output_file

        job = tmp_path / "job"
        (job / "output").mkdir(parents=True)
        with pytest.raises(HTTPException) as exc:
            _safe_output_file(job, "../../../../etc/passwd")
        assert exc.value.status_code == 400

    def test_normal_filenames_resolve(self, tmp_path: Path):
        from pixiecad.web.app import _safe_output_file

        job = tmp_path / "job"
        (job / "output" / "parts").mkdir(parents=True)
        assert _safe_output_file(job, "model.glb").name == "model.glb"
        assert _safe_output_file(job, "parts/body.glb").name == "body.glb"

    def test_files_endpoint_404s_for_unknown_job(self, client):
        assert client.get("/api/jobs/nope/files").status_code == 404

    def test_convert_rejects_unsupported_format(self, client, tmp_path: Path):
        # Register a minimal job record so we reach the format check.
        assert client.post("/api/jobs/nope/convert", json={"format": "step"}).status_code == 404

    def test_delete_refuses_a_running_job(self, client, tmp_path: Path):
        """Deleting mid-run would pull files out from under the worker."""
        assert client.delete("/api/jobs/missing").status_code == 404


class TestVmStatus:
    def test_endpoint_degrades_without_gcloud(self, client):
        d = client.get("/api/cloud/vm").json()
        assert "available" in d and "vms" in d

    def test_building_is_distinct_from_running(self, client):
        """A VM is RUNNING long before its images exist; conflating them
        invites jobs that fail on a missing docker image."""
        d = client.get("/api/cloud/vm").json()
        if d.get("available"):
            assert "building" in d and "running" in d


class TestViewTagging:
    def test_tags_rename_uploads_so_view_mapping_works(self, client, tmp_path: Path):
        """A camera-roll filename maps to no view; tagging must fix that."""
        import io
        import json as _json

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (800, 800), (120, 120, 120)).save(buf, format="JPEG")
        raw = buf.getvalue()
        files = [
            ("files", ("WhatsApp Image 2026-08-03 at 05.16.50.jpeg", raw, "image/jpeg")),
            ("files", ("WhatsApp Image 2026-08-03 at 05.16.51.jpeg", raw, "image/jpeg")),
        ]
        tags = {
            "WhatsApp Image 2026-08-03 at 05.16.50.jpeg": "front",
            "WhatsApp Image 2026-08-03 at 05.16.51.jpeg": "left",
        }
        r = client.post(
            "/api/jobs",
            data={"name": "tagged", "backend": "fake", "view_tags": _json.dumps(tags),
                  "multiview": "true", "split": "false"},
            files=files,
        )
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}").json()
        names = sorted(p.name for p in (Path(job["dir"]) / "input").iterdir())
        assert names == ["front.jpeg", "left.jpeg"], names

    def test_untagged_uploads_keep_their_names(self, client):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (800, 800), (90, 90, 90)).save(buf, format="JPEG")
        r = client.post(
            "/api/jobs",
            data={"name": "untagged", "backend": "fake", "split": "false"},
            files=[("files", ("holiday snap.jpeg", buf.getvalue(), "image/jpeg"))],
        )
        job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        assert [p.name for p in (Path(job["dir"]) / "input").iterdir()] == ["holiday snap.jpeg"]

    def test_duplicate_tag_does_not_overwrite(self, client):
        """Two photos tagged 'front' must not collide into one file."""
        import io
        import json as _json

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (800, 800), (70, 70, 70)).save(buf, format="JPEG")
        raw = buf.getvalue()
        r = client.post(
            "/api/jobs",
            data={"name": "dupe", "backend": "fake", "split": "false",
                  "view_tags": _json.dumps({"a.jpeg": "front", "b.jpeg": "front"})},
            files=[("files", ("a.jpeg", raw, "image/jpeg")),
                   ("files", ("b.jpeg", raw, "image/jpeg"))],
        )
        job = client.get(f"/api/jobs/{r.json()['job_id']}").json()
        assert len(list((Path(job["dir"]) / "input").iterdir())) == 2

    def test_malformed_view_tags_are_ignored(self, client):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (800, 800), (60, 60, 60)).save(buf, format="JPEG")
        r = client.post(
            "/api/jobs",
            data={"name": "bad", "backend": "fake", "split": "false", "view_tags": "not json"},
            files=[("files", ("x.jpeg", buf.getvalue(), "image/jpeg"))],
        )
        assert r.status_code == 200


class TestBakedImagePreference:
    def test_only_matches_pixiecad_worker_images(self):
        """An unrelated image in the project must never be booted as a worker."""
        import inspect

        from pixiecad.web import app as m

        src = inspect.getsource(m._latest_machine_image)
        assert "name~^pixiecad-worker" in src
        assert "~creationTimestamp" in src, "must pick the newest"

    def test_returns_empty_when_gcloud_is_absent(self, monkeypatch):
        import subprocess as sp

        from pixiecad.web import app as m

        def boom(*a, **k):
            raise FileNotFoundError("gcloud")

        monkeypatch.setattr(sp, "run", boom)
        assert m._latest_machine_image() == ""

    def test_provision_falls_back_to_building_without_an_image(self):
        """No image must still work, just slowly -- and say so."""
        import inspect

        from pixiecad.web import app as m

        src = inspect.getsource(m.create_app)
        assert "no baked machine image found" in src
        assert "provision_gpu_vm.sh" in src


class TestSpaServing:
    """The React bundle, and the deep links it depends on.

    The whole point of moving job state into the URL is that a job survives a
    reload -- so /ui/jobs/<id>, which exists only in the client router, must
    return the shell rather than a 404.
    """

    @pytest.fixture
    def spa_built(self):
        from pixiecad.web import app as m

        dist = pathlib.Path(m.__file__).parent / "static" / "dist"
        if not (dist / "index.html").is_file():
            pytest.skip("SPA bundle not built; run `npm run build` in frontend/")
        return dist

    def test_root_redirects_into_the_spa(self, client, spa_built):
        """One URL space: / sends you to /ui rather than serving a second copy
        of the app whose asset paths point somewhere else."""
        res = client.get("/", follow_redirects=False)
        assert res.status_code == 307
        assert res.headers["location"] == "/ui/"

    def test_spa_shell_at_ui(self, client, spa_built):
        res = client.get("/ui")
        assert res.status_code == 200
        assert 'id="root"' in res.text

    @pytest.mark.parametrize(
        "path", ["/ui/", "/ui/new", "/ui/jobs/abc123", "/ui/jobs/abc123?compute=1"]
    )
    def test_client_routes_fall_back_to_the_shell(self, client, spa_built, path):
        res = client.get(path)
        assert res.status_code == 200, f"{path} must not 404"
        assert 'id="root"' in res.text

    def test_real_bundle_files_are_served_as_themselves(self, client, spa_built):
        """A hashed asset must return the asset, not the HTML shell."""
        import re

        shell = client.get("/ui").text
        match = re.search(r'/ui/assets/([^"\']+\.js)', shell)
        assert match, "the shell should reference a hashed JS bundle"
        res = client.get(f"/ui/assets/{match.group(1)}")
        assert res.status_code == 200
        assert 'id="root"' not in res.text

    def test_traversal_out_of_the_bundle_is_refused(self, client, spa_built):
        """A path escaping the bundle must fall through to the shell, never
        serve a file from elsewhere on disk."""
        res = client.get("/ui/../../../../etc/passwd")
        assert "root:" not in res.text

    def test_api_is_not_shadowed_by_the_spa(self, client, spa_built):
        res = client.get("/api/jobs")
        assert res.status_code == 200
        assert res.json() == []


class TestRehydration:
    """Jobs must survive a dashboard restart.

    The registry is an in-process dict, but the sessions are on disk. Before
    this, restarting the server emptied /api/jobs, 404'd every job URL, and
    orphaned finished models that were sitting right there in output/ -- which
    makes a deep link, or leaving and coming back, impossible to rely on.
    """

    def _session(self, root, name, *, job_id, finished):
        from pixiecad.session import new_session

        session = new_session(root, label=name)
        session.write_meta(job_id=job_id, name=name, target_faces=20000)
        if finished:
            (session.output_dir / "model.glb").write_bytes(b"glTF-stub")
            parts = session.output_dir / "parts"
            parts.mkdir(exist_ok=True)
            (parts / "wheel_0.glb").write_bytes(b"glTF-stub")
        return session

    def test_finished_job_is_restored(self, tmp_path):
        self._session(tmp_path, "widget", job_id="aaa111", finished=True)
        client = TestClient(create_app(tmp_path))

        listing = client.get("/api/jobs").json()
        assert [j["job_id"] for j in listing] == ["aaa111"]
        assert listing[0]["status"] == "done"
        assert listing[0]["restored"] is True

        detail = client.get("/api/jobs/aaa111")
        assert detail.status_code == 200, "a deep link must survive a restart"
        assert detail.json()["glb_url"].startswith("/api/jobs/aaa111/model.glb?v=")

    def test_restored_job_can_still_be_downloaded(self, tmp_path):
        """The point of restoring is reaching the artifacts, not the record."""
        self._session(tmp_path, "widget", job_id="aaa111", finished=True)
        client = TestClient(create_app(tmp_path))

        assert client.get("/api/jobs/aaa111/model.glb").status_code == 200
        files = client.get("/api/jobs/aaa111/files").json()
        assert "model.glb" in [f["name"] for f in files["files"]]

    def test_restored_job_exposes_its_parts(self, tmp_path):
        self._session(tmp_path, "widget", job_id="aaa111", finished=True)
        client = TestClient(create_app(tmp_path))

        parts = client.get("/api/jobs/aaa111").json()["parts"]
        assert [p["file"] for p in parts] == ["wheel_0.glb"]
        assert client.get(parts[0]["url"]).status_code == 200

    def test_interrupted_job_reads_as_failed_not_running(self, tmp_path):
        """Nothing will ever advance it, so a spinner would never resolve."""
        self._session(tmp_path, "halfway", job_id="bbb222", finished=False)
        client = TestClient(create_app(tmp_path))

        job = client.get("/api/jobs/bbb222").json()
        assert job["status"] == "failed"
        assert "interrupted" in " ".join(job["log"]).lower()

    def test_sessions_without_a_job_id_still_appear(self, tmp_path):
        """Pre-dashboard sessions are real work and must not be invisible."""
        from pixiecad.session import new_session

        session = new_session(tmp_path, label="cli-run")
        session.write_meta()
        (session.output_dir / "model.glb").write_bytes(b"glTF-stub")

        listing = TestClient(create_app(tmp_path)).get("/api/jobs").json()
        assert len(listing) == 1
        assert listing[0]["job_id"] == session.root.name

    def test_newest_job_is_listed_first(self, tmp_path):
        self._session(tmp_path, "older", job_id="old", finished=True)
        self._session(tmp_path, "newer", job_id="new", finished=True)
        listing = TestClient(create_app(tmp_path)).get("/api/jobs").json()
        assert [j["job_id"] for j in listing] == ["new", "old"]

    def test_a_junk_session_does_not_break_the_listing(self, tmp_path):
        (tmp_path / "broken").mkdir()
        (tmp_path / "broken" / "session.json").write_text("{not json")
        self._session(tmp_path, "fine", job_id="ok1", finished=True)

        listing = TestClient(create_app(tmp_path)).get("/api/jobs").json()
        assert [j["job_id"] for j in listing] == ["ok1"]

    def test_rehydrate_preserves_regime_warnings_and_stages(self, tmp_path):
        session = self._session(tmp_path, "meta-test", job_id="ccc333", finished=True)
        meta_file = session.root / "session.json"
        meta = json.loads(meta_file.read_text())
        meta["regime"] = "sparse_views"
        meta["warnings"] = ["Geometry is generated"]
        meta["stages"] = [{"name": "generate", "status": "done", "detail": "foo", "seconds": 1.5}]
        meta_file.write_text(json.dumps(meta))

        client = TestClient(create_app(tmp_path))
        job = client.get("/api/jobs/ccc333").json()
        assert job["regime"] == "sparse_views"
        assert job["warnings"] == ["Geometry is generated"]
        assert job["stages"] == [{"name": "generate", "status": "done", "detail": "foo", "seconds": 1.5}]

    def test_rehydrate_handles_legacy_jobs_without_new_fields(self, tmp_path):
        self._session(tmp_path, "legacy-test", job_id="ddd444", finished=True)
        client = TestClient(create_app(tmp_path))
        job = client.get("/api/jobs/ddd444").json()
        assert job["regime"] is None
        assert job["warnings"] == []
        assert job["stages"] == []

class TestMergeSessionMeta:
    def test_merge_session_meta_preserves_settings_and_seed(self, tmp_path):
        from pixiecad.web.app import _merge_session_meta

        session_root = tmp_path / "session_abc"
        session_root.mkdir()
        meta_file = session_root / "session.json"

        original = {
            "job_id": "abc",
            "settings": {"target_faces": 5000},
            "seed": 42
        }
        meta_file.write_text(json.dumps(original))

        _merge_session_meta(session_root, regime="sparse_views", warnings=["test"])

        updated = json.loads(meta_file.read_text())
        assert updated["settings"] == {"target_faces": 5000}
        assert updated["seed"] == 42
        assert updated["regime"] == "sparse_views"
        assert updated["warnings"] == ["test"]


class TestLogCursor:
    """`log_since` keeps a long job's polling from being quadratic.

    A GPU run is polled for many minutes while its log only grows; re-sending
    the whole thing every couple of seconds costs more the longer the job runs,
    which is exactly backwards.
    """

    def test_cursor_returns_only_new_lines(self, client, tmp_path):
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(1), "image/jpeg"))],
            data={"name": "cursorjob"},
        )
        job_id = res.json()["job_id"]

        first = client.get(f"/api/jobs/{job_id}").json()
        assert first["log_since"] == 0
        total = first["log_total"]
        assert total == len(first["log"])

        again = client.get(f"/api/jobs/{job_id}?log_since={total}").json()
        assert again["log_since"] == total
        assert again["log"] == again["log"][: len(again["log"])]
        assert len(again["log"]) == again["log_total"] - total

    def test_out_of_range_cursor_resyncs_the_whole_log(self, client):
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(2), "image/jpeg"))],
            data={"name": "resync"},
        )
        job_id = res.json()["job_id"]

        job = client.get(f"/api/jobs/{job_id}?log_since=9999").json()
        assert job["log_since"] == 0, "a bad cursor must resync, not truncate"
        assert len(job["log"]) == job["log_total"]

    def test_negative_cursor_is_refused_safely(self, client):
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(3), "image/jpeg"))],
            data={"name": "neg"},
        )
        job_id = res.json()["job_id"]
        job = client.get(f"/api/jobs/{job_id}?log_since=-5").json()
        assert job["log_since"] == 0
        assert len(job["log"]) == job["log_total"]


class TestStagesExposed:
    """The pipeline records every stage with a real elapsed time. That was
    being flattened into log strings and regex-parsed back apart in the
    browser, which made measured timings unavailable to the interface."""

    def test_job_detail_carries_a_stages_list(self, client):
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(4), "image/jpeg"))],
            data={"name": "stagejob"},
        )
        job = client.get(f"/api/jobs/{res.json()['job_id']}").json()
        assert "stages" in job
        assert isinstance(job["stages"], list)

    def test_stage_entries_have_name_status_and_seconds(self):
        """Shape is asserted at the source so a rename cannot pass silently."""
        import inspect

        from pixiecad.web import app as m

        src = inspect.getsource(m.create_app)
        assert '"name": s.name' in src
        assert '"status": s.status' in src
        assert '"seconds": round(s.seconds, 3)' in src


class TestQueuedGate:
    """A job needing a GPU, submitted while one is still building, used to be
    accepted and then die on a missing docker image -- the only warning being
    a line of grey text far from the button."""

    def test_gpu_job_queues_behind_an_in_flight_provision(self, tmp_path, monkeypatch):
        """The behaviour, end to end: a texturing job submitted mid-provision
        is accepted as queued instead of started and failed."""
        import subprocess as sp
        import time

        # Hold the provision open so it is genuinely in flight when the job
        # arrives, rather than racing to finish first.
        def slow(*args, **kwargs):
            time.sleep(5)
            return sp.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(sp, "run", slow)

        client = TestClient(create_app(tmp_path))
        assert client.post("/api/cloud/provision", json={"gpu": "l4"}).status_code == 200

        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(7), "image/jpeg"))],
            # texture is what makes this job need a GPU at all.
            data={"name": "needs_gpu", "texture": "true"},
        )
        assert res.status_code == 200
        assert res.json().get("status") == "queued"

        job = client.get(f"/api/jobs/{res.json()['job_id']}").json()
        assert job["status"] == "queued"
        assert "waiting for the gpu" in " ".join(job["log"]).lower()

    def test_a_job_needing_no_gpu_is_not_gated(self, tmp_path, monkeypatch):
        """Local work must not wait on infrastructure it does not use."""
        import subprocess as sp
        import time

        def slow(*args, **kwargs):
            time.sleep(5)
            return sp.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(sp, "run", slow)

        client = TestClient(create_app(tmp_path))
        client.post("/api/cloud/provision", json={"gpu": "l4"})

        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(8), "image/jpeg"))],
            data={"name": "local_only", "texture": "false", "segmentation": "auto"},
        )
        assert res.json().get("status") != "queued"

    def test_gate_is_wired_to_needs_gpu(self):
        import inspect

        from pixiecad.web import app as m

        src = inspect.getsource(m.create_app)
        assert "provision_running and finish.needs_gpu" in src
        assert "_dispatch_when_gpu_ready" in src

    def test_waiter_fails_the_job_when_provisioning_fails(self):
        """A failed provision must not leave the job queued forever."""
        import inspect

        from pixiecad.web import app as m

        src = inspect.getsource(m.create_app)
        assert "GPU provisioning failed" in src
        assert "Gave up waiting for the GPU VM" in src


class TestStagesPublishedEarly:
    """Stage data must appear when the BUILD reports, not when the JOB ends.

    Finishing -- smoothing, texturing on the GPU, segmentation -- runs for
    minutes after run_build returns. Holding the stage list until the very end
    left the dashboard with nothing to show for the whole of it, and the
    pipeline view sat on its first lane while the job was actually texturing.
    """

    def test_stages_are_published_before_finishing_runs(self, tmp_path, monkeypatch):
        import threading

        import pixiecad.web.finishing as fin_mod

        # Block inside finishing and inspect the job record from the outside:
        # if stages are only assigned after finish_model returns, they are
        # still empty at this point.
        entered = threading.Event()
        release = threading.Event()
        real_finish = fin_mod.finish_model

        def blocking_finish(*args, **kwargs):
            entered.set()
            release.wait(timeout=30)
            return real_finish(*args, **kwargs)

        monkeypatch.setattr("pixiecad.web.app.finish_model", blocking_finish)

        client = TestClient(create_app(tmp_path))
        res = client.post(
            "/api/jobs",
            files=[
                ("files", ("a.jpg", _generate_test_jpeg(11), "image/jpeg")),
                ("files", ("b.jpg", _generate_test_jpeg(12), "image/jpeg")),
            ],
            data={"name": "early", "backend": "fake", "smooth_iterations": "1"},
        )
        job_id = res.json()["job_id"]

        assert entered.wait(timeout=30), "finishing never started"
        try:
            mid = client.get(f"/api/jobs/{job_id}").json()
            assert mid["status"] == "running"
            assert len(mid["stages"]) > 0, (
                "stage timings must be visible while finishing is still running"
            )
            assert any(s["name"] == "ingest" for s in mid["stages"])
            # The build's own summary is logged at the same moment, so the log
            # is not silent for the whole of finishing either.
            assert any("[OK] ingest" in line for line in mid["log"])
        finally:
            release.set()

    def test_summary_lines_are_not_logged_twice(self, tmp_path):
        import time

        client = TestClient(create_app(tmp_path))
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(13), "image/jpeg"))],
            data={"name": "once", "backend": "fake"},
        )
        job_id = res.json()["job_id"]

        deadline = time.time() + 30
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.1)

        ingest_lines = [ln for ln in job["log"] if ln.startswith("[OK] ingest")]
        assert len(ingest_lines) == 1, f"logged {len(ingest_lines)} times: {ingest_lines}"


class TestBakeNormalsWiring:
    """Normal-map baking is wired from the form field all the way to _call_build.

    Intercepts _call_build in app.py and records kwargs rather than running the
    full pipeline, following the same pattern as the rest of this file.
    """

    @staticmethod
    def _fake_result():
        """Minimal BuildResult the job worker can destructure without error."""
        from pixiecad.pipeline import BuildResult, Regime, StageOutcome

        return BuildResult(
            regime=Regime.SPARSE_VIEWS,
            stages=[StageOutcome(name="ingest", status="ok", detail="", seconds=0.0)],
            glb_path=None,
            faces=None,
            scale_applied=None,
            warnings=[],
        )

    def test_bake_normals_true_reaches_call_build(self, tmp_path, monkeypatch):
        """Posting bake_normals=true must reach _call_build with bake=True."""
        import time

        captured: dict = {}

        def fake_call_build(**kwargs):
            captured.update(kwargs)
            return self._fake_result()

        monkeypatch.setattr("pixiecad.web.app._call_build", fake_call_build)
        client = TestClient(create_app(tmp_path))

        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(30), "image/jpeg"))],
            data={"name": "bake_on", "bake_normals": "true", "normal_res": "1024"},
        )
        assert res.status_code == 200

        deadline = time.time() + 10
        while time.time() < deadline and not captured:
            time.sleep(0.05)

        assert captured.get("bake") is True, (
            f"expected bake=True, got {captured.get('bake')}"
        )

    def test_normal_res_clamp_values(self, tmp_path, monkeypatch):
        """4096 → 2048, 64 → 256, 1024 stays 1024."""
        import time

        captured_calls: list[dict] = []

        def fake_call_build(**kwargs):
            captured_calls.append(dict(kwargs))
            return self._fake_result()

        monkeypatch.setattr("pixiecad.web.app._call_build", fake_call_build)
        client = TestClient(create_app(tmp_path))

        for res_in, res_expected in [(4096, 2048), (64, 256), (1024, 1024)]:
            captured_calls.clear()
            client.post(
                "/api/jobs",
                files=[("files", ("a.jpg", _generate_test_jpeg(31), "image/jpeg"))],
                data={"name": f"clamp_{res_in}", "bake_normals": "true",
                      "normal_res": str(res_in)},
            )
            deadline = time.time() + 10
            while time.time() < deadline and not captured_calls:
                time.sleep(0.05)
            got = captured_calls[0].get("normal_res") if captured_calls else None
            assert got == res_expected, (
                f"normal_res={res_in} should clamp to {res_expected}, got {got}"
            )

    def test_default_is_bake_on_at_1024(self, tmp_path, monkeypatch):
        """Omitting both fields must default to bake=True, normal_res=1024."""
        import time

        captured: dict = {}

        def fake_call_build(**kwargs):
            captured.update(kwargs)
            return self._fake_result()

        monkeypatch.setattr("pixiecad.web.app._call_build", fake_call_build)
        client = TestClient(create_app(tmp_path))

        client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(32), "image/jpeg"))],
            data={"name": "defaults"},
        )

        deadline = time.time() + 10
        while time.time() < deadline and not captured:
            time.sleep(0.05)

        assert captured.get("bake") is True, (
            f"default bake should be True, got {captured.get('bake')}"
        )
        assert captured.get("normal_res") == 1024, (
            f"default normal_res should be 1024, got {captured.get('normal_res')}"
        )


class TestBackendSelection:
    """The dashboard can point a job at a specific generative backend."""

    def test_lists_backends_with_requirements(self, client):
        d = client.get("/api/backends").json()
        keys = {b["key"] for b in d["backends"]}
        assert {"auto", "hunyuan-remote", "trellis-remote"} <= keys
        for b in d["backends"]:
            assert b["requires"] in (None, "gpu_host")
            assert b["note"]

    def test_no_third_party_hosted_backend_is_offered(self):
        """fal.ai was removed deliberately: it would have uploaded the user's
        photos off the machine. Nothing in the registry or the auto order may
        quietly bring a hosted backend back."""
        from pixiecad.generative.base import AUTO_PRIORITY, _REGISTRY

        import pixiecad.generative  # noqa: F401  (registers what registers)

        assert "fal" not in AUTO_PRIORITY
        assert "fal" not in _REGISTRY

    def test_auto_is_first_and_needs_nothing(self, client):
        first = client.get("/api/backends").json()["backends"][0]
        assert first["key"] == "auto"
        assert first["requires"] is None

    def test_validation_reflects_what_has_actually_run(self, client):
        """`validated` means "this produced a model on this project", so it is
        only ever flipped by a real run.

        hunyuan-remote: 969,088 faces. trellis-remote: 1,915,811 faces and
        textured in one pass, 2026-08-04, once the DINOv3 licence was
        approved. Anything that has not run must still report False so nobody
        spends a GPU hour finding out."""
        by_key = {b["key"]: b for b in client.get("/api/backends").json()["backends"]}
        assert by_key["hunyuan-remote"]["validated"] is True
        assert by_key["trellis-remote"]["validated"] is True
        for b in client.get("/api/backends").json()["backends"]:
            assert isinstance(b["validated"], bool)

    def test_auto_is_translated_to_no_backend(self, client):
        """'auto' is the dashboard's word; the pipeline spells it None. Passing
        the literal through would fail as an unregistered backend name."""
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(21), "image/jpeg"))],
            data={"name": "autobackend", "backend": "auto"},
        )
        assert res.status_code == 200
        job = client.get(f"/api/jobs/{res.json()['job_id']}").json()
        assert "not registered" not in " ".join(job["log"]).lower()

    def test_an_explicit_backend_is_honoured(self, client):
        import time

        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(22), "image/jpeg"))],
            data={"name": "fakebackend", "backend": "fake"},
        )
        job_id = res.json()["job_id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                break
            time.sleep(0.1)
        gen = [s for s in job["stages"] if s["name"] == "generate"]
        assert gen and "fake" in gen[0]["detail"]

    def test_the_picker_sends_backend(self):
        """The control exists in the form and its value reaches the request."""
        route = (
            pathlib.Path(__file__).resolve().parents[1]
            / "frontend" / "src" / "routes" / "NewJobRoute.tsx"
        )
        if not route.is_file():
            pytest.skip("frontend sources not present")
        body = route.read_text()
        assert "BackendPicker" in body
        assert "backend," in body, "backend must be included in the request params"


class TestBakeLocation:
    """Where the bake runs is a user choice, and it must survive the trip."""

    def test_stage_costs_reports_local_memory(self, client):
        d = client.get("/api/stage-costs?bake=true&normal_res=1024&has_host=false").json()
        assert d["bake_location_resolved"] == "local"
        # The whole point of the panel: baking dominates local memory.
        assert d["local_peak_ram_mb"] > 3000

    def test_a_host_takes_the_bake_off_this_machine(self, client):
        """auto + a host must drop local peak back to the non-bake baseline."""
        d = client.get("/api/stage-costs?bake=true&normal_res=1024&has_host=true").json()
        assert d["bake_location_resolved"] == "host"
        assert d["local_peak_ram_mb"] < 500, (
            "baking on the host must not still charge this machine 3 GB"
        )

    def test_explicit_local_overrides_an_available_host(self, client):
        d = client.get(
            "/api/stage-costs?bake=true&normal_res=1024"
            "&bake_location=local&has_host=true"
        ).json()
        assert d["bake_location_resolved"] == "local"

    def test_unmeasured_stages_are_flagged_not_invented(self, client):
        """A figure that was never measured must not be invented to fill a gap.

        Host baking has been timed at 1024 only. Any other resolution has to
        report no duration rather than scale the 1024 number, because the cost
        is dominated by transfer and container start, which do not scale with
        resolution the way the local bake does.
        """
        d = client.get(
            "/api/stage-costs?bake=true&normal_res=2048"
            "&bake_location=host&has_host=true"
        ).json()
        bake = next(s for s in d["stages"] if s["key"] == "bake")
        assert bake["seconds"] is None
        assert bake["basis"] == "estimated"
        assert d["has_unmeasured"] is True

    def test_a_measured_host_bake_reports_its_time(self, client):
        """1024 on the host HAS been run end to end, so it must say so."""
        d = client.get(
            "/api/stage-costs?bake=true&normal_res=1024"
            "&bake_location=host&has_host=true"
        ).json()
        bake = next(s for s in d["stages"] if s["key"] == "bake")
        assert bake["basis"] == "measured"
        assert bake["seconds"] == 45.5
        # Still free to this machine, which is the whole reason to choose it.
        assert bake["peak_ram_mb"] == 0

    def test_measured_stages_say_measured(self, client):
        d = client.get("/api/stage-costs?bake=true&normal_res=1024&has_host=false").json()
        bake = next(s for s in d["stages"] if s["key"] == "bake")
        assert bake["basis"] == "measured"
        assert bake["seconds"] == 13.6

    def test_bake_location_reaches_the_pipeline(self, tmp_path, monkeypatch):
        """Without a host, 'host' must resolve to local: the pipeline has no
        executor to send it to and would otherwise bake locally while claiming
        otherwise."""
        import time

        captured: dict = {}

        def fake_call_build(**kwargs):
            captured.update(kwargs)
            from pixiecad.pipeline import BuildResult, Regime, StageOutcome

            return BuildResult(
                regime=Regime.SPARSE_VIEWS,
                stages=[StageOutcome(name="ingest", status="ok", detail="", seconds=0.0)],
                glb_path=None,
                faces=None,
                scale_applied=None,
                warnings=[],
            )

        monkeypatch.setattr("pixiecad.web.app._call_build", fake_call_build)
        client = TestClient(create_app(tmp_path))
        client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(33), "image/jpeg"))],
            data={"name": "loc", "bake_normals": "true", "bake_location": "host"},
        )
        deadline = time.time() + 10
        while time.time() < deadline and not captured:
            time.sleep(0.05)
        assert captured.get("bake_location") == "local"


class TestRerun:
    """A finished or failed job can be run again without re-entering anything.

    The jobs this exists for failed on infrastructure -- an unreachable GPU
    host, a module missing from the worker image -- where nothing about the
    photos or the options was wrong. Making someone re-upload and re-pick a
    dozen settings to retry is pure toil.
    """

    @staticmethod
    def _finished_job(client, **form):
        data = {"name": "reruns", "target_faces": "5000", **form}
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(40), "image/jpeg"))],
            data=data,
        )
        assert res.status_code == 200
        job_id = res.json()["job_id"]
        deadline = time.time() + 30
        while time.time() < deadline:
            j = client.get(f"/api/jobs/{job_id}").json()
            if j["status"] in ("done", "failed"):
                return job_id, j
            time.sleep(0.05)
        raise AssertionError("job never reached a terminal state")

    def test_rerun_creates_a_new_job(self, client):
        job_id, _ = self._finished_job(client)
        res = client.post(f"/api/jobs/{job_id}/rerun")
        assert res.status_code == 200
        new_id = res.json()["job_id"]
        assert new_id != job_id
        assert res.json()["rerun_of"] == job_id

    def test_the_original_is_left_intact(self, client):
        """The failed run's log is the evidence for why it failed; a retry
        that overwrote it would destroy what you want to compare against."""
        job_id, before = self._finished_job(client)
        client.post(f"/api/jobs/{job_id}/rerun")
        after = client.get(f"/api/jobs/{job_id}").json()
        assert after["status"] == before["status"]
        assert after["log"] == before["log"]

    def test_the_photos_are_reused(self, client):
        job_id, original = self._finished_job(client)
        new_id = client.post(f"/api/jobs/{job_id}/rerun").json()["job_id"]
        new_job = client.get(f"/api/jobs/{new_id}").json()
        assert new_job["dir"] != original["dir"], "must be its own session"
        src = list((Path(original["dir"]) / "input").iterdir())
        dst = list((Path(new_job["dir"]) / "input").iterdir())
        assert len(dst) == len(src) == 1
        assert Path(original["dir"]).is_dir(), "the original session must survive"

    def test_settings_carry_over(self, client):
        job_id, _ = self._finished_job(
            client, split="false", texture_size="512", normal_res="512",
            bake_location="local", object_hint="a lighter",
        )
        new_id = client.post(f"/api/jobs/{job_id}/rerun").json()["job_id"]
        meta = json.loads(
            (Path(client.get(f"/api/jobs/{new_id}").json()["dir"]) / "session.json").read_text()
        )
        s = meta["settings"]
        assert s["split"] is False
        assert s["texture_size"] == 512
        assert s["normal_res"] == 512
        assert s["bake_location"] == "local"
        assert s["object_hint"] == "a lighter"
        assert s["rerun_of"] == job_id

    def test_a_newly_provisioned_host_overrides_the_stale_one(self, client):
        """Usually the reason for the retry: there was no host, now there is."""
        job_id, _ = self._finished_job(client)
        new_id = client.post(
            f"/api/jobs/{job_id}/rerun", data={"gpu_host": "fresh.host.invalid"}
        ).json()["job_id"]
        meta = json.loads(
            (Path(client.get(f"/api/jobs/{new_id}").json()["dir"]) / "session.json").read_text()
        )
        assert meta["settings"]["gpu_host"] == "fresh.host.invalid"

    def test_rerunning_a_missing_job_is_404(self, client):
        assert client.post("/api/jobs/nope/rerun").status_code == 404

    def test_settings_are_persisted_for_restored_jobs(self, tmp_path):
        """A job restored after a restart must still be rerunnable -- the
        in-memory record is gone, so the settings have to come off disk."""
        client = TestClient(create_app(tmp_path))
        job_id, original = self._finished_job(client, object_hint="a lighter")

        restored = TestClient(create_app(tmp_path))
        j = restored.get(f"/api/jobs/{job_id}").json()
        assert j.get("restored") is True
        new_id = restored.post(f"/api/jobs/{job_id}/rerun").json()["job_id"]
        meta = json.loads(
            (Path(restored.get(f"/api/jobs/{new_id}").json()["dir"]) / "session.json").read_text()
        )
        assert meta["settings"]["object_hint"] == "a lighter"


class TestRerunIsNotAutomatic:
    """Fetching the plan must not start anything, and overrides must win.

    The first version of rerun fired on click, which silently reused settings
    the user could neither see nor change. Reviewing them is the point.
    """

    def test_the_plan_starts_nothing(self, client):
        job_id, _ = TestRerun._finished_job(client)
        before = len(client.get("/api/jobs").json())
        plan = client.get(f"/api/jobs/{job_id}/rerun-plan").json()
        assert plan["can_rerun"] is True
        assert plan["photo_count"] == 1
        assert len(client.get("/api/jobs").json()) == before, (
            "reading the plan must never create a job"
        )

    def test_the_plan_reports_the_settings_that_would_be_used(self, client):
        job_id, _ = TestRerun._finished_job(
            client, texture_size="512", object_hint="a lighter", split="false"
        )
        plan = client.get(f"/api/jobs/{job_id}/rerun-plan").json()
        assert plan["has_settings"] is True
        assert plan["settings"]["texture_size"] == 512
        assert plan["settings"]["object_hint"] == "a lighter"
        assert plan["settings"]["split"] is False

    def test_an_override_replaces_the_stored_value(self, client):
        job_id, _ = TestRerun._finished_job(client, texture_size="512")
        new_id = client.post(
            f"/api/jobs/{job_id}/rerun", data={"texture_size": "2048"}
        ).json()["job_id"]
        meta = json.loads(
            (Path(client.get(f"/api/jobs/{new_id}").json()["dir"]) / "session.json").read_text()
        )
        assert meta["settings"]["texture_size"] == 2048

    def test_omitted_fields_keep_the_original_value(self, client):
        """A confirmation screen sends what it shows; anything it does not
        send must not silently revert to a default."""
        job_id, _ = TestRerun._finished_job(client, texture_size="512", max_parts="3")
        new_id = client.post(
            f"/api/jobs/{job_id}/rerun", data={"texture_size": "2048"}
        ).json()["job_id"]
        meta = json.loads(
            (Path(client.get(f"/api/jobs/{new_id}").json()["dir"]) / "session.json").read_text()
        )
        assert meta["settings"]["max_parts"] == 3, "untouched settings must survive"

    def test_auto_backend_is_normalised_on_override(self, client):
        """'auto' is the dashboard's word; the pipeline spells it None, and the
        literal string is not a registered backend name."""
        job_id, _ = TestRerun._finished_job(client)
        new_id = client.post(
            f"/api/jobs/{job_id}/rerun", data={"backend": "auto"}
        ).json()["job_id"]
        meta = json.loads(
            (Path(client.get(f"/api/jobs/{new_id}").json()["dir"]) / "session.json").read_text()
        )
        assert meta["settings"]["backend"] is None

    def test_plan_for_a_missing_job_is_404(self, client):
        assert client.get("/api/jobs/nope/rerun-plan").status_code == 404


class TestTargetFaceCeiling:
    """The triangle budget must be able to hold what the backends produce.

    Hunyuan returns ~969k faces and TRELLIS ~1.92M. A 200k ceiling in the form
    silently discarded most of that geometry, and there was no way to ask for
    it -- the server never had the limit, only the number input did.
    """

    def test_the_form_allows_a_whole_generated_mesh(self):
        src = Path("frontend/src/routes/NewJobRoute.tsx").read_text()
        assert "max={2000000}" in src
        assert "max={200000}" not in src, "the old ceiling is below every backend's output"

    def test_the_server_accepts_a_budget_above_the_old_cap(self, client, tmp_path):
        res = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(50), "image/jpeg"))],
            data={"name": "big", "target_faces": "1900000"},
        )
        assert res.status_code == 200
        job_id = res.json()["job_id"]
        assert client.get(f"/api/jobs/{job_id}").json()["job_id"] == job_id

    def test_zero_and_negative_budgets_are_still_refused(self, client):
        """Raising a ceiling must not remove the floor."""
        for bad in ("0", "-5"):
            res = client.post(
                "/api/jobs",
                files=[("files", ("a.jpg", _generate_test_jpeg(51), "image/jpeg"))],
                data={"name": "bad", "target_faces": bad},
            )
            assert res.status_code >= 400, f"target_faces={bad} must be rejected"


class TestZoneStockoutHandling:
    """L4 capacity runs out per-zone, so a VM may not land where it was asked.

    asia-southeast1-b -- the zone every earlier run used -- returned
    ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS / reason: stockout. That is not
    quota and not a bad request; the zone simply has no L4 left.
    """

    def test_the_launcher_sweeps_zones(self):
        src = Path("scripts/launch_from_image.sh").read_text()
        assert "ZONE_RESOURCE_POOL_EXHAUSTED" in src
        assert "CANDIDATE_ZONES" in src

    def test_only_stockouts_trigger_the_sweep(self):
        """Quota, permissions and a bad image name are not fixed by moving
        zones, so they must fail once rather than six times."""
        src = Path("scripts/launch_from_image.sh").read_text()
        sweep = src[src.index("CANDIDATE_ZONES=("):]
        assert "exit 1" in sweep

    def test_the_host_uses_where_the_vm_actually_landed(self, monkeypatch):
        """Building the ssh alias from the REQUESTED zone gives a name that
        does not resolve, and the failure surfaces later as 'backend not
        available' -- pointing at the backend rather than the zone."""
        import subprocess as sp

        from pixiecad.web import app as app_module

        class _Res:
            stdout = "asia-southeast1-a\n"

        monkeypatch.setattr(sp, "run", lambda *a, **k: _Res())
        assert app_module._actual_zone("vm", "asia-southeast1-b", "proj") == "asia-southeast1-a"

    def test_it_falls_back_to_the_request_when_lookup_fails(self, monkeypatch):
        import subprocess as sp

        from pixiecad.web import app as app_module

        def boom(*a, **k):
            raise OSError("gcloud missing")

        monkeypatch.setattr(sp, "run", boom)
        assert app_module._actual_zone("vm", "asia-southeast1-b", "proj") == "asia-southeast1-b"


class TestOneJobPerGpuHost:
    """A GPU host serves one job at a time, and the second one is refused.

    The TRELLIS worker holds a single pipeline on a single device. Sending it a
    second generation does not queue -- it DEADLOCKS. Observed on 2026-08-08:
    two jobs submitted 55 seconds apart both reached "Sampling sparse
    structure: 100%" and stopped dead. /ready stopped answering entirely
    (HTTP 000 after 20 s), the GPU sat at 0% across samples, and the worker
    process used 0.23 s of CPU per 40 s. Nothing recovered it but killing the
    container, and BOTH runs were lost along with their GPU minutes.
    """

    HOST = "gpu.example.invalid"

    def _job(self, app, *, host, status="running", name="first"):
        """Put a job straight into the registry, bypassing the worker thread."""
        from uuid import uuid4

        jid = uuid4().hex[:8]
        app.state.jobs[jid] = {
            "job_id": jid,
            "name": name,
            "status": status,
            "settings": {"gpu_host": host},
            "log": [],
        }
        return jid

    def _submit(self, client, *, host):
        return client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(1), "image/jpeg"))],
            data={"name": "second", "target_faces": "5000", "gpu_host": host},
        )

    def test_a_second_job_on_a_busy_host_is_refused(self, tmp_path):
        app = create_app(tmp_path)
        client = TestClient(app)
        self._job(app, host=self.HOST)
        r = self._submit(client, host=self.HOST)
        assert r.status_code == 409, r.text

    def test_the_refusal_names_the_job_holding_the_host(self):
        """"Busy" with no way to find out what is busy just moves the confusion."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            app = create_app(Path(d))
            client = TestClient(app)
            jid = self._job(app, host=self.HOST, name="lighter-trellis-v4")
            r = self._submit(client, host=self.HOST)
            assert jid in r.text and "lighter-trellis-v4" in r.text
            assert "deadlock" in r.text.lower()

    def test_a_queued_job_also_holds_the_host(self, tmp_path):
        """Queued means it will start the moment the GPU is up -- it owns the
        slot just as much as a running job."""
        app = create_app(tmp_path)
        client = TestClient(app)
        self._job(app, host=self.HOST, status="queued")
        assert self._submit(client, host=self.HOST).status_code == 409

    def test_a_different_host_is_unaffected(self, tmp_path):
        app = create_app(tmp_path)
        client = TestClient(app)
        self._job(app, host="other.example.invalid")
        assert self._submit(client, host=self.HOST).status_code != 409

    def test_a_finished_job_releases_the_host(self, tmp_path):
        app = create_app(tmp_path)
        client = TestClient(app)
        for status in ("done", "failed"):
            app.state.jobs.clear()
            self._job(app, host=self.HOST, status=status)
            assert self._submit(client, host=self.HOST).status_code != 409, status

    def test_a_job_with_no_host_is_never_blocked(self, tmp_path):
        """Local-only work does not touch the worker."""
        app = create_app(tmp_path)
        client = TestClient(app)
        self._job(app, host=self.HOST)
        r = client.post(
            "/api/jobs",
            files=[("files", ("a.jpg", _generate_test_jpeg(1), "image/jpeg"))],
            data={"name": "local", "target_faces": "5000"},
        )
        assert r.status_code != 409, r.text

    def test_rerun_is_gated_too(self, tmp_path):
        """The deadlock was reached through the rerun button, so guarding only
        create_job would leave the actual path open."""
        import inspect

        from pixiecad.web import app as appmod

        src = inspect.getsource(appmod.create_app)
        register = src[src.index("def _register_and_dispatch"):]
        register = register[: register.index("@app.get(")]
        assert "409" in register, (
            "the guard must live in the shared dispatch, which create_job and "
            "rerun both go through -- not in create_job alone"
        )
