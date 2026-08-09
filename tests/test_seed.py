import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from pixiecad.web.app import create_app
from pixiecad.generative.base import get_backend, GenerateRequest
from pixiecad.workspace import Workspace, fingerprint_file
from pixiecad.spec import ObjectSpec
import numpy as np
import cv2
import trimesh
import time

def _generate_test_jpeg(seed: int = 0) -> bytes:
    img = np.zeros((700, 700, 3), dtype=np.uint8)
    grid_size = 50
    for i in range(0, 700, grid_size):
        for j in range(0, 700, grid_size):
            if ((i + j) // grid_size) % 2 == 0:
                img[i:i + grid_size, j:j + grid_size] = (200, 150, 100)
            else:
                img[i:i + grid_size, j:j + grid_size] = (50, 100, 200)
    # Draw a massive distinct shape to avoid deduplication
    cv2.circle(img, (350, 350), 50 + seed * 40, (255, 255, 255), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

@pytest.fixture
def client(tmp_path: Path):
    app = create_app(tmp_path)
    return TestClient(app)

def test_auto_generate_seed_if_missing(client, tmp_path):
    img = _generate_test_jpeg(1)
    res = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img, "image/jpeg"))],
        data={"name": "test_obj"},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    
    # Check session meta
    detail = client.get(f"/api/jobs/{job_id}").json()
    session_file = Path(detail["dir"]) / "session.json"
    meta = json.loads(session_file.read_text())
    seed = meta["settings"]["seed"]
    assert isinstance(seed, int)
    assert seed is not None

def test_explicit_seed_is_used(client, tmp_path):
    img = _generate_test_jpeg(1)
    res = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img, "image/jpeg"))],
        data={"name": "test_obj", "seed": 42},
    )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    
    detail = client.get(f"/api/jobs/{job_id}").json()
    session_file = Path(detail["dir"]) / "session.json"
    meta = json.loads(session_file.read_text())
    assert meta["settings"]["seed"] == 42

def test_cache_keys_with_seeds(tmp_path):
    ws = Workspace.create(tmp_path / "ws", ObjectSpec(name="test"))
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(_generate_test_jpeg())
    
    req1 = GenerateRequest(images=[img_path], seed=123)
    fp1 = [fingerprint_file(p) for p in req1.images]
    run1 = ws.begin_stage("s3-generate", {"backend": "fake", "seed": req1.seed, "texture": True, "options": {}}, fp1)
    
    req2 = GenerateRequest(images=[img_path], seed=123)
    fp2 = [fingerprint_file(p) for p in req2.images]
    run2 = ws.begin_stage("s3-generate", {"backend": "fake", "seed": req2.seed, "texture": True, "options": {}}, fp2)
    
    req3 = GenerateRequest(images=[img_path], seed=456)
    fp3 = [fingerprint_file(p) for p in req3.images]
    run3 = ws.begin_stage("s3-generate", {"backend": "fake", "seed": req3.seed, "texture": True, "options": {}}, fp3)
    
    assert run1.key == run2.key
    assert run1.key != run3.key

def test_rerun_inherits_and_overrides_seed(client, tmp_path):
    img = _generate_test_jpeg(1)
    res1 = client.post(
        "/api/jobs",
        files=[("files", ("photo1.jpg", img, "image/jpeg"))],
        data={"name": "test_obj", "seed": 42},
    )
    job1_id = res1.json()["job_id"]
    
    for _ in range(50):
        if client.get(f"/api/jobs/{job1_id}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    
    # Rerun without passing seed
    res2 = client.post(f"/api/jobs/{job1_id}/rerun")
    assert res2.status_code == 200
    job2_id = res2.json()["job_id"]
    
    detail2 = client.get(f"/api/jobs/{job2_id}").json()
    meta2 = json.loads((Path(detail2["dir"]) / "session.json").read_text())
    assert meta2["settings"]["seed"] == 42
    
    for _ in range(50):
        if client.get(f"/api/jobs/{job2_id}").json()["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    
    # Rerun with new seed
    res3 = client.post(f"/api/jobs/{job1_id}/rerun", data={"seed": 99})
    assert res3.status_code == 200
    job3_id = res3.json()["job_id"]
    
    detail3 = client.get(f"/api/jobs/{job3_id}").json()
    meta3 = json.loads((Path(detail3["dir"]) / "session.json").read_text())
    assert meta3["settings"]["seed"] == 99

def test_fake_backend_determinism(tmp_path):
    backend = get_backend("fake")
    
    img_path = tmp_path / "dummy.jpg"
    img_path.touch()
    
    req1 = GenerateRequest(images=[img_path], seed=123)
    res1 = backend.generate(req1, tmp_path / "out1")
    mesh1 = trimesh.load(res1.mesh_path, force="mesh", process=False)
    
    req2 = GenerateRequest(images=[img_path], seed=123)
    res2 = backend.generate(req2, tmp_path / "out2")
    mesh2 = trimesh.load(res2.mesh_path, force="mesh", process=False)
    
    req3 = GenerateRequest(images=[img_path], seed=999)
    res3 = backend.generate(req3, tmp_path / "out3")
    mesh3 = trimesh.load(res3.mesh_path, force="mesh", process=False)
    
    assert np.allclose(mesh1.vertices, mesh2.vertices)
    assert np.array_equal(mesh1.faces, mesh2.faces)
    assert mesh1.vertices.shape != mesh3.vertices.shape or not np.allclose(mesh1.vertices, mesh3.vertices)

def test_unused_photos_are_named_in_the_warnings():
    """Photos are dropped by the BACKEND, not by view selection.

    _select_conditioning_views hands over up to four; on the single-view path
    the backend keeps ONE and the other three never touch the geometry. So the
    drop happens after selection and only the backend knows -- it records the
    answer in result.metadata["conditioning_views"]. A tetrapod that came out
    well and one that did not differed by exactly which photo was kept, and
    the log gave no way to tell.
    """
    import inspect

    from pixiecad import pipeline

    src = inspect.getsource(pipeline._run_generative)
    assert "conditioning_views" in src, "the warning must read what the backend used"
    assert "were not used for geometry" in src
    # It must NOT be driven by the selection step, which drops nothing in the
    # common 4-photo case and would leave the real loss silent.
    warn_block = src[src.index("conditioning_views"):]
    assert "len(all_images)" in warn_block


def test_the_conditioning_limit_is_unchanged():
    """TRELLIS conditions on up to FOUR images and needs no tags. An earlier
    attempt at the warning cut the limit to 1 unless multiview was on, which
    would have quietly crippled it."""
    import inspect

    from pixiecad import pipeline

    assert pipeline._select_conditioning_views.__defaults__ == (4,)
    src = inspect.getsource(pipeline._run_generative)
    assert "_select_conditioning_views(all_images)" in src
