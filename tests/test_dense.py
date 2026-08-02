"""Tests for Stage 2b dense reconstruction backend."""

from pathlib import Path

import pytest
import trimesh

from pixiecad.executors import Capabilities, GPUInfo, Job, JobResult
from pixiecad.geometry import (
    COLMAP_MIN_VRAM_MB,
    DenseResult,
    DenseUnavailable,
    build_dense_script,
    docker_prefix,
    run_dense,
)
from pixiecad.spec import ObjectSpec
from pixiecad.workspace import Workspace


class FakeExecutor:
    def __init__(
        self,
        caps: Capabilities,
        run_ok: bool = True,
        stderr_tail: str = "",
    ):
        self.caps = caps
        self.run_ok = run_ok
        self.stderr_tail = stderr_tail
        self.recorded_jobs: list[Job] = []
        self.run_call_count = 0

    def probe(self) -> Capabilities:
        return self.caps

    def run(self, job: Job) -> JobResult:
        self.run_call_count += 1
        self.recorded_jobs.append(job)
        if self.run_ok:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            mesh_path = job.output_dir / "mesh.ply"
            box = trimesh.creation.box()
            box.export(str(mesh_path))
            return JobResult(
                returncode=0,
                stdout_tail="dense reconstruction finished successfully",
                stderr_tail="",
                duration_s=5.0,
            )
        else:
            return JobResult(
                returncode=1,
                stdout_tail="",
                stderr_tail=self.stderr_tail or "Dense reconstruction error",
                duration_s=1.0,
            )


@pytest.fixture
def workspace(tmp_path):
    spec = ObjectSpec(name="test_dense_obj")
    return Workspace.create(tmp_path / "ws", spec)


@pytest.fixture
def sample_sparse_inputs(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "img1.jpg").write_bytes(b"dummy_image_1")
    (images_dir / "img2.jpg").write_bytes(b"dummy_image_2")

    sparse_model_dir = tmp_path / "sparse"
    sparse_model_dir.mkdir(parents=True, exist_ok=True)
    (sparse_model_dir / "cameras.bin").write_bytes(b"dummy_bin_cameras")
    (sparse_model_dir / "images.bin").write_bytes(b"dummy_bin_images")
    (sparse_model_dir / "points3D.bin").write_bytes(b"dummy_bin_points")

    return images_dir, sparse_model_dir


def test_build_dense_script():
    script = build_dense_script()
    assert "colmap image_undistorter --image_path images --input_path sparse --output_path dense" in script
    assert "colmap patch_match_stereo --workspace_path dense" in script
    assert "colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply" in script
    assert "mkdir -p out" in script
    assert "colmap poisson_mesher --input_path dense/fused.ply --output_path out/mesh.ply" in script

    # COLMAP stages run in dependency order; `mkdir -p out` comes first so the
    # output dir belongs to the login user rather than root-in-container.
    idx_mkdir = script.index("mkdir -p out")
    idx1 = script.index("colmap image_undistorter")
    idx2 = script.index("colmap patch_match_stereo")
    idx3 = script.index("colmap stereo_fusion")
    idx5 = script.index("colmap poisson_mesher")
    assert idx_mkdir < idx1 < idx2 < idx3 < idx5


def test_build_dense_script_docker():
    """Docker mode wraps every COLMAP call but leaves mkdir on the host."""
    script = build_dense_script(colmap_cmd=f"{docker_prefix()} colmap")

    assert script.count("docker run --rm --gpus all") == 4  # one per colmap stage
    assert "colmap/colmap" in script
    # mkdir must NOT be containerised
    mkdir_segment = next(s for s in script.split(" && ") if "mkdir" in s)
    assert "docker" not in mkdir_segment


def test_no_gpu_probe(workspace, sample_sparse_inputs):
    images_dir, sparse_model_dir = sample_sparse_inputs
    caps = Capabilities(hostname="local", gpu=None, reachable=True)
    executor = FakeExecutor(caps)

    with pytest.raises(DenseUnavailable) as exc_info:
        run_dense(images_dir, sparse_model_dir, workspace, executor)

    assert "VRAM" in str(exc_info.value)
    assert "local" in str(exc_info.value)


def test_unreachable_probe(workspace, sample_sparse_inputs):
    images_dir, sparse_model_dir = sample_sparse_inputs
    caps = Capabilities(
        hostname="remote-box",
        gpu=GPUInfo(name="RTX 3090", vram_mb=24000, compute_cap=8.6),
        reachable=False,
    )
    executor = FakeExecutor(caps)

    with pytest.raises(DenseUnavailable) as exc_info:
        run_dense(images_dir, sparse_model_dir, workspace, executor)

    assert "reachable=False" in str(exc_info.value)
    assert "remote-box" in str(exc_info.value)


def test_low_vram_probe(workspace, sample_sparse_inputs):
    images_dir, sparse_model_dir = sample_sparse_inputs
    caps = Capabilities(
        hostname="old-gpu-box",
        gpu=GPUInfo(name="GTX 1050", vram_mb=2000, compute_cap=6.1),
        reachable=True,
    )
    executor = FakeExecutor(caps)

    with pytest.raises(DenseUnavailable) as exc_info:
        run_dense(images_dir, sparse_model_dir, workspace, executor)

    assert "VRAM" in str(exc_info.value)
    assert "2000" in str(exc_info.value)
    assert str(COLMAP_MIN_VRAM_MB) in str(exc_info.value)


def test_happy_path_and_cache(workspace, sample_sparse_inputs):
    images_dir, sparse_model_dir = sample_sparse_inputs
    caps = Capabilities(
        hostname="cuda-box",
        gpu=GPUInfo(name="RTX 3080", vram_mb=10000, compute_cap=8.6),
        reachable=True,
    )
    executor = FakeExecutor(caps)

    res1 = run_dense(images_dir, sparse_model_dir, workspace, executor)

    assert isinstance(res1, DenseResult)
    assert res1.cached is False
    assert res1.n_faces == 12
    assert Path(res1.mesh_path).exists()
    assert executor.run_call_count == 1

    recorded_job = executor.recorded_jobs[0]
    # run_dense defaults to docker (distro COLMAP has no CUDA)
    assert recorded_job.command[:2] == ["sh", "-c"]
    assert recorded_job.command[2] == build_dense_script(
        colmap_cmd=f"{docker_prefix()} colmap"
    )
    assert recorded_job.inputs == [images_dir, sparse_model_dir]

    script = recorded_job.command[2]
    assert "colmap image_undistorter" in script
    assert "colmap patch_match_stereo" in script
    assert "colmap stereo_fusion" in script
    assert "colmap poisson_mesher" in script

    # Second call must hit cache without invoking executor.run again
    res2 = run_dense(images_dir, sparse_model_dir, workspace, executor)
    assert res2.cached is True
    assert res2.n_faces == 12
    assert res2.mesh_path == res1.mesh_path
    assert executor.run_call_count == 1


def test_job_failure_raises_dense_unavailable(workspace, sample_sparse_inputs):
    images_dir, sparse_model_dir = sample_sparse_inputs
    caps = Capabilities(
        hostname="cuda-box",
        gpu=GPUInfo(name="RTX 3080", vram_mb=10000, compute_cap=8.6),
        reachable=True,
    )
    executor = FakeExecutor(caps, run_ok=False, stderr_tail="PatchMatchStereo CUDA error")

    with pytest.raises(DenseUnavailable) as exc_info:
        run_dense(images_dir, sparse_model_dir, workspace, executor)

    assert "PatchMatchStereo CUDA error" in str(exc_info.value)
