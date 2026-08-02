"""Tests for TRELLIS remote generative backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import trimesh

from pixiecad.executors.base import Capabilities, GPUInfo, Job, JobResult
from pixiecad.generative.trellis_remote import (
    DEFAULT_TRELLIS_IMAGE,
    TRELLIS_MIN_VRAM_MB,
    RemoteTrellisBackend,
    _get_generative_base,
    build_trellis_script,
    docker_prefix,
    make_backend,
)

GenerateResult, GenerativeError, BackendUnavailable = _get_generative_base()

try:
    from pixiecad.generative.base import GenerateRequest
except ImportError:
    @dataclass
    class GenerateRequest:  # type: ignore[no-redef]
        images: list[Path]
        seed: int | None = None
        texture: bool = True
        options: dict = field(default_factory=dict)


class FakeExecutor:
    """Fake executor recording jobs without network or docker calls."""

    def __init__(
        self,
        caps: Capabilities | None = None,
        run_result: JobResult | None = None,
        write_mesh: bool = True,
        raise_on_probe: bool = False,
    ) -> None:
        self.caps = caps
        self.run_result = run_result
        self.write_mesh = write_mesh
        self.raise_on_probe = raise_on_probe
        self.jobs: list[Job] = []
        self.recorded_inputs: list[list[str]] = []

    def probe(self) -> Capabilities:
        if self.raise_on_probe:
            raise RuntimeError("SSH connection failed")
        if self.caps is None:
            return Capabilities(hostname="fake-host", gpu=None, reachable=False)
        return self.caps

    def run(self, job: Job) -> JobResult:
        self.jobs.append(job)

        # Save snapshot of files in job inputs before temp dir cleanup
        saved_names = []
        for inp in job.inputs:
            if inp.is_dir():
                for f in inp.iterdir():
                    saved_names.append(f.name)
        self.recorded_inputs.append(saved_names)

        if self.write_mesh and job.output_dir:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            mesh_file = job.output_dir / "mesh.glb"
            glb_bytes = trimesh.creation.icosphere().export(file_type="glb")
            mesh_file.write_bytes(glb_bytes)

        if self.run_result is not None:
            return self.run_result

        return JobResult(returncode=0, stdout_tail="", stderr_tail="", duration_s=0.5)


def test_build_trellis_script() -> None:
    script_default = build_trellis_script()
    assert "mkdir -p out" in script_default
    assert script_default.startswith("mkdir -p out && ")
    assert docker_prefix(DEFAULT_TRELLIS_IMAGE) in script_default
    assert DEFAULT_TRELLIS_IMAGE in script_default
    assert "--seed" not in script_default
    assert "--no-texture" not in script_default

    # Ensure mkdir segment is NOT containerised
    parts = script_default.split(" && ")
    assert parts[0] == "mkdir -p out"
    assert parts[1].startswith("docker run ")

    # Seed flag testing
    script_seed = build_trellis_script(seed=7)
    assert "--seed 7" in script_seed

    # Texture flag testing
    script_no_tex = build_trellis_script(texture=False)
    assert "--no-texture" in script_no_tex

    # Custom image testing
    custom_img = "myrepo/trellis:v2"
    script_custom = build_trellis_script(image=custom_img)
    assert custom_img in script_custom


def test_available() -> None:
    high_gpu = GPUInfo(name="A100", vram_mb=40000, compute_cap=8.0)
    low_gpu = GPUInfo(name="T4", vram_mb=16000, compute_cap=7.5)

    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    caps_low_vram = Capabilities(hostname="gpu-node", gpu=low_gpu, reachable=True)
    caps_unreachable = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=False)

    exec_ok = FakeExecutor(caps=caps_ok)
    exec_low_vram = FakeExecutor(caps=caps_low_vram)
    exec_unreachable = FakeExecutor(caps=caps_unreachable)
    exec_error = FakeExecutor(raise_on_probe=True)

    assert RemoteTrellisBackend(exec_ok).available() is True
    assert RemoteTrellisBackend(exec_low_vram).available() is False
    assert RemoteTrellisBackend(exec_unreachable).available() is False
    assert RemoteTrellisBackend(exec_error).available() is False


def test_generate_insufficient_vram(tmp_path: Path) -> None:
    low_gpu = GPUInfo(name="T4", vram_mb=16000, compute_cap=7.5)
    caps_low_vram = Capabilities(hostname="test-host", gpu=low_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_low_vram)

    backend = make_backend(executor)

    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"dummy image data")
    req = GenerateRequest(images=[img_file])

    out_dir = tmp_path / "out"
    with pytest.raises(BackendUnavailable) as exc_info:
        backend.generate(req, out_dir)

    msg = str(exc_info.value)
    assert "test-host" in msg
    assert str(16000) in msg
    assert str(TRELLIS_MIN_VRAM_MB) in msg


def test_generate_happy_path(tmp_path: Path) -> None:
    high_gpu = GPUInfo(name="A100", vram_mb=40000, compute_cap=8.0)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_ok, write_mesh=True)

    backend = RemoteTrellisBackend(executor)

    img1 = tmp_path / "photo.png"
    img2 = tmp_path / "photo2.jpg"
    img1.write_bytes(b"img1")
    img2.write_bytes(b"img2")

    req = GenerateRequest(images=[img1, img2], seed=42, texture=True)
    out_dir = tmp_path / "output"

    res = backend.generate(req, out_dir)

    assert res.backend == "trellis-remote"
    assert res.mesh_path == str(out_dir / "mesh.glb")
    assert res.n_faces is not None and res.n_faces > 0
    assert res.metadata["image"] == DEFAULT_TRELLIS_IMAGE
    assert res.metadata["remote_subdir"].startswith("trellis-")

    # Assert composed Job inputs contain directory with files named 00.* and 01.*
    assert len(executor.jobs) == 1
    recorded = executor.recorded_inputs[0]
    assert any(name.startswith("00.") for name in recorded)
    assert any(name.startswith("01.") for name in recorded)


def test_generate_executor_failure(tmp_path: Path) -> None:
    high_gpu = GPUInfo(name="A100", vram_mb=40000, compute_cap=8.0)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    fail_result = JobResult(
        returncode=1,
        stdout_tail="",
        stderr_tail="Error: CUDA out of memory in trellis worker",
        duration_s=1.2,
    )
    executor = FakeExecutor(caps=caps_ok, run_result=fail_result)
    backend = RemoteTrellisBackend(executor)

    img = tmp_path / "input.png"
    img.write_bytes(b"data")
    req = GenerateRequest(images=[img])

    out_dir = tmp_path / "out"
    with pytest.raises(GenerativeError) as exc_info:
        backend.generate(req, out_dir)

    assert "CUDA out of memory in trellis worker" in str(exc_info.value)


def test_generate_missing_output_file(tmp_path: Path) -> None:
    high_gpu = GPUInfo(name="A100", vram_mb=40000, compute_cap=8.0)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_ok, write_mesh=False)
    backend = RemoteTrellisBackend(executor)

    img = tmp_path / "input.png"
    img.write_bytes(b"data")
    req = GenerateRequest(images=[img])

    out_dir = tmp_path / "out"
    with pytest.raises(GenerativeError) as exc_info:
        backend.generate(req, out_dir)

    assert "mesh.glb" in str(exc_info.value)
