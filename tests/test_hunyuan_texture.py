"""Tests for the remote Hunyuan3D texturing stage. No GPU or network."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import trimesh

from pixiecad.executors.base import JobResult
from pixiecad.generative.base import GenerativeError
from pixiecad.generative.hunyuan_remote import build_texture_script, texture_mesh


class FakeExecutor:
    def __init__(self, *, ok: bool = True, emit: bool = True) -> None:
        self.ok = ok
        self.emit = emit
        self.jobs: list = []

    def probe(self):  # pragma: no cover - unused by texturing
        raise NotImplementedError

    def run(self, job):
        self.jobs.append(job)
        if not self.ok:
            return JobResult(1, "", "CUDA out of memory", 1.0)
        job.output_dir.mkdir(parents=True, exist_ok=True)
        if self.emit:
            mesh = next(p for p in job.inputs if p.name == "mesh.glb")
            shutil.copyfile(mesh, job.output_dir / "textured.glb")
        return JobResult(0, "ok", "", 1.0)


@pytest.fixture
def inputs(tmp_path: Path):
    mesh_path = tmp_path / "model.glb"
    trimesh.creation.box().export(mesh_path)
    image_path = tmp_path / "front.png"
    from PIL import Image
    import numpy as np

    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    return mesh_path, image_path


class TestBuildTextureScript:
    def test_writes_into_the_collected_directory(self):
        """SSHExecutor only rsyncs 'out/' back."""
        assert "--out out/textured.glb" in build_texture_script()

    def test_mounts_weight_caches(self):
        script = build_texture_script()
        assert '-v "$HOME/.cache/huggingface":/root/.cache/huggingface' in script
        assert "--gpus all" in script

    def test_numeric_arguments_are_coerced(self):
        script = build_texture_script(max_views="8", resolution="256")  # type: ignore[arg-type]
        assert "--max-views 8" in script
        assert "--resolution 256" in script


class TestTextureMesh:
    def test_returns_the_textured_mesh(self, tmp_path: Path, inputs):
        mesh_path, image_path = inputs
        out = tmp_path / "out"
        result = texture_mesh(FakeExecutor(), mesh_path, image_path, out)
        assert result == out / "textured.glb"
        assert result.is_file()

    def test_uploads_mesh_image_and_script(self, tmp_path: Path, inputs):
        mesh_path, image_path = inputs
        ex = FakeExecutor()
        texture_mesh(ex, mesh_path, image_path, tmp_path / "out")
        names = {p.name for p in ex.jobs[0].inputs}
        assert {"mesh.glb", "images", "hunyuan_texture.py"} == names

    def test_missing_mesh_is_rejected_before_any_remote_call(self, tmp_path: Path, inputs):
        _, image_path = inputs
        ex = FakeExecutor()
        with pytest.raises(GenerativeError, match="no mesh to texture"):
            texture_mesh(ex, tmp_path / "absent.glb", image_path, tmp_path / "out")
        assert not ex.jobs

    def test_missing_image_is_rejected(self, tmp_path: Path, inputs):
        mesh_path, _ = inputs
        with pytest.raises(GenerativeError, match="no conditioning image"):
            texture_mesh(FakeExecutor(), mesh_path, tmp_path / "absent.png", tmp_path / "out")

    def test_remote_failure_surfaces_stderr(self, tmp_path: Path, inputs):
        mesh_path, image_path = inputs
        with pytest.raises(GenerativeError, match="CUDA out of memory"):
            texture_mesh(FakeExecutor(ok=False), mesh_path, image_path, tmp_path / "out")

    def test_silent_no_output_is_an_error(self, tmp_path: Path, inputs):
        """A zero exit code with no mesh must not read as success."""
        mesh_path, image_path = inputs
        with pytest.raises(GenerativeError, match="produced no mesh"):
            texture_mesh(FakeExecutor(emit=False), mesh_path, image_path, tmp_path / "out")
