"""Tests for post-build finishing and web sizing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pixiecad.meshops.webexport import (
    downscale_texture,
    export_for_web,
    part_texture_size,
)
from pixiecad.web.finishing import FinishOptions, finish_model


def _textured_box(size: int = 512) -> trimesh.Trimesh:
    from PIL import Image

    mesh = trimesh.creation.box()
    rng = np.random.default_rng(0)
    # Noise, not a flat colour: a solid image compresses to nothing and would
    # make every size comparison meaningless.
    img = Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    uv = rng.random((len(mesh.vertices), 2))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=img)
    )
    return mesh


class TestWebExport:
    def test_downscale_shrinks_texture(self):
        mesh = _textured_box(512)
        out = downscale_texture(mesh, 128)
        assert max(out.visual.material.baseColorTexture.size) == 128

    def test_downscale_leaves_geometry_and_uvs_untouched(self):
        """UVs are normalised, so resizing the image must need no remapping."""
        mesh = _textured_box(256)
        out = downscale_texture(mesh, 64)
        assert np.array_equal(out.faces, mesh.faces)
        assert np.allclose(out.visual.uv, mesh.visual.uv)

    def test_upscaling_is_not_attempted(self):
        mesh = _textured_box(128)
        out = downscale_texture(mesh, 512)
        assert max(out.visual.material.baseColorTexture.size) == 128

    def test_export_reports_a_real_saving(self, tmp_path: Path):
        mesh = _textured_box(512)
        report = export_for_web(mesh, tmp_path / "web.glb", texture_size=64)
        assert report.path.is_file()
        assert report.bytes_after < report.bytes_before
        assert report.texture_after == (64, 64)

    def test_untextured_mesh_passes_through(self, tmp_path: Path):
        report = export_for_web(trimesh.creation.box(), tmp_path / "plain.glb")
        assert report.path.is_file()
        assert report.texture_before is None

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="max_size"):
            downscale_texture(_textured_box(64), 0)


class TestPartTextureSize:
    def test_scales_with_face_share(self):
        """A tiny part must not ship the same atlas as the body."""
        big = part_texture_size(9000, 10000, base=1024)
        small = part_texture_size(100, 10000, base=1024)
        assert small < big <= 1024

    def test_never_exceeds_base_or_drops_below_floor(self):
        assert part_texture_size(10000, 10000, base=512) <= 512
        assert part_texture_size(1, 100000, base=1024, floor=128) == 128

    def test_degenerate_inputs_return_floor(self):
        assert part_texture_size(0, 0, floor=64) == 64


class TestFinishModel:
    def test_smoothing_preserves_face_count(self, tmp_path: Path):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        path = tmp_path / "model.glb"
        mesh.export(path)
        finish_model(path, tmp_path, FinishOptions(smooth_iterations=3, segmentation="auto"))
        after = trimesh.load(path, force="mesh", process=False)
        assert len(after.faces) == len(mesh.faces)

    def test_gpu_stages_warn_instead_of_failing_without_a_host(self, tmp_path: Path):
        """A missing GPU host must degrade, not lose the whole build."""
        path = tmp_path / "model.glb"
        trimesh.creation.box().export(path)
        report = finish_model(
            path, tmp_path, FinishOptions(texture=True, segmentation="semantic")
        )
        assert any("GPU host" in w for w in report.warnings)
        assert path.is_file()
        assert "segment:geometric" in report.steps

    def test_semantic_falls_back_to_geometric_on_failure(self, tmp_path: Path):
        path = tmp_path / "model.glb"
        trimesh.creation.box().export(path)
        report = finish_model(
            path,
            tmp_path,
            FinishOptions(segmentation="semantic", gpu_host="nonexistent.invalid"),
        )
        assert "segment:geometric" in report.steps
        assert any("semantic segmentation failed" in w for w in report.warnings)

    def test_web_export_produces_a_second_smaller_file(self, tmp_path: Path):
        mesh = _textured_box(512)
        path = tmp_path / "model.glb"
        mesh.export(path)
        report = finish_model(
            path, tmp_path, FinishOptions(web_export=True, texture_size=64)
        )
        web = tmp_path / "model_web.glb"
        assert web.is_file()
        assert report.web_bytes == web.stat().st_size
        assert web.stat().st_size < path.stat().st_size

    def test_needs_gpu_flag(self):
        assert not FinishOptions().needs_gpu
        assert FinishOptions(texture=True).needs_gpu
        assert FinishOptions(segmentation="semantic").needs_gpu
