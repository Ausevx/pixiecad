"""Tests for S6b UV unwrap and object-space normal map bake."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import trimesh

from pixiecad.meshops.bake import UnwrapResult, unwrap_uv, bake_object_space_normals, save_normal_map


def test_unwrap_uv():
    sphere = trimesh.creation.icosphere(subdivisions=2)
    orig_verts = sphere.vertices.copy()
    orig_faces = sphere.faces.copy()

    res = unwrap_uv(sphere)

    assert isinstance(res, UnwrapResult)
    assert isinstance(res.mesh, trimesh.Trimesh)
    assert res.n_charts_hint > 0
    assert res.orig_face_count == len(orig_faces)

    # Verify input mesh not mutated
    assert np.array_equal(sphere.vertices, orig_verts)
    assert np.array_equal(sphere.faces, orig_faces)

    # Verify TextureVisuals and UV coordinates
    assert hasattr(res.mesh.visual, "uv")
    uvs = res.mesh.visual.uv
    assert uvs is not None
    assert len(uvs) == len(res.mesh.vertices)
    assert np.all(uvs >= 0.0) and np.all(uvs <= 1.0)
    assert res.mesh.faces.max() < len(res.mesh.vertices)
    assert res.mesh.faces.min() >= 0


def test_bake_object_space_normals(monkeypatch):
    dense = trimesh.creation.icosphere(subdivisions=4)
    low_raw = trimesh.creation.icosphere(subdivisions=2)
    low_unwrapped = unwrap_uv(low_raw).mesh

    call_count = 0
    orig_on_surface = trimesh.proximity.ProximityQuery.on_surface

    def mock_on_surface(self, points):
        nonlocal call_count
        call_count += 1
        return orig_on_surface(self, points)

    monkeypatch.setattr(trimesh.proximity.ProximityQuery, "on_surface", mock_on_surface)

    img = bake_object_space_normals(dense, low_unwrapped, resolution=128, padding_px=4)

    # Verify return properties
    assert isinstance(img, np.ndarray)
    assert img.shape == (128, 128, 3)
    assert img.dtype == np.uint8

    # Verify single batched on_surface query call
    assert call_count == 1

    # Verify non-background pixels exist
    bg = np.array([128, 128, 255], dtype=np.uint8)
    is_bg = np.all(img == bg, axis=-1)
    covered = img[~is_bg]
    assert len(covered) > 0

    # Verify normals vary across sphere (std > 10)
    covered_std = np.std(covered)
    assert covered_std > 10.0


def test_save_normal_map(tmp_path: Path):
    img = np.full((128, 128, 3), [128, 128, 255], dtype=np.uint8)
    out_path = tmp_path / "normal_map.png"

    save_normal_map(img, out_path)

    assert out_path.exists()
    with Image.open(out_path) as loaded:
        assert loaded.size == (128, 128)
        assert loaded.mode == "RGB"
