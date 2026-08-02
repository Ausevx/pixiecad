"""Tests for Taubin surface smoothing."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from pixiecad.meshops.smooth import smooth_mesh


def _noisy_sphere(seed: int = 0, amplitude: float = 0.03) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    rng = np.random.default_rng(seed)
    mesh.vertices = mesh.vertices + rng.normal(0, amplitude, mesh.vertices.shape)
    return mesh


def _roughness(mesh: trimesh.Trimesh) -> float:
    """Mean deviation from a fitted sphere: lower is smoother."""
    r = np.linalg.norm(mesh.vertices - mesh.vertices.mean(axis=0), axis=1)
    return float(r.std())


class TestSmoothMesh:
    def test_reduces_high_frequency_noise(self):
        noisy = _noisy_sphere()
        smoothed, _ = smooth_mesh(noisy, iterations=5)
        assert _roughness(smoothed) < _roughness(noisy) * 0.6

    def test_preserves_vertex_and_face_count(self):
        """The budget promise: smoothing must not change file size."""
        mesh = _noisy_sphere()
        smoothed, _ = smooth_mesh(mesh, iterations=5)
        assert len(smoothed.vertices) == len(mesh.vertices)
        assert len(smoothed.faces) == len(mesh.faces)
        assert np.array_equal(smoothed.faces, mesh.faces)

    def test_does_not_shrink_like_laplacian(self):
        """Taubin's whole point: volume must survive many iterations."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        smoothed, report = smooth_mesh(mesh, iterations=20)
        assert abs(report.volume_change) < 0.05

    def test_seam_duplicated_vertices_stay_together(self):
        """Smoothing must not tear a UV seam open.

        Duplicating a vertex splits it in the index buffer but not in space;
        smoothing them independently would pull the copies apart and crack the
        surface. They must end up at the same position.
        """
        mesh = _noisy_sphere()
        # Duplicate every vertex of face 0, as a UV seam would.
        extra = mesh.faces[0]
        verts = np.vstack([mesh.vertices, mesh.vertices[extra]])
        faces = mesh.faces.copy()
        faces[0] = [len(mesh.vertices) + i for i in range(3)]
        seamed = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        smoothed, _ = smooth_mesh(seamed, iterations=5)
        for original, duplicate in zip(extra, faces[0]):
            assert np.allclose(
                smoothed.vertices[original], smoothed.vertices[duplicate], atol=1e-9
            )

    def test_boundary_is_pinned_by_default(self):
        """An open hole must not creep outward with each iteration."""
        plane = trimesh.creation.box(extents=[1, 1, 1])
        plane.faces = plane.faces[:6]  # open shell, so it has a real boundary
        plane = trimesh.Trimesh(vertices=plane.vertices, faces=plane.faces, process=False)
        smoothed, report = smooth_mesh(plane, iterations=5, preserve_boundary=True)
        assert report.boundary_pinned > 0

    def test_zero_iterations_is_identity(self):
        mesh = _noisy_sphere()
        smoothed, report = smooth_mesh(mesh, iterations=0)
        assert np.array_equal(smoothed.vertices, mesh.vertices)
        assert report.iterations == 0

    def test_rejects_non_shrinking_parameters(self):
        mesh = _noisy_sphere()
        with pytest.raises(ValueError, match="mu must be < 0"):
            smooth_mesh(mesh, mu=0.1)
        with pytest.raises(ValueError, match="lamb must be > 0"):
            smooth_mesh(mesh, lamb=-0.1)
        with pytest.raises(ValueError, match="iterations must be >= 0"):
            smooth_mesh(mesh, iterations=-1)

    def test_empty_mesh_is_returned_unchanged(self):
        empty = trimesh.Trimesh(
            vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
        )
        smoothed, report = smooth_mesh(empty)
        assert len(smoothed.faces) == 0
        assert report.iterations == 0
