"""Tests for generated-mesh structural sanity checks."""

import numpy as np
import trimesh

from pixiecad.meshops.sanity import check_mesh


def test_plausible_object_passes():
    """A car-shaped box is exactly what should NOT be flagged."""
    r = check_mesh(trimesh.creation.box(extents=[4.5, 1.8, 1.0]))
    assert r.ok
    assert not r.warnings


def test_cube_of_noise_is_flagged():
    """The real failure mode: geometry filling its bounding box uniformly.

    Every other check passed on the mesh this reproduces -- exact face count,
    valid UVs, successful bake -- so the cubic bounding box is the signal.
    """
    rng = np.random.default_rng(0)
    pts = rng.random((400, 3))
    noise = trimesh.Trimesh(
        vertices=pts, faces=trimesh.convex.convex_hull(pts).faces, process=False
    )
    noise.vertices = pts[: len(noise.vertices)] if False else noise.vertices
    r = check_mesh(noise)
    assert not r.ok
    assert any("cubic" in w for w in r.warnings)


def test_round_objects_can_opt_out():
    """A ball really is cubic in aspect; the check must be suppressible."""
    ball = trimesh.creation.icosphere()
    assert not check_mesh(ball).ok
    assert check_mesh(ball, expect_elongated=False).ok


def test_report_is_serialisable_and_readable():
    r = check_mesh(trimesh.creation.box(extents=[3, 1, 0.5]))
    assert "ok" in r.summary()
    assert len(r.extents) == 3
    assert r.n_components >= 1
