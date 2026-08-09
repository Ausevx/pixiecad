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


def _make_uv_split_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Duplicate vertices per face to simulate UV seam splits across chart boundaries."""
    split_verts = mesh.vertices[mesh.faces.reshape(-1)].copy()
    split_faces = np.arange(len(mesh.faces) * 3).reshape(-1, 3)
    return trimesh.Trimesh(vertices=split_verts, faces=split_faces, process=False)


def test_uv_split_mesh_reported_as_sealed():
    """A mesh split along UV seams (duplicated coincident vertices) has 0 holes."""
    box = trimesh.creation.box(extents=[4.5, 1.8, 1.0])
    split_box = _make_uv_split_mesh(box)
    assert not split_box.is_watertight  # trimesh native check fails due to seams

    r = check_mesh(split_box)
    assert r.n_holes == 0
    assert r.n_non_manifold == 0
    assert not any("holes" in w for w in r.warnings)


def test_real_hole_is_reported():
    """Deleting a face creates open boundary edges (a real hole)."""
    box = trimesh.creation.box(extents=[4.5, 1.8, 1.0])
    box.update_faces(np.arange(len(box.faces)) != 0)
    r = check_mesh(box)
    assert r.n_holes > 0
    assert any("open boundary edges" in w or "holes" in w for w in r.warnings)


def test_non_manifold_edge_reported_separately_not_as_hole():
    """Attaching a 3rd face to an existing edge creates a non-manifold pinch."""
    # Two tetrahedra sharing edge (0, 1), fully closed (0 open boundary edges)
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1]],
        dtype=float,
    )
    f1 = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    f2 = np.array([[0, 1, 4], [0, 4, 5], [0, 5, 1], [1, 5, 4]])
    nm_mesh = trimesh.Trimesh(
        vertices=verts, faces=np.vstack([f1, f2]), process=False
    )

    r = check_mesh(nm_mesh, expect_elongated=False)
    assert r.n_non_manifold == 1
    assert r.n_holes == 0
    # Counted, but NOT warned about, and it must not flip ok. The pipeline
    # maps report.ok straight to a [FAILED] sanity stage, and a pinch does not
    # leak: measured on real exports, a good Hunyuan model carries 1 and the
    # raw worker mesh 126, with zero holes in both. Failing on that would be
    # the same false alarm this change removes, only louder.
    assert not any("non-manifold" in w for w in r.warnings)
    assert not any("holes" in w for w in r.warnings)
    assert r.ok, "a sealed mesh with a pinch must still pass"


def test_check_mesh_does_not_mutate_input_or_volume():
    """check_mesh must be diagnostic-only and not mutate geometry or volume."""
    box = trimesh.creation.box(extents=[4.5, 1.8, 1.0])
    split_box = _make_uv_split_mesh(box)

    v_count_before = len(split_box.vertices)
    f_count_before = len(split_box.faces)
    vol_before = split_box.volume
    v_copy_before = split_box.vertices.copy()

    check_mesh(split_box)

    assert len(split_box.vertices) == v_count_before
    assert len(split_box.faces) == f_count_before
    assert split_box.volume == vol_before
    assert np.array_equal(split_box.vertices, v_copy_before)
    assert abs(split_box.volume - box.volume) < 1e-5
