"""Tests for multi-view semantic part segmentation."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from pixiecad.parts.render import (
    BACKGROUND_FACE_ID,
    Camera,
    canonical_rotation,
    default_cameras,
    render_mesh,
)
from pixiecad.parts.semantic import (
    SilhouetteSegmenter,
    _felzenszwalb,
    segment_semantic,
)


def _two_spheres() -> trimesh.Trimesh:
    """Two disjoint spheres, far enough apart to never overlap in projection."""
    a = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    b = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    b.apply_translation([6.0, 0.0, 0.0])
    return trimesh.util.concatenate([a, b])


class TestRender:
    def test_face_id_buffer_indexes_real_faces(self):
        mesh = trimesh.creation.box()
        (render,) = render_mesh(mesh, cameras=[Camera("front", (0.0, -1.0, 0.0))], resolution=64)
        ids = render.visible_faces
        assert len(ids)
        assert ids.min() >= 0
        assert ids.max() < len(mesh.faces)

    def test_background_is_marked_not_face_zero(self):
        """A background pixel must be distinguishable from a hit on face 0."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        (render,) = render_mesh(mesh, cameras=[Camera("front", (0, -1, 0))], resolution=64)
        # Corners cannot be covered by a centred sphere with margin.
        assert render.face_id[0, 0] == BACKGROUND_FACE_ID
        assert render.rgb[0, 0].sum() == 0

    def test_all_default_cameras_see_geometry(self):
        mesh = trimesh.creation.box()
        for render in render_mesh(mesh, resolution=48):
            assert len(render.visible_faces), f"{render.camera.name} rendered nothing"

    def test_shared_scale_across_views(self):
        """One scale for all views: coverage must not swing with framing."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        counts = [
            int((r.face_id != BACKGROUND_FACE_ID).sum()) for r in render_mesh(mesh, resolution=64)
        ]
        assert max(counts) - min(counts) < 0.05 * max(counts)

    def test_up_axis_rotation_is_a_rotation(self):
        for axis in ("x", "y", "z"):
            rot = canonical_rotation(axis)
            assert np.allclose(rot @ rot.T, np.eye(3))
            # Determinant +1, not -1: a mirror would flip face winding and
            # invert every normal used for shading.
            assert np.isclose(np.linalg.det(rot), 1.0)

    def test_up_axis_maps_named_axis_to_z(self):
        assert np.allclose(canonical_rotation("y") @ np.array([0.0, 1.0, 0.0]), [0, 0, 1])
        assert np.allclose(canonical_rotation("x") @ np.array([1.0, 0.0, 0.0]), [0, 0, 1])

    def test_rejects_unknown_up_axis(self):
        with pytest.raises(ValueError, match="up_axis"):
            canonical_rotation("w")

    def test_rejects_empty_mesh(self):
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64))
        with pytest.raises(ValueError, match="no geometry"):
            render_mesh(empty)

    def test_default_camera_names_unique(self):
        names = [c.name for c in default_cameras()]
        assert len(names) == len(set(names)) == 14

    def test_occlusion_respects_depth(self):
        """The near box must win every pixel it covers."""
        near = trimesh.creation.box(extents=[1, 1, 1])
        far = trimesh.creation.box(extents=[1, 1, 1])
        far.apply_translation([0.0, 4.0, 0.0])
        combined = trimesh.util.concatenate([near, far])
        n_near = len(near.faces)
        (render,) = render_mesh(
            combined, cameras=[Camera("front", (0, -1, 0))], resolution=64, up_axis="z"
        )
        # Camera sits on -Y, so only the near box may be visible.
        assert set(render.visible_faces.tolist()) <= set(range(n_near))


class TestFelzenszwalb:
    def test_size_penalty_prevents_one_region_swallowing_all(self):
        """The failure this algorithm was brought in to fix.

        A chain of four faces whose boundaries all cost the same non-zero
        amount. Plain agglomerative merging absorbs the lot regardless; here
        the k/area slack is far below that cost, so nothing may merge.
        """
        adjacency = np.array([[0, 1], [1, 2], [2, 3]])
        dissim = np.full(3, 0.5)
        area = np.full(4, 0.25)
        assert len(np.unique(_felzenszwalb(adjacency, dissim, area, 1e-9))) == 4
        # Slack above the boundary cost (k/0.25 > 0.5) and it all merges.
        assert len(np.unique(_felzenszwalb(adjacency, dissim, area, 1.0))) == 1

    def test_larger_k_gives_coarser_segmentation(self):
        adjacency = np.array([[0, 1], [1, 2], [2, 3]])
        dissim = np.array([0.1, 0.9, 0.1])
        area = np.full(4, 0.25)
        fine = len(np.unique(_felzenszwalb(adjacency, dissim, area, 1e-6)))
        coarse = len(np.unique(_felzenszwalb(adjacency, dissim, area, 10.0)))
        assert coarse <= fine

    def test_cuts_at_the_expensive_boundary(self):
        """A strong disagreement must survive as a part boundary."""
        adjacency = np.array([[0, 1], [1, 2], [2, 3]])
        dissim = np.array([0.0, 100.0, 0.0])
        area = np.full(4, 0.25)
        labels = _felzenszwalb(adjacency, dissim, area, 0.01)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[1] != labels[2]


class TestSegmentSemantic:
    def test_no_part_spans_disjoint_objects(self):
        """The real invariant: a part never bridges two separate objects.

        Not asserting exactly two clusters. ``max_parts`` is the part count the
        user asked for, so subdividing two spheres into up to eight patches is
        correct behaviour -- what would be wrong is a single part containing
        surface from both spheres.
        """
        mesh = _two_spheres()
        result = segment_semantic(
            mesh, SilhouetteSegmenter(), resolution=128, up_axis="z", max_parts=8
        )
        assert (result.face_labels >= 0).all()
        assert result.n_clusters <= 8
        left = mesh.triangles_center[:, 0] < 3.0
        for cluster in np.unique(result.face_labels):
            sel = result.face_labels == cluster
            assert left[sel].all() or (~left[sel]).all(), "part spans both spheres"

    def test_respects_max_parts(self):
        mesh = _two_spheres()
        for budget in (2, 4, 6):
            result = segment_semantic(
                mesh, SilhouetteSegmenter(), resolution=128, up_axis="z", max_parts=budget
            )
            assert result.n_clusters <= budget

    def test_every_face_receives_a_label(self):
        """Hidden faces must be filled, or exported parts come out with holes."""
        mesh = trimesh.creation.icosphere(subdivisions=3)
        result = segment_semantic(mesh, SilhouetteSegmenter(), resolution=96, up_axis="z")
        assert len(result.face_labels) == len(mesh.faces)
        assert (result.face_labels >= 0).all()

    def test_deterministic(self):
        mesh = _two_spheres()
        a = segment_semantic(mesh, SilhouetteSegmenter(), resolution=96, up_axis="z")
        b = segment_semantic(mesh, SilhouetteSegmenter(), resolution=96, up_axis="z")
        assert np.array_equal(a.face_labels, b.face_labels)

    def test_segmenter_shape_mismatch_is_reported(self):
        class Wrong:
            def segment(self, rgb):
                return np.zeros((7, 7), dtype=np.int64)

        with pytest.raises(ValueError, match="segmenter returned"):
            segment_semantic(trimesh.creation.box(), Wrong(), resolution=32)

    def test_no_masks_falls_back_to_one_part(self):
        class Blank:
            def segment(self, rgb):
                return np.full(rgb.shape[:2], -1, dtype=np.int64)

        mesh = trimesh.creation.box()
        result = segment_semantic(mesh, Blank(), resolution=32)
        assert result.n_clusters == 1
        assert (result.face_labels == 0).all()

    def test_whole_object_mask_is_ignored(self):
        """A mask covering everything is the segmenter failing, not a part."""
        class All:
            def segment(self, rgb):
                return np.zeros(rgb.shape[:2], dtype=np.int64)

        result = segment_semantic(trimesh.creation.box(), All(), resolution=32)
        assert not result.regions
