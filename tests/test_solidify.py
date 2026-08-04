"""Tests for repairing a shattered generative surface into a solid.

The defect these exist for, measured on a real TRELLIS job: 1,915,811 faces in
40,341 disconnected components, 609,689 open boundary edges, enclosing 1.8% of
its own convex hull. The silhouette was right and the object was recognisable;
there was simply nothing inside it.

Meshes here are small on purpose -- solidifying the real 1.9M face mesh peaks
at about 4 GB, and this suite has to run on a 16 GB laptop.
"""

from __future__ import annotations

import numpy as np
import trimesh

from pixiecad.meshops.solidify import (
    count_components,
    fill_ratio,
    looks_shattered,
    solidify,
)


def _perforated(subdivisions: int = 5, keep: float = 0.7) -> trimesh.Trimesh:
    """Dense unwelded patches with holes: the topology of the real defect.

    Every face is its own component and nothing shares a vertex, which is what
    a real TRELLIS mesh looks like (40,341 components, 609,689 open edges).
    Dense enough that dilation can bridge the gaps, which is the property the
    repair depends on.
    """
    rng = np.random.default_rng(0)
    src = trimesh.creation.icosphere(subdivisions=subdivisions)
    picks = rng.choice(len(src.faces), size=int(len(src.faces) * keep), replace=False)
    verts, faces = [], []
    for i, f in enumerate(src.faces[picks]):
        verts.extend(src.vertices[f])
        faces.append([i * 3, i * 3 + 1, i * 3 + 2])
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=False)


def _hollow(keep: float = 0.10) -> trimesh.Trimesh:
    """Sparse enough that it encloses almost nothing: the detection signal.

    Two fixtures rather than one, deliberately. A synthetic sphere cannot be
    both detectably hollow AND dense enough to repair -- removing enough area
    to drop the fill ratio also opens gaps wider than any affordable dilation.
    The real mesh manages both because 1.9M faces cover the surface while
    fragmenting into open patches, and that is not reproducible at a size this
    suite can afford. So detection is tested here and the repair mechanism on
    _perforated, with the real-world numbers recorded in solidify.py.
    """
    return _perforated(subdivisions=6, keep=keep)


class TestDetection:
    def test_a_solid_is_not_flagged(self):
        assert not looks_shattered(trimesh.creation.box())
        assert not looks_shattered(trimesh.creation.icosphere(subdivisions=3))

    def test_a_hollow_mesh_is_flagged(self):
        assert looks_shattered(_hollow())

    def test_fill_ratio_separates_them(self):
        """The metric the decision rests on, so it gets its own test."""
        assert fill_ratio(trimesh.creation.box()) > 0.9
        assert fill_ratio(_hollow()) < 0.15

    def test_component_counting_avoids_split(self):
        """split() materialises every submesh and OOM-killed a 16 GB machine
        on a 40,000-component mesh. The graph labelling must agree with it on
        a small case where split() is still safe."""
        m = _perforated(subdivisions=3, keep=0.5)
        assert count_components(m) == len(m.split(only_watertight=False))


class TestSolidify:
    def test_shattered_input_becomes_one_watertight_body(self):
        r = solidify(_perforated(), only_if_shattered=False)
        assert r.applied
        assert r.mesh.is_watertight
        assert r.components_after == 1
        assert r.components_before > 100

    def test_the_interior_actually_gets_filled(self):
        """The whole point. On the real mesh, binary_fill_holes alone reaches
        0.13 because the gaps let the interior drain to the border; dilating
        first is what makes the fill mean anything (measured 0.96)."""
        r = solidify(_perforated(), only_if_shattered=False)
        assert r.fill_after > 0.9, f"only reached {r.fill_after:.2f}"

    def test_no_open_edges_remain(self):
        r = solidify(_perforated(), only_if_shattered=False)
        _, counts = np.unique(r.mesh.edges_sorted, axis=0, return_counts=True)
        assert int((counts == 1).sum()) == 0

    def test_the_shape_is_preserved(self):
        """A solid that ignores the input is not a repair. Extents must track
        the original within a voxel or two."""
        src = _perforated()
        r = solidify(src, only_if_shattered=False)
        assert np.allclose(r.mesh.extents, src.extents, atol=0.15 * src.extents.max())

    def test_the_result_is_not_a_featureless_blob(self):
        """Dilation could in principle inflate everything to its hull."""
        r = solidify(_perforated(), only_if_shattered=False)
        assert r.mesh.volume / r.mesh.convex_hull.volume > 0.5

    def test_a_healthy_mesh_is_left_alone(self):
        """Hunyuan output encloses 0.42-0.88 of its hull and must not be
        touched -- solidifying it would only round off real detail."""
        box = trimesh.creation.box()
        r = solidify(box)
        assert not r.applied
        assert r.mesh is box
        assert "left alone" in r.reason

    def test_it_can_be_forced_on_a_healthy_mesh(self):
        r = solidify(trimesh.creation.box(), only_if_shattered=False)
        assert r.applied

    def test_degenerate_input_does_not_raise(self):
        empty = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
        assert solidify(empty) is not None


class TestPipelineWiring:
    def test_the_generative_path_solidifies(self):
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert "solidify" in src

    def test_it_runs_before_the_mesh_tail(self):
        """Decimating confetti only rearranges the fragments, so the repair
        has to happen before clean/decimate, not after."""
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert src.index("solidify") < src.index("_mesh_tail")

    def test_failure_is_not_fatal(self):
        """An unrepaired mesh is still a mesh."""
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        head = src[src.index("if solidify_generated:"):]
        assert "except Exception" in head
