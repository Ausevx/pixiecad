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


def _scattered() -> trimesh.Trimesh:
    """Solid boxes spread far apart: extracts cleanly, encloses almost nothing.

    The deterministic stand-in for a bad generation. _hollow used to serve this
    role, but escalation now rescues it (fill 0.099 -> 0.994 at dilate 8) --
    which is the point of escalation, and why a fixture that cannot be closed
    at ANY radius is needed instead. These boxes are solid, so they survive the
    inward offset and extraction succeeds; they simply occupy a fraction of
    their own convex hull no dilation can fill.
    """
    return trimesh.util.concatenate([
        trimesh.creation.box(extents=(0.08, 0.08, 0.08)).apply_translation(t)
        for t in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    ])


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
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        assert r.applied
        assert r.mesh.is_watertight
        assert r.components_after == 1
        assert r.components_before > 100

    def test_the_interior_actually_gets_filled(self):
        """The whole point. On the real mesh, binary_fill_holes alone reaches
        0.13 because the gaps let the interior drain to the border; dilating
        first is what makes the fill mean anything (measured 0.96)."""
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        assert r.fill_after > 0.9, f"only reached {r.fill_after:.2f}"

    def test_no_open_edges_remain(self):
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        _, counts = np.unique(r.mesh.edges_sorted, axis=0, return_counts=True)
        assert int((counts == 1).sum()) == 0

    def test_the_shape_is_preserved(self):
        """A solid that ignores the input is not a repair. Extents must track
        the original within a voxel or two."""
        src = _perforated()
        r = solidify(src, divisions=128, only_if_shattered=False)
        assert np.allclose(r.mesh.extents, src.extents, atol=0.15 * src.extents.max())

    def test_the_result_is_not_a_featureless_blob(self):
        """Dilation could in principle inflate everything to its hull."""
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
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
        r = solidify(trimesh.creation.box(), divisions=128, only_if_shattered=False)
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


class TestResolutionIsIndependentOfTheBudget:
    """The grid must NOT follow the face budget. This class used to assert the
    opposite, and both positions came from a real bug.

    First a job asked for 300,000 faces and got 89,844, because the grid was
    fixed at 128 and decimation only removes faces. The fix tied divisions to
    the budget. But every error solidify makes is a fixed number of VOXELS, so
    that made dimensional accuracy a function of how many triangles the user
    asked for: a 20,000-face job got a 64^3 grid and came back 8.5% too thick
    on its thinnest axis, on an object whose thinnest axis is 37% of its
    longest.

    The grid now always runs as fine as the machine affords and decimation
    takes the count down, which is what decimation is for.
    """

    def test_the_grid_does_not_follow_the_face_budget(self):
        import inspect

        from pixiecad.meshops import solidify as mod

        assert "target_faces" not in inspect.getsource(mod.solidify)
        assert not hasattr(mod, "divisions_for_target")

    def test_the_pipeline_asks_for_no_particular_size(self):
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert "solidify(mesh)" in src
        assert "target_faces=spec.target_faces" not in src

    def test_the_default_is_the_memory_cap(self):
        """256^3 measured at 2.06 GB end to end with the chunked voxeliser
        (3.91 GB without it), on a 16 GB machine. There is no reason to run
        below it and no room to run above."""
        from pixiecad.meshops.solidify import DEFAULT_DIVISIONS, MAX_DIVISIONS

        assert DEFAULT_DIVISIONS == MAX_DIVISIONS == 256

    def test_the_default_grid_beats_any_realistic_budget(self):
        """Why the old refinement retry could be deleted: at a measured ~9.5
        faces per division squared, 256 yields ~620k -- more than the worker
        can deliver."""
        from pixiecad.meshops.solidify import DEFAULT_DIVISIONS

        assert DEFAULT_DIVISIONS**2 * 9.5 > 300_000

    def test_the_pipeline_warns_when_the_grid_undershoots(self):
        """Dropping target_faces removed the guard against the ORIGINAL bug,
        so the user has to hear about it instead."""
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert "short of the" in src and "voxel" in src


class TestDilationEscalatesInsteadOfGuessing:
    """Dilation is both what seals the mesh and what destroys its detail.

    Dilate-then-fill welds shut any gap narrower than twice the radius, so at
    256 divisions the old rule (divisions/45 = 6 voxels) closed anything under
    4.7% of the object -- wide enough to swallow a lighter's lid seam. It now
    STARTS small and widens only when the result does not actually enclose
    anything, so detail is never given away up front.

    The bug this class originally existed for: holding dilation at 2 while
    raising divisions dropped fill from 0.955 to 0.093 -- a watertight mesh
    that is hollow again, the worst outcome because every other check passes.
    """

    def test_both_resolutions_still_seal(self):
        coarse = solidify(_perforated(), divisions=128, only_if_shattered=False)
        fine = solidify(_perforated(), divisions=256, only_if_shattered=False)
        assert coarse.fill_after > 0.9
        assert fine.fill_after > 0.9, f"fine grid left it at {fine.fill_after:.2f}"

    def test_the_start_is_smaller_than_it_used_to_be(self):
        """Pins the detail gain so it cannot silently drift back to /45."""
        from pixiecad.meshops.solidify import start_dilate

        assert 1 < start_dilate(256) < 6
        assert start_dilate(128) >= 2

    def test_the_schedule_still_reaches_the_measured_minimums(self):
        """Measured on real output: 128 needs 2, 160 needs 3, 192 needs 3,
        256 needs 6. The START no longer covers 256 -- the ESCALATION must."""
        from pixiecad.meshops.solidify import _dilation_schedule, start_dilate

        for divisions, minimum in ((128, 2), (160, 3), (192, 3), (256, 6)):
            schedule = _dilation_schedule(start_dilate(divisions))
            assert max(schedule) >= minimum, f"{divisions} never reaches {minimum}"

    def test_the_schedule_ends_at_the_cap(self):
        from pixiecad.meshops.solidify import MAX_DILATE, _dilation_schedule

        assert _dilation_schedule(2)[-1] == MAX_DILATE
        assert _dilation_schedule(MAX_DILATE) == [MAX_DILATE]

    def test_an_extraction_failure_escalates_rather_than_giving_up(self):
        """A dilation too small to seal leaves thin shells that the inward
        offset erodes away entirely, and extract_surface returns None. That
        used to end the call outright -- which matters far more now that the
        starting radius is deliberately small."""
        r = solidify(_perforated(), divisions=192, dilate=2, only_if_shattered=False)
        assert r.applied, "escalation never ran"
        assert r.fill_after > 0.9, f"escalated but only reached {r.fill_after:.2f}"

    def test_escalation_keeps_the_smoothing(self):
        """The old recursive escalation could lose its parameters across the
        call; a loop cannot, so this is asserted behaviourally instead of by
        grepping the source."""
        r = solidify(_perforated(), divisions=192, dilate=2, only_if_shattered=False)
        angles = np.degrees(r.mesh.face_adjacency_angles)
        assert ((angles >= 40) & (angles < 50)).mean() < 0.005

    def test_the_reason_reports_what_was_welded_shut(self):
        """The user cannot see the dilation, but they can see a missing seam."""
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        assert "gaps under" in r.reason and "%" in r.reason


class TestCurvedSurfacesAreNotTerraced:
    """The user-visible defect: "the flat faces get solidified better and the
    curved ones are still like a mesh".

    Extracting from a BINARY grid gives marching cubes nothing to interpolate,
    so every vertex lands on a voxel corner and a curved surface degenerates
    into 45-degree steps. Flat faces of the object lie on grid planes and are
    unaffected, which is exactly the asymmetry that was reported.

    The tell is the dihedral angle histogram. On a shipped model, 4.2% of edges
    sat at *precisely* 45 degrees while the 5-20 degree band -- where a real
    curved surface lives -- was empty. Watertightness, fill ratio and component
    count all passed throughout, which is why this survived so long.
    """

    @staticmethod
    def _dihedrals(mesh: trimesh.Trimesh) -> np.ndarray:
        return np.degrees(mesh.face_adjacency_angles)

    def test_a_sphere_has_no_staircase(self):
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        angles = self._dihedrals(r.mesh)
        at_45 = ((angles >= 40) & (angles < 50)).mean()
        assert at_45 < 0.005, f"{at_45:.1%} of edges are 45-degree steps"

    def test_curvature_lands_in_the_band_a_real_surface_uses(self):
        """Not just "no 45s" -- a sphere must actually curve. A blob that had
        been smoothed flat would also pass the test above."""
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        angles = self._dihedrals(r.mesh)
        gentle = ((angles > 0.5) & (angles < 20)).mean()
        assert gentle > 0.10, f"only {gentle:.1%} of edges show any curvature"

    def test_smoothing_does_not_move_the_surface(self):
        """Blurring the field is only acceptable because it is nearly free.
        Measured against the true offset on a real mesh: 0.09% of object size
        at p95. Here, extents must survive within a voxel."""
        src = _perforated()
        smoothed = solidify(src, divisions=128, only_if_shattered=False)
        raw = solidify(src, smooth_sigma=0.0, divisions=128, only_if_shattered=False)
        assert np.allclose(smoothed.mesh.extents, raw.mesh.extents, atol=0.03 * src.extents.max())

    def test_it_still_encloses_the_same_volume(self):
        """Smoothing must not deflate the body the repair just built."""
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        assert r.fill_after > 0.9, f"fill fell to {r.fill_after:.2f}"

    def test_sigma_zero_is_still_a_valid_surface(self):
        """The escape hatch has to work: a distance field terraces far less
        than a binary one even unsmoothed, so 0 is a legitimate choice.

        Also the only test that catches the isovalue nudge. An unsmoothed
        distance transform is integer-valued along the axes, so an integer
        offset coincides with thousands of samples exactly and marching cubes
        emits degenerate cells -- watertight goes False. Blurring hides it, so
        without a sigma=0 case the guard could be deleted and every other test
        here would still pass.
        """
        r = solidify(_perforated(), smooth_sigma=0.0, divisions=128, only_if_shattered=False)
        assert r.applied and r.mesh.is_watertight

    def test_the_extractor_matches_trimeshs_index_space(self):
        """The surface is placed by the caller's voxel transform, so swapping
        extractors must not shift it by half a voxel -- an error that would be
        invisible in every other assertion here."""
        from pixiecad.meshops.solidify import extract_surface

        grid = np.zeros((40, 40, 40), dtype=bool)
        grid[10:30, 10:30, 10:30] = True
        binary = trimesh.voxel.VoxelGrid(grid).marching_cubes
        field = extract_surface(grid, offset=0, sigma=0.0)
        assert field is not None
        # Explicit tolerance, not the default. extract_surface nudges the
        # isovalue by 1e-4 on purpose, which displaces the surface by ~5e-5
        # voxels; np.allclose's default rtol scales with the coordinates, so on
        # a small grid like this one it clears by only about 6x and would go
        # flaky. A twentieth of a voxel is the tolerance that actually matters
        # here -- it is a half-voxel shift this test exists to catch.
        assert np.allclose(field.bounds, binary.bounds, atol=0.05)

    def test_eroding_away_the_whole_object_is_reported_not_raised(self):
        """marching_cubes raises when the isosurface is outside the field.
        A thin object plus a large dilation reaches that, and it must come
        back as an unapplied result rather than an exception."""
        from pixiecad.meshops.solidify import extract_surface

        grid = np.zeros((30, 30, 30), dtype=bool)
        grid[14:16, 5:25, 5:25] = True  # two voxels thick
        assert extract_surface(grid, offset=8, sigma=0.0) is None


class TestTheWorkerGetsTheFaceBudget:
    """The generative worker chooses its own face count unless told otherwise.

    The TRELLIS worker's default decimation_target is 2,000,000 faces. Nothing
    was setting it, so a job with a 300,000-face budget asked for -- and got --
    1,872,364 faces, and every stage downstream spent its time cleaning up
    after a number the user never chose.
    """

    def _spec(self, target):
        from pixiecad.spec import ObjectSpec

        return ObjectSpec(name="thing", target_faces=target)

    def test_the_budget_becomes_the_decimation_target(self):
        from pixiecad.pipeline import _generative_defaults

        out = _generative_defaults(None, spec=self._spec(300000), solidify_generated=False)
        assert out["decimation_target"] == 300000

    def test_solidify_gets_the_dense_mesh_instead(self):
        """Solidify DILATES fragments until they touch, so it reads the dense
        mesh even though it replaces it. Decimating first starves it -- same
        object, same dilation, same grid:

            1,872,364 faces in -> fill 0.018 -> 0.958, one component
              298,336 faces in -> fill 0.025 -> 0.047, 77 components

        The second is a hollow blob. Decimation belongs after the repair, which
        is where our own decimate stage already is.
        """
        from pixiecad.pipeline import _generative_defaults

        out = _generative_defaults(None, spec=self._spec(300000), solidify_generated=True)
        assert "decimation_target" not in out

    def test_an_explicit_target_is_honoured_even_with_solidify(self):
        """Refusing to default is not the same as refusing to obey."""
        from pixiecad.pipeline import _generative_defaults

        out = _generative_defaults(
            {"decimation_target": 900000}, spec=self._spec(300000), solidify_generated=True
        )
        assert out["decimation_target"] == 900000

    def test_the_pipeline_tells_it_whether_solidify_runs(self):
        """A default that reads the wrong flag is worse than none."""
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert "solidify_generated=solidify_generated" in src

    def test_an_explicit_option_wins(self):
        """Defaults must not overwrite something the caller actually asked for."""
        from pixiecad.pipeline import _generative_defaults

        out = _generative_defaults({"decimation_target": 12345}, spec=self._spec(300000))
        assert out["decimation_target"] == 12345

    def test_no_budget_leaves_the_worker_alone(self):
        """ObjectSpec validates target_faces > 0, so this cannot come from a
        real spec -- the guard is for any other caller, and asking the worker
        for zero faces would be worse than letting it choose."""
        from types import SimpleNamespace

        from pixiecad.pipeline import _generative_defaults

        out = _generative_defaults(None, spec=SimpleNamespace(target_faces=None))
        assert "decimation_target" not in out

    def test_the_caller_dict_is_not_mutated(self):
        from pixiecad.pipeline import _generative_defaults

        original = {}
        _generative_defaults(original, spec=self._spec(300000))
        assert original == {}

    def test_the_mesh_profile_is_selectable(self, monkeypatch):
        """hd (the worker's default, 1024 geometry) versus game_ready (512,
        capped at 300k). Every run so far has silently been hd."""
        from pixiecad.pipeline import _generative_defaults

        monkeypatch.setenv("PIXIECAD_TRELLIS_MESH_PROFILE", "game_ready")
        assert _generative_defaults(None, spec=self._spec(1000))["mesh_profile"] == "game_ready"

    def test_no_profile_set_means_no_opinion(self, monkeypatch):
        from pixiecad.pipeline import _generative_defaults

        monkeypatch.delenv("PIXIECAD_TRELLIS_MESH_PROFILE", raising=False)
        assert "mesh_profile" not in _generative_defaults(None, spec=self._spec(1000))

    def test_the_worker_actually_accepts_these_keys(self):
        """A default the passthrough filter drops is worse than none: it looks
        set and does nothing."""
        from pixiecad.generative.trellis_remote import PASSTHROUGH_OPTIONS

        assert "decimation_target" in PASSTHROUGH_OPTIONS
        assert "mesh_profile" in PASSTHROUGH_OPTIONS

    def test_they_reach_the_request_form(self):
        from pixiecad.generative.trellis_remote import build_trellis_script

        script = build_trellis_script(options={"decimation_target": 300000})
        assert "decimation_target=300000" in script

    def test_an_unknown_profile_is_refused(self, monkeypatch):
        """The worker falls back to hd on an unrecognised profile, so a typo
        would look like it took effect and silently give back hd -- exactly the
        confusion this knob exists to end."""
        import pytest

        from pixiecad.pipeline import _generative_defaults

        monkeypatch.setenv("PIXIECAD_TRELLIS_MESH_PROFILE", "gaem_ready")
        with pytest.raises(ValueError, match="not a profile"):
            _generative_defaults(None, spec=self._spec(1000))

    def test_the_profile_set_matches_the_worker(self):
        from pixiecad.generative.trellis_remote import TRELLIS_MESH_PROFILES

        assert TRELLIS_MESH_PROFILES == {"hd", "game_ready"}


class TestAFailedRepairMustNotLookLikeSuccess:
    """A blob passes every check the pipeline has.

    The output of a failed repair is watertight, one connected component, and
    exactly on the face budget -- so sanity, clean, decimate, bake and export
    all report ok. A real run went out as [OK] at every stage and the user
    spent twenty minutes texturing a melted blob before finding out.

    Measured on that generation: no dilation rescued it (6 -> 0.047,
    8 -> 0.067, 10 -> 0.137, 14 -> 0.204), while a good generation of the SAME
    object from the SAME photos and settings reached 0.966 at dilation 6. Low
    fill therefore means the generation is unusable, not that the repair needs
    tuning -- and the pipeline has to say so before the expensive stages run.
    """

    def test_a_good_repair_is_marked_repaired(self):
        r = solidify(_perforated(), divisions=128, only_if_shattered=False)
        assert r.repaired
        assert r.fill_after > 0.9

    def test_an_unclosable_surface_is_not_marked_repaired(self):
        r = solidify(_scattered(), divisions=96, only_if_shattered=False)
        assert not r.repaired, f"fill {r.fill_after:.3f} was accepted"

    def test_the_reason_says_it_failed_and_what_to_do(self):
        r = solidify(_scattered(), divisions=96, only_if_shattered=False)
        assert "could not close" in r.reason
        assert "re-run" in r.reason.lower()

    def test_the_mesh_is_still_returned(self):
        """No worse than the input, and the caller may still want to look."""
        r = solidify(_scattered(), divisions=96, only_if_shattered=False)
        assert r.mesh is not None and len(r.mesh.faces) > 0

    def test_the_low_fill_branch_reports_the_same_way(self):
        """The other failure shape: the surface DOES extract, it just encloses
        nothing. Scattered solid boxes reproduce it deterministically -- they
        survive erosion, so extraction succeeds, but they occupy a fraction of
        their own convex hull. This is the branch the real blob took."""
        boxes = [
            trimesh.creation.box(extents=(0.08, 0.08, 0.08)).apply_translation(t)
            for t in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
        ]
        scattered = trimesh.util.concatenate(boxes)
        r = solidify(scattered, divisions=96, only_if_shattered=False)
        assert r.applied, "the surface must extract; this is not the erosion path"
        assert not r.repaired, f"fill {r.fill_after:.3f} was accepted"
        assert "could not close" in r.reason and "re-run" in r.reason.lower()

    def test_escalation_rescues_what_it_can(self):
        """_hollow is unsealable at the starting radius -- extraction returns
        None outright -- and escalation turns it into a clean solid. Before
        escalation fired on that path it came back applied=False."""
        r = solidify(_hollow(), divisions=128, only_if_shattered=False)
        assert r.repaired and r.fill_after > 0.9

    def test_the_floor_sits_between_a_blob_and_real_output(self):
        """Hunyuan output that never needed repair sits at 0.42-0.88, a good
        repair at 0.95+, the blob at 0.05-0.22. The floor must not reject
        legitimate geometry."""
        from pixiecad.meshops.solidify import FILL_FLOOR

        assert 0.25 < FILL_FLOOR < 0.42

    def test_the_pipeline_marks_the_stage_failed(self):
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        # Three outcomes, not two. Collapsing "not applied" into "failed"
        # reported a healthy Hunyuan mesh (fill 0.97, deliberately left alone)
        # as [FAILED], telling the user their best result was broken.
        assert 'solid_status = "failed"' in src
        assert 'solid_status = "skipped"' in src
        assert 'elif outcome.repaired:' in src

    def test_the_pipeline_warns_the_model_is_unusable(self):
        """The warning is the only thing standing between the user and twenty
        minutes of texturing."""
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert "NOT USABLE" in src
        assert "re-running" in src


class TestTheSurfaceLandsWhereItShould:
    """The half-voxel bug, on exact binary grids so nothing else can hide it.

    ``distance_transform_edt`` measures centre to centre, but the interface
    sits half a voxel in from the last occupied centre: a voxel touching
    background gets edt == 1 while being 0.5 from the surface. So the raw field
    is sign*(d + 0.5) and asking for level -n lands at depth n - 0.5 -- half a
    voxel too far out on every side, at every resolution and every dilation.

    Worth exactly +1.000 voxel of extent, and the dominant term in a real model
    coming back 8.5% too thick on its thinnest axis. At offset 0 the +-1
    samples bracket the interface symmetrically and the error vanishes, which
    is why the index-space alignment test never caught it.
    """

    #: True extent of the slab below, in voxels.
    TRUE = 60.0

    def _slab(self):
        g = np.zeros((100, 100, 100), dtype=bool)
        g[20:80, 20:80, 20:80] = True
        return g

    def test_the_level_set_round_trip_is_exact(self):
        from pixiecad.meshops.solidify import extract_surface

        for offset in (0, 2, 4, 6):
            for sigma in (0.0, 1.5):
                s = extract_surface(self._slab(), offset=offset, sigma=sigma)
                assert s is not None
                got = float(np.mean(s.extents))
                want = self.TRUE - 2 * offset
                assert abs(got - want) < 0.02, (
                    f"offset {offset} sigma {sigma}: {got:.3f} vs {want:.3f}"
                )

    def test_the_error_does_not_depend_on_the_dilation(self):
        """Pins it as a constant, so nobody 'fixes' it with a percentage."""
        from pixiecad.meshops.solidify import extract_surface

        errs = [
            float(np.mean(extract_surface(self._slab(), offset=o, sigma=0.0).extents))
            - (self.TRUE - 2 * o)
            for o in (2, 4, 8)
        ]
        assert max(errs) - min(errs) < 0.02

    def test_smoothing_does_not_move_a_true_sdf(self):
        """Blurring a locally-linear function leaves its level sets alone.
        Before the correction, sigma 0 and 1.5 differed by 0.28 voxels."""
        from pixiecad.meshops.solidify import extract_surface

        a = np.mean(extract_surface(self._slab(), offset=2, sigma=0.0).extents)
        b = np.mean(extract_surface(self._slab(), offset=2, sigma=1.5).extents)
        assert abs(float(a) - float(b)) < 0.05

    def test_an_offset_of_two_insets_by_exactly_two(self):
        """The property that was actually broken: it inset by 1.5."""
        from pixiecad.meshops.solidify import extract_surface

        zero = extract_surface(self._slab(), offset=0, sigma=0.0)
        two = extract_surface(self._slab(), offset=2, sigma=0.0)
        inset = (np.asarray(zero.extents) - np.asarray(two.extents)) / 2.0
        assert np.allclose(inset, 2.0, atol=0.05), f"inset by {inset}"


class TestDimensionalAccuracy:
    """Known solids must come back their real size.

    The user-facing complaint: "the lighter is always a bit thicker than it
    actually is". Measured on a real job before the fix, growth per axis in
    voxels was [-0.93, +1.74, +2.02] -- a constant absolute offset, so the
    thinnest axis (37% of the longest) paid +8.52% while the longest paid
    +2.72%. After the fix that job's thinnest axis moved +0.13%.

    Two terms: the half-voxel EDT bug above (exactly removable) and voxelising
    marking the nearest centre (statistical, mean 0.5, cancelled by
    SHELL_BIAS). The residual +-0.5 spread is irreducible -- where the surface
    falls inside its voxel is genuinely unknown at this resolution.
    """

    SHAPES = {
        "box": trimesh.creation.box(extents=(1.0, 0.5, 0.3)),
        "sphere": trimesh.creation.icosphere(subdivisions=4, radius=0.5),
        "slab": trimesh.creation.box(extents=(1.0, 1.0, 0.1)),
    }

    @staticmethod
    def _error_voxels(mesh, divisions, phase=0.0):
        """Extent error in VOXELS, so the assertion holds at any resolution."""
        pitch = float(np.max(mesh.extents)) / divisions
        moved = mesh.copy()
        moved.apply_translation(np.array([phase, phase * 0.5, phase * 0.25]) * pitch)
        r = solidify(moved, divisions=divisions, only_if_shattered=False)
        return (np.asarray(r.mesh.extents) - np.asarray(mesh.extents)) / pitch

    def test_a_known_solid_comes_back_the_right_size(self):
        for name, mesh in self.SHAPES.items():
            err = self._error_voxels(mesh, 96)
            assert np.abs(err).max() < 1.2, f"{name}: {np.round(err, 2)} voxels"

    def test_there_is_no_systematic_growth_across_grid_phases(self):
        """The calibration, executable. Sweeping the sub-voxel phase is the
        whole point -- a box at phase 0 reads a clean artefact and would
        calibrate the bias to the wrong number."""
        errs = np.array([
            self._error_voxels(mesh, 72, phase)
            for mesh in self.SHAPES.values()
            for phase in (0.0, 0.25, 0.5, 0.75)
        ])
        assert -0.7 < errs.mean() < 0.4, f"mean {errs.mean():+.3f} voxels"
        assert np.abs(errs).max() < 1.2

    def test_the_error_is_constant_in_voxels_not_proportional(self):
        """Guards against anyone re-fixing this with a percentage."""
        sphere = self.SHAPES["sphere"]
        coarse = self._error_voxels(sphere, 64).mean()
        fine = self._error_voxels(sphere, 128).mean()
        assert abs(coarse - fine) < 0.4

    def test_a_thin_axis_is_not_thickened(self):
        """The complaint, in test form. A 0.1-thick slab is the shape that
        punishes an absolute offset hardest."""
        err = self._error_voxels(self.SHAPES["slab"], 128)
        assert abs(err[2]) < 1.0, f"thin axis moved {err[2]:+.2f} voxels"

    def test_a_thin_feature_survives_the_correction(self):
        """Every surface now moves inward a voxel further than before, so the
        new risk is eroding thin features away entirely."""
        from pixiecad.meshops.solidify import extract_surface

        grid = np.zeros((40, 40, 40), dtype=bool)
        grid[17:23, 5:35, 5:35] = True  # 6 voxels thick
        s = extract_surface(grid, offset=0, sigma=0.0)
        assert s is not None
        assert abs(float(np.min(s.extents)) - 6.0) < 1.0


class TestTheGridIsAffordable:
    """256^3 on every job only works because voxelising is chunked.

    trimesh subdivides the WHOLE mesh until every edge is under half a voxel
    and holds all of it at once: 2.6 GB on a 1.9M-face input at 256, which is
    65% of the module's entire footprint and what made running at full
    resolution unaffordable. Chunked it is 311 MB, and the grids are identical.
    """

    def test_chunked_voxelisation_matches_trimesh(self):
        from pixiecad.meshops.solidify import _voxelize

        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        pitch = float(np.max(mesh.extents)) / 48
        grid, lo = _voxelize(mesh, pitch, pad=2)
        ours = {tuple(v) for v in (np.argwhere(grid) + lo)}

        vox = mesh.voxelized(pitch=pitch)
        origin = np.round(np.asarray(vox.transform)[:3, 3] / pitch).astype(np.int64)
        theirs = {tuple(v) for v in (np.argwhere(vox.matrix) + origin)}
        assert ours == theirs, f"{len(ours ^ theirs)} voxels differ"

    def test_the_padding_keeps_the_object_off_the_border(self):
        """binary_fill_holes treats the border as outside, so an object flush
        against it drains and the whole repair silently produces a shell."""
        from pixiecad.meshops.solidify import _voxelize

        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        grid, _ = _voxelize(mesh, float(np.max(mesh.extents)) / 32, pad=5)
        assert not grid[0].any() and not grid[-1].any()
        assert not grid[:, 0].any() and not grid[:, :, 0].any()

    def test_it_does_not_hold_the_whole_subdivision(self):
        """Coarse, but it is the only thing that would catch a regression to
        mesh.voxelized on the production path."""
        import resource

        from pixiecad.meshops.solidify import _voxelize

        mesh = trimesh.creation.icosphere(subdivisions=5, radius=0.5)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _voxelize(mesh, float(np.max(mesh.extents)) / 128, pad=4)
        grew_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1e6
        assert grew_mb < 500, f"voxelisation grew RSS by {grew_mb:.0f} MB"
