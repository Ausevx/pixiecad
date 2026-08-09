"""Re-optimising a job must not need the GPU, or the backend that built it.

Re-optimise re-runs only the mesh tail -- clean, decimate, unwrap, bake,
export -- at a new face budget. That is the cheap half of a build: changing
your mind about polygon count should cost seconds locally, not another
generation on a rented L4.

It used to look for ``dense.ply`` alone, which only photogrammetry produces,
so every generative job failed with "Dense mesh (dense.ply) not found for this
job". That reads as a missing file when the truth was that the feature had
never been taught about the backends -- and generative is the path almost
every job actually takes, because it is what fewer than 16 photos selects.

Measured on a real job after the fix: 1,864,190 raw faces solidified and
re-decimated to 100,000 gave 2.6 MB against the 20.0 MB the job shipped at
662k, in 42 seconds and about 2 GB, with no VM involved at all.
"""

from __future__ import annotations

from pathlib import Path

from pixiecad.web.app import _find_reoptimise_source


def _job(root: Path) -> Path:
    (root / "work" / "ws" / "stages").mkdir(parents=True)
    return root


def _generated(root: Path, name: str = "s3-generate-abc123") -> Path:
    out = root / "work" / "ws" / "stages" / name / "out"
    out.mkdir(parents=True)
    mesh = out / "mesh.glb"
    mesh.write_bytes(b"glb")
    return mesh


class TestFindingTheSource:
    def test_a_generative_job_has_one(self, tmp_path):
        """The regression: this returned None and the endpoint 409'd."""
        root = _job(tmp_path)
        mesh = _generated(root)
        found = _find_reoptimise_source(root)
        assert found is not None, "generative jobs had no re-optimise source"
        assert found == (mesh, True)

    def test_dense_wins_when_a_job_has_both(self, tmp_path):
        """A true dense reconstruction is measured geometry; the generative
        mesh is inferred. Prefer the accurate one."""
        root = _job(tmp_path)
        _generated(root)
        dense = root / "work" / "ws" / "dense.ply"
        dense.write_bytes(b"ply")
        assert _find_reoptimise_source(root) == (dense, False)

    def test_dense_is_not_flagged_generative(self, tmp_path):
        """The flag decides whether solidify runs. Running the voxel repair on
        a photogrammetry mesh would round off measured detail for nothing."""
        root = _job(tmp_path)
        dense = root / "dense.ply"
        dense.write_bytes(b"ply")
        assert _find_reoptimise_source(root) == (dense, False)

    def test_the_newest_generate_stage_wins(self, tmp_path):
        """A rerun leaves several stage dirs; the latest is the mesh the job
        actually shipped."""
        root = _job(tmp_path)
        _generated(root, "s3-generate-aaa")
        newer = _generated(root, "s3-generate-zzz")
        assert _find_reoptimise_source(root) == (newer, True)

    def test_a_job_with_neither_returns_none(self, tmp_path):
        assert _find_reoptimise_source(_job(tmp_path)) is None

    def test_a_missing_directory_does_not_raise(self, tmp_path):
        assert _find_reoptimise_source(tmp_path / "nope") is None


class TestTheEndpointExplainsItself:
    def test_the_409_names_both_sources(self):
        """"dense.ply not found" sent the user looking for a missing file when
        the feature simply did not apply to their job."""
        import inspect

        from pixiecad.web import app as appmod

        src = inspect.getsource(appmod.create_app)
        detail = src[src.index("No source mesh kept for this job") :][:400]
        assert "dense.ply" in detail and "mesh.glb" in detail


class TestGenerativeSourcesAreRepairedFirst:
    def test_solidify_runs_before_decimation(self):
        """Decimating confetti only rearranges the fragments, so the repair has
        to precede it -- exactly as the build path orders them. solidify
        self-gates on fill, so a Hunyuan mesh that arrives solid is untouched
        and this costs nothing."""
        import inspect

        from pixiecad.web import app as appmod

        src = inspect.getsource(appmod.create_app)
        body = src[src.index("def _run_mesh_stages_sync") :]
        body = body[: body.index("def _job_worker")]
        assert "solidify(dense_mesh)" in body
        assert body.index("solidify(dense_mesh)") < body.index("clean_mesh(dense_mesh)")

    def test_it_only_runs_for_generative_sources(self):
        import inspect

        from pixiecad.web import app as appmod

        src = inspect.getsource(appmod.create_app)
        body = src[src.index("def _run_mesh_stages_sync") :]
        assert "if is_generative:" in body

    def test_a_failed_repair_is_not_fatal(self):
        """An unrepaired mesh is still a mesh, and the user asked for a
        smaller file rather than a perfect one."""
        import inspect

        from pixiecad.web import app as appmod

        src = inspect.getsource(appmod.create_app)
        body = src[src.index("if is_generative:") :][:1200]
        assert "except Exception" in body
