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

import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

from pixiecad.web.app import (
    _current_model,
    _export_stem,
    _find_reoptimise_source,
    _model_files,
    _model_list,
    _model_url,
    _variant_name,
    create_app,
)


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


def _out(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(out: Path, name: str, *, age: float = 0.0) -> Path:
    """A model file with a controllable mtime.

    mtime is what decides which model is current, so the tests have to be able
    to say "this one is older" without sleeping for a real second.
    """
    path = out / name
    path.write_bytes(b"glTF-stub")
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


class TestEveryModelIsKept:
    """The regression this whole feature exists for: re-optimise rewrote
    output/model.glb in place, so the geometry the build shipped was gone the
    moment you tried a different budget -- and getting it back meant another
    generation on a rented L4."""

    def test_a_variant_is_named_for_its_face_count(self):
        assert _variant_name(60000) == "model-60000.glb"

    def test_the_build_model_and_its_variants_are_all_listed(self, tmp_path):
        out = _out(tmp_path)
        _write(out, "model.glb", age=300)
        _write(out, "model-60000.glb", age=100)
        assert [p.name for p in _model_files(out)] == [
            "model.glb",
            "model-60000.glb",
        ]

    def test_the_newest_is_current(self, tmp_path):
        out = _out(tmp_path)
        _write(out, "model.glb", age=300)
        newest = _write(out, "model-60000.glb", age=10)
        _write(out, "model-120000.glb", age=100)
        assert _current_model(out) == newest

    def test_the_web_export_is_not_mistaken_for_a_model(self, tmp_path):
        """model_web.glb is a texture-shrunk copy, not a budget of its own; it
        must never become the current model or be offered as one."""
        out = _out(tmp_path)
        _write(out, "model.glb", age=300)
        _write(out, "model_web.glb", age=1)
        assert [p.name for p in _model_files(out)] == ["model.glb"]

    def test_no_models_is_not_an_error(self, tmp_path):
        assert _model_files(_out(tmp_path)) == []
        assert _current_model(tmp_path / "nope") is None


class TestTheModelListTellsTheUiWhatItNeeds:
    def test_variants_carry_their_face_count_in_the_name(self, tmp_path):
        out = _out(tmp_path)
        _write(out, "model.glb", age=300)
        _write(out, "model-60000.glb", age=10)
        listed = _model_list(
            "j1", out, job_name="lighter", build_faces=297100, current_faces=60000
        )
        assert [m["faces"] for m in listed] == [297100, 60000]
        assert [m["original"] for m in listed] == [True, False]
        assert [m["current"] for m in listed] == [False, True]
        assert listed[1]["url"] == "/api/jobs/j1/download/model-60000.glb"

    def test_an_untouched_job_labels_its_only_model(self, tmp_path):
        """Jobs built before build_faces was recorded still know their count --
        nothing has superseded model.glb, so `faces` is still describing it."""
        out = _out(tmp_path)
        _write(out, "model.glb")
        listed = _model_list("j1", out, build_faces=None, current_faces=297100)
        assert listed[0]["faces"] == 297100

    def test_an_unknown_build_count_is_left_blank(self, tmp_path):
        """A job re-optimised before build_faces existed. Saying nothing beats
        attributing the variant's count to the build."""
        out = _out(tmp_path)
        _write(out, "model.glb", age=300)
        _write(out, "model-60000.glb", age=10)
        listed = _model_list("j1", out, build_faces=None, current_faces=60000)
        assert listed[0]["faces"] is None
        assert listed[1]["faces"] == 60000


class TestConversionsDoNotCollide:
    def test_the_build_model_keeps_the_plain_name(self):
        assert _export_stem("lighter", "model.glb") == "lighter"

    def test_a_variant_is_suffixed(self):
        """Without this, converting a re-optimised model overwrites the STL of
        the model it was derived from -- same name, different geometry."""
        assert _export_stem("lighter", "model-60000.glb") == "lighter-60000"


class TestTheViewerCanSeeTheChange:
    def test_the_url_carries_a_version(self, tmp_path):
        """A constant URL is served from the browser's cache and never
        remounts the viewer, so a working re-optimise and one that did nothing
        looked identical on screen."""
        out = _out(tmp_path)
        model = _write(out, "model.glb")
        url = _model_url("j1", model)
        assert url is not None and url.startswith("/api/jobs/j1/model.glb?v=")

    def test_a_new_model_gets_a_new_url(self, tmp_path):
        out = _out(tmp_path)
        first = _write(out, "model.glb", age=300)
        second = _write(out, "model-60000.glb")
        assert _model_url("j1", first) != _model_url("j1", second)

    def test_no_model_has_no_url(self, tmp_path):
        assert _model_url("j1", tmp_path / "missing.glb") is None


class TestTheApiSurfacesTheModels:
    def _restored(self, tmp_path: Path, models: dict[str, float]) -> TestClient:
        import json

        session = tmp_path / "20260810-000000-lighter"
        out = session / "output"
        out.mkdir(parents=True)
        (session / "session.json").write_text(
            json.dumps({
                "job_id": "abc123",
                "name": "lighter",
                "created_at": "2026-08-10T00:00:00",
                "faces": 60000,
                "build_faces": 297100,
            })
        )
        for name, age in models.items():
            _write(out, name, age=age)
        return TestClient(create_app(tmp_path))

    def test_the_list_counts_them(self):
        """"How many models does this job have" is the one thing that tells you
        which jobs you have already re-optimised without opening each one."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = self._restored(
                Path(tmp), {"model.glb": 300, "model-60000.glb": 10}
            )
            assert client.get("/api/jobs").json()[0]["models"] == 2

    def test_the_detail_lists_them(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = self._restored(
                Path(tmp), {"model.glb": 300, "model-60000.glb": 10}
            )
            detail = client.get("/api/jobs/abc123").json()
            assert [m["name"] for m in detail["models"]] == [
                "model.glb",
                "model-60000.glb",
            ]
            assert [m["faces"] for m in detail["models"]] == [297100, 60000]

    def test_a_restart_comes_back_on_the_newest_model(self):
        """Restoring to model.glb would silently undo the re-optimise: the
        viewer, the sidebar and every conversion would go back to the geometry
        the user had already replaced."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = self._restored(
                Path(tmp), {"model.glb": 300, "model-60000.glb": 10}
            )
            body = client.get("/api/jobs/abc123/model.glb")
            assert body.status_code == 200
            detail = client.get("/api/jobs/abc123").json()
            assert next(m for m in detail["models"] if m["current"])["name"] == (
                "model-60000.glb"
            )

    def test_a_job_with_only_a_variant_still_counts_as_finished(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            client = self._restored(Path(tmp), {"model-60000.glb": 10})
            assert client.get("/api/jobs").json()[0]["status"] == "done"


class TestARealReoptimiseKeepsBothModels:
    """End to end through the endpoint, on a real mesh.

    Kept cheap on purpose: an icosphere at normal_res 128. The heavy part of
    re-optimise is the normal bake, which is the one stage on this machine
    measured in gigabytes.
    """

    def _job(self, tmp_path: Path):
        import trimesh
        from fastapi.testclient import TestClient

        # tests/ is not a package, so this is the import pytest's rootdir
        # insertion actually supports.
        from test_web import _generate_test_jpeg

        client = TestClient(create_app(tmp_path))
        res = client.post(
            "/api/jobs",
            files=[("files", ("photo1.jpg", _generate_test_jpeg(1), "image/jpeg"))],
            data={"name": "lighter"},
        )
        job_id = res.json()["job_id"]
        job_dir = Path(client.get(f"/api/jobs/{job_id}").json()["dir"])
        trimesh.creation.icosphere(subdivisions=3).export(job_dir / "dense.ply")
        # Stand in for the model the build shipped -- a real GLB, and
        # deliberately not the same shape as dense.ply, so "the build's model
        # survived" cannot pass by accident. This is the file that used to be
        # destroyed by the first re-optimise.
        out = job_dir / "output"
        out.mkdir(parents=True, exist_ok=True)
        trimesh.creation.box().export(out / "model.glb")
        return client, job_id, job_dir

    def test_the_build_model_survives_and_the_variant_appears(self, tmp_path):
        client, job_id, job_dir = self._job(tmp_path)
        out = job_dir / "output"
        before = (out / "model.glb").read_bytes()

        res = client.post(
            f"/api/jobs/{job_id}/optimize",
            json={"target_faces": 200, "normal_res": 128},
        )
        assert res.status_code == 200, res.text
        body = res.json()

        assert (out / "model.glb").read_bytes() == before, (
            "re-optimise destroyed the model the build produced"
        )
        variant = out / _variant_name(body["faces"])
        assert variant.is_file()
        assert body["model"] == variant.name
        assert body["models"] == 2

        detail = client.get(f"/api/jobs/{job_id}").json()
        assert [m["name"] for m in detail["models"]] == ["model.glb", variant.name]
        assert detail["glb_url"].startswith(f"/api/jobs/{job_id}/model.glb?v=")
        assert client.get("/api/jobs").json()[0]["models"] == 2

    def test_the_same_budget_twice_is_the_same_model(self, tmp_path):
        """Not a third file. Re-running at a budget you already tried is the
        same geometry, and accumulating copies of it would be noise."""
        client, job_id, job_dir = self._job(tmp_path)
        for _ in range(2):
            res = client.post(
                f"/api/jobs/{job_id}/optimize",
                json={"target_faces": 200, "normal_res": 128},
            )
            assert res.status_code == 200, res.text
        assert res.json()["models"] == 2

    def test_converting_after_it_uses_the_new_geometry(self, tmp_path):
        """The trap this replaced: convert defaulted to model.glb, so pressing
        "download STL" after a re-optimise handed back the geometry that had
        just been superseded -- under a name that overwrote nothing and
        explained nothing."""
        client, job_id, job_dir = self._job(tmp_path)
        opt = client.post(
            f"/api/jobs/{job_id}/optimize",
            json={"target_faces": 200, "normal_res": 128},
        ).json()

        res = client.post(f"/api/jobs/{job_id}/convert", json={"format": "stl"})
        assert res.status_code == 200, res.text
        assert res.json()["file"] == f"lighter-{opt['faces']}.stl"

        # And asking for the build's own model still gets the build's own
        # name, so both STLs sit in the directory at once instead of one
        # silently overwriting the other.
        explicit = client.post(
            f"/api/jobs/{job_id}/convert",
            json={"format": "stl", "source": "model.glb"},
        )
        assert explicit.status_code == 200, explicit.text
        assert explicit.json()["file"] == "lighter.stl"
        names = {f["name"] for f in client.get(f"/api/jobs/{job_id}/files").json()["files"]}
        assert {"lighter.stl", f"lighter-{opt['faces']}.stl"} <= names
