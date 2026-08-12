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

    def test_semantic_segmentation_passes_cache_dir(self, tmp_path: Path):
        """Dashboard finishing path must thread a non-None stages cache_dir to SAM."""
        from unittest.mock import patch

        path = tmp_path / "model.glb"
        trimesh.creation.box().export(path)

        captured_cache_dir: list[Path | None] = []

        def fake_run_semantic_segmentation(mesh, executor, max_parts, *, cache_dir=None, log=lambda _m: None):
            captured_cache_dir.append(cache_dir)
            return [], False

        with patch("pixiecad.web.finishing.run_semantic_segmentation", fake_run_semantic_segmentation):
            finish_model(
                path,
                tmp_path,
                FinishOptions(segmentation="semantic", gpu_host="gpu.example.invalid"),
            )

        assert len(captured_cache_dir) == 1
        assert captured_cache_dir[0] is not None
        assert captured_cache_dir[0] == tmp_path / "stages"

    def test_finish_model_accepts_custom_cache_dir(self, tmp_path: Path):
        """Passing an explicit cache_dir to finish_model overrides the default stages dir."""
        from unittest.mock import patch

        path = tmp_path / "model.glb"
        trimesh.creation.box().export(path)

        captured_cache_dir: list[Path | None] = []

        def fake_run_semantic_segmentation(mesh, executor, max_parts, *, cache_dir=None, log=lambda _m: None):
            captured_cache_dir.append(cache_dir)
            return [], False

        custom_cache = tmp_path / "custom_cache"
        with patch("pixiecad.web.finishing.run_semantic_segmentation", fake_run_semantic_segmentation):
            finish_model(
                path,
                tmp_path,
                FinishOptions(segmentation="semantic", gpu_host="gpu.example.invalid"),
                cache_dir=custom_cache,
            )

        assert len(captured_cache_dir) == 1
        assert captured_cache_dir[0] == custom_cache


class TestTexturingKeepsTheFaceBudget:
    """Texturing must not silently discard the face budget.

    Hunyuan's paint stage remeshes before texturing, hardcoded to 40,000
    faces, and finish_model publishes that mesh as the model. So a job that
    asked for 200,000 faces, decimated to exactly 200,000 and exported them,
    came back with 40,000 the moment texturing was switched on -- with nothing
    in the log to say so.
    """

    def test_the_budget_is_passed_to_the_paint_stage(self):
        from pixiecad.generative.hunyuan_remote import build_texture_script

        script = build_texture_script(simplify_target=200000)
        assert "HUNYUAN_SIMPLIFY_TARGET=200000" in script

    def test_no_target_leaves_upstream_behaviour_alone(self):
        """Absent means the image keeps its own 40,000 default."""
        from pixiecad.generative.hunyuan_remote import build_texture_script

        assert "HUNYUAN_SIMPLIFY_TARGET" not in build_texture_script()

    def test_the_env_var_precedes_the_image_name(self):
        """docker run flags must come before the image, or they are passed to
        the container as arguments instead of to docker."""
        from pixiecad.generative.hunyuan_remote import build_texture_script

        script = build_texture_script(simplify_target=123456)
        assert script.index("HUNYUAN_SIMPLIFY_TARGET") < script.index("python /work/")

    def test_finishing_asks_for_the_job_budget(self):
        """FinishOptions.total_budget is the number the user chose."""
        import inspect

        from pixiecad.web import finishing

        src = inspect.getsource(finishing.finish_model)
        assert "simplify_target=options.total_budget" in src

    def test_a_remesh_loss_is_reported(self):
        """If the paint stage shrinks it anyway, that must reach the warnings
        rather than leaving someone to compare face counts and guess."""
        import inspect

        from pixiecad.web import finishing

        src = inspect.getsource(finishing.finish_model)
        assert "texturing remeshed" in src


class TestCachedWeightsAreAuthoritative:
    """A job must not phone Hugging Face for weights already on the disk.

    The weights are baked into the machine image so a run never downloads
    them. But huggingface_hub does not treat a full cache as sufficient --
    snapshot_download contacts the Hub to resolve the revision on every call.
    A Hub CDN 500 therefore killed a job that needed nothing from the Hub,
    after generation and solidify had already succeeded.

    Shipped first as an absolute, which was worse. The premise that the image
    holds every repo turned out to be false -- tencent/Hunyuan3D-2.1, which
    texturing requires, was never in the cache at all -- so offline mode turned
    a slow download into a failed build. It is now a preference: see
    TestOfflineFallsBackOnACacheMiss.
    """

    def test_texturing_runs_offline_by_default(self, monkeypatch):
        from pixiecad.generative.hunyuan_remote import build_texture_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        assert "HF_HUB_OFFLINE=1" in build_texture_script()

    def test_generation_runs_offline_by_default(self, monkeypatch):
        from pixiecad.generative.hunyuan_remote import build_hunyuan_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        assert "HF_HUB_OFFLINE=1" in build_hunyuan_script()

    def test_the_trellis_worker_runs_offline_by_default(self, monkeypatch):
        from pixiecad.generative.trellis_remote import build_trellis_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        assert "HF_HUB_OFFLINE=1" in build_trellis_script()

    def test_it_can_be_turned_off_to_add_a_model(self, monkeypatch):
        """Offline has to be escapable, or fetching something genuinely new
        becomes impossible."""
        from pixiecad.generative.hunyuan_remote import build_texture_script

        monkeypatch.setenv("PIXIECAD_HF_ONLINE", "1")
        # The name still appears in hf_run's own cache-miss pattern, so assert
        # on the assignment: it is the flag that must be empty.
        script = build_texture_script()
        assert 'HF_OFFLINE=""' in script
        assert 'HF_OFFLINE="-e HF_HUB_OFFLINE=1"' not in script

    def test_the_flag_precedes_the_image_name(self, monkeypatch):
        """docker run flags after the image are passed to the container as
        arguments, not to docker."""
        from pixiecad.generative.hunyuan_remote import build_texture_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        script = build_texture_script()
        assert script.index("HF_HUB_OFFLINE") < script.index("python /work/")


class TestOfflineFallsBackOnACacheMiss:
    """Offline mode must never be the reason a build fails.

    The regression, from a real job whose geometry had already succeeded:

        WARNING: texturing failed: ...
        LocalEntryNotFoundError: Cannot find an appropriate cached snapshot
        folder for the specified revision on the local disk and outgoing
        traffic has been disabled.

    tencent/Hunyuan3D-2.1 -- the paint weights -- was not in the baked cache.
    The earlier CDN 500 that motivated offline mode had been that same repo
    downloading live and failing; offline did not cause the gap, it made it
    fatal. So it now retries with the network on a genuine cache miss, and
    only on a cache miss.

    These run the emitted shell for real against a stub, because the whole
    thing is control flow: asserting on substrings would have passed for a
    wrapper that never actually retried.
    """

    # The stub reads $HF_OFFLINE from the environment rather than its argv,
    # exactly as the generated _hf_cmd function does. This is load-bearing:
    # hf_run re-invokes with "$@", so a caller that expanded the flag at the
    # call site would hand the retry the SAME flag and never actually go
    # online -- a bug this harness would hide if it took the flag as an arg.
    PREAMBLE_STUB = """
attempts=0
fake() {
  attempts=$((attempts+1))
  case "$HF_OFFLINE" in
    *HF_HUB_OFFLINE*)
      echo "LocalEntryNotFoundError: Cannot find an appropriate cached snapshot" >&2
      return 1 ;;
    *) echo "fetched over the network" ; return 0 ;;
  esac
}
"""

    @staticmethod
    def _sh(script: str):
        import subprocess

        return subprocess.run(
            ["sh", "-c", script], capture_output=True, text=True, timeout=30
        )

    def _run(self, body: str, monkeypatch, *, online: bool = False):
        from pixiecad.generative.hunyuan_remote import hf_offline_preamble

        if online:
            monkeypatch.setenv("PIXIECAD_HF_ONLINE", "1")
        else:
            monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        # set -e exactly as every generated script opens, because that is what
        # turns a failed hf_run into a failed job. Without it here the suite
        # would happily pass a wrapper that swallowed every error.
        return self._sh("set -e\n" + hf_offline_preamble() + self.PREAMBLE_STUB + body)

    def test_a_cache_miss_is_retried_with_the_network(self, monkeypatch):
        r = self._run("hf_run fake\necho rc=$?", monkeypatch)
        assert r.returncode == 0, r.stderr
        assert "fetched over the network" in r.stdout
        assert "rc=0" in r.stdout

    def test_the_retry_actually_drops_the_offline_flag(self, monkeypatch):
        """Retrying with the same flags would just fail identically."""
        r = self._run(
            "hf_run fake\necho attempts=$attempts", monkeypatch
        )
        assert "attempts=2" in r.stdout

    def test_an_unrelated_failure_is_not_retried(self, monkeypatch):
        """A CUDA OOM is not a cache miss. Running the whole job twice to
        rediscover that would double the GPU bill for nothing."""
        r = self._run(
            "broken() { attempts=$((attempts+1)); echo 'CUDA out of memory' >&2; return 1; }\n"
            "hf_run broken || true\necho attempts=$attempts",
            monkeypatch,
        )
        assert "attempts=1" in r.stdout

    def test_an_unrelated_failure_still_fails(self, monkeypatch):
        r = self._run(
            "broken() { echo 'CUDA out of memory' >&2; return 1; }\n"
            "hf_run broken\necho SHOULD-NOT-REACH",
            monkeypatch,
        )
        assert "SHOULD-NOT-REACH" not in r.stdout
        assert r.returncode != 0

    def test_a_cache_hit_never_touches_the_network(self, monkeypatch):
        """The property offline mode exists for: one attempt, offline, done."""
        r = self._run(
            "cached() { attempts=$((attempts+1)); echo 'loaded from cache'; }\n"
            "hf_run cached\necho attempts=$attempts",
            monkeypatch,
        )
        assert r.returncode == 0
        assert "attempts=1" in r.stdout

    def test_output_reaches_the_log_either_way(self, monkeypatch):
        """hf_run buffers to a file to inspect it; it must still print it, or
        every remote failure becomes undiagnosable."""
        r = self._run(
            "chatty() { echo 'step 1 of 3'; return 0; }\nhf_run chatty",
            monkeypatch,
        )
        assert "step 1 of 3" in r.stdout

    def test_online_mode_skips_straight_to_the_network(self, monkeypatch):
        r = self._run(
            "hf_run fake\necho attempts=$attempts", monkeypatch, online=True
        )
        assert "attempts=1" in r.stdout
        assert "fetched over the network" in r.stdout

    def test_every_generated_script_is_valid_posix_sh(self, monkeypatch):
        """The preamble is spliced into three scripts by string formatting."""
        import subprocess

        from pixiecad.generative.hunyuan_remote import (
            build_hunyuan_script,
            build_texture_script,
        )
        from pixiecad.generative.trellis_remote import build_trellis_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        for script in (build_texture_script(), build_hunyuan_script(), build_trellis_script()):
            check = subprocess.run(
                ["sh", "-n"], input=script, capture_output=True, text=True, timeout=30
            )
            assert check.returncode == 0, check.stderr

    def test_the_trellis_worker_restarts_online_on_a_cache_miss(self, monkeypatch):
        """That container is detached, so hf_run cannot wrap it -- the recovery
        has to be spliced into the readiness loop instead."""
        from pixiecad.generative.trellis_remote import build_trellis_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        script = build_trellis_script()
        assert script.count('HF_OFFLINE=""') >= 1
        assert "restarting the worker with network access" in script
        # It must re-run the container, not just clear the variable and wait.
        after = script[script.index("restarting the worker"):]
        assert "docker run -d" in after

    def test_every_generated_script_sets_e(self, monkeypatch):
        """The retry reports a non-cache-miss failure by returning non-zero,
        which only stops the job because of set -e."""
        from pixiecad.generative.hunyuan_remote import (
            build_hunyuan_script,
            build_texture_script,
        )
        from pixiecad.generative.trellis_remote import build_trellis_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        for script in (build_texture_script(), build_hunyuan_script(), build_trellis_script()):
            assert script.startswith("set -e\n")

    @pytest.mark.parametrize(
        "payload",
        [
            "m\'.glb",
            "m\';touch {M};x=\'.glb",
            "m$(touch {M}).glb",
            "m`touch {M}`.glb",
            "m;touch {M};.glb",
            'm";touch {M};x=".glb',
        ],
    )
    @pytest.mark.parametrize("builder,arg", [("texture", "mesh_name"), ("shape", "image_name")])
    def test_a_filename_cannot_execute_shell(self, payload, builder, arg, tmp_path):
        """Filenames are staged as "00<suffix>" and "mesh.glb", but the SUFFIX
        is the user's, and "photo$(...).glb" is a legal filename.

        Two mistakes were made here in sequence, so both are pinned:

        1. The command was passed to hf_run as a string and ``eval``ed, so it
           was parsed TWICE. shlex.quote defeats the first parse only --
           measured, `photo\'.glb` was stopped but `photo$(touch X).glb` ran.
           hf_run now takes a function NAME and re-invokes it with "$@".
        2. A function body is parsed once but still EXPANDS at call time, so
           the arguments themselves must be quoted as well.
        """
        import subprocess

        from pixiecad.generative.hunyuan_remote import (
            build_hunyuan_script,
            build_texture_script,
        )

        build = build_texture_script if builder == "texture" else build_hunyuan_script
        marker = tmp_path / "PWNED"
        script = build(**{arg: payload.format(M=marker)})
        # docker is stubbed out; an escaped payload would run regardless.
        r = subprocess.run(
            ["sh", "-c", "docker() { :; }\n" + script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert not marker.exists(), f"{payload!r} escaped and executed"
        assert r.returncode == 0, r.stderr

    def test_the_trellis_restart_is_guarded_against_looping(self, monkeypatch):
        """Without the guard, a container that keeps dying would be restarted
        forever: each restart resets tries and continues the wait loop."""
        from pixiecad.generative.trellis_remote import build_trellis_script

        monkeypatch.delenv("PIXIECAD_HF_ONLINE", raising=False)
        script = build_trellis_script()
        for block in script.split("restarting the worker with network access")[:-1]:
            assert '[ -n "$HF_OFFLINE" ]' in block.rsplit("if ", 1)[-1] or \
                   '[ -n "$HF_OFFLINE" ]' in block[-400:], "restart is unguarded"

    def test_the_cache_miss_pattern_does_not_match_a_mere_mention(self):
        """A startup banner echoing HF_HUB_OFFLINE=1, or the error's own advice
        to 'set HF_HUB_OFFLINE=0', must not trigger an expensive GPU restart."""
        import re

        from pixiecad.generative.hunyuan_remote import HF_CACHE_MISS

        innocent = [
            "docker run -e HF_HUB_OFFLINE=1 pixiecad-hunyuan:latest",
            "[worker] starting with HF_HUB_OFFLINE=1",
            "CUDA out of memory",
        ]
        for line in innocent:
            assert not re.search(HF_CACHE_MISS, line), f"false positive on: {line}"

    def test_the_cache_miss_pattern_matches_the_real_error(self):
        """The message from the job this was written for."""
        import re

        from pixiecad.generative.hunyuan_remote import HF_CACHE_MISS

        real = (
            "huggingface_hub.errors.LocalEntryNotFoundError: Cannot find an "
            "appropriate cached snapshot folder for the specified revision on the "
            "local disk and outgoing traffic has been disabled."
        )
        assert re.search(HF_CACHE_MISS, real)


class TestThePaintImageMustHonourTheBudget:
    """Passing an env var the image cannot read is worse than not passing it.

    PixieCAD sends -e HUNYUAN_SIMPLIFY_TARGET so the paint stage keeps the
    user's face budget. The image had NO code reading that variable:
    mesh_simplify_trimesh takes target_count=40000 as a hardcoded default and
    remesh_mesh never passes one. So the flag was set on every textured run and
    did nothing, and a 300,000-face job silently shipped 39,997 faces with only
    a face count as the clue. launch_from_image.sh now patches the image.
    """

    def test_the_launch_script_patches_the_image_to_read_it(self):
        from pathlib import Path

        script = Path("scripts/launch_from_image.sh").read_text()
        assert "HUNYUAN_SIMPLIFY_TARGET" in script
        patch = script[script.index("patching $IMG to honour HUNYUAN_SIMPLIFY_TARGET"):]
        assert "simplify_mesh_utils.py" in patch
        assert "docker commit" in patch, "the patch must persist into the image"

    def test_the_patch_keeps_the_upstream_default(self):
        """A missing variable must mean upstream behaviour, not zero faces."""
        from pathlib import Path

        script = Path("scripts/launch_from_image.sh").read_text()
        # The quotes are \x27 escapes: the patch is python inside a docker
        # command inside an ssh command, and a literal ' would end the shell
        # string it is embedded in.
        assert "HUNYUAN_SIMPLIFY_TARGET\\x27, 40000" in script

    def test_it_is_idempotent(self):
        """Self-heal runs on every launch; re-patching a patched image would
        stack edits and eventually corrupt the file."""
        from pathlib import Path

        script = Path("scripts/launch_from_image.sh").read_text()
        block = script[script.index("HUNYUAN_SIMPLIFY_TARGET already honoured") - 400:]
        assert "grep -q" in block

    def test_the_patch_logic_produces_working_code(self):
        """Run the same two replacements the container patch does, against the
        upstream source, and check the result compiles and reads the env."""
        import os
        import textwrap

        upstream = textwrap.dedent('''
            import trimesh
            import pymeshlab

            def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
                return target_count
        ''')
        s = upstream.replace("import trimesh", "import os\nimport trimesh", 1)
        s = s.replace(
            "def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):",
            "def mesh_simplify_trimesh(inputpath, outputpath, target_count=None):\n"
            "    if target_count is None:\n"
            "        target_count = int(os.environ.get('HUNYUAN_SIMPLIFY_TARGET', 40000))",
        )
        ns: dict = {}
        exec(compile(s.replace("import pymeshlab", ""), "<patched>", "exec"), ns)

        os.environ["HUNYUAN_SIMPLIFY_TARGET"] = "300000"
        try:
            assert ns["mesh_simplify_trimesh"]("a", "b") == 300000
        finally:
            del os.environ["HUNYUAN_SIMPLIFY_TARGET"]
        assert ns["mesh_simplify_trimesh"]("a", "b") == 40000
        assert ns["mesh_simplify_trimesh"]("a", "b", 123) == 123
