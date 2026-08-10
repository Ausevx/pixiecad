"""A generation setting the chosen backend cannot read must never be silent.

Each backend filters ``GenerateRequest.options`` against its own allow-list
before building a remote command. That is right -- a typo must not quietly
change generation settings -- but the same filter dropped DELIBERATE settings
just as quietly, and the difference is invisible from the outside.

It cost real GPU time to find out. The dashboard's "generation detail" control
sets ``octree_resolution``, which only Hunyuan reads. Every TRELLIS build made
from the dashboard therefore generated at the worker's own default however the
control was set: five consecutive runs asked for maximum detail, all five
ignored it, and nothing in the log, the stage list or the warnings said so.

The fix is not to invent a translation. TRELLIS takes ``geometry_resolution``,
whose only values we know to be good are the 512 and 1024 its own two mesh
profiles use -- guessing a mapping and shipping it unverified would swap a
silent no-op for a silent wrong answer, on a leg that costs a rented L4 to
test. So the backend keeps ignoring what it does not understand, and the job
says which settings those were.
"""

from __future__ import annotations

from pixiecad.generative import BACKEND_OPTIONS, unsupported_options
from pixiecad.pipeline import _warn_about_ignored_options


class TestTheManifestMatchesTheBackends:
    """Mirrored from each backend module rather than restated, so a backend
    that learns a knob cannot drift from what we claim it understands."""

    def test_hunyuan_reads_the_detail_knob(self):
        assert "octree_resolution" in BACKEND_OPTIONS["hunyuan-remote"]

    def test_trellis_does_not(self):
        assert "octree_resolution" not in BACKEND_OPTIONS["trellis-remote"]

    def test_each_manifest_is_its_module_s_own_list(self):
        from pixiecad.generative import hunyuan_remote, trellis_remote

        assert BACKEND_OPTIONS["hunyuan-remote"] == frozenset(
            hunyuan_remote.PASSTHROUGH_OPTIONS
        )
        assert BACKEND_OPTIONS["trellis-remote"] == frozenset(
            trellis_remote.PASSTHROUGH_OPTIONS
        )


class TestWhatABackendWillIgnore:
    def test_trellis_ignores_generation_detail(self):
        assert unsupported_options("trellis-remote", {"octree_resolution": 512}) == [
            "octree_resolution"
        ]

    def test_hunyuan_ignores_the_worker_face_budget(self):
        assert unsupported_options("hunyuan-remote", {"decimation_target": 300000}) == [
            "decimation_target"
        ]

    def test_a_setting_the_backend_reads_is_not_reported(self):
        assert unsupported_options("hunyuan-remote", {"octree_resolution": 512}) == []

    def test_an_unknown_backend_says_nothing(self):
        """Silence beats warning about a manifest we do not have. FakeBackend
        and anything added later land here until someone lists it."""
        assert unsupported_options("fake", {"octree_resolution": 512}) == []

    def test_every_ignored_option_is_reported_not_just_the_first(self):
        got = unsupported_options(
            "trellis-remote", {"octree_resolution": 512, "guidance_scale": 7.5}
        )
        assert got == ["guidance_scale", "octree_resolution"]


class TestTheWarningIsUseful:
    def _warn(self, backend: str, options: dict, requested: str | None = None):
        warnings: list[str] = []
        _warn_about_ignored_options(backend, options, warnings, requested=requested)
        return warnings

    def test_it_names_the_control_the_user_touched(self):
        """"octree_resolution" is a worker field. The user chose "generation
        detail", and a warning that does not use their word for it sends them
        looking for a setting that is not on the screen."""
        (msg,) = self._warn(
            "trellis-remote", {"octree_resolution": 512}, requested="trellis-remote"
        )
        assert "generation detail" in msg
        assert "octree_resolution=512" in msg
        assert "trellis-remote" in msg

    def test_it_says_the_backend_was_chosen_for_them(self):
        """An auto-selected backend makes this a surprise rather than a
        mistake, so the sentence has to admit the user never picked it."""
        (msg,) = self._warn("trellis-remote", {"octree_resolution": 512}, requested=None)
        assert "automatically" in msg

    def test_it_does_not_say_that_when_the_backend_was_asked_for(self):
        (msg,) = self._warn(
            "trellis-remote", {"octree_resolution": 512}, requested="trellis-remote"
        )
        assert "automatically" not in msg

    def test_a_backend_that_reads_everything_warns_about_nothing(self):
        assert self._warn("hunyuan-remote", {"octree_resolution": 512}) == []

    def test_no_options_is_not_a_warning(self):
        assert self._warn("trellis-remote", {}) == []
        assert self._warn("trellis-remote", None) == []

    def test_only_the_user_s_own_options_are_reported(self):
        """decimation_target and mesh_profile are filled in by
        _generative_defaults, not chosen by anyone. Warning that Hunyuan
        ignores a TRELLIS knob we added ourselves is noise that would train
        the user to skip the warnings that matter."""
        assert self._warn("hunyuan-remote", {"octree_resolution": 256}) == []


class TestThePipelineRaisesIt:
    def test_generation_reports_ignored_options(self):
        """Wired to the backend that RAN, not the one that was requested --
        with auto-selection those differ, and it is the one that ran whose
        manifest decides what was dropped."""
        import inspect

        from pixiecad import pipeline

        src = inspect.getsource(pipeline._run_generative)
        assert "_warn_about_ignored_options(" in src
        call = src[src.index("_warn_about_ignored_options(") :][:200]
        assert "result.backend" in call
        assert "generative_options" in call, (
            "must report the caller's options, not the defaults-filled dict"
        )
