"""An unusable GPU backend has to say why it is unusable.

Written after a real job died on this line:

    [FAILED] generate: Backend 'hunyuan-remote' is not available.
             Registered backends: ['fake', 'hunyuan-remote', 'trellis-remote']

The name was correct and the list was noise. The actual cause was a stale SSH
HostKeyAlias left behind when the VM was recreated with the same name and IP,
and the fix was one command -- but ssh's explanation had already been thrown
away by ``except Exception: return False``.
"""

from __future__ import annotations

import pytest

from pixiecad.executors.base import Capabilities, GPUInfo
from pixiecad.executors.ssh import _ssh_hint
from pixiecad.generative.base import BackendUnavailable, get_backend, register_backend
from pixiecad.generative.hunyuan_remote import HUNYUAN_MIN_VRAM_MB, RemoteHunyuanBackend

HOST = "gpu.example.invalid"

CHANGED_KEY_STDERR = """@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
Host key verification failed."""


class _Probe:
    """Executor stub returning one fixed Capabilities."""

    def __init__(self, caps) -> None:
        self._caps = caps

    def probe(self):
        if isinstance(self._caps, Exception):
            raise self._caps
        return self._caps

    def run(self, job):  # pragma: no cover - never reached in these tests
        raise AssertionError("no job should run when the backend is unavailable")


class TestSshHint:
    def test_changed_host_key_names_the_fix(self):
        """The whole banner collapses to the one command that resolves it."""
        hint = _ssh_hint(CHANGED_KEY_STDERR)
        assert "gcloud compute config-ssh" in hint
        assert "recreated" in hint
        assert "@@@" not in hint, "the banner must not survive into a job log"

    @pytest.mark.parametrize(
        "stderr,expected",
        [
            ("Permission denied (publickey).", "permission denied"),
            ("ssh: Could not resolve hostname foo", "does not resolve"),
            ("connect to host foo port 22: Connection refused", "refused"),
            ("connect to host foo port 22: Connection timed out", "timed out"),
        ],
    )
    def test_common_failures_are_named(self, stderr, expected):
        assert expected in _ssh_hint(stderr).lower()

    def test_unknown_failure_keeps_the_last_line(self):
        """Never invent a diagnosis, but never return nothing either."""
        assert _ssh_hint("something\nweird happened here") == "weird happened here"

    def test_empty_stays_empty(self):
        assert _ssh_hint("   \n  ") == ""


class TestWhyNot:
    def test_unreachable_reports_the_ssh_reason(self):
        caps = Capabilities(HOST, None, False, detail="host key mismatch")
        why = caps.why_not(HUNYUAN_MIN_VRAM_MB)
        assert "cannot reach" in why and "host key mismatch" in why

    def test_reachable_without_gpu_says_so(self):
        """Distinct from unreachable: the box is fine, the GPU is the problem."""
        caps = Capabilities(HOST, None, True, detail="nvidia-smi failed")
        why = caps.why_not(HUNYUAN_MIN_VRAM_MB)
        assert "no GPU" in why and "nvidia-smi failed" in why

    def test_small_gpu_reports_both_numbers(self):
        caps = Capabilities(HOST, GPUInfo("NVIDIA T4", 4096, 7.5), True)
        why = caps.why_not(HUNYUAN_MIN_VRAM_MB)
        assert "4096" in why and str(HUNYUAN_MIN_VRAM_MB) in why

    def test_a_good_host_has_nothing_to_explain(self):
        caps = Capabilities(HOST, GPUInfo("NVIDIA L4", 23034, 8.9), True)
        assert caps.why_not(HUNYUAN_MIN_VRAM_MB) == ""


class TestBackendErrorMessage:
    def test_the_real_reason_replaces_the_registry_dump(self):
        """The regression this file exists for."""
        caps = Capabilities(
            HOST,
            None,
            False,
            detail=_ssh_hint(CHANGED_KEY_STDERR),
        )
        backend = RemoteHunyuanBackend(_Probe(caps))
        assert backend.available() is False

        register_backend(lambda: backend, "hunyuan-remote-test")
        with pytest.raises(BackendUnavailable) as excinfo:
            get_backend("hunyuan-remote-test")

        msg = str(excinfo.value)
        assert "gcloud compute config-ssh" in msg
        assert "Registered backends" not in msg, (
            "listing registered names implies the NAME was wrong, which sends "
            "the user looking in entirely the wrong place"
        )

    def test_an_actually_wrong_name_still_lists_what_exists(self):
        with pytest.raises(BackendUnavailable, match="Registered backends"):
            get_backend("hunyuan-remot")  # typo

    def test_a_probe_that_raises_is_reported_not_swallowed(self):
        backend = RemoteHunyuanBackend(_Probe(OSError("ssh binary missing")))
        assert backend.available() is False
        assert "ssh binary missing" in backend.last_unavailable_reason

    def test_recovery_clears_the_stale_reason(self):
        """A host that comes back must not keep reporting an old failure."""
        good = Capabilities(HOST, GPUInfo("NVIDIA L4", 23034, 8.9), True)
        backend = RemoteHunyuanBackend(_Probe(Capabilities(HOST, None, False, detail="down")))
        assert backend.available() is False
        backend.executor = _Probe(good)
        assert backend.available() is True
        assert backend.last_unavailable_reason == ""
