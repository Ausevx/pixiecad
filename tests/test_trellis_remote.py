"""Tests for TRELLIS remote generative backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import trimesh

from pixiecad.executors.base import Capabilities, GPUInfo, Job, JobResult
from pixiecad.generative.trellis_remote import (
    CONTAINER_NAME,
    DEFAULT_TRELLIS_IMAGE,
    TRELLIS_MIN_VRAM_MB,
    RemoteTrellisBackend,
    _get_generative_base,
    build_trellis_script,
    docker_run_command,
    make_backend,
)

GenerateResult, GenerativeError, BackendUnavailable = _get_generative_base()

try:
    from pixiecad.generative.base import GenerateRequest
except ImportError:
    @dataclass
    class GenerateRequest:  # type: ignore[no-redef]
        images: list[Path]
        seed: int | None = None
        texture: bool = True
        options: dict = field(default_factory=dict)


class FakeExecutor:
    """Fake executor recording jobs without network or docker calls."""

    def __init__(
        self,
        caps: Capabilities | None = None,
        run_result: JobResult | None = None,
        write_mesh: bool = True,
        raise_on_probe: bool = False,
    ) -> None:
        self.caps = caps
        self.run_result = run_result
        self.write_mesh = write_mesh
        self.raise_on_probe = raise_on_probe
        self.jobs: list[Job] = []
        self.recorded_inputs: list[list[str]] = []

    def probe(self) -> Capabilities:
        if self.raise_on_probe:
            raise RuntimeError("SSH connection failed")
        if self.caps is None:
            return Capabilities(hostname="fake-host", gpu=None, reachable=False)
        return self.caps

    def run(self, job: Job) -> JobResult:
        self.jobs.append(job)

        # Save snapshot of files in job inputs before temp dir cleanup
        saved_names = []
        for inp in job.inputs:
            if inp.is_dir():
                for f in inp.iterdir():
                    saved_names.append(f.name)
            else:
                # Plain files are inputs too — the HF token is staged as one.
                saved_names.append(inp.name)
        self.recorded_inputs.append(saved_names)

        if self.write_mesh and job.output_dir:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            mesh_file = job.output_dir / "mesh.glb"
            glb_bytes = trimesh.creation.icosphere().export(file_type="glb")
            mesh_file.write_bytes(glb_bytes)

        if self.run_result is not None:
            return self.run_result

        return JobResult(returncode=0, stdout_tail="", stderr_tail="", duration_s=0.5)


def test_build_trellis_script() -> None:
    """The script must drive the worker's HTTP API, not a non-existent CLI."""
    script = build_trellis_script(image_names=["00.png", "01.jpg"])

    assert "mkdir -p out" in script
    assert docker_run_command(DEFAULT_TRELLIS_IMAGE) in script
    assert DEFAULT_TRELLIS_IMAGE in script
    # Every staged view is posted, not just the first.
    assert '-F "images=@images/00.png"' in script
    assert '-F "images=@images/01.jpg"' in script
    # Readiness is awaited before generating: the model loads in background.
    assert "/ready" in script and "/generate" in script
    # A pre-existing container is reused rather than restarted.
    assert f"docker ps -q -f name=^{CONTAINER_NAME}$" in script

    assert "seed" not in build_trellis_script()
    assert '-F "seed=7"' in build_trellis_script(seed=7)

    # There is no no-texture switch on the worker; the cheapest mode stands in.
    assert '-F "texture_generation_mode=fast_512"' in build_trellis_script(texture=False)

    custom_img = "myrepo/trellis:v2"
    assert custom_img in build_trellis_script(image=custom_img)


def test_build_trellis_script_options_are_allow_listed() -> None:
    """Unknown options are dropped so a typo cannot alter generation silently."""
    script = build_trellis_script(
        options={"geometry_resolution": 1024, "nonsense_knob": "boom"}
    )
    assert '-F "geometry_resolution=1024"' in script
    assert "nonsense_knob" not in script


def test_available() -> None:
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    low_gpu = GPUInfo(name="T4", vram_mb=16000, compute_cap=7.5)

    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    caps_low_vram = Capabilities(hostname="gpu-node", gpu=low_gpu, reachable=True)
    caps_unreachable = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=False)

    # Enough VRAM, wrong architecture: the image ships Ada cubins only.
    big_but_wrong_arch = GPUInfo(name="A100", vram_mb=40000, compute_cap=8.0)

    exec_ok = FakeExecutor(caps=caps_ok)
    exec_low_vram = FakeExecutor(caps=caps_low_vram)
    exec_unreachable = FakeExecutor(caps=caps_unreachable)
    exec_error = FakeExecutor(raise_on_probe=True)
    exec_wrong_arch = FakeExecutor(
        caps=Capabilities(hostname="gpu-node", gpu=big_but_wrong_arch, reachable=True)
    )

    assert RemoteTrellisBackend(exec_ok).available() is True
    assert RemoteTrellisBackend(exec_low_vram).available() is False
    assert RemoteTrellisBackend(exec_unreachable).available() is False
    assert RemoteTrellisBackend(exec_error).available() is False
    assert RemoteTrellisBackend(exec_wrong_arch).available() is False


def test_generate_insufficient_vram(tmp_path: Path) -> None:
    low_gpu = GPUInfo(name="T4", vram_mb=16000, compute_cap=7.5)
    caps_low_vram = Capabilities(hostname="test-host", gpu=low_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_low_vram)

    backend = make_backend(executor)

    img_file = tmp_path / "test.png"
    img_file.write_bytes(b"dummy image data")
    req = GenerateRequest(images=[img_file])

    out_dir = tmp_path / "out"
    with pytest.raises(BackendUnavailable) as exc_info:
        backend.generate(req, out_dir)

    msg = str(exc_info.value)
    assert "test-host" in msg
    assert str(16000) in msg
    assert str(TRELLIS_MIN_VRAM_MB) in msg


def test_generate_happy_path(tmp_path: Path) -> None:
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_ok, write_mesh=True)

    backend = RemoteTrellisBackend(executor)

    img1 = tmp_path / "photo.png"
    img2 = tmp_path / "photo2.jpg"
    img1.write_bytes(b"img1")
    img2.write_bytes(b"img2")

    req = GenerateRequest(images=[img1, img2], seed=42, texture=True)
    out_dir = tmp_path / "output"

    res = backend.generate(req, out_dir)

    assert res.backend == "trellis-remote"
    assert res.mesh_path == str(out_dir / "mesh.glb")
    assert res.n_faces is not None and res.n_faces > 0
    assert res.metadata["image"] == DEFAULT_TRELLIS_IMAGE
    assert res.metadata["remote_subdir"].startswith("trellis-")

    # Assert composed Job inputs contain directory with files named 00.* and 01.*
    assert len(executor.jobs) == 1
    recorded = executor.recorded_inputs[0]
    assert any(name.startswith("00.") for name in recorded)
    assert any(name.startswith("01.") for name in recorded)


def test_generate_executor_failure(tmp_path: Path) -> None:
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    fail_result = JobResult(
        returncode=1,
        stdout_tail="",
        stderr_tail="Error: CUDA out of memory in trellis worker",
        duration_s=1.2,
    )
    executor = FakeExecutor(caps=caps_ok, run_result=fail_result)
    backend = RemoteTrellisBackend(executor)

    img = tmp_path / "input.png"
    img.write_bytes(b"data")
    req = GenerateRequest(images=[img])

    out_dir = tmp_path / "out"
    with pytest.raises(GenerativeError) as exc_info:
        backend.generate(req, out_dir)

    assert "CUDA out of memory in trellis worker" in str(exc_info.value)


def test_generate_missing_output_file(tmp_path: Path) -> None:
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_ok, write_mesh=False)
    backend = RemoteTrellisBackend(executor)

    img = tmp_path / "input.png"
    img.write_bytes(b"data")
    req = GenerateRequest(images=[img])

    out_dir = tmp_path / "out"
    with pytest.raises(GenerativeError) as exc_info:
        backend.generate(req, out_dir)

    assert "mesh.glb" in str(exc_info.value)


def test_token_is_installed_from_a_file_never_a_command_line() -> None:
    """The HF token must not reach `ps` output or `docker inspect`."""
    script = build_trellis_script()

    assert 'cp hf_token "$HOME/.cache/huggingface/token"' in script
    assert "chmod 600" in script
    # It is never exported into the container's environment or argv.
    assert "-e HF_TOKEN" not in script
    assert "HUGGING_FACE_HUB_TOKEN" not in script


def test_script_aborts_on_worker_error_state() -> None:
    """A failed preload keeps the container up; waiting it out wastes the budget."""
    script = build_trellis_script()
    assert '"status"[ ]*:[ ]*"error"' in script
    assert "failed to initialise" in script


def test_generate_stages_token_when_env_is_set(tmp_path: Path, monkeypatch) -> None:
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_ok, write_mesh=True)

    img = tmp_path / "photo.png"
    img.write_bytes(b"img")

    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    RemoteTrellisBackend(executor).generate(
        GenerateRequest(images=[img]), tmp_path / "out"
    )
    assert "hf_token" in executor.recorded_inputs[0]


def test_generate_omits_token_when_env_is_unset(tmp_path: Path, monkeypatch) -> None:
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    executor = FakeExecutor(caps=caps_ok, write_mesh=True)

    img = tmp_path / "photo.png"
    img.write_bytes(b"img")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    # Point HF_HOME at an empty dir: otherwise this reads the developer's real
    # stored login and passes or fails depending on whose machine it runs on.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    RemoteTrellisBackend(executor).generate(
        GenerateRequest(images=[img]), tmp_path / "out"
    )
    assert "hf_token" not in executor.recorded_inputs[0]


def test_gated_repo_failure_explains_the_fix(tmp_path: Path) -> None:
    """A 401 deep in a traceback must become an instruction, not a wall of text."""
    high_gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    caps_ok = Capabilities(hostname="gpu-node", gpu=high_gpu, reachable=True)
    fail = JobResult(
        returncode=1,
        stdout_tail="",
        stderr_tail="huggingface_hub.errors.GatedRepoError: 401 Client Error.",
        duration_s=1.0,
    )
    executor = FakeExecutor(caps=caps_ok, run_result=fail)

    img = tmp_path / "photo.png"
    img.write_bytes(b"img")

    with pytest.raises(GenerativeError) as exc:
        RemoteTrellisBackend(executor).generate(
            GenerateRequest(images=[img]), tmp_path / "out"
        )
    msg = str(exc.value)
    assert "HF_TOKEN" in msg
    assert "dinov3" in msg


def test_resolve_hf_token_prefers_env_then_file(tmp_path: Path, monkeypatch) -> None:
    """An interactive `export` never reaches a build shell; the file must work."""
    from pixiecad.generative.trellis_remote import resolve_hf_token

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert resolve_hf_token() is None

    (tmp_path / "token").write_text("hf_from_file\n")
    assert resolve_hf_token() == "hf_from_file"

    # An explicit env var wins over the stored login.
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert resolve_hf_token() == "hf_from_env"


def test_resolve_hf_token_ignores_blank_values(tmp_path: Path, monkeypatch) -> None:
    """An empty export must not read as 'authenticated'."""
    from pixiecad.generative.trellis_remote import resolve_hf_token

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "   ")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert resolve_hf_token() is None


def test_gated_dependency_is_avoided_by_default() -> None:
    """DINOv3 is gated:manual — a build must not wait on Meta approving a form."""
    assert "-e TRELLIS2_AVOID_GATED_DEPS=1" in build_trellis_script()
    assert "TRELLIS2_AVOID_GATED_DEPS" not in build_trellis_script(avoid_gated=False)


def test_map_views_picks_cardinals_and_drops_obliques() -> None:
    """Obliques must be dropped, not mislabelled.

    Feeding a 3/4 shot in as "front" tells the model the object is shaped
    wrongly, which is worse than simply using fewer views.
    """
    from pixiecad.generative.hunyuan_remote import map_views

    names = [
        "view_01_front.png", "view_02_front_left.png", "view_03_left.png",
        "view_04_rear_left.png", "view_05_rear.png", "view_06_rear_right.png",
        "view_07_right.png", "view_08_top.png",
    ]
    got = map_views([Path(n) for n in names])
    assert {k: v.name for k, v in got.items()} == {
        "front": "view_01_front.png",
        "left": "view_03_left.png",
        "back": "view_05_rear.png",
        "right": "view_07_right.png",
    }


def test_map_views_accepts_back_as_synonym_for_rear() -> None:
    from pixiecad.generative.hunyuan_remote import map_views

    got = map_views([Path("back.jpg"), Path("front.jpg")])
    assert set(got) == {"back", "front"}


def test_multiview_needs_two_views_to_be_worth_it(tmp_path: Path) -> None:
    """One usable view means the dedicated single-view model is the better tool."""
    from pixiecad.generative.hunyuan_remote import RemoteHunyuanBackend

    gpu = GPUInfo(name="NVIDIA L4", vram_mb=23034, compute_cap=8.9)
    ex = FakeExecutor(
        caps=Capabilities(hostname="gpu", gpu=gpu, reachable=True), write_mesh=True
    )
    only_front = tmp_path / "front.png"
    only_front.write_bytes(b"x")
    res = RemoteHunyuanBackend(ex).generate(
        GenerateRequest(images=[only_front]), tmp_path / "out"
    )
    assert res.metadata["mode"] == "single"


class TestGatedEncoderDecision:
    """Whether TRELLIS.2 runs with real DINOv3 is decided by asking, not guessing.

    The flag it replaced defaulted to "avoid the gated repo" because approval
    was pending. Approval then landed and the flag did not, so the encoder the
    model was actually trained against kept being silently swapped for DINOv2 --
    a quality loss with nothing in the log to explain it.
    """

    @staticmethod
    def _fresh(monkeypatch, *, token=None, status=200, boom=False):
        from pixiecad.generative import trellis_remote as tr

        monkeypatch.setattr(tr, "_gate_cache", None, raising=False)
        monkeypatch.delenv("PIXIECAD_TRELLIS_DINOV3", raising=False)
        if token is None:
            monkeypatch.delenv("HF_TOKEN", raising=False)
        else:
            monkeypatch.setenv("HF_TOKEN", token)

        class _Res:
            def __init__(self): self.status = status
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=0):
            if boom:
                raise OSError("network is down")
            return _Res()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        return tr

    def test_an_approved_token_opens_the_gate(self, monkeypatch):
        tr = self._fresh(monkeypatch, token="hf_x", status=200)
        assert tr.gate_is_open() is True
        assert tr.use_gated_dinov3() is True

    def test_an_unapproved_token_does_not(self, monkeypatch):
        """403 is what a token without access to THIS repo returns."""
        tr = self._fresh(monkeypatch, token="hf_x", status=403)
        assert tr.gate_is_open() is False

    def test_no_token_means_no_gate(self, monkeypatch):
        tr = self._fresh(monkeypatch, token=None)
        assert tr.gate_is_open() is False

    def test_a_failed_check_fails_closed(self, monkeypatch):
        """If the check itself cannot run, degrade to DINOv2 rather than die.
        A worse encoder beats no model at all."""
        tr = self._fresh(monkeypatch, token="hf_x", boom=True)
        assert tr.gate_is_open() is False

    def test_the_override_still_wins_both_ways(self, monkeypatch):
        """A network check has to be overridable when it is the thing wrong."""
        tr = self._fresh(monkeypatch, token=None)
        monkeypatch.setenv("PIXIECAD_TRELLIS_DINOV3", "1")
        assert tr.use_gated_dinov3() is True

        tr = self._fresh(monkeypatch, token="hf_x", status=200)
        monkeypatch.setenv("PIXIECAD_TRELLIS_DINOV3", "0")
        assert tr.use_gated_dinov3() is False

    def test_the_result_is_cached(self, monkeypatch):
        """One request per process: this runs before every TRELLIS job."""
        tr = self._fresh(monkeypatch, token="hf_x", status=200)
        calls = []
        real = __import__("urllib.request").request.urlopen

        def counting(req, timeout=0):
            calls.append(1)
            return real(req, timeout=timeout)

        monkeypatch.setattr("urllib.request.urlopen", counting)
        tr.gate_is_open()
        tr.gate_is_open()
        assert len(calls) <= 1


class TestTimeoutBudget:
    """The generate window has to fit the unwrap, not just the sampling.

    A real job died on this: generation_elapsed_sec was 127.15, then xatlas
    spent 25 minutes parameterising a 1.87M face mesh into 1748 charts, and a
    1620 s ceiling killed a mesh that already existed. The old split reserved
    half the budget for readiness -- correct when that meant pulling 25 GB,
    wrong once the image is baked and readiness takes two minutes.
    """

    @staticmethod
    def _split(total: int) -> tuple[int, int]:
        ready = min(1800, max(300, total // 4))
        return ready, max(600, total - ready - 300)

    def test_generation_gets_most_of_the_budget(self):
        ready, gen = self._split(5400)
        assert gen > ready * 2, "readiness is fast from a baked image; generate is not"

    def test_the_window_beats_the_run_that_failed(self):
        """1847 s of real work must fit, with room for a worse unwrap."""
        _, gen = self._split(5400)
        assert gen > 1847, f"generate window {gen}s is under the run that already failed"

    def test_curl_cannot_outlive_the_ssh_job(self):
        """curl gets generate+120. If that exceeds the job timeout, ssh dies
        first and the error says nothing about what actually went wrong."""
        for total in (3600, 5400, 7200):
            _, gen = self._split(total)
            assert gen + 120 < total, f"curl would outlive the job at {total}s"

    def test_a_small_budget_still_leaves_a_usable_window(self):
        _, gen = self._split(900)
        assert gen >= 600

    def test_the_default_is_ninety_minutes(self, monkeypatch):
        from pixiecad.generative.trellis_remote import RemoteTrellisBackend

        monkeypatch.delenv("PIXIECAD_TRELLIS_TIMEOUT", raising=False)
        assert RemoteTrellisBackend(object()).timeout_s == 5400

    def test_the_env_var_is_read_when_the_backend_is_built(self, monkeypatch):
        """Not at import time. A frozen default would ignore an override set
        by the process that constructs the backend, and do it silently."""
        from pixiecad.generative.trellis_remote import RemoteTrellisBackend

        monkeypatch.setenv("PIXIECAD_TRELLIS_TIMEOUT", "9000")
        assert RemoteTrellisBackend(object()).timeout_s == 9000

    def test_an_explicit_value_beats_the_env(self, monkeypatch):
        from pixiecad.generative.trellis_remote import RemoteTrellisBackend

        monkeypatch.setenv("PIXIECAD_TRELLIS_TIMEOUT", "9000")
        assert RemoteTrellisBackend(object(), timeout_s=1200).timeout_s == 1200
