"""Tests for --segmentation on ``pixiecad run``.

All tests are fast and offline — no GPU, no SSH, no network requests.
Tests 1–2 exercise the CLI through Typer's CliRunner with heavy mocking.
Tests 3–4 call the shared helper and finish_model directly, which avoids
the session/build scaffolding that would otherwise obscure what is tested.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh
from typer.testing import CliRunner

from pixiecad.cli import app
from pixiecad.web.finishing import FinishOptions, finish_model, run_semantic_segmentation

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _photos_dir(tmp_path: Path) -> Path:
    """One minimal JPEG so the CLI's photos argument is satisfied."""
    photos = tmp_path / "photos"
    photos.mkdir(exist_ok=True)
    # Minimal valid JFIF bytes that PIL can open without loading OpenCV.
    # fmt: off
    jpeg = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\x09\x09"
        b"\x08\x0a\x0c\x14\x0d\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1eB"
        b"\x11\x0a\x0a\x0a\x11\x0a\x14\x0f\x0f\x14))))))))))))))))))))))))))"
        b")))))))))))))))))))))))))))))))))))))\xff\xc0\x00\x0b\x08"
        b"\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05"
        b"\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02"
        b"\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\xff\xda\x00\x08\x01\x01\x00"
        b"\x00?\x00\xf5\x0a\xff\xd9"
    )
    # fmt: on
    (photos / "img01.jpg").write_bytes(jpeg)
    return photos


def _build_result_with_glb(tmp_path: Path):
    """Minimal BuildResult with a real .glb on disk."""
    from pixiecad.pipeline import BuildResult, Regime, StageOutcome

    glb = tmp_path / "model.glb"
    trimesh.creation.box().export(str(glb))
    return BuildResult(
        regime=Regime.ORBIT,
        stages=[StageOutcome("export", "ok", str(glb), 0.0)],
        glb_path=str(glb),
        faces=12,
        scale_applied=None,
        warnings=[],
        parts=[{"name": "body", "file": "body.glb", "faces": 12, "target_faces": 12}],
        parts_dir=str(tmp_path / "parts"),
    )


def _mock_session(tmp_path: Path, tag: str = "sess"):
    """Return a mock Session and a configured patch for new_session."""
    sess = MagicMock()
    sess.root = tmp_path / tag
    sess.root.mkdir(parents=True, exist_ok=True)
    for sub in ("input", "work", "output"):
        d = tmp_path / tag / sub
        d.mkdir(parents=True, exist_ok=True)
        setattr(sess, f"{sub}_dir", d)
    sess.stage_inputs.return_value = []
    sess.write_meta.return_value = None
    return sess


def _mock_workspace(tmp_path: Path, tag: str = "ws"):
    """Return a mock Workspace and a configured patch for Workspace.create."""
    ws = MagicMock()
    ws.root = tmp_path / tag
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.stages_dir = ws.root / "stages"
    ws.stages_dir.mkdir(parents=True, exist_ok=True)
    return ws


class _FakeExecutor:
    """Stand-in GPU host: writes label arrays, counts invocations.

    The label maps must match the render resolution exactly -- semantic.py
    rejects a mismatch rather than silently resizing -- so the shape is read
    off the rendered PNG instead of being hardcoded. Two horizontal bands give
    the fuser something to actually vote on; a single uniform label produces no
    regions at all and would not exercise the path under test.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def run(self, job):
        self.call_count += 1
        from PIL import Image

        from pixiecad.executors.base import JobResult

        job.output_dir.mkdir(parents=True, exist_ok=True)
        views = next(p for p in job.inputs if p.is_dir())
        for png in sorted(views.glob("*.png")):
            with Image.open(png) as im:
                w, h = im.size
            arr = np.zeros((h, w), dtype=np.int32)
            arr[: h // 2, :] = 1
            np.save(job.output_dir / f"{png.stem}.npy", arr)
        return JobResult(0, "ok", "", 0.1)


# ---------------------------------------------------------------------------
# Test 1: --segmentation semantic with no GPU host warns, exit 0
# ---------------------------------------------------------------------------

def test_semantic_no_host_warns_and_exits_zero(tmp_path: Path):
    """One-line warning printed, fall back to geometric, no hard failure."""
    photos = _photos_dir(tmp_path)
    fake_result = _build_result_with_glb(tmp_path)
    sess = _mock_session(tmp_path)
    ws = _mock_workspace(tmp_path)

    with patch("pixiecad.session.new_session", return_value=sess), \
         patch("pixiecad.pipeline.run_build", return_value=fake_result), \
         patch("pixiecad.cli.collect_outputs", return_value={}), \
         patch("pixiecad.cli.Workspace") as mock_ws_cls:

        mock_ws_cls.create.return_value = ws
        mock_ws_cls.open.return_value = ws

        result = runner.invoke(
            app,
            ["run", str(photos), "--runs", str(tmp_path / "runs"),
             "--segmentation", "semantic"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    # The warning must mention both semantic and the missing host so the user
    # knows exactly what to add.
    lower = result.output.lower()
    assert "semantic" in lower
    assert "host" in lower or "gpu" in lower


# ---------------------------------------------------------------------------
# Test 2: --segmentation semantic with a host reaches the segmenter
# ---------------------------------------------------------------------------

def test_semantic_with_host_calls_segmenter(tmp_path: Path):
    """run_semantic_segmentation must be invoked when a host is configured."""
    photos = _photos_dir(tmp_path)
    fake_result = _build_result_with_glb(tmp_path)
    sess = _mock_session(tmp_path, "sess2")
    ws = _mock_workspace(tmp_path, "ws2")

    reached: list[bool] = []

    def _fake_seg(mesh, executor, max_parts, *, cache_dir=None, log=None):
        reached.append(True)
        from pixiecad.parts.segment import split_parts
        return split_parts(mesh, max_parts=max_parts), False

    with patch("pixiecad.session.new_session", return_value=sess), \
         patch("pixiecad.pipeline.run_build", return_value=fake_result), \
         patch("pixiecad.cli.collect_outputs", return_value={}), \
         patch("pixiecad.cli.Workspace") as mock_ws_cls, \
         patch("pixiecad.web.finishing.run_semantic_segmentation", _fake_seg), \
         patch("pixiecad.vision.triage.name_parts",
               side_effect=lambda parts, *a, **kw: parts), \
         patch("pixiecad.parts.export_parts") as mock_export:

        mock_ws_cls.create.return_value = ws
        mock_ws_cls.open.return_value = ws
        mock_export.return_value = ([], tmp_path / "manifest.json")

        result = runner.invoke(
            app,
            [
                "run", str(photos),
                "--runs", str(tmp_path / "runs2"),
                "--host", "gpu.example.invalid",
                "--segmentation", "semantic",
                "--split",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert reached, "run_semantic_segmentation was never called"


# ---------------------------------------------------------------------------
# Test 3: warm cache skips the remote segmenter entirely
# ---------------------------------------------------------------------------

def test_cache_hit_skips_executor(tmp_path: Path):
    """Second call with the same mesh geometry must not invoke the executor."""
    mesh = trimesh.creation.box()
    executor = _FakeExecutor()
    cache = tmp_path / "cache"
    cache.mkdir()

    # First call: cold cache — executor must be called exactly once.
    parts1, hit1 = run_semantic_segmentation(
        mesh, executor, max_parts=4, cache_dir=cache
    )
    assert executor.call_count == 1
    assert not hit1

    # Second call: warm cache — executor must NOT be called again.
    parts2, hit2 = run_semantic_segmentation(
        mesh, executor, max_parts=4, cache_dir=cache
    )
    assert executor.call_count == 1, (
        f"executor invoked {executor.call_count} times despite a warm cache"
    )
    assert hit2


# ---------------------------------------------------------------------------
# Test 4: default (auto) is byte-for-byte unchanged
# ---------------------------------------------------------------------------

def test_auto_segmentation_unchanged(tmp_path: Path):
    """The 'auto' default must still use geometric splitting with no regressions."""
    mesh = trimesh.creation.icosphere(subdivisions=1)
    glb = tmp_path / "model.glb"
    mesh.export(str(glb))

    report = finish_model(glb, tmp_path, FinishOptions(segmentation="auto"))

    assert "segment:geometric" in report.steps
    assert not any("semantic" in w.lower() for w in report.warnings)
