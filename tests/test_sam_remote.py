"""Tests for the remote SAM segmenter. No GPU or network involved."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pixiecad.executors.base import JobResult
from pixiecad.parts.render import render_mesh
from pixiecad.parts.sam_remote import (
    PrecomputedSegmenter,
    RemoteSAMSegmenter,
    SegmentationError,
    build_sam_script,
)
from pixiecad.parts.semantic import segment_semantic


class FakeExecutor:
    """Stands in for a GPU host: records the job, writes plausible labels."""

    def __init__(self, *, ok: bool = True, skip: set[str] | None = None) -> None:
        self.ok = ok
        self.skip = skip or set()
        self.jobs: list = []

    def probe(self):  # pragma: no cover - not used by the segmenter
        raise NotImplementedError

    def run(self, job):
        self.jobs.append(job)
        if not self.ok:
            return JobResult(1, "", "CUDA out of memory", 1.0)
        views = next(p for p in job.inputs if p.is_dir())
        job.output_dir.mkdir(parents=True, exist_ok=True)
        for png in sorted(views.glob("*.png")):
            if png.stem in self.skip:
                continue
            # Two vertical stripes: a partition, which is what the real
            # script guarantees by painting masks largest-first.
            labels = np.zeros((8, 8), dtype=np.int32)
            labels[:, 4:] = 1
            np.save(job.output_dir / f"{png.stem}.npy", labels)
        return JobResult(0, "ok", "", 1.0)


def _renders(n: int = 3):
    mesh = trimesh.creation.box()
    return render_mesh(mesh, resolution=8)[:n]


class TestBuildScript:
    def test_mounts_checkpoint_from_host_not_image(self):
        script = build_sam_script()
        assert '-v "$HOME/.cache/sam":/opt/sam' in script
        assert "--gpus all" in script

    def test_passes_through_model_settings(self):
        script = build_sam_script(model_type="vit_b", points_per_side=16)
        assert "--model-type vit_b" in script
        assert "--points-per-side 16" in script

    def test_points_per_side_is_coerced_to_int(self):
        """Values reach a shell command line, so no unsanitised passthrough."""
        assert "--points-per-side 24" in build_sam_script(points_per_side="24")  # type: ignore[arg-type]


class TestRemoteSAMSegmenter:
    def test_returns_labels_per_view(self, tmp_path: Path):
        renders = _renders()
        ex = FakeExecutor()
        labels = RemoteSAMSegmenter(ex).segment_renders(renders, work_dir=tmp_path)
        assert set(labels) == {r.camera.name for r in renders}
        assert all(v.shape == (8, 8) for v in labels.values())

    def test_one_remote_call_for_all_views(self, tmp_path: Path):
        """A round trip per view would dominate runtime; batch or bust."""
        ex = FakeExecutor()
        RemoteSAMSegmenter(ex).segment_renders(_renders(6), work_dir=tmp_path)
        assert len(ex.jobs) == 1

    def test_uploads_one_png_per_render(self, tmp_path: Path):
        renders = _renders(4)
        ex = FakeExecutor()
        RemoteSAMSegmenter(ex).segment_renders(renders, work_dir=tmp_path)
        views = next(p for p in ex.jobs[0].inputs if p.is_dir())
        assert {p.stem for p in views.glob("*.png")} == {r.camera.name for r in renders}

    def test_failure_surfaces_remote_stderr(self, tmp_path: Path):
        ex = FakeExecutor(ok=False)
        with pytest.raises(SegmentationError, match="CUDA out of memory"):
            RemoteSAMSegmenter(ex).segment_renders(_renders(), work_dir=tmp_path)

    def test_missing_view_is_an_error_not_a_silent_gap(self, tmp_path: Path):
        """A dropped view would quietly weaken the vote; fail loudly instead."""
        renders = _renders(3)
        ex = FakeExecutor(skip={renders[1].camera.name})
        with pytest.raises(SegmentationError, match="no labels for view"):
            RemoteSAMSegmenter(ex).segment_renders(renders, work_dir=tmp_path)

    def test_empty_input_makes_no_remote_call(self, tmp_path: Path):
        ex = FakeExecutor()
        assert RemoteSAMSegmenter(ex).segment_renders([], work_dir=tmp_path) == {}
        assert not ex.jobs


class TestPrecomputedSegmenter:
    def test_feeds_labels_into_fusion(self, tmp_path: Path):
        mesh = trimesh.creation.box()
        renders = render_mesh(mesh, resolution=8)
        labels = RemoteSAMSegmenter(FakeExecutor()).segment_renders(renders, work_dir=tmp_path)
        result = segment_semantic(
            mesh, PrecomputedSegmenter(renders, labels), renders=renders
        )
        assert len(result.face_labels) == len(mesh.faces)
        assert (result.face_labels >= 0).all()

    def test_unknown_render_is_reported_clearly(self):
        renders = _renders(2)
        seg = PrecomputedSegmenter(renders, {})
        with pytest.raises(SegmentationError, match="no labels for"):
            seg.segment(renders[0].rgb)
