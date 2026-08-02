"""Tests for end-to-end build orchestrator."""

import pytest
import trimesh
from pathlib import Path

from pixiecad.pipeline import BuildResult, Regime, StageOutcome, detect_regime, run_build
from pixiecad.spec import Dimensions, ObjectSpec
from pixiecad.workspace import Workspace
from pixiecad.ingest.pipeline import IngestReport, PhotoRecord
from pixiecad.geometry.sparse import SparseResult, SparseFailure
from pixiecad.geometry.dense import DenseResult, DenseUnavailable
from pixiecad.executors.local import LocalExecutor


def test_detect_regime_boundaries():
    with pytest.raises(ValueError):
        detect_regime(0)

    with pytest.raises(ValueError):
        detect_regime(-1)

    assert detect_regime(1) == Regime.SINGLE_IMAGE
    assert detect_regime(2) == Regime.SPARSE_VIEWS
    assert detect_regime(15) == Regime.SPARSE_VIEWS
    assert detect_regime(16) == Regime.ORBIT
    assert detect_regime(40) == Regime.ORBIT


def test_orbit_build_happy_path(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    spec = ObjectSpec(name="test_object", target_faces=80, dimensions=Dimensions.parse(length="2m"))
    ws = Workspace.create(ws_dir, spec)

    images_dir = tmp_path / "working_images"
    images_dir.mkdir()
    fake_img_path = images_dir / "photo0.jpg"
    fake_img_path.write_text("fake_image_content")

    fake_report = IngestReport(
        accepted=20,
        duplicates=0,
        rejected=0,
        unreadable=0,
        photos=[PhotoRecord(source="photo0.jpg", status="accepted", working_path=str(fake_img_path))],
        advice=[],
    )

    sparse_dir = tmp_path / "sparse_model"
    sparse_dir.mkdir()
    fake_sparse_res = SparseResult(
        n_images_in=20,
        n_registered=20,
        n_points3d=500,
        mean_reproj_error=0.4,
        model_dir=str(sparse_dir),
    )

    mesh_dir = tmp_path / "dense_out"
    mesh_dir.mkdir()
    ply_path = mesh_dir / "mesh.ply"
    ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    ico.export(str(ply_path))

    fake_dense_res = DenseResult(
        mesh_path=str(ply_path),
        n_faces=len(ico.faces),
    )

    monkeypatch.setattr("pixiecad.pipeline.run_ingest", lambda p, w: fake_report)
    monkeypatch.setattr("pixiecad.pipeline.run_sparse", lambda img_dir, w: fake_sparse_res)
    monkeypatch.setattr("pixiecad.pipeline.run_dense", lambda img_dir, s_dir, w, ex: fake_dense_res)

    executor = LocalExecutor()
    result = run_build(photos_dir, ws, executor=executor)

    assert isinstance(result, BuildResult)
    assert result.regime == Regime.ORBIT
    assert result.glb_path is not None
    assert Path(result.glb_path).exists()
    assert result.faces == spec.target_faces
    assert all(s.status == "ok" for s in result.stages)
    assert len(result.summary_lines()) == len(result.stages)


def test_sparse_failure_returns_partial_build_result(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    ws = Workspace.create(ws_dir, ObjectSpec())

    images_dir = tmp_path / "working_images"
    images_dir.mkdir()
    fake_img_path = images_dir / "photo0.jpg"
    fake_img_path.write_text("fake_image_content")

    fake_report = IngestReport(
        accepted=20,
        duplicates=0,
        rejected=0,
        unreadable=0,
        photos=[PhotoRecord(source="photo0.jpg", status="accepted", working_path=str(fake_img_path))],
        advice=[],
    )

    monkeypatch.setattr("pixiecad.pipeline.run_ingest", lambda p, w: fake_report)

    def raise_sparse_failure(img_dir, w):
        raise SparseFailure("pycolmap produced no valid reconstructions")

    monkeypatch.setattr("pixiecad.pipeline.run_sparse", raise_sparse_failure)

    result = run_build(photos_dir, ws)

    assert result.glb_path is None
    assert result.faces is None
    sparse_stage = next(s for s in result.stages if s.name == "sparse")
    assert sparse_stage.status == "failed"
    assert "pycolmap" in sparse_stage.detail


def test_dense_unavailable_skips_subsequent_stages(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    ws = Workspace.create(ws_dir, ObjectSpec())

    images_dir = tmp_path / "working_images"
    images_dir.mkdir()
    fake_img_path = images_dir / "photo0.jpg"
    fake_img_path.write_text("fake_image_content")

    fake_report = IngestReport(
        accepted=20,
        duplicates=0,
        rejected=0,
        unreadable=0,
        photos=[PhotoRecord(source="photo0.jpg", status="accepted", working_path=str(fake_img_path))],
        advice=[],
    )

    sparse_dir = tmp_path / "sparse_model"
    sparse_dir.mkdir()
    fake_sparse_res = SparseResult(
        n_images_in=20,
        n_registered=20,
        n_points3d=500,
        mean_reproj_error=0.4,
        model_dir=str(sparse_dir),
    )

    monkeypatch.setattr("pixiecad.pipeline.run_ingest", lambda p, w: fake_report)
    monkeypatch.setattr("pixiecad.pipeline.run_sparse", lambda img_dir, w: fake_sparse_res)

    def raise_dense_unavailable(img_dir, s_dir, w, ex):
        raise DenseUnavailable("No GPU VRAM available")

    monkeypatch.setattr("pixiecad.pipeline.run_dense", raise_dense_unavailable)

    executor = LocalExecutor()
    result = run_build(photos_dir, ws, executor=executor)

    assert result.glb_path is None
    dense_stage = next(s for s in result.stages if s.name == "dense")
    assert dense_stage.status == "skipped"
    assert "GPU VRAM" in dense_stage.detail

    dense_idx = [i for i, s in enumerate(result.stages) if s.name == "dense"][0]
    subsequent_stages = result.stages[dense_idx + 1:]
    assert len(subsequent_stages) > 0
    assert all(s.status == "skipped" for s in subsequent_stages)


def test_scale_applied(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    spec = ObjectSpec(name="scaled_obj", target_faces=80, dimensions=Dimensions.parse(length="4m"))
    ws = Workspace.create(ws_dir, spec)

    images_dir = tmp_path / "working_images"
    images_dir.mkdir()
    fake_img_path = images_dir / "photo0.jpg"
    fake_img_path.write_text("fake_image_content")

    fake_report = IngestReport(
        accepted=20,
        duplicates=0,
        rejected=0,
        unreadable=0,
        photos=[PhotoRecord(source="photo0.jpg", status="accepted", working_path=str(fake_img_path))],
        advice=[],
    )

    sparse_dir = tmp_path / "sparse_model"
    sparse_dir.mkdir()
    fake_sparse_res = SparseResult(
        n_images_in=20,
        n_registered=20,
        n_points3d=500,
        mean_reproj_error=0.4,
        model_dir=str(sparse_dir),
    )

    mesh_dir = tmp_path / "dense_out"
    mesh_dir.mkdir()
    ply_path = mesh_dir / "mesh.ply"
    # Sphere of radius 1.0 has bounding box extents (2.0, 2.0, 2.0)
    ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    ico.export(str(ply_path))

    fake_dense_res = DenseResult(
        mesh_path=str(ply_path),
        n_faces=len(ico.faces),
    )

    monkeypatch.setattr("pixiecad.pipeline.run_ingest", lambda p, w: fake_report)
    monkeypatch.setattr("pixiecad.pipeline.run_sparse", lambda img_dir, w: fake_sparse_res)
    monkeypatch.setattr("pixiecad.pipeline.run_dense", lambda img_dir, s_dir, w, ex: fake_dense_res)

    executor = LocalExecutor()
    result = run_build(photos_dir, ws, executor=executor)

    # 4m target extent / 2m actual extent = scale factor 2.0
    assert result.scale_applied is not None
    assert pytest.approx(result.scale_applied, abs=1e-3) == 2.0
    scale_stage = next(s for s in result.stages if s.name == "scale")
    assert scale_stage.status == "ok"
