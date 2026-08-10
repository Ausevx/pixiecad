"""Tests for end-to-end build orchestrator."""

import pytest
import trimesh
from pathlib import Path

from pixiecad.pipeline import BuildResult, Regime, detect_regime, run_build
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
    result = run_build(photos_dir, ws, bake=False, executor=executor)

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

    result = run_build(photos_dir, ws, bake=False)

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
    result = run_build(photos_dir, ws, bake=False, executor=executor)

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
    result = run_build(photos_dir, ws, bake=False, executor=executor)

    # 4m target extent / 2m actual extent = scale factor 2.0
    assert result.scale_applied is not None
    assert pytest.approx(result.scale_applied, abs=1e-3) == 2.0
    scale_stage = next(s for s in result.stages if s.name == "scale")
    assert scale_stage.status == "ok"


def test_split_parts_happy_path(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    spec = ObjectSpec(name="test_parts_object", target_faces=120, dimensions=Dimensions.parse(length="2m"))
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

    box = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    cyl1 = trimesh.creation.cylinder(radius=0.3, height=0.5)
    cyl1.apply_translation([3.0, 0.0, 0.0])
    cyl2 = trimesh.creation.cylinder(radius=0.3, height=0.5)
    cyl2.apply_translation([-3.0, 0.0, 0.0])
    multi_mesh = trimesh.util.concatenate([box, cyl1, cyl2])
    multi_mesh.export(str(ply_path))

    fake_dense_res = DenseResult(
        mesh_path=str(ply_path),
        n_faces=len(multi_mesh.faces),
    )

    monkeypatch.setattr("pixiecad.pipeline.run_ingest", lambda p, w: fake_report)
    monkeypatch.setattr("pixiecad.pipeline.run_sparse", lambda img_dir, w: fake_sparse_res)
    monkeypatch.setattr("pixiecad.pipeline.run_dense", lambda img_dir, s_dir, w, ex: fake_dense_res)

    executor = LocalExecutor()
    result = run_build(photos_dir, ws, bake=False, executor=executor, split=True, max_parts=8, object_hint="toy car")

    assert isinstance(result, BuildResult)
    assert result.glb_path is not None
    assert Path(result.glb_path).exists()
    assert result.parts_dir is not None
    assert Path(result.parts_dir).exists()
    assert len(result.parts) == 3
    for part_info in result.parts:
        assert part_info["name"]
        part_file = Path(result.parts_dir) / part_info["file"]
        assert part_file.exists()
    assert sum(p["target_faces"] for p in result.parts) == spec.target_faces

    parts_stage = next(s for s in result.stages if s.name == "parts")
    assert parts_stage.status == "ok"


def test_split_false_no_parts_stage(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    spec = ObjectSpec(name="test_object", target_faces=80)
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
    result = run_build(photos_dir, ws, bake=False, executor=executor, split=False)

    assert result.glb_path is not None
    assert not any(s.name == "parts" for s in result.stages)
    assert result.parts_dir is None
    assert result.parts == []


def test_parts_failure_containment(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    spec = ObjectSpec(name="test_object", target_faces=80)
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

    def raise_split_error(mesh, max_parts=8):
        raise RuntimeError("Splitting failed catastrophically")

    monkeypatch.setattr("pixiecad.pipeline.split_parts", raise_split_error)

    executor = LocalExecutor()
    result = run_build(photos_dir, ws, bake=False, executor=executor, split=True)

    assert result.glb_path is not None
    assert Path(result.glb_path).exists()
    parts_stage = next(s for s in result.stages if s.name == "parts")
    assert parts_stage.status == "failed"
    assert "Splitting failed catastrophically" in parts_stage.detail



def test_conditioning_views_prefer_cardinals_over_filename_order():
    """Naive truncation silently halved real coverage.

    A normal 8-view capture sorts as front, front-left, left, rear-left; taking
    the first four hands a multi-view model two obliques it discards, so the
    back and right of the object are never seen despite being photographed.
    """
    from pathlib import Path

    from pixiecad.pipeline import _select_conditioning_views

    shots = [
        Path(n) for n in [
            "view_01_front.png", "view_02_front_left.png", "view_03_left.png",
            "view_04_rear_left.png", "view_05_rear.png", "view_06_rear_right.png",
            "view_07_right.png", "view_08_top.png",
        ]
    ]
    assert [p.name for p in _select_conditioning_views(shots)] == [
        "view_01_front.png", "view_03_left.png",
        "view_05_rear.png", "view_07_right.png",
    ]


def test_conditioning_views_fall_back_when_no_cardinals():
    """Arbitrary filenames must still yield a usable primary image."""
    from pathlib import Path

    from pixiecad.pipeline import _select_conditioning_views

    shots = [Path(f"IMG_{i}.jpg") for i in range(6)]
    got = _select_conditioning_views(shots)
    assert len(got) == 4
    assert got[0].name == "IMG_0.jpg"


def test_no_duplicate_stage_names_on_zero_accepted_photos(tmp_path, monkeypatch):
    """When ingest succeeds but yields 0 accepted photos, detect_regime fails.

    The stage outcome list must record 'ingest' as ok and 'regime' as failed, with
    no duplicate stage names in the list.
    """
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    ws = Workspace.create(ws_dir, ObjectSpec())

    fake_report = IngestReport(
        accepted=0,
        duplicates=0,
        rejected=4,
        unreadable=0,
        photos=[],
        advice=[],
    )

    monkeypatch.setattr("pixiecad.pipeline.run_ingest", lambda p, w: fake_report)

    result = run_build(photos_dir, ws, bake=False)

    stage_names = [s.name for s in result.stages]
    assert len(stage_names) == len(set(stage_names)), f"Duplicate stage names found: {stage_names}"

    ingest_stage = next(s for s in result.stages if s.name == "ingest")
    assert ingest_stage.status == "ok"
    assert "0 accepted" in ingest_stage.detail

    regime_stage = next(s for s in result.stages if s.name == "regime")
    assert regime_stage.status == "failed"
    assert "Invalid photo count: 0" in regime_stage.detail


def test_no_duplicate_stage_names_on_happy_path(tmp_path, monkeypatch):
    """Happy path build must not produce duplicate stage names."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    ws_dir = tmp_path / "ws"
    spec = ObjectSpec(name="test_object", target_faces=80)
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
    result = run_build(photos_dir, ws, bake=False, executor=executor)

    stage_names = [s.name for s in result.stages]
    assert len(stage_names) == len(set(stage_names)), f"Duplicate stage names found: {stage_names}"
    assert "ingest" in stage_names
    assert "regime" in stage_names

