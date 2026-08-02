import json

import cv2

from pixiecad.ingest import run_ingest
from pixiecad.spec import ObjectSpec
from pixiecad.workspace import Workspace

from conftest import checkerboard, save_jpg


def _photo_set(root, sharp, blurry):
    """5 photos: 3 distinct sharp, 1 duplicate of the first, 1 blurry."""
    photos = root / "photos"
    save_jpg(photos / "a.jpg", sharp)
    save_jpg(photos / "a_dupe.jpg", sharp)
    save_jpg(photos / "b.jpg", cv2.rotate(sharp, cv2.ROTATE_90_CLOCKWISE)[:, ::-1].copy())
    save_jpg(photos / "c.jpg", checkerboard(tile=40))
    save_jpg(photos / "blur.jpg", blurry)
    return photos


def test_ingest_end_to_end(tmp_path, sharp, blurry):
    photos = _photo_set(tmp_path, sharp, blurry)
    ws = Workspace.create(tmp_path / "ws", ObjectSpec())

    report = run_ingest(photos, ws, working_long_edge=640)

    assert report.rejected == 1
    assert report.duplicates >= 1
    assert report.accepted == 5 - report.rejected - report.duplicates
    accepted = [p for p in report.photos if p.status == "accepted"]
    assert all(p.working_path for p in accepted)
    # working copies exist and respect the resolution cap
    for p in accepted:
        img = cv2.imread(p.working_path)
        assert img is not None and max(img.shape[:2]) <= 640

    dupe = next(p for p in report.photos if p.status == "duplicate")
    assert dupe.duplicate_of.endswith("a.jpg")


def test_ingest_cache_hit(tmp_path, sharp, blurry):
    photos = _photo_set(tmp_path, sharp, blurry)
    ws = Workspace.create(tmp_path / "ws", ObjectSpec())

    first = run_ingest(photos, ws, working_long_edge=640)
    stage_dirs = list(ws.stages_dir.iterdir())
    second = run_ingest(photos, ws, working_long_edge=640)  # must hit cache

    assert list(ws.stages_dir.iterdir()) == stage_dirs  # no new stage dir
    assert second.accepted == first.accepted

    # changing params produces a fresh stage dir
    run_ingest(photos, ws, working_long_edge=512)
    assert len(list(ws.stages_dir.iterdir())) == len(stage_dirs) + 1


def test_manifest_records_stage(tmp_path, sharp, blurry):
    photos = _photo_set(tmp_path, sharp, blurry)
    ws = Workspace.create(tmp_path / "ws", ObjectSpec())
    run_ingest(photos, ws)
    manifest = json.loads(ws.manifest_path.read_text())
    (key, summary), = manifest["stages"].items()
    assert key.startswith("s0-ingest-")
    assert summary["rejected"] == 1
