"""Tests for per-run session directories."""

from datetime import datetime
from pathlib import Path

import pytest

from pixiecad.session import latest_session, new_session, slugify


def _png(path: Path, data: bytes = b"\x89PNG\r\n\x1a\n") -> Path:
    path.write_bytes(data)
    return path


def test_slugify():
    assert slugify("F1 Car — Rear Wing!") == "f1-car-rear-wing"
    assert slugify("   ") == ""
    assert len(slugify("x" * 100)) <= 32


def test_new_session_layout(tmp_path: Path):
    s = new_session(tmp_path / "runs", label="F1 Car", now=datetime(2026, 8, 2, 17, 5, 9))

    assert s.session_id == "20260802-170509-f1-car"
    assert s.root.name == s.session_id
    for d in (s.input_dir, s.work_dir, s.output_dir):
        assert d.is_dir()
        assert d.parent == s.root


def test_session_name_falls_back_to_source_dir(tmp_path: Path):
    photos = tmp_path / "my photos"
    photos.mkdir()
    s = new_session(tmp_path / "runs", source=photos, now=datetime(2026, 1, 1, 0, 0, 0))
    assert s.session_id == "20260101-000000-my-photos"


def test_two_sessions_in_the_same_second_do_not_collide(tmp_path: Path):
    """The dashboard can take simultaneous uploads; folders must stay distinct."""
    when = datetime(2026, 8, 2, 12, 0, 0)
    a = new_session(tmp_path / "runs", label="job", now=when)
    b = new_session(tmp_path / "runs", label="job", now=when)

    assert a.root != b.root
    assert b.session_id.endswith("-2")
    assert a.root.is_dir() and b.root.is_dir()


def test_stage_inputs_copies_only_images(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _png(photos / "a.png")
    _png(photos / "b.JPG")
    (photos / "notes.txt").write_text("ignore me")
    (photos / "subdir").mkdir()

    s = new_session(tmp_path / "runs", label="t")
    copied = s.stage_inputs(photos)

    assert [p.name for p in copied] == ["a.png", "b.JPG"]
    assert (s.input_dir / "a.png").read_bytes() == (photos / "a.png").read_bytes()
    assert not (s.input_dir / "notes.txt").exists()


def test_stage_inputs_rejects_empty_dir(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    s = new_session(tmp_path / "runs", label="t")
    with pytest.raises(ValueError, match="no images"):
        s.stage_inputs(empty)


def test_latest_points_at_newest(tmp_path: Path):
    root = tmp_path / "runs"
    first = new_session(root, label="one", now=datetime(2026, 8, 1, 9, 0, 0))
    first.write_meta()
    second = new_session(root, label="two", now=datetime(2026, 8, 2, 9, 0, 0))
    second.write_meta()

    assert latest_session(root).session_id == second.session_id
    # The convenience symlink tracks it too, when the filesystem allows one.
    link = root / "latest"
    if link.is_symlink():
        assert link.resolve() == second.root.resolve()


def test_latest_session_ignores_incomplete_dirs(tmp_path: Path):
    """A session is only 'latest' once it has written its metadata."""
    root = tmp_path / "runs"
    done = new_session(root, label="done", now=datetime(2026, 8, 1, 9, 0, 0))
    done.write_meta()
    new_session(root, label="crashed", now=datetime(2026, 8, 3, 9, 0, 0))

    assert latest_session(root).session_id == done.session_id


def test_latest_session_on_missing_root(tmp_path: Path):
    assert latest_session(tmp_path / "nope") is None


def test_write_meta_roundtrip(tmp_path: Path):
    import json

    s = new_session(tmp_path / "runs", label="meta")
    path = s.write_meta(regime="sparse_views", faces=5000)
    data = json.loads(path.read_text())

    assert data["session_id"] == s.session_id
    assert data["regime"] == "sparse_views"
    assert data["faces"] == 5000
    assert Path(data["output_dir"]) == s.output_dir
