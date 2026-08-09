import json

import cv2
import numpy as np

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


class TestARemovedBackgroundIsNotUnderexposure:
    """A cut-out on black is not a dark photograph.

    Background removal is standard practice for generative 3D -- both backends
    run rembg internally anyway -- and it leaves the subject on pure black.
    Measuring exposure over the whole frame counted that as clipped shadow:
    five real cut-outs measured 35-40% "underexposed" whole-frame while their
    subjects clipped 0-6%, and all five were rejected. The job died with
    "Invalid photo count: 0; must be >= 1" before doing anything at all.
    """

    @staticmethod
    def _cutout(size=800, subject_grey=90, hole=False):
        """A mid-grey blob on pure black, as background removal produces."""
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.circle(img, (size // 2, size // 2), size // 3, (subject_grey,) * 3, -1)
        # Texture, so it is not rejected as blurry -- a flat disc has no edges.
        rng = np.random.default_rng(0)
        noise = rng.integers(-25, 25, img.shape, dtype=np.int16)
        mask = img[:, :, 0] > 0
        img[mask] = np.clip(img[mask].astype(np.int16) + noise[mask], 0, 255).astype(np.uint8)
        if hole:
            cv2.circle(img, (size // 2, size // 2), size // 12, (0, 0, 0), -1)
        return img

    def test_a_cutout_is_not_called_underexposed(self):
        from pixiecad.ingest.quality import assess_quality

        r = assess_quality(self._cutout())
        assert not any("underexposed" in x for x in r.reject_reasons), r.reject_reasons

    def test_the_background_is_excluded_from_the_measurement(self):
        from pixiecad.ingest.quality import assess_quality

        # Well over half the frame is pure black, yet the subject is not dark.
        r = assess_quality(self._cutout())
        assert r.dark_clip < 0.10, f"measured {r.dark_clip:.0%} on a lit subject"

    def test_dark_INSIDE_the_subject_still_counts(self):
        """Only black connected to the BORDER is a removed background. A hole
        in the middle of the object is a genuine exposure problem."""
        from pixiecad.ingest.quality import assess_quality

        plain = assess_quality(self._cutout()).dark_clip
        holed = assess_quality(self._cutout(hole=True)).dark_clip
        assert holed > plain

    def test_a_genuinely_dark_photo_is_still_rejected(self):
        """The escape hatch must not swallow real underexposure: if flood-fill
        leaves almost nothing, the whole frame is the honest measurement."""
        from pixiecad.ingest.quality import assess_quality

        r = assess_quality(np.zeros((800, 800, 3), dtype=np.uint8))
        assert any("underexposed" in x for x in r.reject_reasons), r.reject_reasons

    def test_the_resolution_message_names_the_usual_cause(self):
        """Every rejected photo was 550-639px straight from WhatsApp, which
        re-compresses to ~600px. The transport is the fix, not the threshold."""
        from pixiecad.ingest.quality import assess_quality

        r = assess_quality(self._cutout(size=500))
        assert any("messaging apps" in x for x in r.reject_reasons), r.reject_reasons
