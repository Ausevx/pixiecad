import numpy as np

from pixiecad.ingest.dedup import dhash, hamming, mark_duplicates
from pixiecad.ingest.quality import assess_quality


def test_sharp_passes(sharp):
    q = assess_quality(sharp)
    assert q.ok and not q.reject_reasons


def test_blurry_rejected(blurry):
    q = assess_quality(blurry)
    assert not q.ok
    assert any("blurry" in r for r in q.reject_reasons)


def test_overexposed_rejected(sharp):
    white = np.full_like(sharp, 255)
    q = assess_quality(white)
    assert not q.ok


def test_low_resolution_rejected(sharp):
    import cv2

    tiny = cv2.resize(sharp, (320, 240))
    q = assess_quality(tiny)
    assert any("resolution" in r for r in q.reject_reasons)


def test_dhash_stability_and_distance(sharp, blurry):
    import cv2

    same = dhash(sharp)
    _, buf = cv2.imencode(".jpg", sharp, [cv2.IMWRITE_JPEG_QUALITY, 70])
    reencoded = dhash(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    assert hamming(same, reencoded) <= 6  # re-encoded copy ≈ duplicate

    noise = (np.random.default_rng(7).integers(0, 255, sharp.shape, dtype=np.uint8))
    assert hamming(same, dhash(noise)) > 6  # different content ≠ duplicate


def test_mark_duplicates_first_wins():
    hashes = [0b0, 0b1, 0b1111111111]  # h1 within 6 bits of h0; h2 is 10 bits away
    assert mark_duplicates(hashes, threshold=6) == [None, 0, None]
