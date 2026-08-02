import numpy as np

from pixiecad.ingest.dedup import dhash, dhash_pair, hamming, mark_duplicates
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
    # (hash, flipped_hash) pairs; flips far away so the mirror rule stays out.
    far = 0xFFFFFFFFFFFFFFFF
    pairs = [(0b0, far), (0b1, far), (0b1111111111, far)]
    assert mark_duplicates(pairs, threshold=6) == [None, 0, None]


def test_mirror_views_are_not_duplicates():
    """Opposite sides of a symmetric object must be kept (F1-car regression)."""
    import numpy as np

    rng = np.random.default_rng(3)
    # Asymmetric scene: bright blob left of center on noise.
    img = rng.integers(40, 200, (400, 400, 3), dtype=np.uint8)
    img[150:250, 40:140] = 245
    mirrored = img[:, ::-1].copy()

    pairs = [dhash_pair(img), dhash_pair(mirrored)]
    # A mirrored view may hash near its twin, but must NOT be marked duplicate.
    assert mark_duplicates(pairs, threshold=64)[1] is None

    # A true re-encode of the SAME frame is still caught.
    import cv2

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    re_enc = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    pairs2 = [dhash_pair(img), dhash_pair(re_enc)]
    assert mark_duplicates(pairs2)[1] == 0
