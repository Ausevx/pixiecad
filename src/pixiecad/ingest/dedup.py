"""Near-duplicate detection via difference hash (dHash).

Fewer, more distinct frames means less SfM work and better matches;
burst shots and accidental doubles add cost without adding parallax.
No external dependency: 64-bit dHash + Hamming distance.
"""

from __future__ import annotations

import cv2
import numpy as np

# Hamming distance at or below which two frames count as near-duplicates.
# 64-bit dHash: identical ≈ 0-5, same scene slightly moved ≈ 6-12.
DEFAULT_THRESHOLD = 6


def dhash(image_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return int.from_bytes(np.packbits(diff).tobytes(), "big")


def dhash_pair(image_bgr: np.ndarray) -> tuple[int, int]:
    """dHash of the image and of its horizontal mirror.

    Opposite-side views of a left-right symmetric object (cars!) hash close to
    each other's *mirror*; comparing against both separates "same frame twice"
    from "the other side of a symmetric object", which must be kept.
    """
    return dhash(image_bgr), dhash(image_bgr[:, ::-1])


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def mark_duplicates(
    hash_pairs: list[tuple[int, int]], threshold: int = DEFAULT_THRESHOLD
) -> list[int | None]:
    """For each item, index of the earlier near-duplicate it matches, else None.

    Greedy first-wins. An item close to a kept frame is only a duplicate if it
    is not *better* explained as that frame's mirror image: strictly smaller
    distance to the mirror means it's the opposite side of a symmetric object.
    """
    kept: list[int] = []
    result: list[int | None] = []
    for i, (h, _) in enumerate(hash_pairs):
        match = None
        for j in kept:
            hj, hj_flip = hash_pairs[j]
            d_plain = hamming(h, hj)
            if d_plain <= threshold and not hamming(h, hj_flip) < d_plain:
                match = j
                break
        result.append(match)
        if match is None:
            kept.append(i)
    return result
