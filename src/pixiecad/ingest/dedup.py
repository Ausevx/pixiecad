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


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def mark_duplicates(
    hashes: list[int], threshold: int = DEFAULT_THRESHOLD
) -> list[int | None]:
    """For each item, index of the earlier near-duplicate it matches, else None.

    Greedy first-wins: the first occurrence is kept, later lookalikes point
    back at it.
    """
    kept: list[int] = []
    result: list[int | None] = []
    for i, h in enumerate(hashes):
        match = next((j for j in kept if hamming(h, hashes[j]) <= threshold), None)
        result.append(match)
        if match is None:
            kept.append(i)
    return result
