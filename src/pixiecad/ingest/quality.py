"""Per-photo quality scoring: blur, exposure, resolution.

Cheap CPU heuristics that catch the inputs which most often sink a
reconstruction. Thresholds are deliberately conservative — we warn more
than we reject.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Variance-of-Laplacian below this ⇒ almost certainly motion/defocus blur.
BLUR_REJECT = 40.0
# Below this ⇒ soft; usable but warned.
BLUR_WARN = 100.0
# Fraction of pixels allowed at the histogram extremes before we call it clipped.
CLIP_WARN = 0.10
CLIP_REJECT = 0.30
# Minimum long-edge resolution for useful feature extraction.
MIN_LONG_EDGE = 640


@dataclass
class QualityResult:
    blur_score: float
    dark_clip: float  # fraction of pixels ≤ 5
    bright_clip: float  # fraction of pixels ≥ 250
    long_edge: int
    ok: bool
    warnings: list[str] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)


def assess_quality(image_bgr: np.ndarray) -> QualityResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    long_edge = max(h, w)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark = float(np.mean(gray <= 5))
    bright = float(np.mean(gray >= 250))

    warnings: list[str] = []
    rejects: list[str] = []

    if long_edge < MIN_LONG_EDGE:
        rejects.append(f"resolution too low ({w}x{h}, need long edge ≥ {MIN_LONG_EDGE})")

    if blur < BLUR_REJECT:
        rejects.append(f"too blurry (score {blur:.0f} < {BLUR_REJECT:.0f})")
    elif blur < BLUR_WARN:
        warnings.append(f"soft focus (score {blur:.0f})")

    for name, frac in (("underexposed", dark), ("overexposed", bright)):
        if frac > CLIP_REJECT:
            rejects.append(f"{name} ({frac:.0%} of pixels clipped)")
        elif frac > CLIP_WARN:
            warnings.append(f"partially {name} ({frac:.0%} clipped)")

    return QualityResult(
        blur_score=blur,
        dark_clip=dark,
        bright_clip=bright,
        long_edge=long_edge,
        ok=not rejects,
        warnings=warnings,
        reject_reasons=rejects,
    )
