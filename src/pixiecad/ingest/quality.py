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


def _subject_mask(gray: np.ndarray) -> np.ndarray:
    """True where the photo has content, False over a removed background.

    Background removal is standard practice for generative 3D -- both backends
    run rembg internally anyway -- and it leaves the subject on pure black.
    Measuring exposure over the whole frame then counts that background as
    clipped shadow: five cut-outs measured 35-40% "underexposed" while their
    subjects clipped 0-6%, so every one was rejected for being dark when none
    of them was.

    Only black CONNECTED TO THE BORDER counts as background, so a genuinely
    dark hole inside the object still counts against exposure. The threshold is
    5, not something higher, because a real photograph carries sensor noise and
    does not produce large regions of pure 0 -- that is the signature of an
    alpha channel flattened onto black.
    """
    h, w = gray.shape
    flood = (gray <= 5).astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x, y in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x]:
            cv2.floodFill(flood, mask, (x, y), 2)
    subject = flood != 2

    # A genuinely underexposed photo can have dark corners that flood inward.
    # If almost nothing survives, this was not a cut-out and the whole frame
    # is the honest measurement.
    return subject if subject.mean() > 0.10 else np.ones_like(subject, dtype=bool)


def assess_quality(image_bgr: np.ndarray) -> QualityResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    long_edge = max(h, w)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    subject = _subject_mask(gray)
    lit = gray[subject]
    dark = float(np.mean(lit <= 5))
    bright = float(np.mean(lit >= 250))

    warnings: list[str] = []
    rejects: list[str] = []

    if long_edge < MIN_LONG_EDGE:
        # Name the usual cause. Every rejected photo in the run that prompted
        # this was 550-639px and came straight from WhatsApp, which
        # re-compresses images to roughly 600px on the long edge -- the
        # original off the camera was several thousand. The fix is the
        # transport, not the threshold, and saying so beats leaving someone to
        # conclude their photos are unusable.
        rejects.append(
            f"resolution too low ({w}x{h}, need long edge ≥ {MIN_LONG_EDGE}); "
            "messaging apps re-compress photos -- send the original as a file"
        )

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
