"""Scale factor calculation and geometric transform application."""

from __future__ import annotations

import numpy as np
from pixiecad.spec import Dimensions


def scale_factor_from_dimensions(
    bbox_extents: tuple[float, float, float],
    dims: Dimensions,
) -> float:
    """Calculate scale factor by comparing declared dimensions against measured bbox extents."""
    if len(bbox_extents) != 3 or any(e <= 1e-9 for e in bbox_extents):
        raise ValueError(
            f"Bounding box extents must contain 3 positive non-zero values, got {bbox_extents}"
        )

    declared_dims = [
        v for v in (dims.length_m, dims.width_m, dims.height_m) if v is not None
    ]
    if not declared_dims:
        raise ValueError("No dimensions declared in Dimensions object")

    sorted_extents = sorted(bbox_extents, reverse=True)
    sorted_declared = sorted(declared_dims, reverse=True)

    ratios = [dec / ext for dec, ext in zip(sorted_declared, sorted_extents)]
    return float(np.mean(ratios))


def scale_factor_from_marker(measured_side_m: float, true_side_m: float) -> float:
    """Calculate scale factor from measured vs true physical side length of a marker."""
    if measured_side_m <= 0 or true_side_m <= 0:
        raise ValueError(
            f"Side lengths must be positive, got measured={measured_side_m}, true={true_side_m}"
        )
    return float(true_side_m / measured_side_m)


def apply_scale(vertices: np.ndarray, factor: float) -> np.ndarray:
    """Apply scale factor to vertex positions (N, 3), returning a new array."""
    if factor <= 0:
        raise ValueError(f"Scale factor must be positive, got {factor}")

    arr = np.asarray(vertices)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Vertices array must have shape (N, 3), got {arr.shape}")

    return (arr * factor).copy()
