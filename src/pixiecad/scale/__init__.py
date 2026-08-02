"""Scale recovery stage: ArUco marker detection and dimension-based scaling."""

from pixiecad.scale.aruco import MarkerDetection, detect_markers, scan_images
from pixiecad.scale.transform import (
    apply_scale,
    scale_factor_from_dimensions,
    scale_factor_from_marker,
)

__all__ = [
    "MarkerDetection",
    "detect_markers",
    "scan_images",
    "scale_factor_from_dimensions",
    "scale_factor_from_marker",
    "apply_scale",
]
