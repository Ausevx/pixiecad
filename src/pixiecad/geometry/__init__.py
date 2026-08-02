"""Geometry package for PixieCAD."""

from .dense import COLMAP_MIN_VRAM_MB, DenseResult, DenseUnavailable, build_dense_script, run_dense
from .sparse import SparseFailure, SparseResult, run_sparse

__all__ = [
    "SparseFailure",
    "SparseResult",
    "run_sparse",
    "COLMAP_MIN_VRAM_MB",
    "DenseResult",
    "DenseUnavailable",
    "build_dense_script",
    "run_dense",
]

