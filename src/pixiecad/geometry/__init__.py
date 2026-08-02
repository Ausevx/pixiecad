"""Geometry package for PixieCAD."""

from .dense import (
    COLMAP_MIN_VRAM_MB,
    DOCKER_COLMAP_IMAGE,
    DenseResult,
    DenseUnavailable,
    build_dense_script,
    docker_prefix,
    run_dense,
)
from .sparse import SparseFailure, SparseResult, run_sparse

__all__ = [
    "SparseFailure",
    "SparseResult",
    "run_sparse",
    "COLMAP_MIN_VRAM_MB",
    "DenseResult",
    "DenseUnavailable",
    "build_dense_script",
    "docker_prefix",
    "DOCKER_COLMAP_IMAGE",
    "run_dense",
]

