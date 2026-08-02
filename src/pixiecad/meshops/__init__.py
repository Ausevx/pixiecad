"""Mesh processing operations."""

from .bake import UnwrapResult, bake_object_space_normals, save_normal_map, unwrap_uv
from .cleanup import CleanupReport, clean_mesh
from .decimate import DecimateResult, budget_for_parts, decimate_to_budget

__all__ = [
    "CleanupReport",
    "clean_mesh",
    "DecimateResult",
    "decimate_to_budget",
    "budget_for_parts",
    "UnwrapResult",
    "unwrap_uv",
    "bake_object_space_normals",
    "save_normal_map",
]
