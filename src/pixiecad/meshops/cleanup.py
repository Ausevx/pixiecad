"""Mesh cleanup: remove duplicate vertices, degenerate faces, floaters, fix normals."""

from __future__ import annotations

from dataclasses import dataclass

import trimesh


@dataclass
class CleanupReport:
    faces_before: int
    faces_after: int
    vertices_before: int
    vertices_after: int
    components_removed: int
    watertight: bool


def clean_mesh(
    mesh: trimesh.Trimesh,
    *,
    min_component_area_frac: float = 0.01,
) -> tuple[trimesh.Trimesh, CleanupReport]:
    """Clean mesh by deduplicating vertices/faces, removing floating debris, and fixing normals.

    Args:
        mesh: Input trimesh object (not mutated).
        min_component_area_frac: Minimum area fraction for a connected component to be kept.

    Returns:
        Tuple of (cleaned_mesh, CleanupReport).
    """
    faces_before = len(mesh.faces)
    vertices_before = len(mesh.vertices)

    work_mesh = mesh.copy()

    # Deduplicate vertices, remove degenerate/duplicate faces, remove unreferenced vertices
    work_mesh.merge_vertices()
    work_mesh.update_faces(work_mesh.nondegenerate_faces())
    work_mesh.update_faces(work_mesh.unique_faces())
    work_mesh.remove_unreferenced_vertices()

    # Split into connected components and remove small floating debris (floaters)
    components = work_mesh.split(only_watertight=False)
    components_removed = 0
    if components:
        total_area = sum(c.area for c in components)
        min_area = min_component_area_frac * total_area
        kept_components = [c for c in components if c.area >= min_area]
        if not kept_components:
            kept_components = [max(components, key=lambda c: c.area)]
        components_removed = len(components) - len(kept_components)

        if len(kept_components) == 1:
            work_mesh = kept_components[0]
        else:
            work_mesh = trimesh.util.concatenate(kept_components)

    # Fix face orientation / normals
    trimesh.repair.fix_normals(work_mesh)

    report = CleanupReport(
        faces_before=faces_before,
        faces_after=len(work_mesh.faces),
        vertices_before=vertices_before,
        vertices_after=len(work_mesh.vertices),
        components_removed=components_removed,
        watertight=bool(work_mesh.is_watertight),
    )

    return work_mesh, report
