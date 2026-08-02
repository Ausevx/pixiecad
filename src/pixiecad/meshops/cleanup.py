"""Mesh cleanup: remove duplicate vertices, degenerate faces, floaters, fix normals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh import graph, grouping


def _positional_face_adjacency(mesh: trimesh.Trimesh) -> np.ndarray:
    """Face adjacency judged by vertex POSITION only.

    trimesh treats vertices that share a location but differ in UV or normal as
    distinct, so every UV seam reads as a crack. On a real textured TRELLIS mesh
    that inflated 2,502 genuine shells into 21,895 "components", which made the
    floater filter delete most of the object. Welding positionally just for the
    connectivity question fixes that without touching the mesh's own UVs.
    """
    unique, inverse = grouping.unique_rows(mesh.vertices, digits=6)
    welded = trimesh.Trimesh(
        vertices=mesh.vertices[unique], faces=inverse[mesh.faces], process=False
    )
    return welded.face_adjacency


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
    min_component_area_frac: float = 1e-4,
) -> tuple[trimesh.Trimesh, CleanupReport]:
    """Clean mesh by deduplicating vertices/faces, removing floating debris, and fixing normals.

    Args:
        mesh: Input trimesh object (not mutated).
        min_component_area_frac: Minimum share of total surface area a connected
            component must have to survive. The default is deliberately small:
            objects like a car are legitimately many separate shells (wheels,
            wings, mirrors), and a 1% threshold deleted over half the model's
            surface. At 1e-4 the specks go and the real parts stay.

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

    # Remove small floating debris.
    #
    # This deliberately does NOT use mesh.split(): that builds one Trimesh
    # object per component, and a generative mesh is highly fragmented -- a
    # real TRELLIS output here had 21,895 components, enough to exhaust a
    # 16 GB machine before any cleaning happened. Labelling faces and masking
    # them is the same operation in bounded memory.
    components_removed = 0
    n_faces = len(work_mesh.faces)
    if n_faces:
        labels = np.full(n_faces, -1, dtype=np.int64)
        components = graph.connected_components(
            _positional_face_adjacency(work_mesh), nodes=np.arange(n_faces)
        )
        for i, comp in enumerate(components):
            labels[comp] = i

        if len(components):
            areas = np.bincount(
                labels, weights=work_mesh.area_faces, minlength=len(components)
            )
            min_area = min_component_area_frac * areas.sum()
            keep = areas >= min_area
            if not keep.any():
                # Everything is "debris" relative to itself: keep the largest
                # rather than returning an empty mesh.
                keep = np.zeros(len(components), dtype=bool)
                keep[int(areas.argmax())] = True
            components_removed = int((~keep).sum())

            if components_removed:
                work_mesh.update_faces(keep[labels])
                work_mesh.remove_unreferenced_vertices()

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
