"""Surface smoothing that does not cost vertices, faces, or file size.

Taubin's lambda/mu filter rather than plain Laplacian. Laplacian smoothing
shrinks: every pass pulls each vertex toward its neighbours' centroid, so a
wheel visibly deflates and a nose cone rounds off, and the loss compounds with
each iteration. Taubin alternates a positive (smoothing) pass with a slightly
larger negative (unshrinking) pass, which suppresses high-frequency noise
while leaving low-frequency shape -- the actual silhouette -- close to intact.

Two properties matter for this pipeline specifically:

- Vertex and face counts are unchanged, so a smoothed model is byte-for-byte
  the same size as the unsmoothed one. Smoothing is free in the budget the
  user sets, unlike remeshing or subdivision.
- It runs on the welded topology. A generative mesh duplicates vertices along
  every UV seam, and smoothing those independently would pull the two sides of
  a seam apart and tear a visible crack down the model. Coincident vertices
  are smoothed as one and the result is scattered back, so seams stay shut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh import grouping

# Taubin's classic pair. mu must be more negative than lambda is positive, or
# the filter still shrinks; this ratio has a near-unity passband, meaning
# coarse shape is preserved while per-face noise is attenuated hard.
DEFAULT_LAMBDA = 0.5
DEFAULT_MU = -0.53
DEFAULT_ITERATIONS = 5


@dataclass
class SmoothReport:
    iterations: int
    moved_mean: float
    moved_max: float
    volume_change: float
    boundary_pinned: int


def _welded(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Positionally-unique vertices and the map from original -> unique."""
    unique, inverse = grouping.unique_rows(mesh.vertices, digits=6)
    return unique, inverse


def _neighbour_sums(
    verts: np.ndarray, edges: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Summed neighbour positions and valence, via scatter-add."""
    total = np.zeros((n, 3), dtype=np.float64)
    count = np.zeros(n, dtype=np.float64)
    a, b = edges[:, 0], edges[:, 1]
    np.add.at(total, a, verts[b])
    np.add.at(total, b, verts[a])
    np.add.at(count, a, 1.0)
    np.add.at(count, b, 1.0)
    return total, count


def weld_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Recompute vertex normals across UV seams. Free: geometry is untouched.

    A textured mesh duplicates vertices along every UV seam -- our F1 carries
    17,848 vertices for 14,873 distinct positions. trimesh averages face
    normals per *index*, so each copy of a seam vertex only sees the faces on
    its own side and the two sides disagree. The renderer then draws a hard
    crease along every seam, which reads as faceting even though the geometry
    is perfectly continuous there.

    Averaging on welded topology and scattering the result back fixes the
    shading without adding a single vertex or triangle, so it costs nothing in
    the polygon budget and nothing at runtime.
    """
    out = mesh.copy()
    if not len(mesh.faces):
        return out

    unique, inverse = _welded(mesh)
    welded = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[unique],
        faces=inverse[np.asarray(mesh.faces)],
        process=False,
    )
    out.vertex_normals = np.asarray(welded.vertex_normals)[inverse]
    return out


def smooth_mesh(
    mesh: trimesh.Trimesh,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    lamb: float = DEFAULT_LAMBDA,
    mu: float = DEFAULT_MU,
    preserve_boundary: bool = True,
) -> tuple[trimesh.Trimesh, SmoothReport]:
    """Return a Taubin-smoothed copy of ``mesh`` plus a report.

    ``preserve_boundary`` pins vertices on an open edge. A generative mesh is
    rarely watertight, and letting a hole's rim drift inward makes the hole
    grow with every iteration.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if lamb <= 0:
        raise ValueError(f"lamb must be > 0, got {lamb}")
    if mu >= 0:
        raise ValueError(f"mu must be < 0 to counteract shrinkage, got {mu}")

    out = mesh.copy()
    if iterations == 0 or not len(mesh.faces):
        return out, SmoothReport(0, 0.0, 0.0, 0.0, 0)

    unique, inverse = _welded(mesh)
    verts = np.asarray(mesh.vertices, dtype=np.float64)[unique].copy()
    start = verts.copy()
    faces = inverse[np.asarray(mesh.faces)]

    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    # Drop degenerate self-edges, which a collapsed sliver triangle produces
    # and which would otherwise inflate a vertex's valence.
    keep = edges[:, 0] != edges[:, 1]
    edges, counts = edges[keep], counts[keep]

    n = len(verts)
    pinned = np.zeros(n, dtype=bool)
    if preserve_boundary:
        # An edge used by one face is a boundary edge on the welded topology.
        border = edges[counts == 1]
        pinned[np.unique(border)] = True

    free = ~pinned
    for _ in range(iterations):
        for factor in (lamb, mu):
            total, count = _neighbour_sums(verts, edges, n)
            valid = count > 0
            delta = np.zeros_like(verts)
            delta[valid] = total[valid] / count[valid, None] - verts[valid]
            delta[~free] = 0.0
            verts += factor * delta

    moved = np.linalg.norm(verts - start, axis=1)
    out.vertices = verts[inverse]

    before = float(abs(mesh.volume)) if len(mesh.faces) else 0.0
    after = float(abs(out.volume)) if len(out.faces) else 0.0
    return out, SmoothReport(
        iterations=iterations,
        moved_mean=float(moved.mean()),
        moved_max=float(moved.max()),
        volume_change=float((after - before) / before) if before > 0 else 0.0,
        boundary_pinned=int(pinned.sum()),
    )
