"""Turn a shattered generative surface into a closed solid.

Some generative backends return a *surface sample*, not a body. Measured on
this project, a TRELLIS mesh straight out of the worker:

    1,915,811 faces
    40,341 disconnected components, the largest just 4.9% of the area
    609,689 open boundary edges
    volume / convex-hull volume = 0.019

That is not a shell with a few holes -- it is confetti: tens of thousands of
small patches that together read as the right silhouette and enclose nothing.
It renders as "a hollow mesh of polygons", it has no interior, and decimating
it only rearranges the fragments. Hunyuan, by contrast, decodes an occupancy
field and lands at 0.42-0.88 of its hull: an actual body.

The repair is voxel-based and deliberately not clever:

    voxelise -> dilate -> fill interior -> erode -> marching cubes

Dilation is the load-bearing step. ``binary_fill_holes`` alone fills only
regions that are *not* connected to the border, so with gaps between the
patches the interior drains straight out and nothing gets filled -- measured,
that path reaches 0.13 where this one reaches 0.96. Dilating first bridges the
gaps, filling then seals a genuine interior, and eroding by the same amount
puts the surface back where it started.

Measured on a real job, 250k-face input at 128 divisions, dilate 2:

    fill 0.033 -> 0.96 of the convex hull, watertight, 0 open edges
    surface deviates from the original by 0.55% of the object's size (p95 0.78%)
    4.2 s, well inside the local memory budget

The honest cost: any concavity narrower than the dilation radius is filled in,
and a hole passing through the object closes if it is narrower than that too.
For a lighter or a bracket that is invisible. For a mesh screen it is wrong,
which is why this is opt-in per backend rather than applied to everything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class SolidifyResult:
    mesh: trimesh.Trimesh
    applied: bool
    reason: str
    components_before: int
    components_after: int
    fill_before: float
    fill_after: float


def count_components(mesh: trimesh.Trimesh) -> int:
    """Connected components, without ``split()``.

    ``split()`` materialises every submesh; on a confetti mesh that is tens of
    thousands of copies and it OOM-killed a 16 GB machine outright. Labelling
    the face-adjacency graph answers the same question for a few hundred MB.
    """
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return len(mesh.faces)

    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    graph = sp.coo_matrix(
        (np.ones(len(adj)), (adj[:, 0], adj[:, 1])),
        shape=(len(mesh.faces),) * 2,
    )
    return int(connected_components(graph, directed=False)[0])


def fill_ratio(mesh: trimesh.Trimesh) -> float:
    """Volume as a fraction of the convex hull's, the shattered-mesh tell.

    Scale-free, so it compares across objects and backends. A solid body sits
    at 0.4-0.9; confetti sits near zero because the enclosed volume of a cloud
    of open patches is close to nothing.
    """
    try:
        hull = mesh.convex_hull.volume
        return float(abs(mesh.volume) / hull) if hull > 0 else 0.0
    except Exception:
        return 0.0


def looks_shattered(mesh: trimesh.Trimesh, *, fill_threshold: float = 0.15) -> bool:
    """True when the mesh encloses almost nothing relative to its own hull.

    Deliberately keyed on fill rather than component count. A legitimately
    complex model can have many components; what no real body does is occupy a
    fifteenth of its own convex hull.
    """
    return fill_ratio(mesh) < fill_threshold


#: Faces produced per division squared. Marching cubes triangulates a surface,
#: so the count grows with the square of the grid resolution, not its cube.
#:
#: This is only a starting guess. Measured on real output it is not constant --
#: about 5.5 below ~150 divisions and 11.5 above, because finer grids resolve
#: detail the coarse ones smooth away, and it varies per object besides. So the
#: first pass uses this and then _refine measures what actually came out and
#: corrects, rather than trusting a number that is wrong half the time.
FACES_PER_DIVISION_SQ = 11.5

#: 256 costs about 3.4 GB and 20 s. Past that the local memory budget on a
#: 16 GB machine is the binding constraint, not the mesh.
MAX_DIVISIONS = 256
MIN_DIVISIONS = 64


def divisions_for_target(target_faces: int | None) -> int:
    """Grid resolution that yields roughly ``target_faces`` after extraction.

    Solidify replaces the mesh outright, so its resolution sets a ceiling the
    face budget cannot exceed -- decimation only removes faces. A job that
    asked for 300,000 came back with 89,844 because the grid was fixed at 128,
    and nothing downstream could recover the difference.
    """
    if not target_faces or target_faces <= 0:
        return 128
    est = int((target_faces / FACES_PER_DIVISION_SQ) ** 0.5)
    return max(MIN_DIVISIONS, min(MAX_DIVISIONS, est))


def solidify(
    mesh: trimesh.Trimesh,
    *,
    divisions: int | None = None,
    target_faces: int | None = None,
    dilate: int | None = None,
    only_if_shattered: bool = True,
    fill_threshold: float = 0.15,
    _retry: bool = True,
) -> SolidifyResult:
    """Close a surface into a watertight solid.

    ``divisions`` is resolution along the longest axis. Leave it None and pass
    ``target_faces`` instead: the grid then produces enough geometry for the
    budget the user actually asked for, since this mesh replaces theirs and
    decimation downstream can only take faces away.

    ``dilate`` is how many voxels of gap to bridge. Left None it scales with
    ``divisions`` so the bridged distance stays constant in world units;
    raise it for a more shattered input, at the cost of filling narrower
    concavities.
    """
    if divisions is None:
        divisions = divisions_for_target(target_faces)
    if dilate is None:
        # Dilation must scale with resolution: the gaps are a fixed size in
        # world units, so bridging them takes more voxels as the voxels shrink.
        # Holding it at 2 while raising divisions silently stopped sealing the
        # interior -- fill fell from 0.955 at 128 to 0.093 at 256, straight
        # back to the hollow mesh this module exists to fix.
        #
        # The relationship is not linear, so this is measured rather than
        # derived. Minimum dilation that seals, by divisions:
        #
        #     128 -> 2      160 -> 3      192 -> 3      256 -> 6
        #
        # divisions/45 covers all of them with a margin, and over-dilating is
        # cheap: at 128 divisions everything from d2 to d8 gives the same 0.96
        # fill. Erosion undoes the size change; what a larger radius does cost
        # is concavities narrower than itself.
        dilate = max(2, -(-divisions // 45))
    before_fill = fill_ratio(mesh)
    before_components = count_components(mesh)

    if only_if_shattered and not looks_shattered(mesh, fill_threshold=fill_threshold):
        return SolidifyResult(
            mesh=mesh,
            applied=False,
            reason=f"mesh already encloses {before_fill:.2f} of its hull; left alone",
            components_before=before_components,
            components_after=before_components,
            fill_before=before_fill,
            fill_after=before_fill,
        )

    from scipy import ndimage

    extent = float(np.max(mesh.extents))
    if not np.isfinite(extent) or extent <= 0:
        return SolidifyResult(
            mesh, False, "degenerate bounds", before_components,
            before_components, before_fill, before_fill,
        )

    pitch = extent / max(16, divisions)
    voxels = mesh.voxelized(pitch=pitch)

    pad = dilate + 2
    grid = np.pad(voxels.matrix, pad)
    grid = ndimage.binary_dilation(grid, iterations=dilate)
    grid = ndimage.binary_fill_holes(grid)
    grid = ndimage.binary_erosion(grid, iterations=dilate)

    if not grid.any():
        return SolidifyResult(
            mesh, False, "solidify produced an empty grid", before_components,
            before_components, before_fill, before_fill,
        )

    solid = trimesh.voxel.VoxelGrid(grid).marching_cubes
    # marching_cubes comes back in voxel index space; put it back in world
    # coordinates, then undo the padding that was added to give room to dilate.
    solid.apply_transform(voxels.transform)
    solid.vertices -= pad * pitch

    # One correction pass. The face count a grid yields varies by object, so
    # rather than trust the estimate, measure it and rescale from what this
    # mesh actually produced -- but only when we undershot badly enough to
    # matter, since overshooting is free (decimation removes the excess).
    if (
        target_faces
        and _retry
        and len(solid.faces) < target_faces * 0.8
        and divisions < MAX_DIVISIONS
    ):
        observed = len(solid.faces) / (divisions * divisions)
        if observed > 0:
            better = int((target_faces / observed) ** 0.5)
            better = max(divisions + 8, min(MAX_DIVISIONS, better))
            if better > divisions:
                return solidify(
                    mesh,
                    divisions=better,
                    target_faces=target_faces,
                    # dilate=None, NOT the value computed for the old
                    # resolution: the retry runs at a finer grid, and reusing
                    # the coarse dilation stops bridging the gaps entirely --
                    # it put fill back to 0.09, undoing the whole repair.
                    dilate=None,
                    only_if_shattered=False,
                    fill_threshold=fill_threshold,
                    _retry=False,
                )

    return SolidifyResult(
        mesh=solid,
        applied=True,
        reason=(
            f"solidified at {divisions}^3 (dilate {dilate}): "
            f"fill {before_fill:.3f} -> {fill_ratio(solid):.3f}"
        ),
        components_before=before_components,
        components_after=count_components(solid),
        fill_before=before_fill,
        fill_after=fill_ratio(solid),
    )
