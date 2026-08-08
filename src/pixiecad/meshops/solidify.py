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

    voxelise -> dilate -> fill interior -> extract the surface back at -dilate

Dilation is the load-bearing step. ``binary_fill_holes`` alone fills only
regions that are *not* connected to the border, so with gaps between the
patches the interior drains straight out and nothing gets filled -- measured,
that path reaches 0.13 where this one reaches 0.96. Dilating first bridges the
gaps, filling then seals a genuine interior, and pulling the surface back in by
the same amount puts it where it started.

That last step is a *level set*, not an erosion, and the difference is the
whole visual quality of the result. Extracting a surface from a binary grid
gives marching cubes nothing to interpolate: every vertex lands on a voxel
corner, so a face is either exactly axis-aligned or exactly diagonal. Flat
faces of the object happen to lie on grid planes and come out clean, while
every curved surface becomes a 45-degree staircase -- "the flat faces get
solidified better and the curved ones are still like a mesh". Measured on a
shipped model, 4.2% of edges sat at exactly 45 degrees and *nothing at all*
fell in the 5-20 degree band a genuinely curved surface produces.

So the grid is converted to a signed distance field first, lightly blurred,
and the surface is taken at the -dilate isosurface, which is where erosion
would have put it. Distances are continuous, so marching cubes interpolates
and the terracing has nowhere to come from. Measured on the same mesh:

    45-degree edges   4.2%  ->  0.0%
    mean dihedral     3.20  ->  0.82 degrees
    fill              0.958 ->  0.969
    surface moved     0.02% of object size mean, 0.09% p95

That error is an order of magnitude below what voxelising costs in the first
place (0.55% mean), so the smoothing is close to free in accuracy terms. It is
also faster than the erosion it replaces.

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


#: Below this the repair did not work, whatever the other numbers say.
#:
#: A healthy result lands at 0.95+; Hunyuan output that never needed repairing
#: sits at 0.42-0.88. A run that ends at 0.05 produced a watertight mesh with
#: one component and every stage green, and it was a blob -- the surface was
#: too fragmented to bridge and the interior drained out through the gaps.
#: Measured on that mesh, no dilation rescued it: 6 -> 0.047, 8 -> 0.067,
#: 10 -> 0.137, 14 -> 0.204, while a good generation of the same object from
#: the same photos reached 0.966 at dilation 6. So a low fill means the
#: GENERATION is unusable, not that the repair needs tuning, and the only
#: useful response is to say so before twenty minutes of texturing go into it.
FILL_FLOOR = 0.35

#: Dilation is doubled up to here while the fill is unacceptable. Past this the
#: bridged distance is a sixth of the object and any real concavity is gone --
#: at that point a better result would not be the object anyway.
MAX_DILATE = 16


@dataclass
class SolidifyResult:
    mesh: trimesh.Trimesh
    applied: bool
    reason: str
    components_before: int
    components_after: int
    fill_before: float
    fill_after: float
    #: False when the surface could not be closed into a plausible body. The
    #: mesh is still returned -- it is no worse than the input -- but the
    #: caller must not present it as a repaired model.
    repaired: bool = True


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


#: Grid resolution along the longest axis. Deliberately NOT derived from the
#: face budget any more.
#:
#: It used to be: divisions = sqrt(target_faces / 11.5), so a job asking for
#: 20,000 faces got a 64^3 grid where one voxel is 1.6% of the object. Since
#: every error this module makes is a fixed number of VOXELS, that made
#: dimensional accuracy a function of how many triangles the user happened to
#: ask for -- a 20k job came back 8.5% too thick on its thinnest axis. The grid
#: now always runs as fine as the machine affords and ``decimate_to_budget``
#: takes the face count down afterwards, which is what it was always for.
#:
#: 256 yields ~9.5 * 256^2 = 620k faces, above any budget the worker can
#: deliver, so nothing downstream is starved. Cost measured end to end on a
#: 1.9M-face input: 2.06 GB and ~5 s with the chunked voxeliser below (3.91 GB
#: without it). Past 256 the distance transforms grow as D^3 for a fraction of
#: a percent of accuracy, so this is also the ceiling.
DEFAULT_DIVISIONS = 256
MAX_DIVISIONS = 256
MIN_DIVISIONS = 64

#: Blur applied to the distance field before extraction, in VOXELS -- so it is
#: resolution-independent and needs no scaling with ``divisions``.
#:
#: The distance transform is exact but quantised, which leaves fine noise on
#: the extracted surface. Swept against the true offset surface on a real mesh:
#:
#:     sigma   mean dihedral   surface error (p95, % of object size)
#:     0.0         8.21              0.00
#:     1.0         3.41              0.05
#:     1.5         0.82              0.09
#:     2.0         0.72              0.14
#:
#: 1.5 is where the noise is fully suppressed; past it the curve flattens and
#: only the error keeps growing. The cost is that a genuinely sharp convex edge
#: picks up a fillet of roughly this radius.
#:
#: Note this blur does NOT move the surface, now that the field is a true
#: signed distance: blurring a locally-linear function leaves its level sets
#: where they were. Measured, dilate 2: sigma 0 and sigma 1.5 agree to 0.000
#: voxels. Before the half-voxel correction below they differed by 0.28.
SMOOTH_SIGMA = 1.5

#: Voxelising is extensive, and this cancels the average of it.
#:
#: ``voxelize_subdivide`` marks the NEAREST voxel centre to each surface sample
#: (``round(v / pitch)``), so the marked voxel's outer face lands somewhere in
#: [0, 1) voxels beyond the true surface -- uniformly distributed, expectation
#: exactly 0.5. Measured over 3 shapes x 4 sub-voxel phases x 3 axes at 72
#: divisions, after the EDT correction in ``extract_surface``:
#:
#:     bias 0.0 -> mean +0.821 voxels, rms 0.90
#:     bias 0.5 -> mean -0.191 voxels, rms 0.42
#:
#: The residual +-0.5 spread is irreducible: where the surface falls inside its
#: voxel is genuinely unknown at this resolution. This cancels the mean, not
#: the spread. Derived rather than fitted -- a curve fit on three shapes gave
#: 0.41, and 0.09 voxels is 0.03% of the object at 256 divisions.
SHELL_BIAS = 0.5

#: Dilation starts at divisions/64 rather than divisions/45.
#:
#: Dilation is what bridges the gaps between fragments, and it is also what
#: destroys detail: dilate-then-fill welds shut any gap or concavity narrower
#: than twice this radius. At 256 divisions the old rule gave 6 voxels, a
#: closing radius of 2.3% of the longest axis -- wide enough to swallow a
#: lighter's lid seam. Starting at 4 (1.6%) keeps more, and the escalation in
#: ``solidify`` recovers the cases that genuinely need more, so nothing is
#: traded away permanently.
DILATE_DIVISOR = 64


def start_dilate(divisions: int) -> int:
    """First dilation to try at this resolution, in voxels."""
    return max(2, -(-divisions // DILATE_DIVISOR))


def _dilation_schedule(start: int) -> list[int]:
    """Dilations to try, in order, ending at ``MAX_DILATE``."""
    out: list[int] = []
    d = start
    while d < MAX_DILATE:
        out.append(d)
        d = max(d + 2, d * 2)
    out.append(MAX_DILATE)
    return out


def _voxelize(
    mesh: trimesh.Trimesh, pitch: float, *, pad: int, chunk: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """Padded boolean occupancy grid, plus the voxel index of its [0,0,0].

    Equivalent to ``mesh.voxelized(pitch).matrix`` -- same rule, same result --
    but built a few thousand faces at a time. ``trimesh`` subdivides the WHOLE
    mesh until every edge is under half a voxel and holds all of it at once,
    which on a 1.9M-face input at 256 divisions costs 2.6 GB in that one call:
    65% of the module's entire footprint, and the thing that made running at
    256 unaffordable. Chunked it is 311 MB, and the grids are bit-identical.

    ``pad`` is in voxels and is applied on every side, leaving room to dilate
    without the object touching the border -- ``binary_fill_holes`` treats the
    border as outside, so an object flush against it drains.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    lo = np.floor(np.asarray(mesh.bounds[0]) / pitch).astype(np.int64) - pad
    hi = np.ceil(np.asarray(mesh.bounds[1]) / pitch).astype(np.int64) + pad
    grid = np.zeros(tuple((hi - lo + 1).tolist()), dtype=bool)
    limit = np.asarray(grid.shape, dtype=np.int64) - 1

    for start in range(0, len(faces), chunk):
        block = faces[start : start + chunk]
        # Renumber to just this block's vertices so a 1.9M-vertex array is not
        # copied per chunk.
        used, remap = np.unique(block, return_inverse=True)
        sub_v, _ = trimesh.remesh.subdivide_to_size(
            verts[used],
            remap.reshape(block.shape),
            # Half a voxel, so no triangle can step over one uninterrupted.
            max_edge=pitch * 0.5,
            # trimesh defaults to 10 doublings; at 256 divisions a single edge
            # spanning the object needs 9, so the default is one step from
            # under-subdividing and silently leaving holes in the grid.
            max_iter=20,
        )
        idx = np.round(sub_v / pitch).astype(np.int64) - lo
        np.clip(idx, 0, limit, out=idx)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    return grid, lo


def extract_surface(
    grid: np.ndarray, *, offset: float, sigma: float = SMOOTH_SIGMA
) -> trimesh.Trimesh | None:
    """Surface of ``grid`` pulled ``offset`` voxels inward, without terracing.

    Returns vertices in the same index space ``trimesh``'s own
    ``VoxelGrid.marching_cubes`` uses, so the caller's transform is unchanged.

    None when the requested isosurface does not exist in the field -- an offset
    larger than the object's own half-thickness erodes it away completely, and
    ``marching_cubes`` raises rather than returning an empty mesh.
    """
    from scipy import ndimage
    from skimage import measure

    # Signed: positive outside, negative inside, zero on the dilated boundary.
    field = ndimage.distance_transform_edt(~grid) - ndimage.distance_transform_edt(grid)

    # The distance transform measures CENTRE TO CENTRE, but the interface sits
    # half a voxel in from the last occupied centre. A voxel touching
    # background gets edt == 1 while being 0.5 from the surface, so the raw
    # field is sign * (d + 0.5) and asking for level -n lands at true depth
    # n - 0.5: half a voxel too far out, on every side.
    #
    # This was worth exactly +1.000 voxel of extent on an exact binary slab,
    # for every dilation >= 1 and at every resolution -- the dominant term in
    # a real model coming back 8.5% too thick on its thinnest axis. At
    # offset == 0 the +-1 samples bracket the interface symmetrically so the
    # error vanishes, which is why the index-space alignment test never saw it.
    #
    # Done in place: at 256 divisions a second copy of this array is 200 MB.
    field[field > 0] -= 0.5
    field[field < 0] += 0.5

    if sigma > 0:
        field = ndimage.gaussian_filter(field, sigma=sigma)

    # Nudge off the exact isovalue. An unsmoothed distance transform is
    # integer-valued along the axes, so an integer offset lands exactly on
    # thousands of samples (measured: 10,331 on a 128^3 grid) and marching
    # cubes emits degenerate cells there -- the surface comes back with holes
    # and ``is_watertight`` is False, which is the one property this whole
    # module exists to guarantee. A ten-thousandth of a voxel is far below any
    # geometric tolerance here and a no-op once the field is blurred.
    level = -float(offset) - 1e-4
    if not (field.min() < level < field.max()):
        return None

    verts, faces, _, _ = measure.marching_cubes(field, level=level)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def solidify(
    mesh: trimesh.Trimesh,
    *,
    divisions: int | None = None,
    dilate: int | None = None,
    smooth_sigma: float = SMOOTH_SIGMA,
    only_if_shattered: bool = True,
    fill_threshold: float = 0.15,
) -> SolidifyResult:
    """Close a surface into a watertight solid.

    ``divisions`` is resolution along the longest axis, defaulting to
    ``DEFAULT_DIVISIONS`` regardless of any face budget -- see that constant
    for why it used to follow the budget and why that was wrong.

    ``dilate`` is how many voxels of gap to bridge. Left None it starts at
    ``start_dilate(divisions)`` and escalates only if the result does not
    enclose anything, so fine detail is not given away up front.

    ``smooth_sigma`` is the distance-field blur, in voxels. Set it to 0 for a
    faithful-but-noisier surface; it is what keeps curved regions from coming
    out as staircases, and it does not move the surface.
    """
    if divisions is None:
        divisions = DEFAULT_DIVISIONS
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
            before_components, before_fill, before_fill, repaired=False,
        )

    pitch = extent / max(16, divisions)
    # Padded for the LARGEST dilation we might escalate to, so the same grid
    # serves every attempt and voxelising happens once -- it is the expensive
    # step (~4.5 s), and the old recursive escalation paid it again each time.
    base, lo = _voxelize(mesh, pitch, pad=MAX_DILATE + 2)

    if not base.any():
        return SolidifyResult(
            mesh, False, "solidify produced an empty grid", before_components,
            before_components, before_fill, before_fill, repaired=False,
        )

    schedule = _dilation_schedule(start_dilate(divisions) if dilate is None else dilate)
    best: trimesh.Trimesh | None = None
    best_fill = -1.0
    best_dilate = schedule[0]

    for d in schedule:
        grid = ndimage.binary_fill_holes(ndimage.binary_dilation(base, iterations=d))
        # offset carries the shell bias as well as the dilation, so the surface
        # lands where the input's actually was rather than a voxel outside it.
        solid = extract_surface(grid, offset=d + SHELL_BIAS, sigma=smooth_sigma)
        if solid is None:
            # A dilation too small to seal leaves thin shells that the inward
            # offset erodes away completely. That used to end the whole call;
            # now it is just a reason to try a wider radius, which matters far
            # more now that the starting radius is deliberately small.
            continue

        solid.vertices = (solid.vertices + lo) * pitch
        after = fill_ratio(solid)
        if after > best_fill:
            best, best_fill, best_dilate = solid, after, d
        # Enclosing something AND being a single closed body. A fill that
        # leaked through the gaps leaves an inner surface too, so the component
        # count catches an under-sealed body that the fill alone would pass.
        if after >= FILL_FLOOR and count_components(solid) == 1:
            break

    if best is None:
        return SolidifyResult(
            mesh, False,
            f"could not close this surface: nothing survives eroding even "
            f"{schedule[-1]} voxels. This is a bad generation, not a tuning "
            f"problem. Re-run it.",
            before_components, before_components, before_fill, before_fill,
            repaired=False,
        )

    repaired = best_fill >= FILL_FLOOR
    welded = 200.0 * best_dilate / divisions
    if repaired:
        reason = (
            f"solidified at {divisions}^3 (dilate {best_dilate}): "
            f"fill {before_fill:.3f} -> {best_fill:.3f}; "
            f"gaps under {welded:.1f}% of the longest axis are closed"
        )
    else:
        # Say what it is. A watertight, single-component, correctly budgeted
        # blob passes every other check in the pipeline, so this is the only
        # place the truth is available.
        reason = (
            f"could not close this surface: fill {before_fill:.3f} -> "
            f"{best_fill:.3f} at {divisions}^3 even with dilation "
            f"{schedule[-1]}. The generated surface is too fragmented to "
            "enclose anything -- this is a bad generation, not a tuning "
            "problem. Re-run it."
        )

    return SolidifyResult(
        mesh=best,
        applied=True,
        reason=reason,
        components_before=before_components,
        components_after=count_components(best),
        fill_before=before_fill,
        fill_after=best_fill,
        repaired=repaired,
    )
