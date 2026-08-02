"""Semantic segmentation of a fused mesh by fusing 2D masks across many views.

Why this exists: a generative mesh is one welded shell. Our F1 has 99.96% of
its surface area in a single connected component, so connectivity splitting
finds nothing and the old k-means fallback carved spatial blobs -- "upper-rear-
centre" -- rather than wheels. The part structure simply is not present in the
geometry, so it has to come from somewhere else: a 2D segmenter that has seen
enough cars to know a wheel when it looks at one.

The pipeline:

1. Render the mesh from 14 directions, keeping a face-index buffer (``render``).
2. Segment each RGB render into masks (any ``Segmenter2D``).
3. Back-project: face ids under a mask are that mask's face set. Free, because
   the renderer already recorded which face owns each pixel.
4. Fuse. This is the hard step and the design worth explaining.

On fusing, which is where the design earns its keep. A dense face-by-face
affinity is out: 20k faces squared is 3.2 GB and this runs on a 16 GB laptop.
Clustering the ~200 *masks* instead was tried and abandoned -- SAM splits one
wheel into rim and tyre within a single view, so no threshold separates "two
masks of the same part" from "two parts", and the result was 90+ fragments.

What works is voting on mesh *edges*. For every pair of faces sharing an edge,
count the views placing them in the same mask against those splitting them.
That is one integer per edge, ~30k of them, and it is the entire multi-view
signal. Parts then come out spatially connected by construction, and a single
bad view is outvoted instead of being able to sever or merge a part alone.

Turning those votes into regions needs care, and two plausible methods failed
outright before the third worked:

- Cut low-agreement edges, then take connected components. No threshold is
  right: net>=1 leaves one region holding 94.6% of the car (wheels fused to
  the body), while net>=6 shatters it into 17,807 fragments with nothing above
  1%. A ratio instead of a count is worse still -- an edge is co-visible in
  only ~4 views, so a ratio has no resolution.
- Cut aggressively, then merge regions back by mean boundary agreement. This
  snowballs: mean agreement carries no notion of size, so the biggest region
  always has *some* strongly-agreeing boundary left and keeps winning. One
  region grew from 13.6% of the surface to 97.9%.

What works is Felzenszwalb-Huttenlocher: a region absorbs a neighbour only
when the boundary between them is no worse than its own internal variation
plus ``k / area``. That slack shrinks as a region grows, which is exactly the
pressure the merging approach lacked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import trimesh

from .render import BACKGROUND_FACE_ID, Render, render_mesh


def _adjacency(mesh: trimesh.Trimesh) -> np.ndarray:
    """Face adjacency that survives UV seams.

    trimesh.face_adjacency keys on vertex indices, and a textured mesh
    duplicates vertices along every UV seam -- so the graph is cut wherever
    the texture wraps. Measured on the F1: 25,676 adjacent pairs instead of
    ~30,000, leaving 390 regions with no neighbour at all, which stalled
    merging at 606 parts because most regions had nothing to merge *into*.
    cleanup solved this already; reuse it rather than rediscover it.
    """
    from ..meshops.cleanup import _positional_face_adjacency

    return _positional_face_adjacency(mesh)

# A mask covering almost the entire object is the segmenter returning the
# whole car rather than a part; keeping it would swallow every other mask
# during clustering.
MAX_MASK_AREA_FRAC = 0.95
MIN_MASK_AREA_FRAC = 0.002



class Segmenter2D(Protocol):
    """Turns one RGB render into a label map.

    Returns an ``(H, W)`` integer array, one distinct non-negative value per
    mask, negative for background. Deliberately narrow: it lets a local stub, a
    remote SAM, and a text-prompted detector all drop in unchanged.
    """

    def segment(self, rgb: np.ndarray) -> np.ndarray: ...


class SilhouetteSegmenter:
    """Baseline segmenter: connected regions of the rendered silhouette.

    Not semantic and not a substitute for SAM -- it can only separate parts
    that happen to be disjoint *in projection*, so it finds a wheel only from
    angles where nothing overlaps it. It earns its place as the zero-dependency
    default: it needs no GPU, no weights and no network, so the fusion path
    stays exercisable on the laptop and in CI.
    """

    def __init__(self, threshold: int = 8) -> None:
        self.threshold = threshold

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        from scipy import ndimage

        fg = rgb.max(axis=2) > self.threshold
        labels, _ = ndimage.label(fg)
        # ndimage uses 0 for background; this protocol uses negatives.
        return labels.astype(np.int64) - 1


@dataclass
class MaskRegion:
    """One 2D mask, resolved to the mesh faces it covers."""

    view: str
    label: int
    faces: np.ndarray  # face indices, sorted
    weights: np.ndarray  # surface area per face, parallel to ``faces``

    @property
    def area(self) -> float:
        return float(self.weights.sum())


@dataclass
class SemanticSegmentation:
    """Result: one cluster id per face, plus the masks that produced it."""

    face_labels: np.ndarray  # (n_faces,) int, cluster id per face
    n_clusters: int
    regions: list[MaskRegion] = field(default_factory=list)
    cluster_views: dict[int, list[str]] = field(default_factory=dict)


def _regions_for_view(
    render: Render, labels: np.ndarray, face_area: np.ndarray
) -> list[MaskRegion]:
    """Resolve each 2D mask in one view to its covered faces."""
    if labels.shape != render.face_id.shape:
        raise ValueError(
            f"segmenter returned {labels.shape} for a {render.face_id.shape} render"
        )

    out: list[MaskRegion] = []
    visible = render.face_id != BACKGROUND_FACE_ID
    # Measured against what this view can actually see, not the whole mesh. A
    # single view sees roughly half the surface, so a segmenter returning the
    # entire object scores only ~0.5 against the total and would sail through
    # the "too big to be a part" filter.
    total = float(face_area[render.visible_faces].sum())
    for label in np.unique(labels):
        if label < 0:
            continue
        sel = (labels == label) & visible
        if not sel.any():
            continue
        faces = np.unique(render.face_id[sel])
        weights = face_area[faces]
        frac = float(weights.sum()) / total if total else 0.0
        # Whole-object and speck masks are both noise for clustering.
        if frac > MAX_MASK_AREA_FRAC or frac < MIN_MASK_AREA_FRAC:
            continue
        out.append(MaskRegion(render.camera.name, int(label), faces, weights))
    return out


def _face_mask_labels(render: Render, labels: np.ndarray, n_faces: int) -> np.ndarray:
    """Per-face mask id for one view; -1 where the face is not visible.

    A face straddling a mask boundary collects pixels from both, so it takes
    whichever mask owns the most of it rather than whichever pixel happens to
    be read last.
    """
    valid = render.face_id >= 0
    out = np.full(n_faces, -1, dtype=np.int64)
    if not valid.any():
        return out

    faces = render.face_id[valid].astype(np.int64)
    masks = labels[valid].astype(np.int64)
    keep = masks >= 0
    faces, masks = faces[keep], masks[keep]
    if not len(faces):
        return out

    stride = int(masks.max()) + 2
    combos, counts = np.unique(faces * stride + masks, return_counts=True)
    # Sort by count ascending within each face, then let the last write win:
    # that leaves the majority mask in place without a per-face loop.
    order = np.lexsort((counts, combos // stride))
    out[combos[order] // stride] = combos[order] % stride
    return out


def _edge_votes(per_view: list[np.ndarray], adjacency: np.ndarray) -> np.ndarray:
    """Net agreement per adjacent face pair: agreeing views minus dissenting.

    This is the whole multi-view signal, reduced to one integer per mesh edge.
    Faces not co-visible in a view simply do not vote, so a pair seen well
    from four angles carries more weight than one glimpsed edge-on once.
    """
    a, b = adjacency[:, 0], adjacency[:, 1]
    score = np.zeros(len(adjacency), dtype=np.int32)
    for fm in per_view:
        both = (fm[a] >= 0) & (fm[b] >= 0)
        same = both & (fm[a] == fm[b])
        score += same.astype(np.int32) - (both & ~same).astype(np.int32)
    return score


def _count_significant(labels: np.ndarray, area: np.ndarray, min_area_frac: float) -> int:
    """Number of regions large enough to count as a part."""
    _, inverse = np.unique(labels, return_inverse=True)
    areas = np.bincount(inverse, weights=area)
    return int((areas >= min_area_frac).sum())


def _felzenszwalb(
    adjacency: np.ndarray,
    dissim: np.ndarray,
    area: np.ndarray,
    k: float,
) -> np.ndarray:
    """Graph segmentation with a size-adaptive merge threshold.

    Plain agglomerative merging by boundary agreement snowballs: measured on
    the F1, one region grew from 13.6% of the surface to 97.9%, because a big
    region always has *some* strongly-agreeing boundary left and keeps winning
    the merge. Mean agreement carries no notion of size, so nothing pushes
    back.

    Felzenszwalb-Huttenlocher's criterion supplies exactly that push-back. A
    region only absorbs a neighbour when the boundary between them is no worse
    than the region's own internal variation plus ``k / area``. That slack
    term shrinks as a region grows, so a large region must meet an
    increasingly strict standard, while genuinely small parts stay easy to
    form.
    """
    order = np.argsort(dissim, kind="stable")
    n = len(area)
    parent = np.arange(n)
    size = area.astype(np.float64).copy()
    internal = np.zeros(n, dtype=np.float64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for e in order:
        a, b = find(int(adjacency[e, 0])), find(int(adjacency[e, 1]))
        if a == b:
            continue
        d = float(dissim[e])
        if d <= min(
            internal[a] + k / max(size[a], 1e-12),
            internal[b] + k / max(size[b], 1e-12),
        ):
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            internal[a] = d
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _drop_slivers(labels: np.ndarray, area: np.ndarray, min_area_frac: float) -> np.ndarray:
    """Mark sub-threshold regions as unlabelled so propagation reclaims them.

    Marking and re-propagating rather than merging directly, because slivers
    come in chains: a rim fragment may touch only other fragments, so a single
    "fold into best neighbour" pass leaves most of them stranded (232 regions
    survived on the two-sphere test). Breadth-first propagation from the
    surviving regions reaches all of them and assigns each to the nearest real
    part. Iterative merging was tried first and cascaded -- each absorption
    grew a region until it swallowed the whole model.
    """
    uniq, inverse = np.unique(labels, return_inverse=True)
    areas = np.bincount(inverse, weights=area, minlength=len(uniq))
    small = areas < min_area_frac
    if not small.any() or small.all():
        return inverse.astype(np.int64)

    out = inverse.astype(np.int64).copy()
    out[small[inverse]] = -1
    # Renumber survivors densely; -1 stays as the "fill me" marker.
    survivors = np.flatnonzero(~small)
    remap = np.full(len(uniq), -1, dtype=np.int64)
    remap[survivors] = np.arange(len(survivors))
    keep = out >= 0
    out[keep] = remap[inverse[keep]]
    return out


def _propagate(mesh: trimesh.Trimesh, face_labels: np.ndarray) -> np.ndarray:
    """Fill unlabelled faces from their labelled neighbours.

    Concavities and the underside of a wheel arch are hidden from all 14
    cameras -- about 1,300 of our 20,000 faces. Leaving them unassigned would
    punch holes in every exported part, so they inherit the label that reaches
    them first across the adjacency graph.
    """
    out = face_labels.copy()
    missing = np.flatnonzero(out < 0)
    if not len(missing):
        return out

    adj = _adjacency(mesh)
    if not len(adj):
        out[missing] = 0
        return out

    # Breadth-first over the adjacency graph, one wave per iteration, so a
    # hidden pocket takes the label of the nearest visible surface rather than
    # whichever face happens to be scanned first.
    for _ in range(64):
        unknown = out < 0
        if not unknown.any():
            break
        a, b = adj[:, 0], adj[:, 1]
        moved = False
        for src, dst in ((a, b), (b, a)):
            fill = unknown[dst] & (out[src] >= 0)
            if fill.any():
                out[dst[fill]] = out[src[fill]]
                moved = True
        if not moved:
            break
    # Anything still unreached is an isolated speck; park it on cluster 0.
    out[out < 0] = 0
    return out


def segment_semantic(
    mesh: trimesh.Trimesh,
    segmenter: Segmenter2D,
    *,
    renders: list[Render] | None = None,
    resolution: int = 512,
    up_axis: str = "y",
    max_parts: int = 8,
    min_area_frac: float = 0.01,
) -> SemanticSegmentation:
    """Label every face of ``mesh`` with a semantic part id."""
    renders = renders if renders is not None else render_mesh(
        mesh, resolution=resolution, up_axis=up_axis
    )
    n_faces = len(mesh.faces)
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)

    regions: list[MaskRegion] = []
    per_view: list[np.ndarray] = []
    for render in renders:
        labels = segmenter.segment(render.rgb)
        regions.extend(_regions_for_view(render, labels, face_area))
        per_view.append(_face_mask_labels(render, labels, n_faces))

    if not regions:
        return SemanticSegmentation(np.zeros(n_faces, dtype=np.int64), 1, [], {0: []})

    adjacency = _adjacency(mesh)
    edge_score = _edge_votes(per_view, adjacency)
    # Dissimilarity: high agreement -> low cost to keep faces together.
    dissim = float(edge_score.max()) - edge_score.astype(np.float64)
    area = np.asarray(mesh.area_faces, dtype=np.float64)
    area = area / (area.sum() or 1.0)

    # k sets the region scale, and its useful value depends on the mesh and on
    # how many parts were asked for, so search it rather than hard-code one.
    # Bisection on a monotone-in-k region count: ~40 passes over 30k edges,
    # which is milliseconds and far more robust than a tuned constant.
    # Scan k rather than bisect. Two things rule bisection out: the count of
    # regions above min_area is not monotone in k (at small k nearly every
    # region is sub-threshold, so the count is low at *both* ends), and total
    # region count is monotone but useless as a target, because a generative
    # mesh carries a handful of stray 2-face components that can never merge
    # -- bisecting to "10 regions" happily returned one real part plus nine
    # specks. Scanning costs ~50 union-find passes, a fraction of a second.
    best, best_count = None, -1
    for k in np.geomspace(1e-7, 1.0, 50):
        labels = _felzenszwalb(adjacency, dissim, area, float(k))
        count = _count_significant(labels, area, min_area_frac)
        # Prefer the segmentation that gets closest to the requested part
        # count from below; ties go to the larger k, i.e. the coarser split.
        if best_count <= count <= max_parts:
            best, best_count = labels, count
    if best is None:
        best = _felzenszwalb(adjacency, dissim, area, 1.0)

    merged = _drop_slivers(best, area, min_area_frac)
    face_labels = _propagate(mesh, merged)
    n_clusters = int(face_labels.max()) + 1

    # Which views contributed to each final part, for diagnostics.
    cluster_views: dict[int, set[str]] = {}
    for render, fm in zip(renders, per_view):
        for cluster in np.unique(face_labels[fm >= 0]):
            cluster_views.setdefault(int(cluster), set()).add(render.camera.name)

    return SemanticSegmentation(
        face_labels=face_labels,
        n_clusters=n_clusters,
        regions=regions,
        cluster_views={k: sorted(v) for k, v in cluster_views.items()},
    )
