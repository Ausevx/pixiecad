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

On fusing: the obvious approach is a face-by-face affinity matrix, but 20k
faces squared is 3.2 GB and this has to run on a 16 GB laptop. So we cluster
*masks* instead of faces. There are only ~200 masks across all views, and two
masks that cover the same faces in different views are the same part almost by
definition. That turns an intractable 20k-node problem into a 200-node one,
and the face labelling falls out as a weighted vote afterwards.

Mask overlap is measured with area-weighted Jaccard rather than pixel counts.
A face near the silhouette gets few pixels in one view and many in another;
weighting by true surface area makes the comparison independent of which
views happen to see it well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import trimesh

from .render import BACKGROUND_FACE_ID, Render, render_mesh

# Two masks are the same part above this area-weighted Jaccard. Set from the
# geometry of the problem: adjacent-but-distinct parts (a tyre and the
# suspension arm touching it) share only their boundary faces and score well
# under 0.3, while the same wheel seen from two angles shares its whole
# visible surface and scores well over it.
DEFAULT_MERGE_THRESHOLD = 0.35

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


def _similarity(regions: list[MaskRegion], n_faces: int) -> np.ndarray:
    """Area-weighted Jaccard between every pair of masks."""
    n = len(regions)
    sim = np.zeros((n, n), dtype=np.float64)
    if not n:
        return sim

    # Dense per-region indicator over faces is n_masks x n_faces of float32 --
    # 200 x 20k is 16 MB, small enough to keep flat and fast.
    ind = np.zeros((n, n_faces), dtype=np.float32)
    for i, r in enumerate(regions):
        ind[i, r.faces] = r.weights.astype(np.float32)

    areas = ind.sum(axis=1)
    # min() over the two indicators is the weighted intersection; since every
    # region stores the same per-face area, this reduces to the shared faces.
    for i in range(n):
        inter = np.minimum(ind[i], ind[i + 1 :]).sum(axis=1)
        union = areas[i] + areas[i + 1 :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(union > 0, inter / union, 0.0)
        sim[i, i + 1 :] = s
        sim[i + 1 :, i] = s
    return sim


def _agglomerate(sim: np.ndarray, threshold: float, views: list[str] | None = None) -> np.ndarray:
    """Cluster masks into parts; returns a cluster id per mask.

    Single linkage, deliberately, plus a cannot-link constraint. The reasoning
    is worth recording because the obvious choice is wrong.

    Masks of one part from *opposite* views share no faces whatsoever -- the
    front view of a wheel covers its front hemisphere, the back view its back
    hemisphere, and the intersection is empty. So similarity between them is
    zero and average linkage refuses to join them: two spheres came out as
    eight clusters. A part is only ever connected around the object as a
    *chain* of overlapping viewpoints, which is precisely what single linkage
    is for.

    Single linkage alone chains too eagerly, fusing distinct parts through one
    bridging mask. The constraint fixes that without a threshold fight: two
    masks that appear as *separate* masks in the same view are, by the
    segmenter's own judgement, different parts. Clusters may therefore only
    merge when they have no view in common. That makes the sphere chain close
    around each sphere and stop, because by then both clusters hold a mask
    from every view.
    """
    n = len(sim)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: list[set[str]] = [{views[i]} if views else set() for i in range(n)]

    iu, ju = np.triu_indices(n, k=1)
    scores = sim[iu, ju]
    keep = scores >= threshold
    iu, ju, scores = iu[keep], ju[keep], scores[keep]
    # Strongest evidence first, so a confident pairing is never pre-empted by
    # a marginal one that happens to be scanned earlier.
    for k in np.argsort(scores)[::-1]:
        a, b = find(int(iu[k])), find(int(ju[k]))
        if a == b or (seen[a] & seen[b]):
            continue
        parent[b] = a
        seen[a] |= seen[b]

    roots = {}
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


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

    adj = mesh.face_adjacency
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
    merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
) -> SemanticSegmentation:
    """Label every face of ``mesh`` with a semantic part id."""
    renders = renders if renders is not None else render_mesh(
        mesh, resolution=resolution, up_axis=up_axis
    )
    n_faces = len(mesh.faces)
    face_area = np.asarray(mesh.area_faces, dtype=np.float64)

    regions: list[MaskRegion] = []
    for render in renders:
        regions.extend(_regions_for_view(render, segmenter.segment(render.rgb), face_area))

    if not regions:
        return SemanticSegmentation(np.zeros(n_faces, dtype=np.int64), 1, [], {0: []})

    mask_labels = _agglomerate(
        _similarity(regions, n_faces), merge_threshold, [r.view for r in regions]
    )
    n_clusters = int(mask_labels.max()) + 1 if len(mask_labels) else 1

    # Weighted vote. A face may sit under masks from several clusters near a
    # part boundary; it goes to whichever cluster saw the most of it, summed
    # over every view, which is exactly the multi-view agreement we rendered
    # 14 angles to obtain.
    votes = np.zeros((n_clusters, n_faces), dtype=np.float32)
    for region, cluster in zip(regions, mask_labels):
        votes[cluster, region.faces] += region.weights.astype(np.float32)

    face_labels = np.where(votes.max(axis=0) > 0, votes.argmax(axis=0), -1).astype(np.int64)
    face_labels = _propagate(mesh, face_labels)

    cluster_views: dict[int, list[str]] = {}
    for region, cluster in zip(regions, mask_labels):
        cluster_views.setdefault(int(cluster), []).append(region.view)

    return SemanticSegmentation(
        face_labels=face_labels,
        n_clusters=n_clusters,
        regions=regions,
        cluster_views=cluster_views,
    )
