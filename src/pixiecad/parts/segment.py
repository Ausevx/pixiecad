"""S5: cut one mesh into separately exportable components.

Segmentation here is pure geometry — no neural model, no GPU, no network. That
keeps the feature usable on the dev laptop and makes it deterministic. Naming
the resulting clusters is a language problem and lives in ``pixiecad.vision``;
this module only ever produces anonymous, numbered parts.

Two methods, because real meshes come in two flavours:

- ``connected``  — split on connectivity. Correct and free when the object
  genuinely has detached components (a car body with separate wheels). Useless
  on a single watertight shell.
- ``cluster``    — k-means over face position + orientation, which does split a
  single shell. Cruder: boundaries land near curvature changes rather than on
  semantic seams. Use when ``connected`` returns one part.

``auto`` tries connectivity first and falls back to clustering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from trimesh import graph

# Debris is judged by surface area, not face count: a coarse box can be huge
# with 12 faces while a fine cylinder is tiny with 64, so face count says
# nothing about physical size. 0.01 matches the floater threshold in
# meshops.clean_mesh.
DEFAULT_MIN_AREA_FRAC = 0.01
MIN_FACES_PER_PART = 4

# Cap on how many connected components are turned into real meshes. Callers
# only ever keep a handful (max_parts), and a generative mesh can contain
# thousands of shells -- materialising them all is what ran the machine out of
# memory. Comfortably above any sane max_parts, cheap at this size.
MAX_MATERIALISED_COMPONENTS = 200


@dataclass
class Part:
    """One segmented component, before anyone has named it."""

    index: int
    mesh: trimesh.Trimesh
    n_faces: int
    volume: float
    area: float
    centroid: tuple[float, float, float]
    extents: tuple[float, float, float]
    method: str
    name: str | None = None  # filled in later by vision.name_parts
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_named(self) -> bool:
        return bool(self.name)

    def describe(self, whole: trimesh.Trimesh) -> dict[str, Any]:
        """Structured description used to name this part without rendering it.

        A VLM can infer "front-left wheel" from position + proportions + shape
        statistics alone, which avoids an offscreen-rendering dependency
        entirely. Positions are normalised to the whole object's bounding box so
        the description is scale-free.
        """
        lo, hi = whole.bounds
        span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
        rel = (np.asarray(self.centroid) - lo) / span
        ext = np.asarray(self.extents)
        longest = float(ext.max()) if ext.max() > 0 else 1.0

        return {
            "index": self.index,
            "position": {
                "x_left_to_right": round(float(rel[0]), 3),
                "y_back_to_front": round(float(rel[1]), 3),
                "z_bottom_to_top": round(float(rel[2]), 3),
            },
            "volume_fraction": round(
                float(self.volume / whole.volume) if whole.volume > 0 else 0.0, 4
            ),
            "face_fraction": round(float(self.n_faces / max(len(whole.faces), 1)), 4),
            "extents_m": [round(float(v), 4) for v in ext],
            "elongation": round(float(longest / max(ext.min(), 1e-9)), 2),
            "flatness": round(float(ext.min() / longest), 3),
        }


def _components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Connected components, largest first, without exploding memory.

    mesh.split() builds a Trimesh per component. A generative mesh is made of
    hundreds of shells whose UV seams read as cracks, so split() produced tens
    of thousands of objects here and was OOM-killed on a 16 GB machine. Faces
    are grouped by label instead, and only the components big enough to matter
    are ever materialised.
    """
    from ..meshops.cleanup import _positional_face_adjacency

    n_faces = len(mesh.faces)
    if not n_faces:
        return [mesh]

    labels = np.full(n_faces, -1, dtype=np.int64)
    groups = graph.connected_components(
        _positional_face_adjacency(mesh), nodes=np.arange(n_faces)
    )
    if not len(groups):
        return [mesh]
    for i, g in enumerate(groups):
        labels[g] = i

    areas = np.bincount(labels, weights=mesh.area_faces, minlength=len(groups))
    # Biggest first: downstream keeps the top N, and a car's body must outrank
    # its wing mirror regardless of which component the mesher emitted first.
    order = np.argsort(areas)[::-1][:MAX_MATERIALISED_COMPONENTS]

    out = []
    for idx in order:
        sub = mesh.submesh([np.flatnonzero(labels == idx)], append=True)
        if len(sub.faces):
            out.append(sub)
    return out or [mesh]


def _cluster_faces(mesh: trimesh.Trimesh, k: int, seed: int = 0) -> np.ndarray:
    """K-means over face centroids plus normals; returns a label per face.

    Normals are weighted so that orientation influences the split as much as
    position — without them a wheel merges into the adjacent body panel; with
    them the boundary tends to land where the surface turns.
    """
    centroids = mesh.triangles_center
    lo, hi = mesh.bounds
    span = np.where((hi - lo) > 1e-9, hi - lo, 1.0)
    pos = (centroids - lo) / span
    feats = np.hstack([pos, mesh.face_normals * 0.5])

    rng = np.random.default_rng(seed)
    # k-means++ style seeding: spread the initial centres out, so the result
    # doesn't hinge on a lucky draw.
    centres = [feats[rng.integers(len(feats))]]
    for _ in range(k - 1):
        d = np.min(
            np.linalg.norm(feats[:, None, :] - np.array(centres)[None, :, :], axis=2),
            axis=1,
        )
        total = d.sum()
        probs = (d / total) if total > 0 else None
        centres.append(feats[rng.choice(len(feats), p=probs)])
    centres = np.array(centres)

    labels = np.zeros(len(feats), dtype=int)
    for _ in range(30):
        dists = np.linalg.norm(feats[:, None, :] - centres[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            member = feats[labels == i]
            if len(member):
                centres[i] = member.mean(axis=0)
    return labels


def _submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh | None:
    if not face_mask.any():
        return None
    sub = mesh.submesh([np.where(face_mask)[0]], append=True)
    return sub if isinstance(sub, trimesh.Trimesh) and len(sub.faces) else None


def split_parts(
    mesh: trimesh.Trimesh,
    *,
    method: str = "auto",
    max_parts: int = 8,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    seed: int = 0,
) -> list[Part]:
    """Segment ``mesh`` into at most ``max_parts`` components.

    Components contributing less than ``min_area_frac`` of total surface area
    are dropped as debris. Always returns at least one Part (the whole mesh) so
    callers never have to special-case an empty result.
    """
    if method not in ("auto", "connected", "cluster"):
        raise ValueError(f"unknown method: {method!r} (auto|connected|cluster)")
    if max_parts < 1:
        raise ValueError(f"max_parts must be >= 1, got {max_parts}")

    total_area = float(mesh.area) or 1.0
    min_area = total_area * min_area_frac

    def _keep(m: trimesh.Trimesh) -> bool:
        return len(m.faces) >= MIN_FACES_PER_PART and float(m.area) >= min_area

    chunks: list[trimesh.Trimesh] = []
    used = method
    if method in ("auto", "connected"):
        chunks = [c for c in _components(mesh) if _keep(c)]
        used = "connected"

    if method == "cluster" or (method == "auto" and len(chunks) <= 1):
        k = max(2, min(max_parts, len(mesh.faces) // MIN_FACES_PER_PART))
        labels = _cluster_faces(mesh, k=k, seed=seed)
        chunks = []
        for i in range(k):
            sub = _submesh(mesh, labels == i)
            if sub is not None and _keep(sub):
                chunks.append(sub)
        used = "cluster"

    if not chunks:
        chunks = [mesh]
        used = "whole"

    # Biggest first — the body before the trim — so budgets and names line up
    # with what a person considers the "main" part. Sort by area, not face
    # count: tessellation density says nothing about physical size.
    chunks.sort(key=lambda m: float(m.area), reverse=True)
    chunks = chunks[:max_parts]

    return [
        Part(
            index=i,
            mesh=chunk,
            n_faces=len(chunk.faces),
            volume=float(abs(chunk.volume)),
            area=float(chunk.area),
            centroid=tuple(float(v) for v in chunk.centroid),
            extents=tuple(float(v) for v in chunk.extents),
            method=used,
        )
        for i, chunk in enumerate(chunks)
    ]
