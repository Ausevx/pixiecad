"""Cheap structural checks on a generated mesh.

Exists because a build once produced a cube-shaped shell of noise and every
numeric check passed: the face count was exact, the UVs were present, the
normal bake succeeded, the file was valid glTF. Nothing looked at whether the
geometry was an object. These are the checks that would have caught it.

Deliberately shape-only and dependency-free — no rendering, no model, no
network — so this can run on every build without slowing it down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

# A generative failure fills its bounding box roughly uniformly, so the box
# comes out near-cubic and the surface sits at a high fraction of the box.
# Real man-made objects are almost never within 5% of cubic in all three axes.
CUBE_TOLERANCE = 0.05
# Surface area over bounding-box area. A solid object is well under 1; noise
# that fills the volume with sheets goes far above it.
MAX_AREA_RATIO = 2.0


@dataclass
class SanityReport:
    ok: bool
    extents: tuple[float, float, float]
    aspect: tuple[float, float, float]
    area_ratio: float
    n_components: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        state = "ok" if self.ok else "SUSPECT"
        return (
            f"[{state}] aspect={tuple(round(a, 2) for a in self.aspect)} "
            f"area_ratio={self.area_ratio:.2f} components={self.n_components}"
        )


def check_mesh(
    mesh: trimesh.Trimesh,
    *,
    expect_elongated: bool = True,
) -> SanityReport:
    """Flag geometry that is unlikely to be a real object.

    ``expect_elongated`` should be False for objects that genuinely are roughly
    cubic (a dice, a box), where the cube test is meaningless.
    """
    ext = np.asarray(mesh.extents, dtype=float)
    longest = float(ext.max()) if ext.size and ext.max() > 0 else 1.0
    aspect = tuple(float(v / longest) for v in ext)

    box_area = 2.0 * (
        ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2]
    ) if ext.size == 3 else 1.0
    area_ratio = float(mesh.area / box_area) if box_area > 0 else 0.0

    warnings: list[str] = []

    if expect_elongated and all(abs(a - 1.0) <= CUBE_TOLERANCE for a in aspect):
        warnings.append(
            f"bounding box is near-cubic {tuple(round(a, 3) for a in aspect)}; "
            "generative failures fill their box uniformly, real objects rarely do"
        )

    if area_ratio > MAX_AREA_RATIO:
        warnings.append(
            f"surface area is {area_ratio:.1f}x its bounding box area; "
            "that is characteristic of noise sheets, not a solid surface"
        )

    n_components = 0
    try:
        from .cleanup import _positional_face_adjacency
        from trimesh import graph

        n_faces = len(mesh.faces)
        if n_faces:
            comps = graph.connected_components(
                _positional_face_adjacency(mesh), nodes=np.arange(n_faces)
            )
            n_components = len(comps)
    except Exception:
        n_components = -1

    return SanityReport(
        ok=not warnings,
        extents=tuple(float(v) for v in ext),
        aspect=aspect,
        area_ratio=area_ratio,
        n_components=n_components,
        warnings=warnings,
    )
