"""GLB export: low-poly mesh + baked maps → a single self-contained asset.

The mesh must already carry UVs (TextureVisuals); the normal map is the
object-space bake from meshops.bake. Provenance metadata (scale source,
generative flags) rides along in the glTF extras.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


def export_glb(
    mesh: trimesh.Trimesh,
    out_path: Path,
    *,
    normal_map: np.ndarray | None = None,
    base_color: tuple[int, int, int, int] = (200, 200, 200, 255),
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write a .glb; returns the path written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = mesh.copy()
    if normal_map is not None:
        if not isinstance(mesh.visual, trimesh.visual.TextureVisuals):
            raise ValueError("normal_map given but mesh has no UV coordinates")
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=base_color,
            normalTexture=Image.fromarray(normal_map, mode="RGB"),
            metallicFactor=0.0,
            roughnessFactor=0.9,
        )
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv, material=material
        )

    scene = trimesh.Scene(geometry={"model": mesh})
    if extras:
        scene.metadata.update(extras)
    scene.export(out_path)
    return out_path
