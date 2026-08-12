"""GLB export: low-poly mesh + baked maps → a single self-contained asset.

The mesh must already carry UVs (TextureVisuals); the normal map is the
object-space bake from meshops.bake. Provenance metadata (scale source,
generative flags) rides along in the glTF extras.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


def _sanitize_node_name(name: str | None) -> str:
    """Sanitise a string for use as a glTF node name.

    Path separators are removed to prevent glTF node paths or scene hierarchy
    from being misinterpreted. If sanitising leaves an empty string or if None
    was passed, falls back to "model".
    """
    if not name:
        return "model"
    clean = re.sub(r"[/\\]+", "-", name).strip("- \t\r\n")
    return clean if clean else "model"


def export_glb(
    mesh: trimesh.Trimesh,
    out_path: Path,
    *,
    node_name: str | None = None,
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

    clean_name = _sanitize_node_name(node_name)
    scene = trimesh.Scene(geometry={clean_name: mesh})
    if extras:
        scene.metadata.update(extras)
    scene.export(out_path)
    return out_path
