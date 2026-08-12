"""Repack a mesh's UVs and resample its texture to a new size.

This avoids shipping the entire atlas for a small part.
"""

from __future__ import annotations

import numpy as np
import trimesh
import xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt
from trimesh.visual.material import PBRMaterial, SimpleMaterial


def _sample_bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample an image using bilinear interpolation."""
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)
    return (
        img[y0, x0] * wa[:, None] +
        img[y0, x1] * wb[:, None] +
        img[y1, x0] * wc[:, None] +
        img[y1, x1] * wd[:, None]
    ).astype(np.uint8)


def _dilate_texture(img: np.ndarray, mask: np.ndarray, padding_px: int) -> np.ndarray:
    """Dilate covered pixels outward by padding_px using nearest neighbor."""
    if padding_px <= 0 or not np.any(mask):
        return img
    dist, inds = distance_transform_edt(~mask, return_distances=True, return_indices=True)
    out = img.copy()
    dilate_mask = (dist > 0) & (dist <= padding_px)
    out[dilate_mask] = img[inds[0, dilate_mask], inds[1, dilate_mask]]
    return out


def repack_texture(mesh: trimesh.Trimesh, resolution: int, padding_px: int = 4) -> trimesh.Trimesh:
    """Repack UVs into a new layout and resample the existing texture.

    Returns a new mesh with the packed UVs and a resized image.
    If the mesh has no texture or UVs, it is returned unchanged.
    """
    if not hasattr(mesh, "visual") or mesh.visual is None or not hasattr(mesh.visual, "uv") or mesh.visual.uv is None:
        return mesh

    material = getattr(mesh.visual, "material", None)
    if material is None:
        return mesh

    img_tex = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if img_tex is None:
        return mesh

    # Keep an alpha channel if the source has one. Flattening to RGB here would
    # silently turn a texture with transparency opaque, which is worse than the
    # atlas duplication this function exists to fix.
    mode = "RGBA" if img_tex.mode in ("RGBA", "LA", "PA") else "RGB"
    old_img = np.array(img_tex.convert(mode))
    h, w = old_img.shape[:2]
    channels = old_img.shape[2]

    try:
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    except Exception:
        return mesh

    old_uvs = mesh.visual.uv[vmapping]
    px_coords = np.stack([uvs[:, 0] * resolution, (1.0 - uvs[:, 1]) * resolution], axis=1)

    points_map = np.zeros((resolution, resolution, 2), dtype=np.float64)
    covered_mask = np.zeros((resolution, resolution), dtype=bool)

    for face in indices:
        p2d = px_coords[face]
        p3d = old_uvs[face]

        xmin = max(0, int(np.floor(p2d[:, 0].min())))
        xmax = min(resolution - 1, int(np.ceil(p2d[:, 0].max())))
        ymin = max(0, int(np.floor(p2d[:, 1].min())))
        ymax = min(resolution - 1, int(np.ceil(p2d[:, 1].max())))

        if xmax < xmin or ymax < ymin:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float64) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(xs, ys)

        x0, y0 = p2d[0]
        x1, y1 = p2d[1]
        x2, y2 = p2d[2]

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-9:
            continue

        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue

        pts3d = (
            w0[inside, None] * p3d[0]
            + w1[inside, None] * p3d[1]
            + w2[inside, None] * p3d[2]
        )

        iy, ix = np.where(inside)
        iy = iy + ymin
        ix = ix + xmin

        points_map[iy, ix] = pts3d
        covered_mask[iy, ix] = True

    # Mid grey behind the islands, fully transparent where the source had an
    # alpha channel, so uncovered padding does not read as real surface.
    bg_color = np.array([128, 128, 128, 0][:channels], dtype=np.uint8)
    new_img = np.full((resolution, resolution, channels), bg_color, dtype=np.uint8)

    if np.any(covered_mask):
        covered_uvs = points_map[covered_mask]
        old_px_x = covered_uvs[:, 0] * w
        old_px_y = (1.0 - covered_uvs[:, 1]) * h
        new_img[covered_mask] = _sample_bilinear(old_img, old_px_x, old_px_y)

    new_img = _dilate_texture(new_img, covered_mask, padding_px)

    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        process=False,
    )
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    # Keep the material's own type: a SimpleMaterial promoted to PBR would gain
    # metallic/roughness defaults the original never asked for.
    if isinstance(material, PBRMaterial):
        new_mesh.visual.material = PBRMaterial(
            baseColorTexture=Image.fromarray(new_img, mode=mode),
            metallicFactor=material.metallicFactor,
            roughnessFactor=material.roughnessFactor,
        )
    else:
        new_mesh.visual.material = SimpleMaterial(
            image=Image.fromarray(new_img, mode=mode)
        )

    return new_mesh
