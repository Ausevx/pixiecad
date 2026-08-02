"""S6b UV unwrap + object-space normal-map bake."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import trimesh
import xatlas


@dataclass
class UnwrapResult:
    mesh: trimesh.Trimesh
    n_charts_hint: int
    orig_face_count: int


def unwrap_uv(mesh: trimesh.Trimesh) -> UnwrapResult:
    """Unwrap low-poly mesh UVs using xatlas.

    Preserves vertex position mapping and attaches TextureVisuals.
    """
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices[vmapping],
        faces=indices,
        process=False,
    )
    new_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)
    return UnwrapResult(
        mesh=new_mesh,
        n_charts_hint=len(uvs),
        orig_face_count=len(mesh.faces),
    )


def _dilate_mask_aware(img: np.ndarray, mask: np.ndarray, padding_px: int) -> np.ndarray:
    """Dilate covered pixels outward by padding_px to prevent chart seam bleeding."""
    if padding_px <= 0 or not np.any(mask):
        return img
    out = img.copy()
    curr_mask = mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(padding_px):
        dilated_mask = cv2.dilate(curr_mask.astype(np.uint8), kernel) > 0
        new_pixels = dilated_mask & ~curr_mask
        if not np.any(new_pixels):
            break
        valid_float = out.astype(np.float32)
        valid_float[~curr_mask] = 0.0

        sum_r = cv2.filter2D(valid_float[:, :, 0], -1, kernel)
        sum_g = cv2.filter2D(valid_float[:, :, 1], -1, kernel)
        sum_b = cv2.filter2D(valid_float[:, :, 2], -1, kernel)
        cnt = cv2.filter2D(curr_mask.astype(np.float32), -1, kernel)

        cnt_nonzero = np.maximum(cnt, 1e-5)
        out[new_pixels, 0] = np.clip(sum_r[new_pixels] / cnt_nonzero[new_pixels], 0, 255).astype(np.uint8)
        out[new_pixels, 1] = np.clip(sum_g[new_pixels] / cnt_nonzero[new_pixels], 0, 255).astype(np.uint8)
        out[new_pixels, 2] = np.clip(sum_b[new_pixels] / cnt_nonzero[new_pixels], 0, 255).astype(np.uint8)

        curr_mask = dilated_mask
    return out


def bake_object_space_normals(
    dense: trimesh.Trimesh,
    low_unwrapped: trimesh.Trimesh,
    *,
    resolution: int = 1024,
    padding_px: int = 4,
) -> np.ndarray:
    """Bake an object-space normal map sampling the dense mesh at low-poly UV texels."""
    if low_unwrapped.visual is None or not hasattr(low_unwrapped.visual, "uv") or low_unwrapped.visual.uv is None:
        raise ValueError("low_unwrapped mesh must have UV coordinates in visual.uv")

    uv_coords = low_unwrapped.visual.uv
    px_coords = np.stack([uv_coords[:, 0] * resolution, (1.0 - uv_coords[:, 1]) * resolution], axis=1)

    points_map = np.zeros((resolution, resolution, 3), dtype=np.float64)
    covered_mask = np.zeros((resolution, resolution), dtype=bool)

    for face in low_unwrapped.faces:
        p2d = px_coords[face]
        p3d = low_unwrapped.vertices[face]

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

    bg_color = np.array([128, 128, 255], dtype=np.uint8)
    img = np.full((resolution, resolution, 3), bg_color, dtype=np.uint8)

    if np.any(covered_mask):
        covered_pts = points_map[covered_mask]

        # Query dense mesh in one single batched call
        query = trimesh.proximity.ProximityQuery(dense)
        _, _, face_ids = query.on_surface(covered_pts)

        normals = dense.face_normals[face_ids]
        rgb_normals = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        img[covered_mask] = rgb_normals

        if padding_px > 0:
            img = _dilate_mask_aware(img, covered_mask, padding_px)

    return img


def save_normal_map(img: np.ndarray, path: Path) -> None:
    """Save normal map image array as RGB PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="RGB").save(path)
