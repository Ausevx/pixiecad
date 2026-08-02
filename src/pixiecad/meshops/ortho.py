"""Orthographic projection drawing export for 3D meshes.

Renders CAD-style orthographic projections (front / side / top) as SVG
line drawings with optional dimension annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh

from pixiecad.spec import Dimensions


@dataclass
class OrthoView:
    name: str
    svg: str
    width_units: float
    height_units: float


def project(mesh: trimesh.Trimesh, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D mesh vertices onto 2D plane for the given view axis.

    Args:
        mesh: Input trimesh object.
        axis: View direction ("front", "side", or "top").

    Returns:
        Tuple of (points2d (V, 2), faces (F, 3)).
    """
    if axis == "front":
        # Look down -Y: keep (X, Z)
        points2d = mesh.vertices[:, [0, 2]].copy()
    elif axis == "side":
        # Look down -X: keep (Y, Z)
        points2d = mesh.vertices[:, [1, 2]].copy()
    elif axis == "top":
        # Look down -Z: keep (X, Y)
        points2d = mesh.vertices[:, [0, 1]].copy()
    else:
        raise ValueError(f"Invalid view axis {axis!r}; must be 'front', 'side', or 'top'")

    faces = mesh.faces.copy()
    return points2d, faces


def feature_edges(mesh: trimesh.Trimesh, angle_deg: float = 25.0) -> np.ndarray:
    """Find feature edges worth drawing (dihedral angle > angle_deg or boundary edges).

    Args:
        mesh: Input trimesh object.
        angle_deg: Minimum dihedral angle in degrees to qualify as a feature edge.

    Returns:
        (E, 2) array of vertex-index pairs (sorted, unique).
    """
    threshold_rad = np.radians(angle_deg)
    adj_edges = np.empty((0, 2), dtype=int)
    if len(mesh.face_adjacency_angles) > 0:
        mask = mesh.face_adjacency_angles > threshold_rad
        adj_edges = mesh.face_adjacency_edges[mask]

    boundary_edges = np.empty((0, 2), dtype=int)
    if len(mesh.edges_sorted) > 0:
        unique_edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]

    if len(adj_edges) > 0 or len(boundary_edges) > 0:
        all_edges = np.vstack([adj_edges, boundary_edges])
        all_edges = np.sort(all_edges, axis=1)
        all_edges = np.unique(all_edges, axis=0)
    else:
        all_edges = np.empty((0, 2), dtype=int)

    return all_edges


def silhouette_edges(mesh: trimesh.Trimesh, axis: str) -> np.ndarray:
    """Find silhouette edges where one adjacent face points toward viewer and the other away.

    Args:
        mesh: Input trimesh object.
        axis: View direction ("front", "side", or "top").

    Returns:
        (E, 2) array of vertex-index pairs (sorted, unique).
    """
    if axis == "front":
        axis_idx = 1
    elif axis == "side":
        axis_idx = 0
    elif axis == "top":
        axis_idx = 2
    else:
        raise ValueError(f"Invalid view axis {axis!r}; must be 'front', 'side', or 'top'")

    if len(mesh.face_adjacency) == 0 or len(mesh.face_normals) == 0:
        return np.empty((0, 2), dtype=int)

    comp = mesh.face_normals[:, axis_idx]
    adj = mesh.face_adjacency
    c0 = comp[adj[:, 0]]
    c1 = comp[adj[:, 1]]

    # Sign change along view axis component: one face > 0 and the other not > 0
    mask = (c0 > 0) != (c1 > 0)
    sil_edges = mesh.face_adjacency_edges[mask]

    if len(sil_edges) > 0:
        sil_edges = np.sort(sil_edges, axis=1)
        sil_edges = np.unique(sil_edges, axis=0)
    else:
        sil_edges = np.empty((0, 2), dtype=int)

    return sil_edges


def render_view(
    mesh: trimesh.Trimesh,
    axis: str,
    *,
    size_px: int = 800,
    margin_px: int = 60,
    dimensions: Dimensions | None = None,
    stroke: str = "#111",
) -> OrthoView:
    """Render a single orthographic projection view as SVG.

    Args:
        mesh: Input trimesh object.
        axis: View direction ("front", "side", or "top").
        size_px: Size of SVG canvas (square).
        margin_px: Margin in pixels around geometry.
        dimensions: Optional real-world dimensions for annotations.
        stroke: Stroke color for drawing lines.

    Returns:
        OrthoView object containing the SVG string and extents.
    """
    points2d, _ = project(mesh, axis)

    if len(points2d) > 0:
        min_x, max_x = float(np.min(points2d[:, 0])), float(np.max(points2d[:, 0]))
        min_y, max_y = float(np.min(points2d[:, 1])), float(np.max(points2d[:, 1]))
    else:
        min_x, max_x = 0.0, 0.0
        min_y, max_y = 0.0, 0.0

    width_units = max_x - min_x
    height_units = max_y - min_y

    # Union of feature and silhouette edges
    fe = feature_edges(mesh)
    se = silhouette_edges(mesh, axis)

    if len(fe) > 0 and len(se) > 0:
        edges = np.vstack([fe, se])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
    elif len(fe) > 0:
        edges = fe
    else:
        edges = se

    avail_w = size_px - 2 * margin_px
    avail_h = size_px - 2 * margin_px

    if width_units > 0 and height_units > 0:
        scale_x = avail_w / width_units
        scale_y = avail_h / height_units
        scale = min(scale_x, scale_y)
    elif width_units > 0:
        scale = avail_w / width_units
    elif height_units > 0:
        scale = avail_h / height_units
    else:
        scale = 1.0

    dw = width_units * scale
    dh = height_units * scale

    offset_x = margin_px + (avail_w - dw) / 2.0
    offset_y = margin_px + (avail_h - dh) / 2.0

    lines_svg: list[str] = []
    for edge in edges:
        i, j = edge[0], edge[1]
        p1 = points2d[i]
        p2 = points2d[j]

        x1 = offset_x + (p1[0] - min_x) * scale
        y1 = offset_y + (max_y - p1[1]) * scale
        x2 = offset_x + (p2[0] - min_x) * scale
        y2 = offset_y + (max_y - p2[1]) * scale

        lines_svg.append(
            f'    <line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="1" stroke-linecap="round" />'
        )

    lines_content = "\n".join(lines_svg)
    group_content = f"  <g>\n{lines_content}\n  </g>" if lines_svg else "  <g>\n  </g>"

    # Dimension annotations
    annotations_svg: list[str] = []
    if dimensions is not None and len(mesh.vertices) > 0:
        mesh_x = float(np.ptp(mesh.vertices[:, 0]))
        mesh_y = float(np.ptp(mesh.vertices[:, 1]))
        mesh_z = float(np.ptp(mesh.vertices[:, 2]))

        scale_factor = None
        if dimensions.length_m is not None and mesh_x > 0:
            scale_factor = dimensions.length_m / mesh_x
        elif dimensions.width_m is not None and mesh_y > 0:
            scale_factor = dimensions.width_m / mesh_y
        elif dimensions.height_m is not None and mesh_z > 0:
            scale_factor = dimensions.height_m / mesh_z

        real_w = None
        real_h = None

        if axis in ("front", "top"):
            if dimensions.length_m is not None:
                real_w = dimensions.length_m
            elif scale_factor is not None:
                real_w = width_units * scale_factor

        if axis == "side":
            if dimensions.width_m is not None:
                real_w = dimensions.width_m
            elif scale_factor is not None:
                real_w = width_units * scale_factor

        if axis in ("front", "side"):
            if dimensions.height_m is not None:
                real_h = dimensions.height_m
            elif scale_factor is not None:
                real_h = height_units * scale_factor

        if axis == "top":
            if dimensions.width_m is not None:
                real_h = dimensions.width_m
            elif scale_factor is not None:
                real_h = height_units * scale_factor

        left_px = offset_x
        right_px = offset_x + dw
        top_px = offset_y
        bottom_px = offset_y + dh

        if real_w is not None and width_units > 0:
            dim_y = bottom_px + 20
            annotations_svg.extend([
                f'  <line x1="{left_px:.2f}" y1="{dim_y:.2f}" x2="{right_px:.2f}" y2="{dim_y:.2f}" stroke="{stroke}" stroke-width="1" />',
                f'  <line x1="{left_px:.2f}" y1="{dim_y - 4:.2f}" x2="{left_px:.2f}" y2="{dim_y + 4:.2f}" stroke="{stroke}" stroke-width="1" />',
                f'  <line x1="{right_px:.2f}" y1="{dim_y - 4:.2f}" x2="{right_px:.2f}" y2="{dim_y + 4:.2f}" stroke="{stroke}" stroke-width="1" />',
                f'  <text x="{(left_px + right_px) / 2.0:.2f}" y="{dim_y + 16:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{stroke}">{real_w:.2f} m</text>'
            ])

        if real_h is not None and height_units > 0:
            dim_x = left_px - 20
            annotations_svg.extend([
                f'  <line x1="{dim_x:.2f}" y1="{top_px:.2f}" x2="{dim_x:.2f}" y2="{bottom_px:.2f}" stroke="{stroke}" stroke-width="1" />',
                f'  <line x1="{dim_x - 4:.2f}" y1="{top_px:.2f}" x2="{dim_x + 4:.2f}" y2="{top_px:.2f}" stroke="{stroke}" stroke-width="1" />',
                f'  <line x1="{dim_x - 4:.2f}" y1="{bottom_px:.2f}" x2="{dim_x + 4:.2f}" y2="{bottom_px:.2f}" stroke="{stroke}" stroke-width="1" />',
                f'  <text x="{dim_x - 8:.2f}" y="{(top_px + bottom_px) / 2.0:.2f}" text-anchor="end" dominant-baseline="middle" font-family="sans-serif" font-size="12" fill="{stroke}">{real_h:.2f} m</text>'
            ])

    ann_content = "\n" + "\n".join(annotations_svg) if annotations_svg else ""

    svg_str = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}" '
        f'viewBox="0 0 {size_px} {size_px}">\n'
        f'{group_content}'
        f'{ann_content}\n'
        f'</svg>'
    )

    return OrthoView(
        name=axis,
        svg=svg_str,
        width_units=width_units,
        height_units=height_units,
    )


def render_all(
    mesh: trimesh.Trimesh,
    *,
    dimensions: Dimensions | None = None,
    **kw,
) -> dict[str, OrthoView]:
    """Render all three standard orthographic views (front, side, top).

    Args:
        mesh: Input trimesh object.
        dimensions: Optional real-world dimensions.
        **kw: Additional arguments passed to render_view.

    Returns:
        Dict mapping view name ("front", "side", "top") to OrthoView.
    """
    return {
        view_name: render_view(mesh, view_name, dimensions=dimensions, **kw)
        for view_name in ("front", "side", "top")
    }


def save_views(views: dict[str, OrthoView], out_dir: Path) -> list[Path]:
    """Save rendered views to SVG files in out_dir.

    Args:
        views: Dictionary mapping view names to OrthoView.
        out_dir: Directory path where SVG files will be saved.

    Returns:
        List of Path objects for the created SVG files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for name, view in views.items():
        file_path = out_dir / f"{name}.svg"
        file_path.write_text(view.svg, encoding="utf-8")
        saved_paths.append(file_path)
    return saved_paths
