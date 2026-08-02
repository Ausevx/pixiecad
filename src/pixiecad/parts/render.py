"""Offscreen renderer producing an RGB image *and* a face-index buffer.

Written by hand in numpy rather than pulled from pyrender/moderngl on purpose.
Every offscreen GL stack needs a display, an EGL build, or a driver, and this
has to run on a headless GPU VM *and* on the dev laptop identically. The mesh
here is 20k faces at 512px, which numpy handles in well under a second per
view, so a GPU rasteriser would buy nothing.

The face-index buffer is the whole point. A 2D segmenter gives back pixel
masks, and a mask is only useful if you can say which faces it covers; keeping
the face id per pixel at render time turns back-projection into an array
lookup instead of a ray cast. Rendering RGB alone would leave us re-deriving
that correspondence later, which is where multi-view part pipelines usually
lose their accuracy.

Projection is orthographic. Parts are identified by which surface a mask
covers, not by how the object recedes, and ortho keeps a face the same size
wherever it sits in frame -- so the area a face contributes to the vote does
not depend on its distance from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

# Enough coverage that no face is visible in fewer than a couple of views,
# without the cost of a dense sphere. Six axis-aligned views catch the flat
# faces; eight corners catch everything the axis views see only edge-on.
DEFAULT_RESOLUTION = 512

BACKGROUND_FACE_ID = -1


@dataclass(frozen=True)
class Camera:
    """A named viewing direction. ``eye`` points from the object to the camera."""

    name: str
    eye: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class Render:
    """One rendered view.

    ``face_id`` holds the index of the face visible at each pixel, or
    ``BACKGROUND_FACE_ID`` where nothing was hit.
    """

    camera: Camera
    rgb: np.ndarray  # (H, W, 3) uint8
    face_id: np.ndarray  # (H, W) int32

    @property
    def visible_faces(self) -> np.ndarray:
        """Unique face indices appearing in this view, background excluded."""
        ids = np.unique(self.face_id)
        return ids[ids >= 0]


def default_cameras() -> list[Camera]:
    """Six axis directions plus eight corners: 14 views covering the sphere.

    Axis views are named for the direction they look *from*, matching the photo
    naming convention used elsewhere in the pipeline (a "front" view sees the
    front of the object).
    """
    cams = [
        Camera("front", (0.0, -1.0, 0.0)),
        Camera("back", (0.0, 1.0, 0.0)),
        Camera("left", (-1.0, 0.0, 0.0)),
        Camera("right", (1.0, 0.0, 0.0)),
        # Looking straight down the up-axis makes the default up vector
        # degenerate, so these two get their own.
        Camera("top", (0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0)),
        Camera("bottom", (0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0)),
    ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                name = (
                    f"{'top' if sz > 0 else 'bottom'}-"
                    f"{'back' if sy > 0 else 'front'}-"
                    f"{'right' if sx > 0 else 'left'}"
                )
                cams.append(Camera(name, (sx, sy, sz)))
    return cams


def canonical_rotation(up_axis: str) -> np.ndarray:
    """Rotation taking ``up_axis`` to +Z, so the cameras above mean what they say.

    glTF is Y-up, and a Hunyuan mesh loaded straight from .glb keeps that
    convention -- our F1 measures 0.70 x 0.44 x 1.99 as (width, height,
    length). Rendered with Z-up cameras it came out lying on its side and the
    "front" view showed the plan, which would have silently mislabelled every
    part downstream.
    """
    axis = up_axis.lower()
    if axis == "z":
        return np.eye(3)
    if axis == "y":
        # Y -> Z, Z -> -Y: a rotation, not a swap, so handedness is preserved
        # and face winding (hence the normals used for shading) still holds.
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    if axis == "x":
        return np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    raise ValueError(f"up_axis must be x, y or z; got {up_axis!r}")


def _basis(eye: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Orthonormal camera basis as rows (right, up, forward)."""
    forward = eye / np.linalg.norm(eye)
    if abs(float(np.dot(forward, up))) > 0.999:
        # Caller handed us a degenerate up vector; any perpendicular will do,
        # since a view down an axis has no natural roll anyway.
        up = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    true_up = np.cross(forward, right)
    return np.stack([right, true_up, forward])


def _rasterize(
    xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, resolution: int
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer rasteriser. Returns (face_id, depth) buffers.

    Loops over triangles but vectorises within each one: a triangle covers a
    handful of pixels, so the per-triangle bounding box keeps the inner work
    tiny and the loop count equal to the face count rather than the pixel
    count.
    """
    face_id = np.full((resolution, resolution), BACKGROUND_FACE_ID, dtype=np.int32)
    zbuf = np.full((resolution, resolution), np.inf, dtype=np.float64)

    tri = xy[faces]  # (F, 3, 2)
    triz = depth[faces]  # (F, 3)

    # Edge function at the third vertex == twice the signed area. Zero means
    # the triangle is edge-on and covers no pixels.
    area2 = (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (
        tri[:, 1, 1] - tri[:, 0, 1]
    ) * (tri[:, 2, 0] - tri[:, 0, 0])

    lo = np.floor(tri.min(axis=1)).astype(np.int64)
    hi = np.ceil(tri.max(axis=1)).astype(np.int64)
    np.clip(lo, 0, resolution - 1, out=lo)
    np.clip(hi, 0, resolution - 1, out=hi)

    # Skip degenerate and fully offscreen triangles up front so the loop body
    # only ever runs on faces that can actually contribute a pixel.
    live = np.flatnonzero((np.abs(area2) > 1e-12) & (hi[:, 0] >= lo[:, 0]) & (hi[:, 1] >= lo[:, 1]))

    for f in live:
        x0, y0 = lo[f]
        x1, y1 = hi[f]
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        px = xs + 0.5
        py = ys + 0.5

        (ax, ay), (bx, by), (cx, cy) = tri[f]
        inv = 1.0 / area2[f]
        # Barycentrics via edge functions; signs follow area2 so winding order
        # does not matter and we never need a separate backface test.
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * triz[f, 0] + w1 * triz[f, 1] + w2 * triz[f, 2]
        window = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        hit = inside & (z < window)
        if not hit.any():
            continue
        window[hit] = z[hit]
        face_id[y0 : y1 + 1, x0 : x1 + 1][hit] = f

    return face_id, zbuf


def _shade(
    face_normals: np.ndarray, face_id: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    """Lambertian headlight shading, so a 2D segmenter can read the form.

    A flat silhouette would give the segmenter nothing to work with -- shading
    is what makes a wheel look like a separate object from the body it is
    fused to.
    """
    rgb = np.zeros((*face_id.shape, 3), dtype=np.uint8)
    hit = face_id >= BACKGROUND_FACE_ID + 1
    if not hit.any():
        return rgb

    normals = face_normals[face_id[hit]]
    # Light sits at the camera, so brightness falls off toward silhouette
    # edges and creases read as tonal boundaries.
    lambert = np.abs(normals @ forward)
    shade = 40 + 215 * np.clip(lambert, 0.0, 1.0)
    rgb[hit] = np.repeat(shade[:, None], 3, axis=1).astype(np.uint8)
    return rgb


def render_mesh(
    mesh: trimesh.Trimesh,
    *,
    cameras: list[Camera] | None = None,
    resolution: int = DEFAULT_RESOLUTION,
    margin: float = 1.05,
    up_axis: str = "y",
) -> list[Render]:
    """Render ``mesh`` from each camera, returning RGB and face-index buffers.

    All views share one scale, derived from the bounding sphere, so a face
    subtends a comparable number of pixels in every view. Per-view autoscaling
    would silently weight whichever view happened to frame the object tightest.
    """
    cameras = cameras or default_cameras()
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("cannot render a mesh with no geometry")

    rot = canonical_rotation(up_axis)
    verts = np.asarray(mesh.vertices, dtype=np.float64) @ rot.T
    # Normals travel with the geometry; a pure rotation lets us reuse them
    # directly instead of recomputing per view.
    normals = np.asarray(mesh.face_normals, dtype=np.float64) @ rot.T

    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    centred = verts - centre
    radius = float(np.linalg.norm(centred, axis=1).max()) or 1.0
    scale = (resolution / 2.0) / (radius * margin)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    out: list[Render] = []
    for cam in cameras:
        eye = np.asarray(cam.eye, dtype=np.float64)
        basis = _basis(eye, np.asarray(cam.up, dtype=np.float64))
        cam_space = centred @ basis.T

        xy = cam_space[:, :2] * scale + resolution / 2.0
        # +forward points at the camera, so nearer surfaces have larger
        # values; negate to make the depth test a plain "less is nearer".
        face_id, _ = _rasterize(xy, -cam_space[:, 2], faces, resolution)
        # Flip vertically: raster rows increase downward, world up increases
        # upward, and an upside-down render confuses a segmenter's priors.
        face_id = np.ascontiguousarray(face_id[::-1])
        out.append(Render(camera=cam, rgb=_shade(normals, face_id, basis[2]), face_id=face_id))
    return out
