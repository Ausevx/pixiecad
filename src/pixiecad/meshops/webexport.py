"""Shrink a textured model for delivery over the web.

The problem this solves is not polygon count. A 20k-triangle car is small by
web standards -- a scroll-driven hero model can afford several times that. The
payload is dominated by texture: our F1's geometry is roughly 1 MB while its
2048x2048 base-colour, metallic and roughness maps are ~4.6 MB, and gzip does
essentially nothing because JPEG is already compressed.

It is far worse for parts. Every part embeds its own full copy of the atlas,
so a 273-face wing element is 6.8 MB -- the same as the 9,170-face body -- and
nine parts total 62 MB for ~20k triangles. Cropping each part's texture to its
UV bounding box does not help, because the UVs are scattered across the whole
atlas (bbox ~1.0) even though real occupancy is only 3-17%.

So this module trades texture resolution, which is the term that actually
dominates, and leaves geometry alone. Repacking each part's UVs into a tight
atlas and rebaking would beat this, but it needs a re-unwrap plus a bake pass;
resolution is the cheap 90%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

# 1024 is the default because it is the point where the F1 stops looking
# resampled at typical hero-model screen sizes while cutting the payload ~4x.
DEFAULT_TEXTURE_SIZE = 1024
DEFAULT_QUALITY = 85


@dataclass
class WebExportReport:
    path: Path
    bytes_before: int
    bytes_after: int
    texture_before: tuple[int, int] | None
    texture_after: tuple[int, int] | None
    n_faces: int

    @property
    def ratio(self) -> float:
        return self.bytes_after / self.bytes_before if self.bytes_before else 1.0


def _texture_of(mesh: trimesh.Trimesh):
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None:
        return None
    # PBR materials expose baseColorTexture; the simple ones expose image.
    return getattr(material, "baseColorTexture", None) or getattr(material, "image", None)


def downscale_texture(
    mesh: trimesh.Trimesh,
    max_size: int = DEFAULT_TEXTURE_SIZE,
    *,
    quality: int = DEFAULT_QUALITY,
) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` whose texture is at most ``max_size`` square.

    UVs are normalised coordinates, so resizing the image needs no change to
    the mesh -- which is exactly why this is safe to apply after every other
    stage, including segmentation.
    """
    if max_size < 1:
        raise ValueError(f"max_size must be >= 1, got {max_size}")

    out = mesh.copy()
    image = _texture_of(out)
    if image is None or max(image.size) <= max_size:
        return out

    from PIL import Image

    scale = max_size / max(image.size)
    size = (max(1, int(image.size[0] * scale)), max(1, int(image.size[1] * scale)))
    resized = image.convert("RGB").resize(size, Image.LANCZOS)

    # Round-trip through JPEG so the size saving is real in the exported file,
    # not just in memory: trimesh re-encodes PNG otherwise, which for a
    # photographic livery is several times larger for no visible gain.
    import io

    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compact = Image.open(buffer)
    compact.load()

    material = out.visual.material
    if hasattr(material, "baseColorTexture"):
        material.baseColorTexture = compact
        # Metallic/roughness maps carry far less perceptible detail than base
        # colour; dropping them entirely is usually the right call for a web
        # hero model, but that is the caller's decision, so only shrink them.
        for attr in ("metallicRoughnessTexture", "normalTexture", "emissiveTexture"):
            extra = getattr(material, attr, None)
            if extra is not None:
                setattr(material, attr, extra.convert("RGB").resize(size, Image.LANCZOS))
    else:
        material.image = compact
    return out


def export_for_web(
    mesh: trimesh.Trimesh,
    path: str | Path,
    *,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    quality: int = DEFAULT_QUALITY,
) -> WebExportReport:
    """Write a web-sized GLB and report what it cost."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    before_image = _texture_of(mesh)
    before_size = tuple(before_image.size) if before_image else None
    raw = mesh.export(file_type="glb")

    shrunk = downscale_texture(mesh, texture_size, quality=quality)
    after_image = _texture_of(shrunk)
    path.write_bytes(shrunk.export(file_type="glb"))

    return WebExportReport(
        path=path,
        bytes_before=len(raw),
        bytes_after=path.stat().st_size,
        texture_before=before_size,
        texture_after=tuple(after_image.size) if after_image else None,
        n_faces=int(len(mesh.faces)),
    )


def part_texture_size(
    part_faces: int, total_faces: int, *, base: int = DEFAULT_TEXTURE_SIZE, floor: int = 128
) -> int:
    """Texture size for one part, scaled by its share of the model.

    A 273-face wing element does not need the same atlas as a 9,170-face body.
    Size follows the square root of the face share, because texture memory
    grows with the square of resolution while surface area grows linearly, and
    is snapped to a power of two.
    """
    if total_faces <= 0 or part_faces <= 0:
        return floor
    share = min(1.0, part_faces / total_faces)
    target = base * (share**0.5)
    power = int(2 ** round(np.log2(max(target, floor))))
    return int(min(base, max(floor, power)))
