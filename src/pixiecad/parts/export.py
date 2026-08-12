"""Export segmented parts as separate files plus a manifest.

Each part gets its own share of the polygon budget (proportional to its current
face count, via ``meshops.budget_for_parts``), is decimated to exactly that,
and is written as its own ``.glb``. The manifest records what each file is and
how it was derived, so a downstream tool — or a person — can tell an invented
part from a measured one.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..export import export_glb
from ..meshops import budget_for_parts, decimate_to_budget
from .segment import Part

_SAFE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, fallback: str = "part") -> str:
    slug = _SAFE.sub("-", name.strip().lower()).strip("-")
    return slug or fallback


@dataclass
class ExportedPart:
    index: int
    name: str
    file: str
    faces: int
    target_faces: int
    volume_fraction: float
    # All three are world space, in the coordinates of the model the parts were
    # split from, so a consumer can reassemble the parts by placing them as
    # exported. centroid and extents describe the part as segmented; bbox is
    # measured on the decimated mesh actually written to the .glb, so it matches
    # the geometry in the file rather than the geometry it was derived from.
    centroid: list[float]
    extents: list[float]
    bbox: list[list[float]]


def export_parts(
    parts: list[Part],
    out_dir: Path,
    *,
    total_budget: int,
    whole_volume: float | None = None,
    extras: dict[str, Any] | None = None,
) -> tuple[list[ExportedPart], Path]:
    """Write one .glb per part plus ``manifest.json``; returns both.

    ``total_budget`` is split across parts by face count, never below 32 faces
    each, summing to exactly the budget.
    """
    if not parts:
        raise ValueError("no parts to export")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = [f"{p.index}" for p in parts]
    budgets = budget_for_parts(total_budget, {k: p.n_faces for k, p in zip(keys, parts)})

    total_volume = whole_volume if whole_volume else sum(p.volume for p in parts) or 1.0

    exported: list[ExportedPart] = []
    seen: set[str] = set()
    for key, part in zip(keys, parts):
        target = budgets[key]
        mesh, _ = decimate_to_budget(part.mesh, target)

        base = slugify(part.name or f"part-{part.index:02d}")
        stem = base
        n = 2
        while stem in seen:  # two "wheel" parts must not overwrite each other
            stem = f"{base}-{n}"
            n += 1
        seen.add(stem)

        path = out_dir / f"{stem}.glb"
        export_glb(
            mesh,
            path,
            node_name=stem,
            extras={
                "part_index": part.index,
                "part_name": part.name,
                "segmentation_method": part.method,
                **(extras or {}),
            },
        )
        exported.append(
            ExportedPart(
                index=part.index,
                name=part.name or stem,
                file=path.name,
                faces=len(mesh.faces),
                target_faces=target,
                volume_fraction=round(part.volume / total_volume, 4),
                centroid=[float(v) for v in part.centroid],
                extents=[float(v) for v in part.extents],
                bbox=[[float(v) for v in pt] for pt in mesh.bounds],
            )
        )

    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "total_budget": total_budget,
                "n_parts": len(exported),
                "segmentation_method": parts[0].method,
                "named": all(p.is_named for p in parts),
                "parts": [asdict(e) for e in exported],
                **(extras or {}),
            },
            indent=2,
        )
        + "\n"
    )
    return exported, manifest
