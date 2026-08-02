"""S0.5 photo triage and part naming.

Both functions do something useful with no API key at all:

- ``triage_photos`` computes coverage structurally (are the photos visually
  distinct enough to imply an orbit?) and only asks a model for viewpoint
  labels and per-photo problems when one is configured.
- ``name_parts`` falls back to geometric names ("largest-part", "front-left")
  derived from position and size when no model is available.

That keeps the pipeline's behaviour identical with or without a key — the model
adds vocabulary, never capability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .base import ProviderUnavailable, VisionError, get_provider

VIEWPOINTS = [
    "front", "front-left", "left", "rear-left",
    "rear", "rear-right", "right", "front-right",
    "top", "bottom", "detail", "unknown",
]


@dataclass
class PhotoNote:
    image: str
    viewpoint: str = "unknown"
    issues: list[str] = field(default_factory=list)


@dataclass
class TriageReport:
    n_photos: int
    object_guess: str | None
    photos: list[PhotoNote]
    covered: list[str]
    missing: list[str]
    advice: list[str]
    provider: str | None
    distinctness: float

    def summary(self) -> str:
        who = self.object_guess or "object"
        return (
            f"{self.n_photos} photos of {who}; "
            f"{len(self.covered)} viewpoints covered, {len(self.missing)} missing"
        )


def _distinctness(images: list[Path]) -> float:
    """Mean pairwise difference of tiny greyscale thumbnails, 0..1.

    A real orbit produces visibly different frames; near-duplicates or a
    tripod-locked burst produce a low score. Cheap, local, and a decent proxy
    for "is there parallax here at all".
    """
    import cv2

    thumbs = []
    for p in images:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            thumbs.append(cv2.resize(img, (32, 32)).astype(np.float32) / 255.0)
    if len(thumbs) < 2:
        return 0.0
    arr = np.stack(thumbs)
    diffs = [
        float(np.abs(arr[i] - arr[j]).mean())
        for i in range(len(arr))
        for j in range(i + 1, len(arr))
    ]
    return round(float(np.mean(diffs)), 4)


_TRIAGE_PROMPT = """You are checking photos intended for 3D reconstruction.

Return ONLY JSON, no prose:
{{"object": "<short name of the subject>",
  "photos": [{{"index": <0-based>, "viewpoint": "<one of: {viewpoints}>",
               "issues": ["<short problem>", ...]}}],
  "missing_viewpoints": ["<viewpoint>", ...],
  "advice": ["<one actionable sentence>", ...]}}

Issues worth reporting: subject partially cut off, something occluding it, a
different object than the other photos, heavy motion blur, strong reflections,
or the subject being too small in frame. Empty list if the photo is fine.
There are {n} photos, in order."""


def triage_photos(
    images: list[Path],
    *,
    provider: str | None = None,
    max_images: int = 12,
) -> TriageReport:
    """Structural coverage analysis, enriched with model labels when available."""
    images = [Path(p) for p in images]
    if not images:
        raise ValueError("no images to triage")

    dist = _distinctness(images)
    notes = [PhotoNote(image=p.name) for p in images]
    advice: list[str] = []
    covered: list[str] = []
    missing: list[str] = []
    obj: str | None = None
    used: str | None = None

    # Local structural verdicts — always produced.
    if len(images) < 16:
        advice.append(
            f"{len(images)} photos: below the ~16 needed to triangulate. "
            "Either shoot a full orbit or use a generative backend."
        )
    if dist and dist < 0.04:
        advice.append(
            "Photos look very similar to each other — if the camera barely moved "
            "there is no parallax to reconstruct from."
        )

    try:
        p = get_provider(provider)
        sample = images[:max_images]
        reply = p.ask(
            _TRIAGE_PROMPT.format(viewpoints=", ".join(VIEWPOINTS), n=len(sample)),
            images=sample,
            max_tokens=2048,
        )
        data = reply.as_json()
        used = p.name
        obj = data.get("object")
        for entry in data.get("photos", []):
            i = entry.get("index")
            if isinstance(i, int) and 0 <= i < len(notes):
                vp = entry.get("viewpoint", "unknown")
                notes[i].viewpoint = vp if vp in VIEWPOINTS else "unknown"
                notes[i].issues = [str(x) for x in entry.get("issues", [])][:4]
        covered = sorted({n.viewpoint for n in notes} - {"unknown"})
        missing = [str(v) for v in data.get("missing_viewpoints", [])]
        advice.extend(str(a) for a in data.get("advice", [])[:4])
    except (ProviderUnavailable, VisionError) as e:
        advice.append(f"viewpoint labelling skipped ({e.__class__.__name__}); geometry is unaffected")

    return TriageReport(
        n_photos=len(images),
        object_guess=obj,
        photos=notes,
        covered=covered,
        missing=missing,
        advice=advice,
        provider=used,
        distinctness=dist,
    )


def _geometric_name(desc: dict[str, Any], rank: int) -> str:
    """Fallback name from position and size, used when no model is available."""
    if rank == 0:
        return "body"
    pos = desc["position"]
    lr = "left" if pos["x_left_to_right"] < 0.4 else "right" if pos["x_left_to_right"] > 0.6 else "centre"
    fb = "front" if pos["y_back_to_front"] > 0.6 else "rear" if pos["y_back_to_front"] < 0.4 else "mid"
    tb = "upper" if pos["z_bottom_to_top"] > 0.66 else "lower" if pos["z_bottom_to_top"] < 0.33 else ""
    return "-".join(x for x in (tb, fb, lr) if x) or f"part-{desc['index']:02d}"


_NAME_PROMPT = """These are components of a single 3D model of: {obj}.

Each is described by its position within the object's bounding box (0..1 along
each axis; x=left→right, y=rear→front, z=bottom→top), its share of the volume,
its extents in metres, elongation (longest/shortest) and flatness.

{parts}

Return ONLY JSON: {{"names": [{{"index": <int>, "name": "<short lowercase name>"}}]}}
Use conventional component names for this kind of object. If two parts are
mirrored, distinguish them (e.g. "front-left wheel" / "front-right wheel")."""


def name_parts(
    parts: list[Any],
    whole,
    *,
    object_hint: str | None = None,
    provider: str | None = None,
) -> list[Any]:
    """Fill in ``part.name`` for each part; mutates and returns the list.

    Uses a model when one is configured, geometry otherwise. Never raises for
    lack of a provider — an unnamed part is still a usable part.
    """
    descs = [p.describe(whole) for p in parts]

    try:
        prov = get_provider(provider)
        reply = prov.ask(
            _NAME_PROMPT.format(
                obj=object_hint or "an unidentified object",
                parts=json.dumps(descs, indent=1),
            ),
            max_tokens=1024,
        )
        names = {
            e["index"]: str(e["name"]).strip().lower()
            for e in reply.as_json().get("names", [])
            if isinstance(e.get("index"), int) and e.get("name")
        }
        for rank, part in enumerate(parts):
            part.name = names.get(part.index) or _geometric_name(descs[rank], rank)
            part.metadata["named_by"] = prov.name if part.index in names else "geometry"
        return parts
    except (ProviderUnavailable, VisionError):
        for rank, part in enumerate(parts):
            part.name = _geometric_name(descs[rank], rank)
            part.metadata["named_by"] = "geometry"
        return parts
