"""Stage 0: ingest & input optimization.

Reads a directory of photos and produces, inside the workspace stage dir:
  - ``images/``   auto-rotated, downscaled working copies of accepted photos
  - ``report.json``  per-photo verdicts (accepted / duplicate / rejected + why)

The goal is to hand later stages the smallest photo set that still
reconstructs well, and to tell the user *exactly* what to reshoot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..spec import ObjectSpec
from ..workspace import Workspace, fingerprint_file
from .dedup import DEFAULT_THRESHOLD, dhash, mark_duplicates
from .quality import assess_quality

PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}
DEFAULT_WORKING_LONG_EDGE = 1600


@dataclass
class PhotoRecord:
    source: str
    status: str  # accepted | duplicate | rejected | unreadable
    working_path: str | None = None
    blur_score: float | None = None
    long_edge: int | None = None
    duplicate_of: str | None = None
    warnings: list[str] = field(default_factory=list)
    reject_reasons: list[str] = field(default_factory=list)


@dataclass
class IngestReport:
    accepted: int
    duplicates: int
    rejected: int
    unreadable: int
    photos: list[PhotoRecord]
    advice: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _load_bgr(path: Path) -> np.ndarray | None:
    """Load with EXIF auto-rotation applied; None if unreadable."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _downscale(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _advice(records: list[PhotoRecord]) -> list[str]:
    advice: list[str] = []
    accepted = [r for r in records if r.status == "accepted"]
    n = len(accepted)
    if n < 12:
        advice.append(
            f"Only {n} usable photos. Aim for 20-40 covering a full orbit "
            "(every ~15°), plus a higher and a lower ring."
        )
    blurry = [r.source for r in records if r.status == "rejected" and any("blurry" in x for x in r.reject_reasons)]
    if blurry:
        advice.append(f"Retake sharper versions of: {', '.join(Path(p).name for p in blurry[:5])}")
    dupes = sum(1 for r in records if r.status == "duplicate")
    if dupes > max(3, len(records) // 4):
        advice.append(
            f"{dupes} near-duplicate shots dropped — move more between frames; "
            "parallax, not repetition, is what reconstruction needs."
        )
    if not advice:
        advice.append("Input set looks healthy.")
    return advice


def run_ingest(
    photos_dir: Path,
    workspace: Workspace,
    working_long_edge: int = DEFAULT_WORKING_LONG_EDGE,
    dedup_threshold: int = DEFAULT_THRESHOLD,
) -> IngestReport:
    sources = sorted(
        p for p in Path(photos_dir).iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_EXTS
    )
    if not sources:
        raise FileNotFoundError(f"No photos found in {photos_dir}")

    params = {"long_edge": working_long_edge, "dedup": dedup_threshold}
    run = workspace.begin_stage("s0-ingest", params, [fingerprint_file(p) for p in sources])
    report_path = run.dir / "report.json"
    if run.cached:
        data = json.loads(report_path.read_text())
        return IngestReport(**{**data, "photos": [PhotoRecord(**p) for p in data["photos"]]})

    images_dir = run.dir / "images"
    images_dir.mkdir()

    records: list[PhotoRecord] = []
    loaded: list[tuple[int, np.ndarray]] = []  # (record index, image)
    for src in sources:
        img = _load_bgr(src)
        if img is None:
            records.append(PhotoRecord(source=str(src), status="unreadable"))
            continue
        q = assess_quality(img)
        rec = PhotoRecord(
            source=str(src),
            status="accepted" if q.ok else "rejected",
            blur_score=round(q.blur_score, 1),
            long_edge=q.long_edge,
            warnings=q.warnings,
            reject_reasons=q.reject_reasons,
        )
        records.append(rec)
        if q.ok:
            loaded.append((len(records) - 1, img))

    # Near-duplicate pass over the survivors only.
    dup_of = mark_duplicates([dhash(img) for _, img in loaded], dedup_threshold)
    for (rec_idx, img), match in zip(loaded, dup_of):
        rec = records[rec_idx]
        if match is not None:
            rec.status = "duplicate"
            rec.duplicate_of = records[loaded[match][0]].source
            continue
        out = images_dir / f"{Path(rec.source).stem}.jpg"
        cv2.imwrite(str(out), _downscale(img, working_long_edge), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec.working_path = str(out)

    report = IngestReport(
        accepted=sum(r.status == "accepted" for r in records),
        duplicates=sum(r.status == "duplicate" for r in records),
        rejected=sum(r.status == "rejected" for r in records),
        unreadable=sum(r.status == "unreadable" for r in records),
        photos=records,
        advice=_advice(records),
    )
    report_path.write_text(report.to_json() + "\n")
    workspace.finish_stage(
        run,
        {"accepted": report.accepted, "duplicates": report.duplicates,
         "rejected": report.rejected, "unreadable": report.unreadable},
    )
    return report
