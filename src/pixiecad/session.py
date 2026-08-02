"""Per-run session directories: one folder per run, never shared.

Every run — headless CLI or dashboard upload — gets its own directory holding
three things that must not be mixed between runs:

    <root>/<timestamp>-<slug>/
        input/    the photos this run was given (copied, so the run is
                  reproducible even if the source folder later changes)
        work/     the workspace: stage cache, intermediates, scratch
        output/   the deliverables: model.glb, parts/, drawings/, build.json

Timestamp-first naming means ``ls`` sorts chronologically, and a ``latest``
symlink points at the newest session so scripts have a stable path.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 32) -> str:
    """Lowercase, hyphen-separated, filesystem-safe fragment of ``text``."""
    slug = _SLUG_STRIP.sub("-", text.lower()).strip("-")
    return slug[:max_len].strip("-")


@dataclass
class Session:
    root: Path
    input_dir: Path
    work_dir: Path
    output_dir: Path
    session_id: str
    label: str
    created_at: str

    def write_meta(self, **extra: object) -> Path:
        """Persist session metadata to ``session.json``, merging ``extra``."""
        meta = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}
        meta.update(extra)
        path = self.root / "session.json"
        path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
        return path

    def stage_inputs(self, photos_dir: Path | str) -> list[Path]:
        """Copy every image in ``photos_dir`` into ``input/``; return the copies.

        Copying rather than referencing is deliberate: a session should still
        be re-runnable after the user reorganises or deletes their source
        folder, and the copies are what the manifest's fingerprints describe.
        """
        photos_dir = Path(photos_dir)
        sources = sorted(
            p
            for p in photos_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not sources:
            raise ValueError(f"no images found in {photos_dir}")

        copied = []
        for src in sources:
            dst = self.input_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            copied.append(dst)
        return copied


def new_session(
    root: Path | str,
    *,
    label: str | None = None,
    source: Path | str | None = None,
    now: datetime | None = None,
) -> Session:
    """Create a fresh session directory under ``root``.

    The name is ``<YYYYmmdd-HHMMSS>-<slug>``, where the slug comes from
    ``label`` or, failing that, the name of the ``source`` photo directory.
    Two runs started in the same second get ``-2``, ``-3``, … rather than
    colliding — the dashboard can accept simultaneous uploads.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    hint = label or (Path(source).name if source else "")
    slug = slugify(hint) or "run"

    base = f"{stamp}-{slug}"
    session_dir = root / base
    n = 2
    while session_dir.exists():
        session_dir = root / f"{base}-{n}"
        n += 1

    session = Session(
        root=session_dir,
        input_dir=session_dir / "input",
        work_dir=session_dir / "work",
        output_dir=session_dir / "output",
        session_id=session_dir.name,
        label=hint or slug,
        created_at=(now or datetime.now()).isoformat(timespec="seconds"),
    )
    for d in (session.input_dir, session.work_dir, session.output_dir):
        d.mkdir(parents=True)

    _point_latest_at(root, session_dir)
    return session


def _point_latest_at(root: Path, session_dir: Path) -> None:
    """Repoint ``root/latest`` at ``session_dir``, ignoring unsupported FS.

    A relative target keeps the whole runs/ tree movable. Symlinks fail on
    some Windows setups and on a few network mounts; a missing convenience
    link must never abort a run, so failures are swallowed.
    """
    link = root / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(session_dir.name, target_is_directory=True)
    except OSError:
        pass


def latest_session(root: Path | str) -> Session | None:
    """Return the most recent session under ``root``, or None if there are none."""
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = [
        d
        for d in root.iterdir()
        if d.is_dir() and not d.is_symlink() and (d / "session.json").exists()
    ]
    if not candidates:
        return None
    newest = max(candidates, key=lambda d: d.name)
    meta = json.loads((newest / "session.json").read_text())
    return Session(
        root=newest,
        input_dir=newest / "input",
        work_dir=newest / "work",
        output_dir=newest / "output",
        session_id=meta.get("session_id", newest.name),
        label=meta.get("label", newest.name),
        created_at=meta.get("created_at", ""),
    )
