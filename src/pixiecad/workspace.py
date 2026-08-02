"""Workspace: content-addressed, resumable stage cache.

Layout:
    workspace/
      manifest.json                  # spec + provenance + stage index
      stages/<stage>-<inputhash>/    # outputs of one stage run
        _done                        # sentinel written only on success

A stage's directory key hashes (stage name, params, input fingerprints), so
changing the face budget re-runs only retopo, never reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .spec import ObjectSpec

_MANIFEST = "manifest.json"
_DONE = "_done"


def fingerprint_file(path: Path, chunk: int = 1 << 20) -> str:
    """Content hash of a file (sha256, first 16 hex chars)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()[:16]


def stage_key(stage: str, params: dict[str, Any], input_fingerprints: list[str]) -> str:
    payload = json.dumps(
        {"stage": stage, "params": params, "inputs": sorted(input_fingerprints)},
        sort_keys=True,
    )
    return f"{stage}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


@dataclass
class StageRun:
    key: str
    dir: Path
    cached: bool


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / _MANIFEST

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def create(cls, root: Path, spec: ObjectSpec) -> "Workspace":
        ws = cls(root)
        ws.stages_dir.mkdir(parents=True, exist_ok=True)
        if not ws.manifest_path.exists():
            ws._write_manifest({"spec": spec.model_dump(mode="json"), "stages": {}})
        return ws

    @classmethod
    def open(cls, root: Path) -> "Workspace":
        ws = cls(root)
        if not ws.manifest_path.exists():
            raise FileNotFoundError(f"No workspace manifest at {ws.manifest_path}")
        return ws

    def spec(self) -> ObjectSpec:
        return ObjectSpec.model_validate(self._read_manifest()["spec"])

    def update_spec(self, spec: ObjectSpec) -> None:
        m = self._read_manifest()
        m["spec"] = spec.model_dump(mode="json")
        self._write_manifest(m)

    # -- stage cache ---------------------------------------------------------

    def begin_stage(
        self, stage: str, params: dict[str, Any], input_fingerprints: list[str]
    ) -> StageRun:
        """Return the stage dir; ``cached`` is True when a finished run exists."""
        key = stage_key(stage, params, input_fingerprints)
        d = self.stages_dir / key
        if (d / _DONE).exists():
            return StageRun(key=key, dir=d, cached=True)
        if d.exists():  # stale partial run
            shutil.rmtree(d)
        d.mkdir(parents=True)
        return StageRun(key=key, dir=d, cached=False)

    def finish_stage(self, run: StageRun, summary: dict[str, Any] | None = None) -> None:
        (run.dir / _DONE).write_text("ok\n")
        m = self._read_manifest()
        m["stages"][run.key] = summary or {}
        self._write_manifest(m)

    # -- internals -----------------------------------------------------------

    def _read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def _write_manifest(self, data: dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n")
        tmp.replace(self.manifest_path)
