"""Base interface, types, registry, and stage wrapper for generative 3D backends."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

import trimesh

from ..workspace import Workspace, fingerprint_file


class GenerativeError(Exception):
    """Raised when a generative backend fails (network, GPU, bad output)."""


class BackendUnavailable(GenerativeError):
    """Raised when a generative backend is not configured or reachable."""


@dataclass
class GenerateRequest:
    images: list[Path]
    seed: int | None = None
    texture: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.images, list):
            self.images = list(self.images)
        self.images = [Path(p) for p in self.images]
        if not (1 <= len(self.images) <= 4):
            raise ValueError(
                f"GenerateRequest requires 1 to 4 images, got {len(self.images)}"
            )
        for img in self.images:
            if not img.exists():
                raise ValueError(f"Image path does not exist: {img}")


@dataclass
class GenerateResult:
    mesh_path: str
    backend: str
    n_faces: int | None
    seconds: float
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerativeBackend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def generate(self, req: GenerateRequest, out_dir: Path) -> GenerateResult:
        ...


_REGISTRY: dict[str, Callable[[], GenerativeBackend]] = {}

# Order tried when no backend is named. hunyuan-remote first: it is the only
# option with no gated dependency, so it cannot be blocked by a third party's
# approval queue, and it is the only one that has ever produced a model here.
#
# Every backend in this list is self-hosted by design. A hosted third-party
# API (fal.ai) lived here briefly and was removed: it would have uploaded the
# user's photos off the machine, which is the exact thing building from
# Tencent's own source was meant to avoid.
AUTO_PRIORITY = ["hunyuan-remote", "trellis-remote"]


def register_backend(factory: Callable[[], GenerativeBackend], name: str) -> None:
    """Register a factory function for a generative backend under name."""
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    """Return names of registered backends that construct and report available()=True."""
    avail = []
    for name, factory in list(_REGISTRY.items()):
        try:
            backend = factory()
            if backend.available():
                avail.append(name)
        except Exception:
            pass
    return avail


def get_backend(name: str | None = None) -> GenerativeBackend:
    """Return the requested backend by name, or the first available one if name is None.

    Raises BackendUnavailable if requested name is missing/unavailable or no backend is usable.
    """
    registered_names = list(_REGISTRY.keys())
    if name is not None:
        if name not in _REGISTRY:
            raise BackendUnavailable(
                f"Backend '{name}' is not registered. Registered backends: {registered_names}"
            )
        try:
            backend = _REGISTRY[name]()
        except Exception as e:
            raise BackendUnavailable(
                f"Failed to instantiate backend '{name}': {e}. Registered backends: {registered_names}"
            ) from e
        if not backend.available():
            # The registered-name list is only useful when the NAME was wrong.
            # When the name was right and the host was not reachable, printing
            # it sends people hunting for a typo instead of at the host -- which
            # is exactly what happened with a stale SSH host key.
            reason = getattr(backend, "last_unavailable_reason", "") or ""
            raise BackendUnavailable(
                f"Backend '{name}' is not available: {reason}"
                if reason
                else f"Backend '{name}' is not available. "
                f"Registered backends: {registered_names}"
            )
        return backend

    # Auto-selection order is AUTO_PRIORITY, then anything else registered;
    # "fake" is a test double and must never be picked implicitly.
    ordered = [n for n in AUTO_PRIORITY if n in _REGISTRY] + [
        n for n in _REGISTRY if n not in AUTO_PRIORITY and n != "fake"
    ]
    for name_key in ordered:
        try:
            backend = _REGISTRY[name_key]()
            if backend.available():
                return backend
        except Exception:
            pass

    raise BackendUnavailable(
        "No generative backend is usable. Provision a GPU host and pass an "
        "executor (hunyuan-remote, or trellis-remote on an L4), or request "
        "'fake' explicitly for tests. "
        f"Registered backends: {registered_names}"
    )


class FakeBackend:
    name: str = "fake"

    def available(self) -> bool:
        return True

    def generate(self, req: GenerateRequest, out_dir: Path) -> GenerateResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh_file = out_dir / "fake.glb"

        t0 = time.monotonic()
        # Seed-sensitive, but the SAME size and topology every time. A fake
        # backend that varied its face count would make seed determinism
        # testable and break every other test that assumes a fixed fake mesh --
        # a jitter proves the seed reached the backend without that cost.
        mesh = trimesh.creation.icosphere()
        if req.seed is not None:
            rng = np.random.default_rng(req.seed)
            mesh.vertices = mesh.vertices + rng.normal(scale=1e-3, size=mesh.vertices.shape)
        mesh.export(str(mesh_file))
        duration = time.monotonic() - t0

        n_faces = int(len(mesh.faces)) if hasattr(mesh, "faces") and mesh.faces is not None else None

        return GenerateResult(
            mesh_path=str(mesh_file),
            backend=self.name,
            n_faces=n_faces,
            seconds=duration,
            cached=False,
            metadata={},
        )


register_backend(FakeBackend, "fake")


def run_generate(
    req: GenerateRequest,
    workspace: Workspace,
    *,
    backend: str | None = None,
) -> GenerateResult:
    """Run generative 3D backend with workspace stage caching (stage 's3-generate')."""
    backend_obj = get_backend(backend)
    resolved_name = backend_obj.name

    fingerprints = [fingerprint_file(p) for p in req.images]
    params = {
        "backend": resolved_name,
        "seed": req.seed,
        "texture": req.texture,
        # Options change the mesh, so they must change the cache key: without
        # this a fast-profile re-run would silently return the HD run's result.
        "options": dict(sorted(req.options.items())),
    }

    run = workspace.begin_stage("s3-generate", params, fingerprints)

    result_path = run.dir / "result.json"
    if run.cached:
        data = json.loads(result_path.read_text())
        data["cached"] = True
        return GenerateResult(**data)

    out_dir = run.dir / "out"
    res = backend_obj.generate(req, out_dir)
    # Record the seed that was actually used, next to the existing mode and
    # conditioning_views. Without this a good result cannot be reproduced --
    # the whole point of pinning one.
    if req.seed is not None:
        res.metadata["seed"] = int(req.seed)

    mesh_p = Path(res.mesh_path)
    if not mesh_p.exists():
        raise GenerativeError(
            f"Backend '{resolved_name}' produced non-existent mesh file '{res.mesh_path}'"
        )

    res.cached = False
    result_path.write_text(json.dumps(asdict(res), indent=2) + "\n")
    workspace.finish_stage(run, asdict(res))

    return res
