"""TRELLIS remote generative 3D backend using Executor and Docker."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trimesh

if TYPE_CHECKING:
    from ..executors.base import Executor

TRELLIS_MIN_VRAM_MB = 22000
DEFAULT_TRELLIS_IMAGE = "kngsly/trellis2-worker:latest"


_base_classes_cache: tuple[type[Any], type[Exception], type[Exception]] | None = None


def _get_generative_base() -> tuple[type[Any], type[Exception], type[Exception]]:
    """Lazy import of generative base classes to avoid hard build dependencies."""
    global _base_classes_cache
    if _base_classes_cache is not None:
        return _base_classes_cache

    try:
        from .base import BackendUnavailable, GenerateResult, GenerativeError

        _base_classes_cache = (GenerateResult, GenerativeError, BackendUnavailable)
        return _base_classes_cache
    except (ImportError, ModuleNotFoundError):
        # Fallback definitions if base.py module is not present during concurrent dev
        from dataclasses import dataclass, field

        @dataclass
        class GenerateResult:  # type: ignore[no-redef]
            mesh_path: str
            backend: str
            n_faces: int | None
            seconds: float
            cached: bool = False
            metadata: dict = field(default_factory=dict)

        class GenerativeError(Exception):  # type: ignore[no-redef]
            pass

        class BackendUnavailable(GenerativeError):  # type: ignore[no-redef]
            pass

        _base_classes_cache = (GenerateResult, GenerativeError, BackendUnavailable)
        return _base_classes_cache


def docker_prefix(image: str = DEFAULT_TRELLIS_IMAGE) -> str:
    """Return command prefix that runs TRELLIS inside container."""
    return f'docker run --rm --gpus all -v "$PWD":/work -w /work {image}'


def build_trellis_script(
    *,
    image: str = DEFAULT_TRELLIS_IMAGE,
    images_dirname: str = "images",
    seed: int | None = None,
    texture: bool = True,
) -> str:
    """Return POSIX shell script string for running TRELLIS reconstruction."""
    cmds = ["mkdir -p out"]
    cli_cmd = f"{docker_prefix(image)} python -m trellis2.cli --input {images_dirname} --output out/mesh.glb"
    if seed is not None:
        cli_cmd += f" --seed {seed}"
    if not texture:
        cli_cmd += " --no-texture"
    cmds.append(cli_cmd)
    return " && ".join(cmds)


class RemoteTrellisBackend:
    """Generative 3D backend that runs TRELLIS on a remote host via Executor."""

    name = "trellis-remote"

    def __init__(
        self,
        executor: Executor,
        *,
        image: str = DEFAULT_TRELLIS_IMAGE,
        timeout_s: int = 1800,
    ) -> None:
        self.executor = executor
        self.image = image
        self.timeout_s = timeout_s

    def available(self) -> bool:
        """Check if remote host is reachable and meets VRAM requirements."""
        try:
            caps = self.executor.probe()
            return caps.cuda_ok(TRELLIS_MIN_VRAM_MB)
        except Exception:
            return False

    def generate(self, req: Any, out_dir: str | Path) -> Any:
        """Run TRELLIS generation on remote host for the given request."""
        GenerateResult, GenerativeError, BackendUnavailable = _get_generative_base()
        from ..executors.base import Job

        out_dir = Path(out_dir)

        caps = None
        try:
            caps = self.executor.probe()
        except Exception:
            caps = None

        if caps is None or not caps.cuda_ok(TRELLIS_MIN_VRAM_MB):
            hostname = caps.hostname if caps else "unknown"
            reachable = caps.reachable if caps else False
            vram_found = caps.gpu.vram_mb if (caps and caps.gpu) else 0
            raise BackendUnavailable(
                f"TRELLIS backend unavailable on host '{hostname}' "
                f"(reachable={reachable}, VRAM found={vram_found} MB, needed={TRELLIS_MIN_VRAM_MB} MB)"
            )

        seed = getattr(req, "seed", None)
        texture = getattr(req, "texture", True)
        req_images = getattr(req, "images", [])

        # Deterministic 8-char hash derived from image filenames and seed
        hasher = hashlib.sha256()
        for img in req_images:
            hasher.update(Path(img).name.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        hash_suffix = hasher.hexdigest()[:8]
        remote_subdir = f"trellis-{hash_suffix}"

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_path = Path(tmp_dir_str)
            images_staging_dir = tmp_path / "images"
            images_staging_dir.mkdir(parents=True, exist_ok=True)

            for i, img_path in enumerate(req_images):
                img_path = Path(img_path)
                suffix = img_path.suffix or ".png"
                target = images_staging_dir / f"{i:02d}{suffix}"
                shutil.copyfile(img_path, target)

            script = build_trellis_script(
                image=self.image,
                images_dirname="images",
                seed=seed,
                texture=texture,
            )

            job = Job(
                command=["sh", "-c", script],
                inputs=[images_staging_dir],
                output_dir=out_dir,
                remote_subdir=remote_subdir,
                timeout_s=self.timeout_s,
            )

            start_time = time.monotonic()
            result = self.executor.run(job)
            elapsed = time.monotonic() - start_time

            if not result.ok:
                stderr_tail = (result.stderr_tail or "")[-500:]
                raise GenerativeError(
                    f"TRELLIS remote execution failed on host '{caps.hostname}': {stderr_tail}"
                )

        mesh_path = out_dir / "mesh.glb"
        if not mesh_path.exists():
            raise GenerativeError(
                f"TRELLIS job reported success but produced no mesh file at '{mesh_path}'"
            )

        n_faces = None
        try:
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            if hasattr(mesh, "faces") and mesh.faces is not None:
                n_faces = int(len(mesh.faces))
        except Exception:
            try:
                mesh = trimesh.load(str(mesh_path), process=False)
                if hasattr(mesh, "faces") and mesh.faces is not None:
                    n_faces = int(len(mesh.faces))
                elif hasattr(mesh, "geometry") and isinstance(mesh.geometry, dict):
                    total = sum(
                        len(g.faces)
                        for g in mesh.geometry.values()
                        if hasattr(g, "faces") and g.faces is not None
                    )
                    n_faces = int(total) if total > 0 else None
            except Exception:
                n_faces = None

        return GenerateResult(
            mesh_path=str(mesh_path),
            backend=self.name,
            n_faces=n_faces,
            seconds=elapsed,
            cached=False,
            metadata={"remote_subdir": remote_subdir, "image": self.image},
        )


def make_backend(executor: Executor, **kwargs: Any) -> RemoteTrellisBackend:
    """Factory to create a RemoteTrellisBackend with the given executor.

    Note: We do not auto-register a factory under "trellis-remote" at import time
    because RemoteTrellisBackend requires a runtime Executor instance.
    """
    return RemoteTrellisBackend(executor, **kwargs)
