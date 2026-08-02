"""Hunyuan3D generative backend, run on a remote GPU host through an Executor.

Chosen over TRELLIS.2 because it has no gated dependency: TRELLIS.2 conditions
on facebook/dinov3-*, which is gated:"manual" and therefore blocks a build on
someone else's approval queue. Substituting a different encoder is not a
workaround — it silently produces noise, because the model was trained on
DINOv3 features specifically.

Unlike the TRELLIS worker this runs a one-shot script rather than a service.
There is no port to bind and nothing left listening between jobs, which suits
the rsync-in / run / rsync-out shape of Executor. The container is built from
Tencent's own Dockerfile by ``scripts/setup_hunyuan_vm.sh``; published
third-party Hunyuan images were inspected and rejected -- one opened a public
Cloudflare tunnel and uploaded results to a stranger's S3 bucket.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trimesh

if TYPE_CHECKING:
    from ..executors.base import Executor

# Shape generation fits comfortably in an L4; this is the floor that leaves
# room for the octree decoder at 256.
HUNYUAN_MIN_VRAM_MB = 16000

DEFAULT_HUNYUAN_IMAGE = "pixiecad-hunyuan:latest"
DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"

# Options a caller may pass through to the remote script.
PASSTHROUGH_OPTIONS = ("steps", "octree_resolution", "guidance_scale", "model")

_SCRIPT = Path(__file__).parent / "remote_scripts" / "hunyuan_gen.py"


def build_hunyuan_script(
    *,
    image: str = DEFAULT_HUNYUAN_IMAGE,
    image_name: str = "00.png",
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """Return the shell script that produces ``out/mesh.glb`` on the GPU host."""
    opts = options or {}
    flags = [
        f"--image images/{image_name}",
        "--out out/mesh.glb",
        f"--steps {int(opts.get('steps', 30))}",
        f"--octree-resolution {int(opts.get('octree_resolution', 256))}",
        f"--guidance-scale {float(opts.get('guidance_scale', 5.0))}",
        f"--model {opts.get('model', DEFAULT_MODEL)}",
    ]
    if seed is not None:
        flags.append(f"--seed {int(seed)}")

    return (
        "set -e\n"
        "mkdir -p out\n"
        # The HF cache is mounted from the host so the ~10 GB of weights
        # survive between jobs; only the first run on a fresh VM downloads.
        "docker run --rm --gpus all "
        '-v "$PWD":/work '
        '-v "$HOME/.cache/huggingface":/root/.cache/huggingface '
        "-w /work "
        f"{image} python /work/hunyuan_gen.py {' '.join(flags)}\n"
    )


def _gpu_supported(caps: Any) -> bool:
    """True when the host has enough VRAM. No architecture allow-list here.

    Our image is built with a wide TORCH_CUDA_ARCH_LIST, so unlike the
    Ada-only TRELLIS image it is not pinned to one GPU generation.
    """
    return bool(caps.cuda_ok(HUNYUAN_MIN_VRAM_MB))


class RemoteHunyuanBackend:
    name = "hunyuan-remote"

    def __init__(
        self,
        executor: Executor,
        *,
        image: str = DEFAULT_HUNYUAN_IMAGE,
        timeout_s: int = 3600,
    ) -> None:
        self.executor = executor
        self.image = image
        self.timeout_s = timeout_s

    def available(self) -> bool:
        try:
            return _gpu_supported(self.executor.probe())
        except Exception:
            return False

    def generate(self, req: Any, out_dir: str | Path) -> Any:
        from .base import BackendUnavailable, GenerateResult, GenerativeError
        from ..executors.base import Job

        out_dir = Path(out_dir)
        try:
            caps = self.executor.probe()
        except Exception:
            caps = None

        if caps is None or not _gpu_supported(caps):
            vram = caps.gpu.vram_mb if (caps and caps.gpu) else 0
            raise BackendUnavailable(
                f"Hunyuan3D backend unavailable on "
                f"'{caps.hostname if caps else 'unknown'}' "
                f"(VRAM {vram} MB / needed {HUNYUAN_MIN_VRAM_MB} MB)"
            )

        images = [Path(p) for p in getattr(req, "images", [])]
        if not images:
            raise GenerativeError("Hunyuan3D needs at least one image")
        # Single-image model: the first view is the conditioning view. Sending
        # more would be silently ignored, so be explicit about what is used.
        chosen = images[0]
        options = dict(getattr(req, "options", {}) or {})
        seed = getattr(req, "seed", None)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            staged_images = tmp_path / "images"
            staged_images.mkdir()
            target = staged_images / f"00{chosen.suffix or '.png'}"
            shutil.copyfile(chosen, target)
            staged_script = tmp_path / "hunyuan_gen.py"
            shutil.copyfile(_SCRIPT, staged_script)

            script = build_hunyuan_script(
                image=self.image,
                image_name=target.name,
                seed=seed,
                options={k: v for k, v in options.items() if k in PASSTHROUGH_OPTIONS},
            )

            job = Job(
                command=["sh", "-c", script],
                inputs=[staged_images, staged_script],
                output_dir=out_dir,
                remote_subdir=f"hunyuan-{abs(hash((chosen.name, seed))) % 10**8:08d}",
                timeout_s=self.timeout_s,
            )
            t0 = time.monotonic()
            result = self.executor.run(job)
            elapsed = time.monotonic() - t0

        if not result.ok:
            raise GenerativeError(
                f"Hunyuan3D failed on '{caps.hostname}': "
                f"{(result.stderr_tail or '')[-1500:]}"
            )

        mesh_path = out_dir / "mesh.glb"
        if not mesh_path.exists():
            raise GenerativeError(
                f"Hunyuan3D reported success but produced no mesh at '{mesh_path}'"
            )

        n_faces = None
        try:
            loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
            n_faces = int(len(loaded.faces))
        except Exception:
            pass

        return GenerateResult(
            mesh_path=str(mesh_path),
            backend=self.name,
            n_faces=n_faces,
            seconds=elapsed,
            cached=False,
            metadata={"image": self.image, "conditioning_view": chosen.name},
        )


def make_backend(executor: Executor, **kwargs: Any) -> RemoteHunyuanBackend:
    """Factory; needs a live Executor so it is not auto-registered at import."""
    return RemoteHunyuanBackend(executor, **kwargs)
