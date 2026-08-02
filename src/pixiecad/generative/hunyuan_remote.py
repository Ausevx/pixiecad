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

import json
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
# The multi-view checkpoint was only released for the 2.x line, so it is a
# different repo and a different package (hy3dgen) from the single-view 2.1.
MULTIVIEW_MODEL = "tencent/Hunyuan3D-2mv"
MULTIVIEW_SUBFOLDER = "hunyuan3d-dit-v2-mv"

# The only tags hy3dgen's preprocessor accepts; it sorts them into its own
# order (front, +90 clockwise, back, +270), so tags matter and order does not.
VIEW_TAGS = ("front", "left", "back", "right")

# Options a caller may pass through to the remote script.
PASSTHROUGH_OPTIONS = (
    "steps",
    "octree_resolution",
    "guidance_scale",
    "model",
    "subfolder",
)

_SCRIPT = Path(__file__).parent / "remote_scripts" / "hunyuan_gen.py"


def map_views(paths: list[Path]) -> dict[str, Path]:
    """Match photo filenames to the four view tags the model understands.

    Only the four cardinal views are usable; obliques ("front_left") and a top
    view have no tag and are dropped rather than mislabelled, because feeding a
    3/4 shot in as "front" tells the model the car is shaped wrongly.
    """
    chosen: dict[str, Path] = {}
    for path in paths:
        stem = path.stem.lower()
        has_front = "front" in stem
        has_back = "rear" in stem or "back" in stem
        has_left = "left" in stem
        has_right = "right" in stem

        # An oblique names two directions at once; ambiguity means skip.
        if (has_front or has_back) and (has_left or has_right):
            continue
        if has_front:
            tag = "front"
        elif has_back:
            tag = "back"
        elif has_left:
            tag = "left"
        elif has_right:
            tag = "right"
        else:
            continue
        # First match wins so the result is stable for a sorted input list.
        chosen.setdefault(tag, path)
    return chosen


def build_hunyuan_script(
    *,
    image: str = DEFAULT_HUNYUAN_IMAGE,
    image_name: str = "00.png",
    views: dict[str, str] | None = None,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """Return the shell script that produces ``out/mesh.glb`` on the GPU host."""
    opts = options or {}
    if views:
        source = [
            "--views views.json",
            f"--model {opts.get('model', MULTIVIEW_MODEL)}",
            f"--subfolder {opts.get('subfolder', MULTIVIEW_SUBFOLDER)}",
        ]
    else:
        source = [
            f"--image images/{image_name}",
            f"--model {opts.get('model', DEFAULT_MODEL)}",
        ]
    flags = [
        *source,
        "--out out/mesh.glb",
        f"--steps {int(opts.get('steps', 30))}",
        f"--octree-resolution {int(opts.get('octree_resolution', 256))}",
        f"--guidance-scale {float(opts.get('guidance_scale', 5.0))}",
    ]
    if seed is not None:
        flags.append(f"--seed {int(seed)}")

    return (
        "set -e\n"
        "mkdir -p out\n"
        # Both caches are mounted from the host so the ~10 GB of weights survive
        # between jobs. hy3dgen keeps its own cache separate from HuggingFace's,
        # and missing it means every --rm run re-downloads the whole model.
        "docker run --rm --gpus all "
        '-v "$PWD":/work '
        '-v "$HOME/.cache/huggingface":/root/.cache/huggingface '
        '-v "$HOME/.cache/hy3dgen":/root/.cache/hy3dgen '
        '-v "$HOME/.u2net":/root/.u2net '
        "-w /work "
        f"{image} python /work/hunyuan_gen.py {' '.join(flags)}\n"
    )


DEFAULT_PAINT_IMAGE = "pixiecad-hunyuan-paint:latest"
_TEXTURE_SCRIPT = Path(__file__).parent / "remote_scripts" / "hunyuan_texture.py"


def build_texture_script(
    *,
    image: str = DEFAULT_PAINT_IMAGE,
    mesh_name: str = "mesh.glb",
    image_name: str = "00.png",
    max_views: int = 6,
    resolution: int = 512,
) -> str:
    """Shell script that textures ``mesh_name`` into ``out/textured.glb``."""
    return (
        "set -e\n"
        "mkdir -p out\n"
        "docker run --rm --gpus all "
        '-v "$PWD":/work '
        '-v "$HOME/.cache/huggingface":/root/.cache/huggingface '
        '-v "$HOME/.cache/hy3dgen":/root/.cache/hy3dgen '
        '-v "$HOME/.u2net":/root/.u2net '
        "-w /work "
        f"{image} python /work/hunyuan_texture.py "
        f"--mesh /work/{mesh_name} --image /work/images/{image_name} "
        f"--out out/textured.glb --max-views {int(max_views)} "
        f"--resolution {int(resolution)}\n"
    )


def texture_mesh(
    executor: Executor,
    mesh_path: str | Path,
    image_path: str | Path,
    out_dir: str | Path,
    *,
    docker_image: str = DEFAULT_PAINT_IMAGE,
    max_views: int = 6,
    resolution: int = 512,
    timeout_s: int = 3600,
) -> Path:
    """Texture an existing mesh on the GPU host; returns the textured GLB.

    Separate from ``generate`` so a good shape can be re-textured without
    paying for regeneration -- texturing is the stage most likely to need a
    second attempt, and the geometry is the expensive part to get right.
    """
    from ..executors.base import Job
    from .base import GenerativeError

    mesh_path, image_path, out_dir = Path(mesh_path), Path(image_path), Path(out_dir)
    if not mesh_path.is_file():
        raise GenerativeError(f"no mesh to texture at '{mesh_path}'")
    if not image_path.is_file():
        raise GenerativeError(f"no conditioning image at '{image_path}'")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        staged_images = tmp_path / "images"
        staged_images.mkdir()
        img_name = f"00{image_path.suffix or '.png'}"
        shutil.copyfile(image_path, staged_images / img_name)
        shutil.copyfile(mesh_path, tmp_path / "mesh.glb")
        shutil.copyfile(_TEXTURE_SCRIPT, tmp_path / "hunyuan_texture.py")

        job = Job(
            command=[
                "sh",
                "-c",
                build_texture_script(
                    image=docker_image,
                    image_name=img_name,
                    max_views=max_views,
                    resolution=resolution,
                ),
            ],
            inputs=[
                staged_images,
                tmp_path / "mesh.glb",
                tmp_path / "hunyuan_texture.py",
            ],
            output_dir=out_dir,
            remote_subdir=f"hytex-{abs(hash(mesh_path.name)) % 10**8:08d}",
            timeout_s=timeout_s,
        )
        result = executor.run(job)

    if not result.ok:
        raise GenerativeError(
            f"Hunyuan3D texturing failed: {(result.stderr_tail or '')[-1500:]}"
        )
    textured = Path(out_dir) / "textured.glb"
    if not textured.exists():
        raise GenerativeError(
            f"texturing reported success but produced no mesh at '{textured}'"
        )
    return textured


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
        multiview: bool = False,
    ) -> None:
        self.executor = executor
        self.image = image
        # Env override lets `pixiecad run --multiview` reach the backend
        # without threading the flag through every call site.
        import os

        self.multiview = multiview or os.environ.get(
            "PIXIECAD_HUNYUAN_MULTIVIEW"
        ) == "1"
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
        options = dict(getattr(req, "options", {}) or {})
        seed = getattr(req, "seed", None)

        # Off by default on the evidence, not on principle. Measured on the F1
        # set: 2mv given all four cardinal views merged the front wheels into a
        # slab and produced 18 components, while single-view 2.1 kept four
        # distinct wheels and 5 components. Swapping left/right in case the
        # tag convention was inverted changed nothing, so it is the model, not
        # the mapping: 2mv is built on the older 2.0 line and the extra views
        # do not compensate for the weaker base architecture.
        view_map = map_views(images) if self.multiview else {}
        if len(view_map) < 2:
            view_map = {}
        chosen = images[0]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            staged_images = tmp_path / "images"
            staged_images.mkdir()
            staged_script = tmp_path / "hunyuan_gen.py"
            shutil.copyfile(_SCRIPT, staged_script)
            job_inputs = [staged_images, staged_script]

            views_json: dict[str, str] = {}
            if view_map:
                for tag, src in sorted(view_map.items()):
                    name = f"{tag}{src.suffix or '.png'}"
                    shutil.copyfile(src, staged_images / name)
                    views_json[tag] = name
                views_file = tmp_path / "views.json"
                views_file.write_text(json.dumps(views_json))
                job_inputs.append(views_file)
                target_name = ""
            else:
                target = staged_images / f"00{chosen.suffix or '.png'}"
                shutil.copyfile(chosen, target)
                target_name = target.name

            script = build_hunyuan_script(
                image=self.image,
                image_name=target_name,
                views=views_json or None,
                seed=seed,
                options={k: v for k, v in options.items() if k in PASSTHROUGH_OPTIONS},
            )

            job = Job(
                command=["sh", "-c", script],
                inputs=job_inputs,
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
            metadata={
                "image": self.image,
                "mode": "multiview" if view_map else "single",
                "conditioning_views": (
                    sorted(f"{t}:{p.name}" for t, p in view_map.items())
                    if view_map
                    else [chosen.name]
                ),
            },
        )


def make_backend(executor: Executor, **kwargs: Any) -> RemoteHunyuanBackend:
    """Factory; needs a live Executor so it is not auto-registered at import."""
    return RemoteHunyuanBackend(executor, **kwargs)
