"""TRELLIS.2 generative backend, run on a remote GPU host through an Executor.

The published worker image is a **service**, not a one-shot CLI: it boots a
FastAPI app on port 8000, loads a multi-GB model in the background, and exposes
``/ready``, ``/generate`` (multipart, NDJSON streaming response) and
``/download``. So the remote command here is a shell script that talks to that
service over the host's own loopback — no GCP firewall rule, no public port,
and the existing rsync-based Executor contract is untouched.

The container is long-lived on purpose. Loading the model costs minutes; a
named container plus a host-side HuggingFace cache means only the first build
on a fresh VM pays that, and later builds reuse a warm GPU.
"""

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

# The image is compiled with TORCH_CUDA_ARCH_LIST=8.9 and no "+PTX", so it ships
# Ada cubins only — there is no JIT fallback for other architectures. A T4 (7.5)
# fails at kernel launch however much VRAM it reports, and so would an A100
# (8.0) or H100 (9.0). Hence an exact allow-list, not a minimum.
TRELLIS_SUPPORTED_COMPUTE_CAPS = (8.9,)

CONTAINER_NAME = "pixiecad-trellis"
WORKER_PORT = 8000
# Written on the remote host, outside the per-job dir, so the model cache and
# the container's output volume survive between jobs.
REMOTE_OUTPUTS_DIR = "/var/tmp/pixiecad-trellis-outputs"

# Form fields the worker accepts that are worth exposing. Anything else in
# GenerateRequest.options is ignored rather than blindly forwarded, so a typo
# can't silently change generation settings.
PASSTHROUGH_OPTIONS = (
    "mesh_profile",
    "geometry_resolution",
    "texture_generation_mode",
    "texture_output_size",
    "steps",
    "decimation_target",
    "preprocess_image",
    "post_scale_z",
)


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


def docker_run_command(image: str = DEFAULT_TRELLIS_IMAGE) -> str:
    """Return the `docker run` line that starts the worker service detached.

    The port is bound to loopback only: the client is a shell on the same host,
    so exposing it on the VM's external interface would be pure attack surface.
    """
    return (
        f"docker run -d --name {CONTAINER_NAME} --gpus all "
        f"--restart unless-stopped "
        f"-p 127.0.0.1:{WORKER_PORT}:{WORKER_PORT} "
        f'-v {REMOTE_OUTPUTS_DIR}:/outputs '
        f'-v "$HOME/.cache/huggingface":/root/.cache/huggingface '
        f"{image}"
    )


def build_trellis_script(
    *,
    image: str = DEFAULT_TRELLIS_IMAGE,
    image_names: list[str] | None = None,
    images_dirname: str = "images",
    seed: int | None = None,
    texture: bool = True,
    ready_timeout_s: int = 1800,
    generate_timeout_s: int = 1800,
    options: dict[str, Any] | None = None,
) -> str:
    """Return the POSIX shell script that produces ``out/mesh.glb`` on the host.

    Steps: ensure the worker container is up, wait for ``/ready`` (first boot
    downloads the model, hence the generous default), POST the views, then
    resolve the NDJSON result to a file. Because ``/outputs`` is bind-mounted,
    the finished .glb is already on the host filesystem and is copied rather
    than re-downloaded through HTTP.
    """
    names = image_names or ["00.png"]
    url = f"http://127.0.0.1:{WORKER_PORT}"

    form = " ".join(
        f'-F "images=@{images_dirname}/{n}"' for n in names
    )
    fields: dict[str, Any] = {}
    if seed is not None:
        fields["seed"] = seed
    if not texture:
        # There is no "no texture" switch; the cheapest mode is the closest
        # thing, and it keeps generation time down when only geometry matters.
        fields["texture_generation_mode"] = "fast_512"
    for key, value in (options or {}).items():
        if key in PASSTHROUGH_OPTIONS and value is not None:
            fields[key] = value
    fields["generate_timeout_sec"] = generate_timeout_s
    form += " " + " ".join(f'-F "{k}={v}"' for k, v in fields.items())

    ready_tries = max(1, ready_timeout_s // 5)

    return f"""set -e
mkdir -p out {REMOTE_OUTPUTS_DIR}
if [ -z "$(docker ps -q -f name=^{CONTAINER_NAME}$ -f status=running)" ]; then
  docker rm -f {CONTAINER_NAME} >/dev/null 2>&1 || true
  {docker_run_command(image)}
fi
tries=0
while :; do
  if curl -sf --max-time 10 {url}/ready | grep -q '"ready"[ ]*:[ ]*true'; then break; fi
  tries=$((tries+1))
  if [ "$tries" -ge {ready_tries} ]; then
    echo "TRELLIS worker not ready after {ready_timeout_s}s" >&2
    docker logs --tail 60 {CONTAINER_NAME} >&2 || true
    exit 1
  fi
  sleep 5
done
curl -sS -N --max-time {generate_timeout_s + 120} -X POST {url}/generate {form} > gen.ndjson
python3 - {REMOTE_OUTPUTS_DIR} <<'PYEOF'
import json, pathlib, shutil, sys

outputs = pathlib.Path(sys.argv[1])
result = None
for line in pathlib.Path("gen.ndjson").read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except ValueError:
        continue
    # Keepalive frames share the stream; the last frame carrying "success" is
    # the verdict, whether or not it is tagged type=result.
    if "success" in obj:
        result = obj
if result is None:
    sys.exit("TRELLIS worker returned no result frame")
if not result.get("success"):
    sys.exit("TRELLIS generation failed: %s" % result.get("error"))
src = outputs / pathlib.Path(result["glb_path"]).name
if not src.is_file():
    sys.exit("TRELLIS reported %s but it is not on the host volume" % result["glb_path"])
shutil.copyfile(src, "out/mesh.glb")
print("wrote out/mesh.glb from %s" % src)
PYEOF"""


def _gpu_supported(caps: Any) -> bool:
    """True when ``caps`` describes a GPU this image's kernels will run on."""
    if not caps.cuda_ok(TRELLIS_MIN_VRAM_MB):
        return False
    return caps.gpu.compute_cap in TRELLIS_SUPPORTED_COMPUTE_CAPS


class RemoteTrellisBackend:
    """Generative 3D backend that runs TRELLIS on a remote host via Executor."""

    name = "trellis-remote"

    def __init__(
        self,
        executor: Executor,
        *,
        image: str = DEFAULT_TRELLIS_IMAGE,
        timeout_s: int = 3600,
    ) -> None:
        self.executor = executor
        self.image = image
        # Budget for the whole remote step. A cold VM spends most of it before
        # any generation starts: pulling an ~9 GB image and downloading model
        # weights. Once warm, the same call is minutes.
        self.timeout_s = timeout_s

    def available(self) -> bool:
        """Check the remote host is reachable and its GPU can run this image."""
        try:
            return _gpu_supported(self.executor.probe())
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

        if caps is None or not _gpu_supported(caps):
            hostname = caps.hostname if caps else "unknown"
            reachable = caps.reachable if caps else False
            vram_found = caps.gpu.vram_mb if (caps and caps.gpu) else 0
            cap_found = caps.gpu.compute_cap if (caps and caps.gpu) else 0
            raise BackendUnavailable(
                f"TRELLIS backend unavailable on host '{hostname}' "
                f"(reachable={reachable}, VRAM {vram_found} MB / needed {TRELLIS_MIN_VRAM_MB} MB, "
                f"compute cap {cap_found} / supported {TRELLIS_SUPPORTED_COMPUTE_CAPS})"
            )

        seed = getattr(req, "seed", None)
        texture = getattr(req, "texture", True)
        req_images = getattr(req, "images", [])
        options = dict(getattr(req, "options", {}) or {})

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

            image_names = []
            for i, img_path in enumerate(req_images):
                img_path = Path(img_path)
                suffix = img_path.suffix or ".png"
                target = images_staging_dir / f"{i:02d}{suffix}"
                shutil.copyfile(img_path, target)
                image_names.append(target.name)

            script = build_trellis_script(
                image=self.image,
                image_names=image_names,
                images_dirname="images",
                seed=seed,
                texture=texture,
                # Leave room inside the job budget for boot and for rsyncing
                # the result back, so the worker's own timeout trips first and
                # reports a real error instead of ssh dying mid-stream.
                ready_timeout_s=max(300, self.timeout_s // 2),
                generate_timeout_s=max(300, self.timeout_s // 2 - 180),
                options=options,
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
