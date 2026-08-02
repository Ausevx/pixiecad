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
import os
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

# TRELLIS.2's image encoder. Gated: access must be granted per-account, and the
# 401 only appears after ~19 GB of unrelated weights have already downloaded.
GATED_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"

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


def docker_run_command(
    image: str = DEFAULT_TRELLIS_IMAGE, *, avoid_gated: bool = True
) -> str:
    """Return the `docker run` line that starts the worker service detached.

    The port is bound to loopback only: the client is a shell on the same host,
    so exposing it on the VM's external interface would be pure attack surface.
    """
    # The image's own escape hatch from the gated DINOv3 encoder: it swaps in
    # DINOv2 (ungated) as the conditioning model. Default on, because
    # facebook/dinov3-* is gated:"manual" -- a human at Meta approves each
    # request, so without this a build is blocked on someone else's inbox.
    gated_env = "-e TRELLIS2_AVOID_GATED_DEPS=1 " if avoid_gated else ""
    return (
        f"docker run -d --name {CONTAINER_NAME} --gpus all {gated_env}"
        # Deliberately NOT --restart unless-stopped. Loading the model needs
        # ~15 GB of host RAM; on an undersized VM the kernel OOM-kills it, and
        # a restart policy turns that into a silent respawn loop that burns the
        # whole job budget. Failing once and surfacing the logs is the point.
        f"--restart no "
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
    avoid_gated: bool = True,
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

    # TRELLIS.2 pulls facebook/dinov3-vitl16-pretrain-lvd1689m, a GATED repo:
    # without credentials the worker downloads 19 GB of other weights, spends
    # minutes loading them, and only then dies with a 401. The token is
    # installed as a file inside the already-mounted HF cache rather than
    # passed as -e or on a command line, so it never appears in `ps` output or
    # in `docker inspect` on a shared host.
    install_token = (
        "if [ -f hf_token ]; then\n"
        '  mkdir -p "$HOME/.cache/huggingface"\n'
        '  cp hf_token "$HOME/.cache/huggingface/token"\n'
        '  chmod 600 "$HOME/.cache/huggingface/token"\n'
        "fi\n"
    )

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
{install_token}if [ -z "$(docker ps -q -f name=^{CONTAINER_NAME}$ -f status=running)" ]; then
  docker rm -f {CONTAINER_NAME} >/dev/null 2>&1 || true
  {docker_run_command(image, avoid_gated=avoid_gated)}
fi
tries=0
while :; do
  if curl -sf --max-time 10 {url}/ready | grep -q '"ready"[ ]*:[ ]*true'; then break; fi
  # A dead container will never become ready, so stop waiting the moment it
  # exits rather than burning the full timeout on a corpse. An OOM kill shows
  # up here as a non-running container with exit code 137.
  if [ -z "$(docker ps -q -f name=^{CONTAINER_NAME}$ -f status=running)" ]; then
    echo "TRELLIS worker exited during startup (code $(docker inspect -f '{{{{.State.ExitCode}}}}' {CONTAINER_NAME} 2>/dev/null))" >&2
    echo "exit 137 means the host kernel OOM-killed it: loading the model peaked past 36 GB in testing, so use a 64 GB VM (g2-standard-16 or larger)." >&2
    docker logs --tail 40 {CONTAINER_NAME} >&2 || true
    exit 1
  fi
  # A failed preload leaves the container up but permanently unusable, so the
  # error state has to end the wait too — otherwise we sit here for the full
  # timeout while the worker cheerfully answers /ready with 200.
  if curl -sf --max-time 10 {url}/ready | grep -q '"status"[ ]*:[ ]*"error"'; then
    echo "TRELLIS worker failed to initialise:" >&2
    curl -sf --max-time 10 {url}/ready >&2 || true
    exit 1
  fi
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


def resolve_hf_token() -> str | None:
    """Find a HuggingFace token the way huggingface_hub itself does.

    Checks both env vars, then the file written by ``huggingface-cli login``.
    Reading the file matters because an interactive ``export`` does not reach a
    non-interactive shell, which is where builds actually run.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(var)
        if value and value.strip():
            return value.strip()

    token_file = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "token"
    try:
        value = token_file.read_text().strip()
    except OSError:
        return None
    return value or None


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

            # Staged as a file in the job's input dir so it travels over the
            # same rsync as the images and never touches a command line.
            hf_token = resolve_hf_token()
            if hf_token:
                token_file = tmp_path / "hf_token"
                token_file.write_text(hf_token.strip())
                token_file.chmod(0o600)
                job_inputs = [images_staging_dir, token_file]
            else:
                job_inputs = [images_staging_dir]

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
                inputs=job_inputs,
                output_dir=out_dir,
                remote_subdir=remote_subdir,
                timeout_s=self.timeout_s,
            )

            start_time = time.monotonic()
            result = self.executor.run(job)
            elapsed = time.monotonic() - start_time

            if not result.ok:
                stderr_tail = (result.stderr_tail or "")[-1500:]
                if "GatedRepoError" in stderr_tail or "gated repo" in stderr_tail:
                    raise GenerativeError(
                        "TRELLIS.2's default encoder is gated. Preferred fix: run "
                        "with avoid_gated=True (the default), which swaps in the "
                        "ungated DINOv2 encoder and needs no account at all. "
                        "To use the gated DINOv3 instead you need a token: it depends on "
                        "facebook/dinov3-vitl16-pretrain-lvd1689m, which is a gated "
                        "repo. Accept the licence at "
                        "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m "
                        "then create a read token at "
                        "https://huggingface.co/settings/tokens and export HF_TOKEN "
                        "before running. Without it the worker downloads 19 GB and "
                        "loads for several minutes before failing with a 401."
                    )
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
