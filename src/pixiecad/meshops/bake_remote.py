"""Run the S6b normal-map bake on the GPU host instead of this machine.

Why this exists: baking is by a wide margin the heaviest thing the local
pipeline does. Measured against a 1.31M face mesh -- the scale a generative
backend actually returns -- on the 16 GB dev machine:

    1024x1024   13.6 s   3.4 GB peak RSS
    2048x2048   52.9 s   4.1 GB peak RSS

against roughly 250 MB for every other local stage combined. Profiling puts
essentially all of it in one call, ``ProximityQuery.on_surface``, which finds
the nearest point on the 1.3M-triangle source for every texel. Chunking that
query was tried first and did not help; peak stayed at 6.5 GB.

An honest caveat, because the name invites the wrong assumption: this is
**remote CPU, not GPU acceleration**. The bake performs no GPU work at all, on
either machine. What moving it buys is that the memory and the wall time land
on a host with 64 GB that is already provisioned and already being billed,
rather than on the laptop its owner is using. Genuine GPU baking would mean
Blender's bake operator (bpy 4.2 already ships in the paint image) and a
different, larger change.

The transfer cost is real and worth stating: the dense mesh is about 24 MB as
raw arrays for 1.31M faces, so this trades roughly 2-19 s of upload -- entirely
dependent on the connection -- for 14 s and 3.4 GB of local pressure.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_BAKE_SOURCE = Path(__file__).parent / "bake.py"
_RUNNER = Path(__file__).parent / "remote_scripts" / "remote_bake.py"

DEFAULT_IMAGE = "pixiecad-hunyuan:latest"


class RemoteBakeError(RuntimeError):
    """The host could not produce a normal map."""


def build_bake_script(
    *,
    image: str = DEFAULT_IMAGE,
    resolution: int = 1024,
    padding: int = 4,
) -> str:
    """Shell script that bakes ``dense.npz`` + ``low.npz`` into ``out/normal.png``.

    No ``--gpus all``: the bake never touches CUDA, and requesting a GPU here
    would make the job queue behind whatever else wants the device for no
    benefit whatsoever.
    """
    return (
        "set -e\n"
        "mkdir -p out\n"
        "docker run --rm "
        '-v "$PWD":/work '
        "-w /work "
        f"{image} python /work/remote_bake.py "
        "--dense /work/dense.npz --low /work/low.npz "
        f"--out /work/out/normal.png --resolution {int(resolution)} "
        f"--padding {int(padding)}\n"
    )


def _save_mesh(path: Path, mesh: Any, *, uv: Any = None) -> None:
    """Write vertices/faces (and UVs) compressed.

    float32 vertices halve the upload and are far finer than the bake's own
    texel resolution, so nothing measurable is lost. Faces stay int32 because
    they are indices.
    """
    arrays = {
        "vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "faces": np.asarray(mesh.faces, dtype=np.int32),
    }
    if uv is not None:
        arrays["uv"] = np.asarray(uv, dtype=np.float32)
    np.savez_compressed(path, **arrays)


def bake_object_space_normals_remote(
    executor: Any,
    dense: Any,
    low_unwrapped: Any,
    *,
    resolution: int = 1024,
    padding_px: int = 4,
    docker_image: str = DEFAULT_IMAGE,
    timeout_s: int = 1800,
    log: Any = None,
) -> np.ndarray:
    """Bake on ``executor``'s host and return the map as an RGB array.

    The return type matches the local ``bake_object_space_normals`` exactly, so
    callers do not branch on where the work happened.

    Raises ``RemoteBakeError`` rather than falling back to a local bake. That
    is deliberate: someone who asked for the host asked precisely because they
    did not want 3.4 GB on their own machine, and quietly doing it there anyway
    would defeat the only reason the option exists.
    """
    from ..executors.base import Job

    uv = getattr(getattr(low_unwrapped, "visual", None), "uv", None)
    if uv is None:
        raise RemoteBakeError("low-poly mesh has no UVs; unwrap must run before baking")

    def _say(msg: str) -> None:
        if log is not None:
            log(msg)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dense_npz = tmp_path / "dense.npz"
        low_npz = tmp_path / "low.npz"
        _save_mesh(dense_npz, dense)
        _save_mesh(low_npz, low_unwrapped, uv=uv)
        shutil.copyfile(_BAKE_SOURCE, tmp_path / "bake.py")
        shutil.copyfile(_RUNNER, tmp_path / "remote_bake.py")

        mb = (dense_npz.stat().st_size + low_npz.stat().st_size) / 1e6
        _say(f"Baking normal map on the GPU host ({mb:.1f} MB to upload)...")

        out_dir = tmp_path / "out"
        job = Job(
            command=[
                "sh",
                "-c",
                build_bake_script(
                    image=docker_image, resolution=resolution, padding=padding_px
                ),
            ],
            inputs=[dense_npz, low_npz, tmp_path / "bake.py", tmp_path / "remote_bake.py"],
            output_dir=out_dir,
            remote_subdir=f"bake-{abs(hash(str(tmp_path))) % 10**8:08d}",
            timeout_s=timeout_s,
        )
        result = executor.run(job)

        if not result.ok:
            raise RemoteBakeError(
                f"remote bake failed: {(result.stderr_tail or '')[-1500:]}"
            )

        png = out_dir / "normal.png"
        if not png.is_file():
            raise RemoteBakeError("remote bake reported success but produced no image")

        from PIL import Image

        with Image.open(png) as im:
            img = np.array(im.convert("RGB"))

    if img.shape[0] != resolution or img.shape[1] != resolution:
        raise RemoteBakeError(
            f"remote bake returned {img.shape[1]}x{img.shape[0]}, expected "
            f"{resolution}x{resolution}"
        )
    _say(f"Normal map baked on the host ({resolution}x{resolution}).")
    return img
