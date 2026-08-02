"""Stage 2b: Dense reconstruction backend using COLMAP via Executor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import trimesh

from ..executors.base import Executor, Job
from ..workspace import Workspace, fingerprint_file

COLMAP_MIN_VRAM_MB = 3500


class DenseUnavailable(Exception):
    """Raised when no capable dense reconstruction backend is available."""


@dataclass
class DenseResult:
    mesh_path: str
    n_faces: int | None
    cached: bool = False


# Distro COLMAP packages are built WITHOUT CUDA (CUDA isn't redistributable in
# Debian main), so patch_match_stereo — the only reason we want a GPU — is
# missing from them. The official image is the reliable CUDA-enabled path.
DOCKER_COLMAP_IMAGE = "colmap/colmap:latest"


def docker_prefix(image: str = DOCKER_COLMAP_IMAGE, workdir: str = "/work") -> str:
    """Command prefix that runs COLMAP inside the CUDA-enabled official image."""
    return (
        f"docker run --rm --gpus all -v \"$PWD\":{workdir} -w {workdir} {image}"
    )


def build_dense_script(
    job_dir: str = ".",
    *,
    images_dirname: str = "images",
    model_dirname: str = "sparse",
    colmap_cmd: str = "colmap",
) -> str:
    """Return a single POSIX shell script string running dense COLMAP commands.

    Inputs land in the remote job dir under their local basenames (rsync
    semantics of SSHExecutor), so the dir names are parameters, not constants.

    ``colmap_cmd`` is how COLMAP is invoked on the target: ``"colmap"`` for a
    native CUDA build, or ``f"{docker_prefix()} colmap"`` to run the official
    image. ``mkdir`` stays outside it so the dir is owned by the login user,
    not by root inside the container.
    """
    cmds = []
    if job_dir and job_dir != ".":
        cmds.append(f"cd {job_dir}")
    cmds.extend([
        "mkdir -p out",
        f"{colmap_cmd} image_undistorter --image_path {images_dirname} "
        f"--input_path {model_dirname} --output_path dense",
        f"{colmap_cmd} patch_match_stereo --workspace_path dense",
        f"{colmap_cmd} stereo_fusion --workspace_path dense --output_path dense/fused.ply",
        f"{colmap_cmd} poisson_mesher --input_path dense/fused.ply --output_path out/mesh.ply",
    ])
    return " && ".join(cmds)


def run_dense(
    images_dir: Path,
    sparse_model_dir: Path,
    workspace: Workspace,
    executor: Executor,
    *,
    timeout_s: int = 7200,
    use_docker: bool = True,
) -> DenseResult:
    images_dir = Path(images_dir)
    sparse_model_dir = Path(sparse_model_dir)

    sparse_files = sorted(p for p in sparse_model_dir.glob("*.bin") if p.is_file())
    if not sparse_files:
        raise ValueError(
            f"No COLMAP model files (*.bin) in {sparse_model_dir}; pass the model "
            "directory itself (e.g. .../sparse/0), not its parent."
        )
    fingerprints = [fingerprint_file(p) for p in sparse_files]

    run = workspace.begin_stage("s2-dense", {"docker": use_docker}, fingerprints)

    result_path = run.dir / "result.json"
    if run.cached:
        data = json.loads(result_path.read_text())
        data["cached"] = True
        return DenseResult(**data)

    caps = executor.probe()
    if not caps.cuda_ok(COLMAP_MIN_VRAM_MB):
        vram_found = caps.gpu.vram_mb if caps.gpu else 0
        raise DenseUnavailable(
            f"Dense reconstruction unavailable on host '{caps.hostname}' "
            f"(reachable={caps.reachable}, VRAM found={vram_found} MB, needed={COLMAP_MIN_VRAM_MB} MB)"
        )

    out_dir = run.dir / "out"
    job = Job(
        command=[
            "sh", "-c",
            build_dense_script(
                images_dirname=images_dir.name,
                model_dirname=sparse_model_dir.name,
                colmap_cmd=f"{docker_prefix()} colmap" if use_docker else "colmap",
            ),
        ],
        inputs=[images_dir, sparse_model_dir],
        output_dir=out_dir,
        remote_subdir=run.key,
        timeout_s=timeout_s,
    )

    result = executor.run(job)
    if not result.ok:
        stderr_tail = (result.stderr_tail or "")[-500:]
        raise DenseUnavailable(
            f"Dense reconstruction job failed on host '{caps.hostname}': {stderr_tail}"
        )

    mesh_path = out_dir / "mesh.ply"
    if not mesh_path.exists():
        raise DenseUnavailable(
            f"Dense job reported success but produced no {mesh_path}; "
            "check the remote COLMAP install and the job's rsync'd outputs."
        )

    n_faces = None
    try:
        mesh = trimesh.load(str(mesh_path), process=False)
        if hasattr(mesh, "faces") and mesh.faces is not None:
            n_faces = int(len(mesh.faces))
    except Exception:
        n_faces = None

    res = DenseResult(
        mesh_path=str(mesh_path),
        n_faces=n_faces,
        cached=False,
    )

    result_path.write_text(json.dumps(asdict(res), indent=2) + "\n")
    workspace.finish_stage(run, asdict(res))

    return res
