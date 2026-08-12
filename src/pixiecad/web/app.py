"""FastAPI application for PixieCAD local web dashboard."""

from __future__ import annotations
from pixiecad.web.finishing import FinishOptions, finish_model

import dataclasses
import inspect
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..export import export_glb
from ..meshops import bake_object_space_normals, clean_mesh, decimate_to_budget, unwrap_uv
from ..session import new_session
from ..spec import Dimensions, ObjectSpec
from ..workspace import Workspace


class OptimizeRequest(BaseModel):
    target_faces: int = Field(..., gt=0)
    normal_res: int = Field(1024, ge=64, le=4096)


def _find_dense_ply(session_root: Path) -> Path | None:
    """Locate a job's dense mesh, wherever the producing stage left it.

    Dense output has landed in a few places across versions and backends
    (session root for a manual drop-in, work/ for the executor, work/ws/ for
    the stage cache), so all of them are searched rather than assuming one.
    """
    candidates = [
        session_root / "dense.ply",
        session_root / "work" / "dense.ply",
        session_root / "work" / "ws" / "dense.ply",
    ]
    return next((p for p in candidates if p.exists()), None)


def _merge_session_meta(session_root: Path, **fields: Any) -> None:
    """Update keys in a session.json without rewriting the rest of it.

    Deliberately NOT Session.write_meta: that rebuilds the file from a
    dataclass, so calling it with a freshly constructed Session would clobber
    everything already recorded -- the settings block and the generation seed
    among them. Best-effort by design; failing to record a face count must
    never fail a build that already succeeded.
    """
    meta_file = Path(session_root) / "session.json"
    try:
        meta = json.loads(meta_file.read_text())
        meta.update(fields)
        meta_file.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    except Exception:
        pass


def _find_reoptimise_source(session_root: Path) -> tuple[Path, bool] | None:
    """The highest-detail mesh a job kept, and whether it came from a backend.

    Re-optimising re-runs only the mesh tail -- clean, decimate, unwrap, bake,
    export -- at a new face budget. That is the cheap half of a build and it
    needs no GPU, so changing your mind about polygon count should cost
    seconds rather than another generation.

    It used to look for dense.ply alone, which only photogrammetry produces.
    Every generative job therefore failed with "Dense mesh (dense.ply) not
    found", which reads as a missing file when the truth was that the feature
    had never been taught about the backends -- and generative is the path
    almost every job actually takes.

    The generative equivalent is the raw mesh the backend returned, kept in
    the generate stage's output. It is the RIGHT source rather than merely an
    available one: re-decimating an already-decimated export would bake a
    normal map from geometry that had itself lost detail.

    Returns (path, is_generative), preferring a true dense reconstruction
    because it is the more accurate of the two when a job has both.
    """
    dense = _find_dense_ply(session_root)
    if dense is not None:
        return dense, False

    stages = session_root / "work" / "ws" / "stages"
    generated = sorted(stages.glob("s3-generate-*/out/mesh.glb")) if stages.is_dir() else []
    if generated:
        return generated[-1], True
    return None


def _call_build(**kwargs: Any) -> Any:
    import pixiecad.generative  # noqa: F401  (registers fake backend)
    from pixiecad.pipeline import run_build

    sig = inspect.signature(run_build)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return run_build(**filtered)


def _ingest_rejections(ws_root: Path) -> list[str]:
    """Per-photo reasons a photo was rejected, for the job log.

    The ingest stage records exactly why each photo failed, but the dashboard
    only ever showed the summary count -- "1 rejected" with no reason, which
    leaves the user guessing at resolution, blur, exposure or their own
    settings. The reasons already exist; they just were not surfaced.
    """
    reports = sorted(
        ws_root.glob("work/ws/stages/s0-ingest-*/report.json"),
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return []
    try:
        data = json.loads(reports[0].read_text())
    except Exception:
        return []

    lines: list[str] = []
    for photo in data.get("photos", []):
        if photo.get("status") == "rejected":
            name = Path(photo.get("source", "?")).name
            lines.append(f"  rejected {name}: {'; '.join(photo.get('reject_reasons', []))}")
        elif photo.get("status") == "unreadable":
            lines.append(f"  unreadable {Path(photo.get('source','?')).name}")
    lines.extend(f"  advice: {a}" for a in data.get("advice", []))
    return lines


def _rehydrate_jobs(root: Path) -> dict[str, dict[str, Any]]:
    """Rebuild the job registry from the session folders on disk.

    The in-memory registry dies with the process, but the sessions do not:
    every job's photos, workspace and finished model outlive any number of
    restarts. Without this, restarting the dashboard emptied /api/jobs, made
    every job URL a 404, and orphaned completed models that were sitting
    right there in output/ -- which makes deep links and "come back later"
    impossible to build on top of.

    A job that was mid-run when the process died is reported as failed rather
    than running: the thread that was doing the work is gone, so nothing will
    ever advance it, and showing a spinner that can never resolve is worse
    than an honest failure.
    """
    restored: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return restored

    for session_dir in sorted(root.iterdir()):
        if not session_dir.is_dir() or session_dir.is_symlink():
            continue
        meta_file = session_dir / "session.json"
        if not meta_file.is_file():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            continue

        # Sessions predating the dashboard have no job_id; the directory name
        # is unique and stable, so it serves as one rather than dropping the
        # session from the list entirely.
        job_id = str(meta.get("job_id") or session_dir.name)
        out_dir = session_dir / "output"
        # The newest model, not model.glb: a job whose last act was a
        # re-optimise must come back showing what it produced, and a job that
        # only ever produced a variant still counts as finished.
        glb = _current_model(out_dir)
        done = glb is not None

        parts: list[dict[str, Any]] = []
        parts_dir = out_dir / "parts"
        if parts_dir.is_dir():
            for part in sorted(parts_dir.glob("*.glb")):
                parts.append({
                    "name": part.stem,
                    "file": part.name,
                    "faces": None,
                    "url": f"/api/jobs/{job_id}/parts/{part.name}",
                })

        restored[job_id] = {
            "job_id": job_id,
            "name": meta.get("name") or meta.get("label") or session_dir.name,
            "target_faces": meta.get("target_faces"),
            # Restored so re-optimise survives a restart; absent for jobs
            # built before it was persisted, which the UI falls back for.
            "faces": meta.get("faces"),
            # What model.glb holds, once `faces` has moved on to a variant.
            # Absent for jobs re-optimised before it was recorded -- the
            # panel shows no count there rather than guessing one.
            "build_faces": meta.get("build_faces"),
            "status": "done" if done else "failed",
            "stage": "complete" if done else "failed",
            "log": [
                "Restored from disk after a dashboard restart."
                if done
                else "Restored from disk: this job was interrupted by a "
                "dashboard restart and did not finish.",
            ],
            "report": None,
            "glb_path": str(glb) if glb else None,
            "glb_url": _model_url(job_id, glb),
            "web_url": (
                f"/api/jobs/{job_id}/model_web.glb"
                if (out_dir / "model_web.glb").is_file()
                else None
            ),
            "dir": str(session_dir),
            "created_at": meta.get("created_at", ""),
            "regime": meta.get("regime"),
            "parts": parts,
            "parts_dir": str(parts_dir) if parts_dir.is_dir() else None,
            "warnings": meta.get("warnings") or [],
            "stages": meta.get("stages") or [],
            "restored": True,
            # Absent for sessions predating settings persistence; a rerun of
            # those falls back to defaults rather than refusing.
            "settings": meta.get("settings") or {},
        }

    return restored


def _latest_machine_image() -> str:
    """Newest baked worker image, or "" if none exists.

    Filtered by name prefix so an unrelated image in the project is never
    launched as a worker.
    """
    try:
        res = subprocess.run(
            ["gcloud", "compute", "machine-images", "list",
             "--filter=name~^pixiecad-worker",
             "--sort-by=~creationTimestamp", "--limit=1",
             "--format=value(name)"],
            capture_output=True, text=True, timeout=60,
        )
        return (res.stdout or "").strip().splitlines()[0] if (res.stdout or "").strip() else ""
    except Exception:
        return ""


def _actual_zone(name: str, requested: str, project: str) -> str:
    """Where the instance really is, which is not always where it was asked for.

    L4 capacity runs out per-zone, so launch_from_image.sh sweeps siblings on a
    stockout and the VM can land somewhere other than the requested zone. The
    ssh alias config-ssh writes is name.zone.project, so trusting the request
    builds an alias that does not resolve -- and the failure surfaces much
    later as "backend not available", pointing at the backend rather than at
    the zone.
    """
    try:
        res = subprocess.run(
            ["gcloud", "compute", "instances", "list",
             f"--filter=name={name}", "--format=value(zone)"]
            + ([f"--project={project}"] if project else []),
            capture_output=True, text=True, timeout=60,
        )
        found = (res.stdout or "").strip().splitlines()
        if found and found[0].strip():
            return found[0].strip()
    except Exception:
        pass
    return requested


def _gcp_project() -> str:
    """Project id from the environment, falling back to the gcloud config."""
    project = os.environ.get("PIXIECAD_GCP_PROJECT", "").strip()
    if project:
        return project
    try:
        res = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, timeout=30,
        )
        value = (res.stdout or "").strip()
        return "" if value in {"", "(unset)"} else value
    except Exception:
        return ""


# The only tags the generative backend understands.
VIEW_TAG_CHOICES = ("front", "back", "left", "right")


class ConvertRequest(BaseModel):
    """GLB -> STL/OBJ/PLY for CAD and slicers."""

    # Empty means the CURRENT model, which is what someone pressing "download
    # STL" means -- naming model.glb explicitly still gets model.glb, so a
    # caller that wants the build's own geometry can still ask for it.
    source: str = ""
    format: str = "stl"
    size_mm: float | None = None
    repair: bool = False


# ── Model variants ─────────────────────────────────────────────────────────
#
# A job keeps every model it has produced: output/model.glb from the build,
# plus one model-<faces>.glb per re-optimise.
#
# Re-optimise used to rewrite model.glb in place, which destroyed the only
# copy of the geometry the build shipped -- trying a smaller budget was a
# one-way door. It also made a working re-optimise indistinguishable from one
# that did nothing: the URL was a constant, so the browser served its cached
# copy, React never remounted the viewer, and the picture on screen did not
# move however far the slider did.
MODEL_VARIANT = re.compile(r"^model-(\d+)\.glb$")


def _variant_name(faces: int) -> str:
    return f"model-{faces}.glb"


def _model_files(out_dir: Path) -> list[Path]:
    """Every model this job holds, oldest first.

    Ordered by mtime because that is the only record of which one is current
    that survives a dashboard restart -- the in-memory job registry does not,
    and the filenames carry a face count rather than a sequence number.
    """
    if not out_dir.is_dir():
        return []
    found = [p for p in out_dir.glob("model-*.glb") if MODEL_VARIANT.match(p.name)]
    original = out_dir / "model.glb"
    if original.is_file():
        found.append(original)
    found.sort(key=lambda p: p.stat().st_mtime)
    return found


def _current_model(out_dir: Path) -> Path | None:
    """The newest model: what the viewer shows and conversions start from."""
    found = _model_files(out_dir)
    return found[-1] if found else None


def _model_url(job_id: str, path: Path | None) -> str | None:
    """The current model's URL, versioned by its mtime.

    The version is the whole point. Without it the URL never changes, so
    re-optimising swapped the bytes behind a constant address and neither the
    browser nor React had any reason to notice.
    """
    if path is None or not path.is_file():
        return None
    return f"/api/jobs/{job_id}/model.glb?v={int(path.stat().st_mtime)}"


def _export_stem(job_name: str, model_name: str) -> str:
    """What a CAD conversion of ``model_name`` is called.

    Named after the job so the file that lands in Downloads says what it is,
    but suffixed with the variant's face count -- otherwise converting a
    re-optimised model silently overwrites the STL of the one it came from.
    """
    stem = Path(job_name or "model").stem
    match = MODEL_VARIANT.match(model_name)
    return f"{stem}-{match.group(1)}" if match else stem


def _model_list(
    job_id: str,
    out_dir: Path,
    *,
    job_name: str = "",
    build_faces: int | None = None,
    current_faces: int | None = None,
) -> list[dict[str, Any]]:
    """The job's models as the UI needs them: what each is, and which is live."""
    files = _model_files(out_dir)
    if not files:
        return []
    current = files[-1].name
    listed: list[dict[str, Any]] = []
    for path in files:
        match = MODEL_VARIANT.match(path.name)
        if match:
            faces: int | None = int(match.group(1))
        elif build_faces is not None:
            faces = build_faces
        elif path.name == current:
            # Nothing has superseded model.glb, so the job's face count is
            # still describing it.
            faces = current_faces
        else:
            # A job re-optimised before the build count was recorded. Saying
            # nothing beats attributing the variant's count to it.
            faces = None
        listed.append({
            "name": path.name,
            "faces": faces,
            "bytes": path.stat().st_size,
            "url": f"/api/jobs/{job_id}/download/{path.name}",
            "original": match is None,
            "current": path.name == current,
            "basename": _export_stem(job_name, path.name),
        })
    return listed


def _safe_output_file(job_dir: Path, relative: str) -> Path:
    """Resolve ``relative`` inside a job's output dir, refusing escapes.

    Filenames reach this from the browser, so a '../../etc/passwd' must not
    resolve outside the job. resolve() then a prefix check is the whole guard.
    """
    out_dir = (Path(job_dir) / "output").resolve()
    target = (out_dir / relative).resolve()
    if not str(target).startswith(str(out_dir) + os.sep) and target != out_dir:
        raise HTTPException(status_code=400, detail="path outside the job directory")
    return target


class ProvisionRequest(BaseModel):
    gpu: str = "l4"
    name: str = "pixiecad-gpu"
    zone: str = "asia-southeast1-b"
    texture: bool = True
    semantic: bool = True


class TeardownRequest(BaseModel):
    name: str = "pixiecad-gpu"
    zone: str = "asia-southeast1-b"


def create_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="PixieCAD Dashboard")
    lock = threading.Lock()

    # Jobs seen this process, seeded from whatever is already on disk so a
    # restart does not orphan finished work.
    jobs: dict[str, dict[str, Any]] = _rehydrate_jobs(root)
    # Published on app.state as well: it is otherwise a closure local, and the
    # one-job-per-GPU-host guard cannot be tested without being able to put a
    # job into the registry without running one. Same dict, not a copy.
    app.state.jobs = jobs
    # Single in-flight provision: two concurrent builds on one project would
    # race for the same L4 quota of 1 and both fail confusingly.
    provisioning: dict[str, Any] = {}

    # The React bundle. Built by `npm run build` in frontend/, which writes
    # here. Absent in a source checkout that has never run the build, so every
    # use of it is guarded rather than assumed.
    spa_dir = Path(__file__).parent / "static" / "dist"

    def _run_mesh_stages_sync(
        job_id: str, ws_root: Path, target_faces: int, normal_res: int = 1024
    ) -> None:
        found = _find_reoptimise_source(ws_root)
        if found is None:
            raise FileNotFoundError("no dense.ply or generate-stage mesh for this job")
        source_path, is_generative = found

        with lock:
            job = jobs.get(job_id)
            if job:
                job["status"] = "running"
                job["stage"] = "clean"
                job["log"].append("Cleaning dense mesh...")

        dense_mesh = trimesh.load(source_path, force="mesh", process=False)

        if is_generative:
            # Same repair the build does, and for the same reason: a shattered
            # backend surface has to be closed BEFORE decimation, because
            # decimating confetti only rearranges the fragments. solidify
            # self-gates on fill, so a Hunyuan mesh that arrives solid is left
            # untouched and this costs nothing.
            from ..meshops.solidify import solidify

            with lock:
                job = jobs.get(job_id)
                if job:
                    job["stage"] = "solidify"
                    job["log"].append("Rebuilding the surface as a solid...")
            try:
                outcome = solidify(dense_mesh)
                if outcome.applied:
                    dense_mesh = outcome.mesh
                with lock:
                    job = jobs.get(job_id)
                    if job:
                        job["log"].append(outcome.reason)
            except Exception as exc:  # never fatal: an unrepaired mesh is a mesh
                with lock:
                    job = jobs.get(job_id)
                    if job:
                        job["log"].append(f"solidify skipped: {exc}")

        cleaned_mesh, _ = clean_mesh(dense_mesh)

        with lock:
            job = jobs.get(job_id)
            if job:
                job["stage"] = "decimate"
                job["log"].append(f"Decimating mesh to {target_faces} faces...")

        decimated_mesh, _ = decimate_to_budget(cleaned_mesh, target_faces=target_faces)

        with lock:
            job = jobs.get(job_id)
            if job:
                job["stage"] = "unwrap"
                job["log"].append("Unwrapping UVs...")

        unwrap_res = unwrap_uv(decimated_mesh)

        with lock:
            job = jobs.get(job_id)
            if job:
                job["stage"] = "bake"
                job["log"].append("Baking object-space normal map...")

        normal_map = bake_object_space_normals(
            dense_mesh, unwrap_res.mesh, resolution=normal_res
        )

        with lock:
            job = jobs.get(job_id)
            if job:
                job["stage"] = "export"
                job["log"].append("Exporting GLB model...")

        out_dir = ws_root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        faces = len(unwrap_res.mesh.faces)
        # A NEW file rather than a rewrite of model.glb. The build's own model
        # is the one thing a re-optimise cannot reproduce -- it would need the
        # GPU again -- so overwriting it threw away the only copy every time
        # someone tried a smaller budget. Re-optimising twice to the same
        # budget lands on the same name, which is the right thing: that is the
        # same model, not a third one.
        glb_path = out_dir / _variant_name(faces)
        export_glb(unwrap_res.mesh, glb_path, normal_map=normal_map)

        with lock:
            job = jobs.get(job_id)
            if job:
                job["status"] = "done"
                job["stage"] = "complete"
                # The count model.glb has. Captured before `faces` below stops
                # describing it and starts describing whichever variant is
                # current, so the Artifacts panel can still label the original.
                if job.get("build_faces") is None and job.get("faces") is not None:
                    job["build_faces"] = job["faces"]
                # Record where we actually wrote it: the download endpoint
                # serves glb_path, and re-optimising must repoint it rather
                # than leave a stale path from an earlier build.
                job["glb_path"] = str(glb_path)
                job["glb_url"] = _model_url(job_id, glb_path)
                # The whole point of re-optimising is a different face count,
                # so the job has to stop reporting the old one -- the sidebar,
                # the re-optimise slider's own starting value and the response
                # all read it.
                job["faces"] = faces
                meta_fields: dict[str, Any] = {"faces": faces}
                if job.get("build_faces") is not None:
                    meta_fields["build_faces"] = job["build_faces"]
                _merge_session_meta(ws_root, **meta_fields)

        with lock:
            job = jobs.get(job_id)
            if job:
                kept = [p.name for p in _model_files(out_dir)]
                if len(kept) > 1:
                    job["log"].append(
                        "Models kept for this job: " + ", ".join(kept) + "."
                    )
                if (out_dir / "parts").is_dir():
                    job["log"].append(
                        "NOTE: the exported parts still come from the build's "
                        "geometry; re-run the build to re-split at this budget."
                    )
                job["log"].append(
                    f"GLB model export complete: {faces:,} faces, "
                    f"{glb_path.stat().st_size / 1e6:.1f} MB "
                    f"({glb_path.name})."
                )

    def _job_worker(
        job_id: str,
        ws_root: Path,
        target_faces: int,
        backend: str | None = None,
        split: bool = True,
        object_hint: str | None = None,
        finish: FinishOptions | None = None,
        generative_options: dict[str, Any] | None = None,
        multiview: bool = False,
        seed: int | None = None,
    ) -> None:
        with lock:
            job = jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["stage"] = "build"
            job["log"].append("Starting full pipeline build...")

        try:
            photos_dir = ws_root / "input"
            ws_dir = ws_root / "work" / "ws"
            # The backend reads this at generate time; set per job rather than
            # at import so two jobs cannot disagree about it.
            os.environ["PIXIECAD_HUNYUAN_MULTIVIEW"] = "1" if multiview else "0"

            # The GPU host is what makes a real generative backend exist at
            # all: run_build registers hunyuan-remote only when it is handed
            # an executor. Without this the pipeline fell back to
            # photogrammetry, which needs COLMAP locally, and every job died
            # straight after ingest with the photos perfectly fine.
            executor = None
            if finish and finish.gpu_host:
                from pixiecad.executors import SSHExecutor

                executor = SSHExecutor(finish.gpu_host)
                with lock:
                    current = jobs.get(job_id)
                    if current:
                        current["log"].append(
                            f"Using GPU host {finish.gpu_host} for generation."
                        )
            else:
                with lock:
                    current = jobs.get(job_id)
                    if current:
                        current["log"].append(
                            "No GPU host set: generation will need local COLMAP "
                            "and 16+ orbit photos. Set a GPU host to generate."
                        )

            result = _call_build(
                photos_dir=photos_dir,
                workspace=ws_dir,
                dense=False,
                bake=finish.bake_normals if finish else False,
                normal_res=finish.normal_res if finish else 1024,
                # "auto" only becomes "host" when there is actually a host to
                # send it to; without an executor the pipeline would have no
                # way to run it there and would silently bake locally anyway.
                bake_location=(
                    "host"
                    if finish
                    and executor is not None
                    and finish.bake_location in ("auto", "host")
                    else "local"
                ),
                executor=executor,
                generative_backend=backend,
                split=split,
                object_hint=object_hint,
                generative_options=generative_options,
                seed=seed,
            )

            regime_val = (
                result.regime.value
                if hasattr(result.regime, "value")
                else (str(result.regime) if getattr(result, "regime", None) else None)
            )
            # The pipeline already records every stage with its status and a
            # real elapsed time. That was being flattened into log strings and
            # regex-parsed back apart in the browser; keeping the structure
            # means the dashboard can show measured timings instead of guesses.
            stage_outcomes = [
                {
                    "name": s.name,
                    "status": s.status,
                    "detail": s.detail,
                    "seconds": round(s.seconds, 3),
                }
                for s in getattr(result, "stages", []) or []
            ]
            # Published the moment the build reports, NOT at the end of the
            # job. Finishing -- smoothing, texturing on the GPU, segmentation
            # -- runs for minutes after this point, and holding the stage list
            # until then left the dashboard with nothing to show for the whole
            # of it. The summary lines are published here for the same reason:
            # they describe work that is already finished.
            with lock:
                current = jobs.get(job_id)
                if current:
                    current["stages"] = stage_outcomes
                    for line in (
                        result.summary_lines() if hasattr(result, "summary_lines") else []
                    ):
                        current["log"].append(line)
                    for line in _ingest_rejections(ws_root):
                        current["log"].append(line)

            glb_path = getattr(result, "glb_path", None)
            faces = getattr(result, "faces", None)
            parts_raw = getattr(result, "parts", None)
            parts_dir = getattr(result, "parts_dir", None)
            warnings = getattr(result, "warnings", None) or []

            # If split requested and run_build didn't compute parts, compute them on output mesh
            if split and (parts_raw is None or len(parts_raw) == 0) and glb_path and Path(glb_path).exists():
                try:
                    from pixiecad.parts import export_parts, split_parts

                    loaded_mesh = trimesh.load(glb_path, force="mesh", process=False)
                    if isinstance(loaded_mesh, trimesh.Scene):
                        loaded_mesh = loaded_mesh.to_mesh()
                    parts_objs = split_parts(loaded_mesh, method="auto", max_parts=8)
                    p_dir = Path(glb_path).parent / "parts"
                    exp_parts, _ = export_parts(
                        parts_objs,
                        p_dir,
                        total_budget=faces or 20000,
                    )
                    parts_raw = [dataclasses.asdict(p) for p in exp_parts]
                    parts_dir = str(p_dir)
                except Exception:
                    parts_raw = []

            parts = []
            if parts_raw:
                for p in parts_raw:
                    p_dict = dict(p) if isinstance(p, dict) else dataclasses.asdict(p)
                    p_file = p_dict.get("file") or p_dict.get("name")
                    p_dict["url"] = f"/api/jobs/{job_id}/parts/{p_file}"
                    parts.append(p_dict)

            # Publish the deliverables into the session's output/ folder. The
            # workspace under work/ is scratch — the user should be able to
            # delete it and still have the model and its parts.
            if glb_path and Path(glb_path).exists():
                out_dir = ws_root / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                published = out_dir / "model.glb"
                shutil.copy2(glb_path, published)
                glb_path = str(published)
                if parts_dir and Path(parts_dir).is_dir():
                    published_parts = out_dir / "parts"
                    shutil.copytree(parts_dir, published_parts, dirs_exist_ok=True)
                    parts_dir = str(published_parts)

            # Finishing runs on the published copy in output/, so the
            # workspace under work/ stays pure build scratch.
            if finish and glb_path and Path(glb_path).exists():
                def _log(message: str) -> None:
                    with lock:
                        current = jobs.get(job_id)
                        if current:
                            current["log"].append(message)

                with lock:
                    current = jobs.get(job_id)
                    if current:
                        current["stage"] = "finishing"

                photos = sorted(
                    q for q in (ws_root / "input").glob("*")
                    if q.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                )
                fin = finish_model(
                    glb_path,
                    Path(glb_path).parent,
                    finish,
                    conditioning_image=photos[0] if photos else None,
                    log=_log,
                )
                warnings = list(warnings) + fin.warnings
                if fin.parts:
                    parts_raw = fin.parts
                    parts_dir = str(fin.parts_dir) if fin.parts_dir else parts_dir
                    parts = []
                    for p_dict in parts_raw:
                        entry = dict(p_dict)
                        entry["url"] = f"/api/jobs/{job_id}/parts/{entry.get('file')}"
                        parts.append(entry)
                web_model = Path(glb_path).parent / "model_web.glb"
                with lock:
                    current = jobs.get(job_id)
                    if current:
                        current["finish_steps"] = fin.steps
                        current["textured"] = fin.textured
                        if web_model.exists():
                            current["web_url"] = f"/api/jobs/{job_id}/model_web.glb"

            # The build's own summary and the per-photo ingest rejections were
            # already logged the moment the build reported, above.
            with lock:
                job = jobs.get(job_id)
                if job:
                    for w in warnings:
                        job["log"].append(f"WARNING: {w}")

                    job["regime"] = regime_val
                    job["glb_path"] = glb_path
                    job["faces"] = faces
            # Outside the lock: this touches the disk. The face count lived
            # only in memory, so every dashboard restart lost it -- and the
            # re-optimise control is gated on it, which meant re-optimising
            # became impossible for every past job the moment the server was
            # restarted, with a finished model sitting right there.
            _merge_session_meta(
                ws_root,
                faces=faces,
                regime=regime_val,
                warnings=warnings,
                stages=stage_outcomes,
            )
            with lock:
                job = jobs.get(job_id)
                if job:
                    job["parts"] = parts
                    job["parts_dir"] = str(parts_dir) if parts_dir else None
                    job["warnings"] = warnings
                    job["stages"] = stage_outcomes

                    if glb_path:
                        job["status"] = "done"
                        job["stage"] = "complete"
                        job["glb_url"] = f"/api/jobs/{job_id}/model.glb"
                    else:
                        job["status"] = "failed"
                        job["stage"] = "failed"
        except Exception as exc:
            with lock:
                job = jobs.get(job_id)
                if job:
                    job["status"] = "failed"
                    job["stage"] = "failed"
                    job["log"].append(f"Pipeline failed: {exc}")

    def _dispatch_when_gpu_ready(
        job_id: str, finish: FinishOptions, worker_args: tuple[Any, ...]
    ) -> None:
        """Hold a queued job until the in-flight provision finishes.

        Polls the same provisioning record the dashboard shows, so the job and
        the UI can never disagree about why it is waiting. A cold image build
        is roughly thirty minutes, so the ceiling is generous; passing it means
        something is wrong and the job is failed rather than left hanging.
        """
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            with lock:
                status = provisioning.get("status")
                host = provisioning.get("host") or ""
                job = jobs.get(job_id)
            if job is None:
                return  # deleted while waiting

            if status == "ready":
                # The freshly built host is what the job was waiting for, so
                # it wins over whatever was in the form when it was submitted.
                if host:
                    finish.gpu_host = host
                with lock:
                    current = jobs.get(job_id)
                    if current:
                        current["log"].append(f"GPU ready at {host}; starting.")
                _job_worker(*worker_args)
                return

            if status in {"failed", None}:
                with lock:
                    current = jobs.get(job_id)
                    if current:
                        current["status"] = "failed"
                        current["stage"] = "failed"
                        current["log"].append(
                            "GPU provisioning failed, so this job never started. "
                            "Provision a VM from Compute and submit again."
                        )
                return

            time.sleep(5)

        with lock:
            current = jobs.get(job_id)
            if current:
                current["status"] = "failed"
                current["stage"] = "failed"
                current["log"].append(
                    "Gave up waiting for the GPU VM after an hour."
                )

    @app.get("/")
    def index():
        """The dashboard.

        Redirects into the React app rather than serving it here, so there is
        exactly one URL space: the bundle's assets and the client router both
        live under /ui, and a job link is the same string wherever it came
        from.
        """
        if (spa_dir / "index.html").is_file():
            return RedirectResponse("/ui/", status_code=307)
        # There is no longer a server-rendered fallback, so say what to do
        # rather than serving a blank page to someone who just cloned this.
        raise HTTPException(
            status_code=503,
            detail="dashboard bundle not built; run `npm run build` in frontend/",
        )

    # ── The React SPA ────────────────────────────────────────────────────
    # Mounted at /ui, with / redirecting there, so the bundle's assets and the
    # client router share one URL space and a job link is the same string
    # wherever it came from.
    if (spa_dir / "index.html").is_file():
        app.mount(
            "/ui/assets",
            StaticFiles(directory=spa_dir / "assets"),
            name="spa-assets",
        )

        @app.get("/ui", response_class=HTMLResponse)
        @app.get("/ui/{spa_path:path}", response_class=HTMLResponse)
        def spa(spa_path: str = ""):
            """Serve the SPA shell for any /ui path.

            Client-side routes like /ui/jobs/abc123 do not exist on disk, so a
            hard reload or a pasted link would 404 without this. Deep links
            surviving a reload is the whole point of moving job state into the
            URL, so the fallback is not optional.
            """
            # A real file under the bundle (favicon, manifest, source map) is
            # served as itself; only unknown paths fall through to the shell.
            if spa_path:
                candidate = (spa_dir / spa_path).resolve()
                if (
                    str(candidate).startswith(str(spa_dir.resolve()) + os.sep)
                    and candidate.is_file()
                ):
                    return FileResponse(candidate)
            return (spa_dir / "index.html").read_text(encoding="utf-8")

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile] = File(default=[]),
        name: str = Form("object"),
        target_faces: int = Form(20000),
        length: str | None = Form(None),
        width: str | None = Form(None),
        height: str | None = Form(None),
        split: bool = Form(True),
        object_hint: str | None = Form(None),
        backend: str | None = Form(None),
        smooth_iterations: int = Form(0),
        texture: bool = Form(False),
        segmentation: str = Form("auto"),
        web_export: bool = Form(False),
        texture_size: int = Form(1024),
        max_parts: int = Form(8),
        gpu_host: str | None = Form(None),
        octree_resolution: int = Form(256),
        view_tags: str | None = Form(None),
        multiview: bool = Form(False),
        bake_normals: bool = Form(True),
        normal_res: int = Form(1024),
        bake_location: str = Form("auto"),
        seed: int | None = Form(None),
    ):
        import random
        seed_val = seed if seed is not None else random.randint(0, 2**31 - 1)

        # One session folder per upload: input/, work/ and output/ are never
        # shared between jobs, so two people uploading at once cannot read or
        # overwrite each other's photos and meshes.
        session = new_session(root, label=name or "object")
        job_id = uuid4().hex[:8]
        job_dir = session.root
        photos_dir = session.input_dir

        # View tags are applied by *renaming* on the way in. The generative
        # backend maps photos to front/back/left/right by filename, and ingest
        # preserves the stem into its working copy, so a file saved as
        # "front.jpg" is picked up as the front view with no other plumbing.
        # That is what lets an arbitrary "WhatsApp Image ....jpeg" participate
        # in multi-view at all -- unrenamed, it maps to nothing and the run
        # silently falls back to a single view.
        try:
            tags = json.loads(view_tags) if view_tags else {}
        except json.JSONDecodeError:
            tags = {}
        if not isinstance(tags, dict):
            tags = {}

        used_tags: set[str] = set()
        for f in files:
            fname = f.filename or f"photo_{uuid4().hex[:4]}.jpg"
            tag = str(tags.get(fname, "") or "").strip().lower()
            if tag in VIEW_TAG_CHOICES and tag not in used_tags:
                used_tags.add(tag)
                stem = tag
            else:
                stem = Path(fname).stem
            dest = photos_dir / f"{stem}{Path(fname).suffix or '.jpg'}"
            n = 2
            while dest.exists():
                dest = photos_dir / f"{stem}-{n}{Path(fname).suffix or '.jpg'}"
                n += 1
            with dest.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)

        # ObjectSpec enforces target_faces > 0, but it raises a pydantic
        # ValidationError deep inside create_job -- a 500 and a traceback for
        # what is plainly a bad request. The ceiling matters too: the budget
        # only ever shrinks a generated mesh, and the largest any backend
        # returns is TRELLIS at ~1.9M, so anything past 2M is a typo that would
        # otherwise be accepted and quietly do nothing.
        if target_faces < 100 or target_faces > 2_000_000:
            raise HTTPException(
                422,
                f"target_faces must be between 100 and 2,000,000 (got {target_faces})",
            )

        l_val = length.strip() if length and length.strip() else None
        w_val = width.strip() if width and width.strip() else None
        h_val = height.strip() if height and height.strip() else None
        dims = Dimensions.parse(length=l_val, width=w_val, height=h_val)
        spec = ObjectSpec(name=name or "object", target_faces=target_faces, dimensions=dims)

        ws_dir = session.work_dir / "ws"
        Workspace.create(ws_dir, spec)

        # "auto" is the dashboard's word for "you choose"; the pipeline spells
        # that None. Without this the literal string reaches get_backend and
        # fails as an unregistered backend name.
        backend_val = (backend or "").strip() or None
        if backend_val == "auto":
            backend_val = None
        hint_val = object_hint.strip() if object_hint and object_hint.strip() else None

        # Clamp to a range the dev machine can handle: baking runs locally,
        # and an unclamped 4096 from a hand-crafted request would exhaust the
        # 16 GB MacBook Air that is the supported dev target.
        normal_res = max(256, min(2048, normal_res))

        finish = FinishOptions(
            smooth_iterations=max(0, smooth_iterations),
            texture=texture,
            segmentation=segmentation,
            web_export=web_export,
            texture_size=texture_size,
            max_parts=max(1, max_parts),
            total_budget=target_faces,
            gpu_host=(gpu_host.strip() or None) if gpu_host else None,
            bake_normals=bake_normals,
            normal_res=normal_res,
            bake_location=(
                bake_location if bake_location in ("auto", "local", "host") else "auto"
            ),
        )

        # Geometric detail of the generative decode. This is the lever that
        # actually separates parts that sit close together -- raising it makes
        # the model resolve a gap the decoder would otherwise bridge -- and it
        # is independent of the polygon budget, which only controls how the
        # result is simplified afterwards.
        gen_options = {"octree_resolution": max(64, octree_resolution)}
        # Needs at least two tagged cardinal views to mean anything; below
        # that the backend falls back to single-view regardless.
        use_multiview = multiview and len(used_tags) >= 2

        # Everything a rerun needs, on disk rather than only in memory. The
        # in-memory record dies with the process, but a job that failed on
        # infrastructure -- an unreachable host, a missing module in the worker
        # image -- is exactly the one you want to retry after a restart, and
        # re-picking a dozen options by hand is not a retry.
        settings = {
            "split": split,
            "object_hint": hint_val,
            "backend": backend_val,
            "length": length,
            "width": width,
            "height": height,
            "octree_resolution": gen_options["octree_resolution"],
            "multiview": use_multiview,
            "seed": seed_val,
            **dataclasses.asdict(finish),
        }
        session.write_meta(
            job_id=job_id,
            name=spec.name,
            target_faces=target_faces,
            settings=settings,
        )

        job_info: dict[str, Any] = {
            "job_id": job_id,
            "name": spec.name,
            "target_faces": target_faces,
            "status": "queued",
            "stage": "queued",
            "log": ["Job initialized."],
            "report": None,
            "glb_url": None,
            "dir": str(job_dir),
            "regime": None,
            "faces": None,
            "parts": [],
            "parts_dir": None,
            "warnings": [],
            "stages": [],
            "created_at": session.created_at,
            "split": split,
            "object_hint": hint_val,
            "backend": backend_val,
            "settings": settings,
            "finish": {
                "smooth_iterations": finish.smooth_iterations,
                "texture": finish.texture,
                "segmentation": finish.segmentation,
                "web_export": finish.web_export,
                "texture_size": finish.texture_size,
                "gpu_host": finish.gpu_host,
                "bake_normals": finish.bake_normals,
                "normal_res": finish.normal_res,
                "bake_location": finish.bake_location,
            },
            "web_url": None,
        }

        return _register_and_dispatch(
            job_info, job_dir, target_faces, backend_val, split, hint_val,
            finish, gen_options, use_multiview, seed_val,
        )

    def _register_and_dispatch(
        job_info: dict[str, Any],
        job_dir: Path,
        target_faces: int,
        backend_val: str | None,
        split: bool,
        hint_val: str | None,
        finish: FinishOptions,
        gen_options: dict[str, Any],
        use_multiview: bool,
        seed: int,
    ) -> dict[str, Any]:
        """Publish a job and start it, honouring the GPU-provisioning gate.

        Shared by create_job and rerun so the two cannot drift. The gate in
        particular is not something to reimplement: a job that needs a GPU and
        is submitted while one is still building has to wait, or it dies on a
        missing docker image.
        """
        job_id = job_info["job_id"]
        host = (job_info.get("settings") or {}).get("gpu_host") or None
        with lock:
            # One GPU host serves one job at a time. The TRELLIS worker holds a
            # single pipeline on a single device, and sending it a second
            # generation does not queue -- it DEADLOCKS. Observed: two jobs
            # submitted 55 s apart both reached "Sampling sparse structure:
            # 100%" and stopped dead, /ready stopped answering entirely, the
            # GPU sat at 0% and the process at 0.23 s of CPU per 40 s. Nothing
            # recovers that but killing the container, and both runs are lost.
            #
            # Refusing up front costs the user a message; accepting costs them
            # both jobs and the GPU minutes already spent. Checked inside the
            # lock and in the same critical section as registration, so two
            # near-simultaneous submissions cannot both pass.
            if host:
                busy = next(
                    (
                        jid
                        for jid, other in jobs.items()
                        if jid != job_id
                        and other.get("status") in {"running", "queued"}
                        and ((other.get("settings") or {}).get("gpu_host") or None) == host
                    ),
                    None,
                )
                if busy is not None:
                    name = jobs[busy].get("name") or busy
                    raise HTTPException(
                        409,
                        f"GPU host {host} is already running job '{name}' ({busy}). "
                        "It serves one job at a time -- a second one deadlocks the "
                        "worker and loses both. Wait for it to finish, or cancel it.",
                    )
            jobs[job_id] = job_info
            provision_running = provisioning.get("status") == "running"

        worker_args = (job_id, job_dir, target_faces, backend_val, split, hint_val,
                       finish, gen_options, use_multiview, seed)

        # A job that needs a GPU, submitted while one is still being built,
        # used to be accepted and then die on a missing docker image -- the
        # only warning being a line of grey text far from the button. Holding
        # it until the VM is ready turns a confusing failure into a wait, and
        # costs the user nothing they were not already spending.
        if provision_running and finish.needs_gpu:
            with lock:
                jobs[job_id]["status"] = "queued"
                jobs[job_id]["stage"] = "waiting for GPU"
                jobs[job_id]["log"].append(
                    "Waiting for the GPU VM to finish building before starting."
                )
            threading.Thread(
                target=_dispatch_when_gpu_ready,
                args=(job_id, finish, worker_args),
                daemon=True,
            ).start()
            return {"job_id": job_id, "status": "queued"}

        threading.Thread(target=_job_worker, args=worker_args, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}/rerun-plan")
    def rerun_plan(job_id: str):
        """What a rerun of this job WOULD use, without starting anything.

        Read-only on purpose. The rerun button used to fire immediately, which
        silently reused settings the user could not see and had no chance to
        change -- fine when the last run failed on infrastructure, wrong when
        they wanted to adjust something. This is what the confirmation screen
        reads.
        """
        with lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "job not found")
            settings = dict(job.get("settings") or {})
            name = job.get("name") or "object"
            target_faces = int(job.get("target_faces") or 20000)
            src_dir = Path(job["dir"])
            status = job.get("status")

        photos_dir = src_dir / "input"
        photos = (
            sorted(p.name for p in photos_dir.iterdir() if p.is_file())
            if photos_dir.is_dir()
            else []
        )
        return {
            "job_id": job_id,
            "name": name,
            "status": status,
            "target_faces": target_faces,
            "photos": photos,
            "photo_count": len(photos),
            # False when the photos are gone: the screen can say so up front
            # rather than letting someone fill in a form that cannot submit.
            "can_rerun": bool(photos) and status not in ("running", "queued"),
            "settings": settings,
            # Sessions written before settings were persisted have none, so the
            # form falls back to its own defaults and must say it is doing that.
            "has_settings": bool(settings),
        }

    @app.post("/api/jobs/{job_id}/rerun")
    def rerun_job(
        job_id: str,
        gpu_host: str | None = Form(None),
        # Every setting is an optional override. Absent means "keep what the
        # original used", so a confirmation screen can send back only what the
        # user actually changed, and a bare POST still reproduces the old run.
        name: str | None = Form(None),
        target_faces: int | None = Form(None),
        split: bool | None = Form(None),
        object_hint: str | None = Form(None),
        backend: str | None = Form(None),
        length: str | None = Form(None),
        width: str | None = Form(None),
        height: str | None = Form(None),
        smooth_iterations: int | None = Form(None),
        texture: bool | None = Form(None),
        segmentation: str | None = Form(None),
        web_export: bool | None = Form(None),
        texture_size: int | None = Form(None),
        max_parts: int | None = Form(None),
        octree_resolution: int | None = Form(None),
        multiview: bool | None = Form(None),
        bake_normals: bool | None = Form(None),
        normal_res: int | None = Form(None),
        bake_location: str | None = Form(None),
        seed: int | None = Form(None),
    ):
        """Start a fresh job from an existing one's photos and settings.

        A new session rather than a reset of the old one. The failed run's log
        and stage timings are the evidence for why it failed, and overwriting
        them to retry would destroy exactly what you would want to compare
        against. Photos are copied, as they are for any upload -- the input
        folder is the session's own, so the original stays intact.

        The common case this exists for is infrastructure, not input: an
        unreachable GPU host, a module missing from the worker image. Nothing
        about the photos or the options was wrong, so re-entering them by hand
        is pure toil.
        """
        with lock:
            original = jobs.get(job_id)
            if original is None:
                raise HTTPException(404, "job not found")
            if original.get("status") in ("running", "queued"):
                raise HTTPException(
                    409, "this job is still running; wait for it to finish or fail"
                )
            src_dir = Path(original["dir"])
            settings = dict(original.get("settings") or {})
            name = (name or "").strip() or original.get("name") or "object"
            target_faces = int(
                target_faces if target_faces is not None
                else (original.get("target_faces") or 20000)
            )

        # Overrides land on top of the stored settings. None means untouched --
        # distinct from an empty string, which is a real value for the text
        # fields (clearing the object hint, say).
        overrides = {
            "split": split, "object_hint": object_hint,
            "backend": backend, "length": length, "width": width,
            "height": height, "smooth_iterations": smooth_iterations,
            "texture": texture, "segmentation": segmentation,
            "web_export": web_export, "texture_size": texture_size,
            "max_parts": max_parts, "octree_resolution": octree_resolution,
            "multiview": multiview, "bake_normals": bake_normals,
            "normal_res": normal_res, "bake_location": bake_location,
        }
        settings.update({k: v for k, v in overrides.items() if v is not None})
        if backend is not None:
            # "auto" is the dashboard's word for "you choose"; the pipeline
            # spells that None, and the literal string is not a registered
            # backend name.
            settings["backend"] = None if backend.strip() in ("", "auto") else backend.strip()
        if object_hint is not None:
            settings["object_hint"] = object_hint.strip() or None

        if seed is not None:
            settings["seed"] = seed
        elif "seed" not in settings or settings["seed"] is None:
            import random
            settings["seed"] = random.randint(0, 2**31 - 1)
        seed_val = settings["seed"]

        with lock:
            if job_id not in jobs:
                raise HTTPException(404, "job not found")

        src_photos = src_dir / "input"
        if not src_photos.is_dir() or not any(src_photos.iterdir()):
            raise HTTPException(
                409,
                "the original job's photos are gone, so it cannot be rerun; "
                "start a new job instead",
            )

        session = new_session(root, label=name)
        new_id = uuid4().hex[:8]
        # stage_inputs copies, and the staged names already carry any view tags
        # applied at upload, so tagging survives the rerun untouched.
        staged = session.stage_inputs(src_photos)

        # Declared dimensions are the only scale reference a generative mesh
        # has, so dropping them on a rerun would silently change the result.
        dims = Dimensions.parse(
            length=(settings.get("length") or None),
            width=(settings.get("width") or None),
            height=(settings.get("height") or None),
        )
        spec = ObjectSpec(name=name, target_faces=target_faces, dimensions=dims)
        Workspace.create(session.work_dir / "ws", spec)

        # A host may have been provisioned since the original failed -- that is
        # frequently the whole reason for the retry -- so an explicitly supplied
        # one wins over the stale value recorded then.
        host = (gpu_host or "").strip() or settings.get("gpu_host") or None
        finish = FinishOptions(
            smooth_iterations=int(settings.get("smooth_iterations", 0)),
            texture=bool(settings.get("texture", False)),
            segmentation=str(settings.get("segmentation", "auto")),
            web_export=bool(settings.get("web_export", False)),
            texture_size=int(settings.get("texture_size", 1024)),
            max_parts=max(1, int(settings.get("max_parts", 8))),
            total_budget=target_faces,
            gpu_host=host,
            bake_normals=bool(settings.get("bake_normals", True)),
            normal_res=max(256, min(2048, int(settings.get("normal_res", 1024)))),
            bake_location=str(settings.get("bake_location", "auto")),
        )
        gen_options = {
            "octree_resolution": max(64, int(settings.get("octree_resolution", 256)))
        }
        new_settings = {**settings, "gpu_host": host, "rerun_of": job_id}
        session.write_meta(
            job_id=new_id, name=name, target_faces=target_faces, settings=new_settings
        )

        job_info: dict[str, Any] = {
            "job_id": new_id,
            "name": name,
            "target_faces": target_faces,
            "status": "queued",
            "stage": "queued",
            "log": [
                f"Rerun of job {job_id} with the same {len(staged)} photo(s) "
                "and settings.",
            ],
            "report": None,
            "glb_url": None,
            "dir": str(session.root),
            "regime": None,
            "faces": None,
            "parts": [],
            "parts_dir": None,
            "warnings": [],
            "stages": [],
            "created_at": session.created_at,
            "split": bool(settings.get("split", True)),
            "object_hint": settings.get("object_hint"),
            "backend": settings.get("backend"),
            "settings": new_settings,
            "finish": dataclasses.asdict(finish),
            "web_url": None,
            "rerun_of": job_id,
        }

        res = _register_and_dispatch(
            job_info,
            session.root,
            target_faces,
            settings.get("backend"),
            bool(settings.get("split", True)),
            settings.get("object_hint"),
            finish,
            gen_options,
            bool(settings.get("multiview", False)),
            seed_val,
        )
        return {**res, "rerun_of": job_id}

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            snapshot = [
                (
                    {
                        "job_id": j["job_id"],
                        "name": j["name"],
                        "status": j["status"],
                        "stage": j["stage"],
                        "glb_url": j["glb_url"],
                        "created_at": j.get("created_at", ""),
                        "restored": bool(j.get("restored")),
                    },
                    j["dir"],
                )
                for j in jobs.values()
            ]
        # Counted outside the lock: it is one directory glob per job, and the
        # list is polled. How many models a job holds is what tells you at a
        # glance which ones you have re-optimised, without opening them.
        res = []
        for entry, job_dir in snapshot:
            entry["models"] = len(_model_files(Path(job_dir) / "output"))
            res.append(entry)
        # Newest first: the job you just started is the one you want to see,
        # and dict insertion order puts it last.
        res.sort(key=lambda j: j.get("created_at") or "", reverse=True)
        return res

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, log_since: int = 0):
        """One job's full state.

        ``log_since`` returns only the log lines after that index. A running
        GPU job is polled for many minutes while its log only grows, and
        re-sending the whole thing every couple of seconds is quadratic in the
        length of the job for no benefit. ``log_total`` is always the true
        length so a client can tell whether it is behind.
        """
        with lock:
            job = jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            log: list[str] = job["log"]
            total = len(log)
            # A negative or over-large cursor means the client and server
            # disagree about history -- most likely the job was restarted.
            # Sending the whole log resynchronises rather than silently
            # leaving the client with a permanent hole in it.
            start = log_since if 0 <= log_since <= total else 0
            return {
                "job_id": job["job_id"],
                "name": job["name"],
                "status": job["status"],
                "stage": job["stage"],
                "log": log[start:],
                "log_since": start,
                "log_total": total,
                "report": job["report"],
                "glb_url": job["glb_url"],
                "web_url": job.get("web_url"),
                "regime": job.get("regime"),
                "faces": job.get("faces"),
                # Every model this job holds: the build's own, plus one per
                # re-optimise. Read from disk so it survives a restart.
                "models": _model_list(
                    job_id,
                    Path(job["dir"]) / "output",
                    job_name=job["name"],
                    build_faces=job.get("build_faces"),
                    current_faces=job.get("faces"),
                ),
                # The budget the job was built with. Survives a restart even
                # when the achieved count does not, so re-optimise seeds its
                # input from it rather than from a generic default.
                "target_faces": job.get("target_faces"),
                "parts": job.get("parts", []),
                "warnings": job.get("warnings", []),
                "stages": job.get("stages", []),
                "created_at": job.get("created_at", ""),
                "restored": bool(job.get("restored")),
                # Where this job's files actually live, so the user can open
                # the session folder instead of hunting through the jobs root.
                "session": Path(job["dir"]).name,
                "dir": job["dir"],
                "output_dir": str(Path(job["dir"]) / "output"),
            }

    @app.get("/api/jobs/{job_id}/parts/{filename:path}")
    def get_job_part(job_id: str, filename: str):
        with lock:
            job = jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            parts_dir_str = job.get("parts_dir")

        if not parts_dir_str:
            raise HTTPException(status_code=404, detail="Parts directory not found")

        parts_dir = Path(parts_dir_str).resolve()
        if not parts_dir.exists() or not parts_dir.is_dir():
            raise HTTPException(status_code=404, detail="Parts directory not found")

        try:
            target_file = (parts_dir / filename).resolve()
            target_file.relative_to(parts_dir)
        except (ValueError, Exception):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if target_file == parts_dir:
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not target_file.exists() or not target_file.is_file():
            raise HTTPException(status_code=404, detail="Part file not found")

        return FileResponse(
            path=target_file,
            media_type="model/gltf-binary",
            filename=target_file.name,
        )

    @app.post("/api/jobs/{job_id}/optimize")
    def optimize_job(job_id: str, req: OptimizeRequest):
        with lock:
            job = jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            job_dir = Path(job["dir"])

        if _find_reoptimise_source(job_dir) is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "No source mesh kept for this job: re-optimising needs "
                    "either a photogrammetry dense.ply or the generate stage's "
                    "mesh.glb, and neither is on disk"
                ),
            )

        with lock:
            job["target_faces"] = req.target_faces

        ws_dir = job_dir / "work" / "ws"
        ws = Workspace.open(ws_dir if ws_dir.exists() else job_dir)
        spec = ws.spec()
        spec.target_faces = req.target_faces
        ws.update_spec(spec)

        try:
            _run_mesh_stages_sync(job_id, job_dir, req.target_faces, req.normal_res)
        except Exception as exc:
            with lock:
                job["status"] = "failed"
                job["log"].append(f"Optimization failed: {exc}")
            # A short message, not a raw traceback -- but it must still point
            # somewhere. The exception text goes to the job log above; saying
            # so is the difference between a dead end and a next step.
            raise HTTPException(
                status_code=500,
                detail="Re-optimise failed. The reason is in this job's log.",
            )

        with lock:
            glb = job.get("glb_path")
            return {
                "status": job["status"],
                "stage": job["stage"],
                "log": list(job["log"]),
                "report": job["report"],
                "glb_url": job["glb_url"],
                # What it actually PRODUCED, not what was asked for. Without
                # these the UI could only echo the requested budget back, so a
                # re-optimise that silently did nothing looked identical to one
                # that worked -- and decimation routinely lands short of the
                # target when the source mesh has fewer faces to give.
                "faces": job.get("faces"),
                "bytes": (
                    Path(glb).stat().st_size
                    if glb and Path(glb).is_file()
                    else None
                ),
                # The file it landed in. The build's model is still there
                # beside it, so saying "re-optimised" without saying which
                # file appeared leaves the user hunting through the list.
                "model": Path(glb).name if glb else None,
                "models": len(_model_files(job_dir / "output")),
            }

    @app.get("/api/jobs/{job_id}/model.glb")
    def get_model_glb(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            job_dir = Path(job["dir"])
            stored_glb = job.get("glb_path")

        glb_path = Path(stored_glb) if stored_glb else (job_dir / "ws" / "output" / f"{job['name']}.glb")
        if not glb_path.exists():
            glb_path = job_dir / "model.glb"
        if not glb_path.exists():
            raise HTTPException(status_code=404, detail="Model GLB not found")

        return FileResponse(
            path=glb_path,
            media_type="model/gltf-binary",
            filename=f"{job['name']}.glb",
        )

    @app.get("/api/jobs/{job_id}/model_web.glb")
    def get_web_glb(job_id: str):
        """The web-sized model: same geometry, smaller texture."""
        with lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        glb = job.get("glb_path")
        if not glb:
            raise HTTPException(status_code=404, detail="no model for this job")
        web = Path(glb).parent / "model_web.glb"
        if not web.is_file():
            raise HTTPException(
                status_code=404,
                detail="no web export for this job; enable 'Optimise for web'",
            )
        return FileResponse(web, media_type="model/gltf-binary", filename="model_web.glb")

    @app.get("/api/jobs/{job_id}/files")
    def job_files(job_id: str):
        """Everything this job produced, read from disk.

        From disk rather than from the in-memory job record: sessions survive
        a server restart but the record does not, and the files are the thing
        the user actually cares about.
        """
        with lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        out_dir = Path(job["dir"]) / "output"
        if not out_dir.is_dir():
            return {"files": [], "total_bytes": 0, "output_dir": str(out_dir)}

        files = []
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(out_dir).as_posix()
                files.append({
                    "name": rel,
                    "bytes": path.stat().st_size,
                    "url": f"/api/jobs/{job_id}/download/{rel}",
                })
        return {
            "files": files,
            "total_bytes": sum(f["bytes"] for f in files),
            "output_dir": str(out_dir),
        }

    @app.get("/api/jobs/{job_id}/download/{relative:path}")
    def job_download(job_id: str, relative: str):
        with lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        target = _safe_output_file(Path(job["dir"]), relative)
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"no such file: {relative}")
        return FileResponse(
            target, filename=target.name, media_type="application/octet-stream"
        )

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        """Delete a job's whole session folder from disk. Not reversible."""
        with lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.get("status") == "running":
            raise HTTPException(
                status_code=409, detail="job is still running; wait for it to finish"
            )

        session_dir = Path(job["dir"]).resolve()
        root_resolved = root.resolve()
        # Never delete outside the jobs root, and never the root itself.
        if session_dir == root_resolved or root_resolved not in session_dir.parents:
            raise HTTPException(status_code=400, detail="refusing to delete outside the jobs root")

        freed = sum(f.stat().st_size for f in session_dir.rglob("*") if f.is_file())
        shutil.rmtree(session_dir, ignore_errors=True)
        with lock:
            jobs.pop(job_id, None)
        return {"deleted": str(session_dir), "freed_bytes": freed}

    @app.post("/api/jobs/{job_id}/convert")
    def convert_job(job_id: str, req: ConvertRequest):
        """Convert a produced model to STL/OBJ/PLY for CAD or a slicer."""
        from pixiecad.meshops.cadexport import CAD_FORMATS, export_cad

        with lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        fmt = req.format.lower().lstrip(".")
        if fmt not in CAD_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unsupported format '{fmt}'; use {', '.join(CAD_FORMATS)}",
            )

        if req.source:
            source = _safe_output_file(Path(job["dir"]), req.source)
            if not source.is_file():
                raise HTTPException(
                    status_code=404, detail=f"no such model: {req.source}"
                )
        else:
            # Convert what the user is looking at. Defaulting to model.glb
            # meant that after a re-optimise the STL button quietly handed
            # back the geometry that had just been superseded.
            found = _current_model(Path(job["dir"]) / "output")
            if found is None:
                raise HTTPException(
                    status_code=404, detail="no model to convert for this job"
                )
            source = found

        mesh = trimesh.load(source, force="mesh", process=False)
        name = f"{_export_stem(job['name'], source.name)}.{fmt}"
        try:
            path, report, actions = export_cad(
                mesh,
                source.parent / name,
                longest_mm=req.size_mm,
                repair=req.repair,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

        return {
            "file": path.name,
            "bytes": path.stat().st_size,
            "url": f"/api/jobs/{job_id}/download/{path.name}",
            "printable": report.printable,
            "solid": report.summary(),
            "actions": actions,
        }

    @app.get("/api/cloud/vm")
    def vm_status():
        """Is there a GPU VM up, and is it usable yet?

        Deliberately cheap -- one gcloud call, no SSH -- because the dashboard
        polls this. "Running" and "ready" are different things: a VM answers
        as RUNNING within a minute of creation but cannot serve a job until
        its worker images are built, which is ~30 minutes on a cold VM.
        """
        try:
            from pixiecad.cloud import gcloud_status, list_instances

            status = gcloud_status()
            if not status.installed:
                return {"available": False, "reason": "gcloud not installed", "vms": []}

            project = status.project or ""
            vms = []
            for inst in list_instances(project=project):
                vms.append({
                    "name": inst.name,
                    "zone": inst.zone,
                    "status": inst.status,
                    "gpu": inst.accelerator,
                    "spot": inst.preemptible,
                    "uptime_hours": inst.uptime_hours,
                    "image": inst.source_machine_image,
                    "max_run_seconds": inst.max_run_seconds,
                    "seconds_remaining": inst.seconds_remaining,
                    "host": f"{inst.name}.{inst.zone}.{project}" if project else "",
                })
            running = [v for v in vms if str(v["status"]).upper() == "RUNNING"]
            with lock:
                prov = dict(provisioning)
            return {
                "available": True,
                "vms": vms,
                "running": len(running),
                # Building means a provision is still in flight, so a job
                # submitted now would fail on a missing docker image.
                "building": prov.get("status") == "running",
                "provision_status": prov.get("status"),
                "host": (prov.get("host") or (running[0]["host"] if running else "")),
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc), "vms": []}

    @app.get("/api/cloud/inventory")
    def cloud_inventory():
        """Everything GCP bills this project for, with console deep links.

        Estimated run-rate, not your actual invoice: real per-resource cost
        needs Cloud Billing export to BigQuery. This answers the question
        people actually ask -- what am I paying to leave things lying around.
        """
        try:
            from pixiecad.cloud import gcloud_status, list_billable

            status = gcloud_status()
            if not status.installed:
                return {"available": False, "reason": "gcloud CLI not installed", "resources": []}
            resources = [dataclasses.asdict(r) for r in list_billable(status.project)]
            known = [r["est_usd_per_month"] for r in resources if r["est_usd_per_month"]]
            return {
                "available": True,
                "project": status.project,
                "resources": resources,
                "est_total_usd_per_month": round(sum(known), 2),
                "unmeasured": sum(1 for r in resources if r["est_usd_per_month"] is None),
                "console_billing_url": "https://console.cloud.google.com/billing",
            }
        except Exception as exc:
            return {"available": False, "reason": str(exc), "resources": []}

    @app.get("/api/backends")
    def list_backends():
        """The generative backends a job may be pointed at.

        Deliberately static rather than probing each one. available() on a
        remote backend opens an SSH connection and asks the host what it has;
        doing that for every backend on every page load would make the new-job
        form wait on the network to render a dropdown. What each backend NEEDS
        is fixed and knowable here, so the client greys out what cannot run
        using the VM state it already tracks.

        `validated` is the honest bit: a backend can be wired up and reachable
        and still never have produced a model on this project. Offering those
        without saying so invites someone to pick one and lose a GPU hour.
        """
        return {
            "backends": [
                {
                    "key": "auto",
                    "label": "Automatic",
                    "requires": None,
                    "validated": True,
                    "note": (
                        "Picks the first backend that is actually usable, "
                        "preferring your own GPU over paid APIs. This is what "
                        "every job has used so far."
                    ),
                },
                {
                    "key": "hunyuan-remote",
                    "label": "Hunyuan3D 2.1",
                    "requires": "gpu_host",
                    "validated": True,
                    "note": (
                        "The measured baseline: 170 s generation and 123 s "
                        "texturing on an L4. Single-view by default; the "
                        "multi-view checkpoint is an older architecture."
                    ),
                },
                {
                    "key": "trellis-remote",
                    "label": "TRELLIS.2",
                    "requires": "gpu_host",
                    "validated": True,
                    "note": (
                        "1.9M faces and textured in one pass, where Hunyuan "
                        "needs a separate texturing stage. Slower, and needs "
                        "an L4 specifically (Ada-only kernels), a 64 GB host "
                        "and an approved DINOv3 licence. First run on a fresh "
                        "VM pays ~11 min for a 25 GB image pull."
                    ),
                },
            ],
        }

    @app.get("/api/stage-costs")
    def stage_costs_endpoint(
        bake: bool = True,
        normal_res: int = 1024,
        bake_location: str = "auto",
        texture: bool = False,
        semantic: bool = False,
        has_host: bool = False,
    ):
        """What a job as configured will cost, before it is submitted.

        Split by where the work lands, because the two are not interchangeable:
        host seconds are billed but cost this machine nothing, while local
        seconds compete with whatever else the user is doing on a 16 GB laptop.
        """
        from pixiecad.cloud_options import stage_costs

        return stage_costs(
            bake=bake,
            normal_res=normal_res,
            bake_location=bake_location,
            texture=texture,
            semantic=semantic,
            has_host=has_host,
        )

    @app.get("/api/gpu-options")
    def gpu_options(texture: bool = True, semantic: bool = True):
        """Hardware choices with measured-where-possible time and cost."""
        from pixiecad.cloud_options import options_payload

        return {"options": options_payload(texture=texture, semantic=semantic)}

    @app.post("/api/cloud/provision")
    def provision(req: ProvisionRequest):
        """Create a GPU VM and build the worker images on it.

        Long-running (a cold build is ~20 minutes), so this returns
        immediately and progress is polled from /api/cloud/provision.
        """
        from pixiecad.cloud_options import GPU_OPTIONS

        option = next((o for o in GPU_OPTIONS if o.key == req.gpu), None)
        if option is None:
            raise HTTPException(status_code=400, detail=f"unknown gpu: {req.gpu}")
        if not option.available:
            raise HTTPException(
                status_code=400,
                detail=f"{option.label} is not available on this project ({option.note})",
            )

        with lock:
            if provisioning.get("status") == "running":
                raise HTTPException(status_code=409, detail="a provision is already running")
            provisioning.clear()
            provisioning.update(
                status="running", gpu=req.gpu, name=req.name, zone=req.zone, log=[]
            )

        state_file = root / ".provision.json"

        def _persist() -> None:
            """Mirror provisioning state to disk.

            The in-memory record dies with the process; the VM and its build
            do not. Persisting means a restarted dashboard can still tell you
            what is happening instead of showing "no VM" while one builds.
            """
            try:
                with lock:
                    snapshot = dict(provisioning)
                snapshot["log"] = snapshot.get("log", [])[-40:]
                state_file.write_text(json.dumps(snapshot))
            except Exception:
                pass

        def _run() -> None:
            scripts = Path(__file__).resolve().parents[3] / "scripts"

            # Prefer a baked machine image when one exists: it already
            # contains docker, all three worker images and the model caches,
            # so this is a ~2 minute boot instead of a ~30 minute rebuild of
            # byte-identical work billed at the GPU rate.
            baked = _latest_machine_image()
            if baked:
                with lock:
                    provisioning["log"].append(f"using baked image: {baked}")
                    provisioning["baked"] = baked
                steps = [[
                    str(scripts / "launch_from_image.sh"), req.name, req.zone, baked,
                ]]
            else:
                with lock:
                    provisioning["log"].append(
                        "no baked machine image found -- building from scratch "
                        "(~30 min). Bake one with scripts/bake_image.sh to make "
                        "future launches ~2 min."
                    )
                steps = [
                    [str(scripts / "provision_gpu_vm.sh"), req.name, req.zone, req.gpu, "hunyuan"],
                    [str(scripts / "setup_hunyuan_vm.sh"), req.name, req.zone],
                ]
                if req.texture:
                    steps.append([str(scripts / "setup_hunyuan_texture.sh"), req.name, req.zone])
                if req.semantic:
                    steps.append([str(scripts / "setup_sam_vm.sh"), req.name, req.zone])
            try:
                for step in steps:
                    with lock:
                        provisioning["log"].append(f"$ {Path(step[0]).name} {' '.join(step[1:])}")
                    _persist()
                    # start_new_session detaches the child into its own
                    # process group. Without it, Ctrl-C or closing the
                    # terminal sends SIGINT/SIGHUP to the whole group and
                    # kills a 30-minute image build half-way, leaving a VM
                    # with some images missing and no sign of it.
                    proc = subprocess.run(
                        step, capture_output=True, text=True, timeout=3600,
                        start_new_session=True,
                    )
                    with lock:
                        provisioning["log"].extend((proc.stdout or "").splitlines()[-15:])
                    if proc.returncode != 0:
                        with lock:
                            provisioning["log"].extend((proc.stderr or "").splitlines()[-15:])
                            provisioning["status"] = "failed"
                        _persist()
                        return
                # The alias the executor needs; without config-ssh the host
                # name does not resolve at all.
                subprocess.run(["gcloud", "compute", "config-ssh", "--quiet"], timeout=300)
                project = _gcp_project()
                with lock:
                    provisioning["status"] = "ready"
                    # config-ssh writes aliases as name.zone.project; without
                    # the project the alias does not resolve at all, so an
                    # unset env var must fall back to the gcloud config rather
                    # than silently producing a two-part name.
                    landed = _actual_zone(req.name, req.zone, project)
                    if landed != req.zone:
                        provisioning["log"].append(
                            f"no L4 capacity in {req.zone}; launched in {landed}"
                        )
                    provisioning["zone"] = landed
                    provisioning["host"] = f"{req.name}.{landed}.{project}" if project else ""
                    if not project:
                        provisioning["log"].append(
                            "WARNING: could not determine GCP project; set "
                            "PIXIECAD_GCP_PROJECT or run 'gcloud config set project'"
                        )
                _persist()
            except Exception as exc:
                with lock:
                    provisioning["status"] = "failed"
                    provisioning["log"].append(str(exc))
                _persist()

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running"}

    @app.get("/api/cloud/provision")
    def provision_status():
        with lock:
            current = dict(provisioning)
        if current:
            return current
        # Nothing in memory: a restart may have lost the record while the
        # detached build carries on, so fall back to the file it writes.
        state_file = root / ".provision.json"
        if state_file.is_file():
            try:
                return json.loads(state_file.read_text())
            except Exception:
                pass
        return {}

    @app.post("/api/cloud/teardown")
    def teardown(req: TeardownRequest):
        """Delete the VM. Billing continues until this succeeds."""
        proc = subprocess.run(
            ["gcloud", "compute", "instances", "delete", req.name,
             f"--zone={req.zone}", "--quiet"],
            capture_output=True, text=True, timeout=600,
        )
        with lock:
            provisioning.clear()
        return {"ok": proc.returncode == 0, "output": (proc.stdout or proc.stderr)[-800:]}

    @app.get("/api/cloud")
    def get_cloud():
        try:
            from pixiecad.cloud import snapshot

            snap = snapshot()

            def _to_dict(obj: Any) -> Any:
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                if hasattr(obj, "__dict__"):
                    res = {}
                    for k, v in obj.__dict__.items():
                        if isinstance(v, list):
                            res[k] = [_to_dict(x) for x in v]
                        elif hasattr(v, "__dict__") or dataclasses.is_dataclass(v):
                            res[k] = _to_dict(v)
                        else:
                            res[k] = v
                    return res
                if isinstance(obj, dict):
                    return {k: _to_dict(v) for k, v in obj.items()}
                return obj

            data = _to_dict(snap)
            if not isinstance(data, dict):
                data = {"snapshot": data}
            data["available"] = True
            if hasattr(snap, "has_running_gpu"):
                data["has_running_gpu"] = bool(snap.has_running_gpu)
            return data
        except ImportError as e:
            return {"available": False, "reason": f"cloud module import error: {e}"}
        except Exception as e:
            return {"available": False, "reason": str(e)}

    return app
