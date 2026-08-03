"""FastAPI application for PixieCAD local web dashboard."""

from __future__ import annotations
from pixiecad.web.finishing import FinishOptions, finish_model

import dataclasses
import inspect
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
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


class ConvertRequest(BaseModel):
    """GLB -> STL/OBJ/PLY for CAD and slicers."""

    source: str = "model.glb"
    format: str = "stl"
    size_mm: float | None = None
    repair: bool = False


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

    # In-process dict storing jobs state
    jobs: dict[str, dict[str, Any]] = {}
    # Single in-flight provision: two concurrent builds on one project would
    # race for the same L4 quota of 1 and both fail confusingly.
    provisioning: dict[str, Any] = {}

    static_index = Path(__file__).parent / "static" / "index.html"

    def _run_mesh_stages_sync(
        job_id: str, ws_root: Path, target_faces: int, normal_res: int = 1024
    ) -> None:
        dense_ply = _find_dense_ply(ws_root)
        if dense_ply is None:
            raise FileNotFoundError("dense.ply not found")

        with lock:
            job = jobs.get(job_id)
            if job:
                job["status"] = "running"
                job["stage"] = "clean"
                job["log"].append("Cleaning dense mesh...")

        dense_mesh = trimesh.load(dense_ply)
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
        glb_path = out_dir / "model.glb"
        export_glb(unwrap_res.mesh, glb_path, normal_map=normal_map)

        with lock:
            job = jobs.get(job_id)
            if job:
                job["status"] = "done"
                job["stage"] = "complete"
                # Record where we actually wrote it: the download endpoint
                # serves glb_path, and re-optimising must repoint it rather
                # than leave a stale path from an earlier build.
                job["glb_path"] = str(glb_path)
                job["glb_url"] = f"/api/jobs/{job_id}/model.glb"
                job["log"].append("GLB model export complete.")

    def _job_worker(
        job_id: str,
        ws_root: Path,
        target_faces: int,
        backend: str | None = None,
        split: bool = True,
        object_hint: str | None = None,
        finish: FinishOptions | None = None,
        generative_options: dict[str, Any] | None = None,
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
                bake=False,
                executor=executor,
                generative_backend=backend,
                split=split,
                object_hint=object_hint,
                generative_options=generative_options,
            )

            regime_val = (
                result.regime.value
                if hasattr(result.regime, "value")
                else (str(result.regime) if getattr(result, "regime", None) else None)
            )
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

            summary_lines = result.summary_lines() if hasattr(result, "summary_lines") else []

            with lock:
                job = jobs.get(job_id)
                if job:
                    for line in summary_lines:
                        job["log"].append(line)
                    # Always, not only on failure: a photo silently dropped
                    # from an otherwise successful run degrades the result and
                    # the user should see which one and why.
                    for line in _ingest_rejections(ws_root):
                        job["log"].append(line)
                    for w in warnings:
                        job["log"].append(f"WARNING: {w}")

                    job["regime"] = regime_val
                    job["glb_path"] = glb_path
                    job["faces"] = faces
                    job["parts"] = parts
                    job["parts_dir"] = str(parts_dir) if parts_dir else None
                    job["warnings"] = warnings

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

    @app.get("/", response_class=HTMLResponse)
    def index():
        if static_index.exists():
            return static_index.read_text(encoding="utf-8")
        return "<html><body><h1>PixieCAD Dashboard</h1></body></html>"

    @app.post("/api/jobs")
    async def create_job(
        files: list[UploadFile] = File(default=[]),
        name: str = Form("object"),
        target_faces: int = Form(20000),
        length: str | None = Form(None),
        width: str | None = Form(None),
        height: str | None = Form(None),
        mode: str = Form("auto"),
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
    ):
        # One session folder per upload: input/, work/ and output/ are never
        # shared between jobs, so two people uploading at once cannot read or
        # overwrite each other's photos and meshes.
        session = new_session(root, label=name or "object")
        job_id = uuid4().hex[:8]
        job_dir = session.root
        photos_dir = session.input_dir

        for f in files:
            fname = f.filename or f"photo_{uuid4().hex[:4]}.jpg"
            dest = photos_dir / Path(fname).name
            with dest.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)

        l_val = length.strip() if length and length.strip() else None
        w_val = width.strip() if width and width.strip() else None
        h_val = height.strip() if height and height.strip() else None
        dims = Dimensions.parse(length=l_val, width=w_val, height=h_val)
        spec = ObjectSpec(name=name or "object", target_faces=target_faces, dimensions=dims)

        ws_dir = session.work_dir / "ws"
        Workspace.create(ws_dir, spec)

        backend_val = backend.strip() if backend and backend.strip() else None
        hint_val = object_hint.strip() if object_hint and object_hint.strip() else None

        finish = FinishOptions(
            smooth_iterations=max(0, smooth_iterations),
            texture=texture,
            segmentation=segmentation,
            web_export=web_export,
            texture_size=texture_size,
            max_parts=max(1, max_parts),
            total_budget=target_faces,
            gpu_host=(gpu_host.strip() or None) if gpu_host else None,
        )

        # Geometric detail of the generative decode. This is the lever that
        # actually separates parts that sit close together -- raising it makes
        # the model resolve a gap the decoder would otherwise bridge -- and it
        # is independent of the polygon budget, which only controls how the
        # result is simplified afterwards.
        gen_options = {"octree_resolution": max(64, octree_resolution)}

        session.write_meta(job_id=job_id, name=spec.name, target_faces=target_faces)

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
            "mode": mode,
            "split": split,
            "object_hint": hint_val,
            "backend": backend_val,
            "finish": {
                "smooth_iterations": finish.smooth_iterations,
                "texture": finish.texture,
                "segmentation": finish.segmentation,
                "web_export": finish.web_export,
                "texture_size": finish.texture_size,
                "gpu_host": finish.gpu_host,
            },
            "web_url": None,
        }

        with lock:
            jobs[job_id] = job_info

        thread = threading.Thread(
            target=_job_worker,
            args=(job_id, job_dir, target_faces, backend_val, split, hint_val, finish,
                  gen_options),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs():
        with lock:
            res = [
                {
                    "job_id": j["job_id"],
                    "name": j["name"],
                    "status": j["status"],
                    "stage": j["stage"],
                    "glb_url": j["glb_url"],
                }
                for j in jobs.values()
            ]
        return res

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        with lock:
            job = jobs.get(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return {
                "job_id": job["job_id"],
                "name": job["name"],
                "status": job["status"],
                "stage": job["stage"],
                "log": list(job["log"]),
                "report": job["report"],
                "glb_url": job["glb_url"],
                "regime": job.get("regime"),
                "faces": job.get("faces"),
                "parts": job.get("parts", []),
                "warnings": job.get("warnings", []),
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

        if not _find_dense_ply(job_dir):
            raise HTTPException(
                status_code=409,
                detail="Dense mesh (dense.ply) not found for this job",
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
            raise HTTPException(status_code=500, detail=str(exc))

        with lock:
            return {
                "status": job["status"],
                "stage": job["stage"],
                "log": list(job["log"]),
                "report": job["report"],
                "glb_url": job["glb_url"],
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

        source = _safe_output_file(Path(job["dir"]), req.source)
        if not source.is_file():
            raise HTTPException(status_code=404, detail=f"no such model: {req.source}")

        mesh = trimesh.load(source, force="mesh", process=False)
        name = f"{Path(job['name'] or 'model').stem}.{fmt}"
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

        def _run() -> None:
            scripts = Path(__file__).resolve().parents[3] / "scripts"
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
                    proc = subprocess.run(step, capture_output=True, text=True, timeout=3600)
                    with lock:
                        provisioning["log"].extend((proc.stdout or "").splitlines()[-15:])
                    if proc.returncode != 0:
                        with lock:
                            provisioning["log"].extend((proc.stderr or "").splitlines()[-15:])
                            provisioning["status"] = "failed"
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
                    provisioning["host"] = f"{req.name}.{req.zone}.{project}" if project else ""
                    if not project:
                        provisioning["log"].append(
                            "WARNING: could not determine GCP project; set "
                            "PIXIECAD_GCP_PROJECT or run 'gcloud config set project'"
                        )
            except Exception as exc:
                with lock:
                    provisioning["status"] = "failed"
                    provisioning["log"].append(str(exc))

        threading.Thread(target=_run, daemon=True).start()
        return {"status": "running"}

    @app.get("/api/cloud/provision")
    def provision_status():
        with lock:
            return dict(provisioning)

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
