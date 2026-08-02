"""FastAPI application for PixieCAD local web dashboard."""

from __future__ import annotations

import dataclasses
import inspect
import shutil
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
from ..spec import Dimensions, ObjectSpec
from ..workspace import Workspace


class OptimizeRequest(BaseModel):
    target_faces: int = Field(..., gt=0)
    normal_res: int = Field(1024, ge=64, le=4096)


def _call_build(**kwargs: Any) -> Any:
    import pixiecad.generative  # noqa: F401  (registers fake backend)
    from pixiecad.pipeline import run_build

    sig = inspect.signature(run_build)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return run_build(**filtered)


def create_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="PixieCAD Dashboard")
    lock = threading.Lock()

    # In-process dict storing jobs state
    jobs: dict[str, dict[str, Any]] = {}

    static_index = Path(__file__).parent / "static" / "index.html"

    def _run_mesh_stages_sync(
        job_id: str, ws_root: Path, target_faces: int, normal_res: int = 1024
    ) -> None:
        dense_ply = ws_root / "dense.ply"
        if not dense_ply.exists() and ws_root.parent.exists():
            dense_ply = ws_root.parent / "dense.ply"
        if not dense_ply.exists():
            dense_ply = ws_root / "ws" / "dense.ply"
        if not dense_ply.exists():
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

        glb_path = ws_root / "model.glb"
        export_glb(unwrap_res.mesh, glb_path, normal_map=normal_map)

        with lock:
            job = jobs.get(job_id)
            if job:
                job["status"] = "done"
                job["stage"] = "complete"
                job["glb_url"] = f"/api/jobs/{job_id}/model.glb"
                job["log"].append("GLB model export complete.")

    def _job_worker(
        job_id: str,
        ws_root: Path,
        target_faces: int,
        backend: str | None = None,
        split: bool = True,
        object_hint: str | None = None,
    ) -> None:
        with lock:
            job = jobs.get(job_id)
            if not job:
                return
            job["status"] = "running"
            job["stage"] = "build"
            job["log"].append("Starting full pipeline build...")

        try:
            photos_dir = ws_root / "photos"
            ws_dir = ws_root / "ws"

            result = _call_build(
                photos_dir=photos_dir,
                workspace=ws_dir,
                dense=False,
                bake=False,
                generative_backend=backend,
                split=split,
                object_hint=object_hint,
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

            summary_lines = result.summary_lines() if hasattr(result, "summary_lines") else []

            with lock:
                job = jobs.get(job_id)
                if job:
                    for line in summary_lines:
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
    ):
        job_id = uuid4().hex[:8]
        job_dir = root / job_id
        photos_dir = job_dir / "photos"
        photos_dir.mkdir(parents=True, exist_ok=True)

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

        ws_dir = job_dir / "ws"
        Workspace.create(ws_dir, spec)

        backend_val = backend.strip() if backend and backend.strip() else None
        hint_val = object_hint.strip() if object_hint and object_hint.strip() else None

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
        }

        with lock:
            jobs[job_id] = job_info

        thread = threading.Thread(
            target=_job_worker,
            args=(job_id, job_dir, target_faces, backend_val, split, hint_val),
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

        dense_ply = job_dir / "dense.ply"
        if not dense_ply.exists():
            dense_ply = job_dir / "ws" / "dense.ply"
        if not dense_ply.exists():
            raise HTTPException(
                status_code=409,
                detail="Dense mesh (dense.ply) not found for this job",
            )

        with lock:
            job["target_faces"] = req.target_faces

        ws_dir = job_dir / "ws" if (job_dir / "ws").exists() else job_dir
        ws = Workspace.open(ws_dir)
        spec = ws.spec()
        spec.target_faces = req.target_faces
        ws.update_spec(spec)

        try:
            _run_mesh_stages_sync(job_id, ws_dir, req.target_faces, req.normal_res)
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
