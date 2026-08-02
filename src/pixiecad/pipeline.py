"""End-to-end build orchestrator for PixieCAD."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time

import trimesh

from .executors.base import Executor
from .export import export_glb
from .geometry.dense import DenseUnavailable, run_dense
from .geometry.sparse import SparseFailure, SparseResult, run_sparse
from .ingest.pipeline import IngestReport, run_ingest
from .meshops import bake_object_space_normals, clean_mesh, decimate_to_budget, unwrap_uv
from .parts import export_parts, split_parts
from .scale.transform import apply_scale, scale_factor_from_dimensions
from .spec import ObjectSpec
from .vision.triage import name_parts
from .workspace import Workspace


class Regime(str, Enum):
    ORBIT = "orbit"
    SPARSE_VIEWS = "sparse_views"
    SINGLE_IMAGE = "single_image"


def detect_regime(n_usable_photos: int) -> Regime:
    """Determine reconstruction regime from count of usable photos."""
    if n_usable_photos >= 16:
        return Regime.ORBIT
    elif 2 <= n_usable_photos <= 15:
        return Regime.SPARSE_VIEWS
    elif n_usable_photos == 1:
        return Regime.SINGLE_IMAGE
    else:
        raise ValueError(f"Invalid photo count: {n_usable_photos}; must be >= 1")


@dataclass
class StageOutcome:
    name: str
    status: str  # "ok" | "skipped" | "failed"
    detail: str
    seconds: float


@dataclass
class BuildResult:
    regime: Regime
    stages: list[StageOutcome]
    glb_path: str | None
    faces: int | None
    scale_applied: float | None
    warnings: list[str]
    parts_dir: str | None = None
    parts: list[dict] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """Return human-readable summary lines, one per stage."""
        return [
            f"[{s.status.upper()}] {s.name}: {s.detail} ({s.seconds:.2f}s)"
            for s in self.stages
        ]


def _ingested_images_dir(report: IngestReport) -> Path | None:
    """Directory holding S0's working copies, or None if there are none.

    S0 writes every accepted photo into one ``images/`` dir inside its stage
    dir, so the parent of any accepted working copy is that dir. Checking all
    accepted records (not just the first) and confirming the dir exists keeps
    this robust if the report is stale or partially populated.
    """
    for photo in report.photos:
        if photo.working_path:
            candidate = Path(photo.working_path).parent
            if candidate.is_dir():
                return candidate
    return None


def _run_generative(
    *,
    report,
    regime: Regime,
    ws: Workspace,
    spec: ObjectSpec,
    stages: list[StageOutcome],
    warnings: list[str],
    backend: str | None,
    bake: bool,
    normal_res: int,
    split: bool = False,
    max_parts: int = 8,
    object_hint: str | None = None,
) -> BuildResult:
    """Regimes with too few photos to triangulate: invent the geometry instead.

    Records the photogrammetry stages as skipped (they genuinely never ran) and
    substitutes S3. The generated mesh is its own bake source — there is no
    higher-detail original, so the normal map only captures decimation loss.
    """
    from .generative import GenerateRequest, GenerativeError, run_generate

    for name in ("sparse", "dense"):
        stages.append(
            StageOutcome(name, "skipped", f"not applicable in {regime.value} regime", 0.0)
        )

    t0 = time.monotonic()
    images = [Path(p.working_path) for p in report.photos if p.working_path][:4]
    try:
        result = run_generate(GenerateRequest(images=images), ws, backend=backend)
        mesh = trimesh.load(result.mesh_path, force="mesh", process=False)
        stages.append(
            StageOutcome(
                "generate",
                "ok",
                f"{result.backend}: {result.n_faces} faces from {len(images)} image(s)"
                + (" (cached)" if result.cached else ""),
                time.monotonic() - t0,
            )
        )
    except (GenerativeError, Exception) as e:
        stages.append(StageOutcome("generate", "failed", str(e), time.monotonic() - t0))
        _fill_skipped_stages(stages, ["scale", "clean", "decimate", "bake", "export"])
        return BuildResult(regime, stages, None, None, None, warnings)

    warnings.append(
        "Geometry is generated, not measured: surfaces no camera saw are "
        "plausible inventions."
    )
    return _mesh_tail(
        mesh,
        mesh,
        ws=ws,
        spec=spec,
        regime=regime,
        stages=stages,
        warnings=warnings,
        bake=bake,
        normal_res=normal_res,
        split=split,
        max_parts=max_parts,
        object_hint=object_hint,
    )


def _fill_skipped_stages(stages: list[StageOutcome], remaining_stage_names: list[str]) -> None:
    for name in remaining_stage_names:
        stages.append(
            StageOutcome(
                name=name,
                status="skipped",
                detail="skipped due to previous stage status",
                seconds=0.0,
            )
        )


def run_build(
    photos_dir: Path | str,
    workspace: Workspace | Path | str,
    *,
    executor: Executor | None = None,
    dense: bool = True,
    bake: bool = True,
    normal_res: int = 1024,
    generative_backend: str | None = None,
    split: bool = False,
    max_parts: int = 8,
    object_hint: str | None = None,
) -> BuildResult:
    """Run the end-to-end PixieCAD pipeline.

    Stage failures never raise: each is recorded as a StageOutcome and a
    partial BuildResult is returned. Only programmer error (bad photos_dir,
    unopenable workspace) raises.

    Note on dense: without a dense surface there is nothing for S4/S6 to
    operate on — sparse SfM yields a point cloud, not a mesh — so when dense is
    disabled, unavailable, or fails, the remaining mesh stages are recorded
    "skipped" and a partial result is returned.
    """
    photos_dir = Path(photos_dir)
    if not photos_dir.exists() or not photos_dir.is_dir():
        raise ValueError(f"photos_dir does not exist or is not a directory: {photos_dir}")

    if isinstance(workspace, Workspace):
        ws = workspace
    elif isinstance(workspace, (str, Path)):
        ws_path = Path(workspace)
        if ws_path.exists() and (ws_path / "manifest.json").exists():
            ws = Workspace.open(ws_path)
        else:
            ws = Workspace.create(ws_path, ObjectSpec())
    else:
        raise ValueError(f"Invalid workspace parameter: {workspace}")

    spec = ws.spec()
    stages: list[StageOutcome] = []
    warnings: list[str] = []

    # S0 Ingest
    t0 = time.monotonic()
    try:
        report = run_ingest(photos_dir, ws)
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="ingest",
                status="ok",
                detail=f"{report.accepted} accepted, {report.duplicates} dupes, {report.rejected} rejected",
                seconds=dt,
            )
        )
        regime = detect_regime(report.accepted)
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="ingest",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["sparse", "dense", "scale", "clean", "decimate", "bake", "export"])
        return BuildResult(
            regime=Regime.SINGLE_IMAGE,
            stages=stages,
            glb_path=None,
            faces=None,
            scale_applied=None,
            warnings=warnings,
        )

    if regime != Regime.ORBIT:
        warnings.append(
            f"Regime is {regime.value}: photogrammetry needs >=16 well-distributed "
            "photos of a real object. Falling back to a generative backend — "
            "unseen surfaces will be invented, and dimensions are only as good "
            "as the ones you declare."
        )
        return _run_generative(
            report=report,
            regime=regime,
            ws=ws,
            spec=spec,
            stages=stages,
            warnings=warnings,
            backend=generative_backend,
            bake=bake,
            normal_res=normal_res,
            split=split,
            max_parts=max_parts,
            object_hint=object_hint,
        )

    # S2a Sparse
    t0 = time.monotonic()
    sparse_result: SparseResult | None = None
    try:
        images_dir = _ingested_images_dir(report)
        if images_dir is None:
            raise SparseFailure(
                "No ingested working images found — every photo was rejected or "
                "de-duplicated at S0; see the ingest report for per-photo reasons."
            )

        sparse_result = run_sparse(images_dir, ws)
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="sparse",
                status="ok",
                detail=f"{sparse_result.n_registered}/{sparse_result.n_images_in} images registered, {sparse_result.n_points3d} points",
                seconds=dt,
            )
        )
    except SparseFailure as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="sparse",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["dense", "scale", "clean", "decimate", "bake", "export"])
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=None,
            scale_applied=None,
            warnings=warnings,
        )
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="sparse",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["dense", "scale", "clean", "decimate", "bake", "export"])
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=None,
            scale_applied=None,
            warnings=warnings,
        )

    # S2b Dense
    t0 = time.monotonic()
    mesh: trimesh.Trimesh | None = None
    dense_mesh: trimesh.Trimesh | None = None

    if dense and executor is not None:
        try:
            dense_res = run_dense(images_dir, Path(sparse_result.model_dir), ws, executor)
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="dense",
                    status="ok",
                    detail=f"dense reconstruction produced {dense_res.n_faces} faces",
                    seconds=dt,
                )
            )
            loaded = trimesh.load(dense_res.mesh_path, process=False)
            # Scene.dump(concatenate=True) is deprecated for removal in trimesh;
            # to_mesh() is the supported concatenation path.
            mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
            dense_mesh = mesh.copy()
        except DenseUnavailable as e:
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="dense",
                    status="skipped",
                    detail=str(e),
                    seconds=dt,
                )
            )
            _fill_skipped_stages(stages, ["scale", "clean", "decimate", "bake", "export"])
            return BuildResult(
                regime=regime,
                stages=stages,
                glb_path=None,
                faces=None,
                scale_applied=None,
                warnings=warnings,
            )
        except Exception as e:
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="dense",
                    status="failed",
                    detail=str(e),
                    seconds=dt,
                )
            )
            _fill_skipped_stages(stages, ["scale", "clean", "decimate", "bake", "export"])
            return BuildResult(
                regime=regime,
                stages=stages,
                glb_path=None,
                faces=None,
                scale_applied=None,
                warnings=warnings,
            )
    else:
        dt = time.monotonic() - t0
        reason = "dense disabled" if not dense else "no executor provided"
        stages.append(
            StageOutcome(
                name="dense",
                status="skipped",
                detail=reason,
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["scale", "clean", "decimate", "bake", "export"])
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=None,
            scale_applied=None,
            warnings=warnings,
        )

    return _mesh_tail(
        mesh,
        dense_mesh,
        ws=ws,
        spec=spec,
        regime=regime,
        stages=stages,
        warnings=warnings,
        bake=bake,
        normal_res=normal_res,
        split=split,
        max_parts=max_parts,
        object_hint=object_hint,
    )


def _mesh_tail(
    mesh,
    dense_mesh,
    *,
    ws: Workspace,
    spec: ObjectSpec,
    regime: Regime,
    stages: list[StageOutcome],
    warnings: list[str],
    bake: bool,
    normal_res: int,
    split: bool = False,
    max_parts: int = 8,
    object_hint: str | None = None,
) -> BuildResult:
    """Scale -> clean -> decimate -> unwrap/bake -> export.

    Shared by the photogrammetry and generative paths: once either has
    produced a surface, everything downstream is identical. ``dense_mesh``
    is the high-detail source the normal map is baked from.
    """
    # S1 Scale
    t0 = time.monotonic()
    scale_applied: float | None = None
    if spec.dimensions.any_known:
        try:
            factor = scale_factor_from_dimensions(tuple(mesh.extents), spec.dimensions)
            new_verts = apply_scale(mesh.vertices, factor)
            mesh = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces, process=False)
            if dense_mesh is not None:
                dense_verts = apply_scale(dense_mesh.vertices, factor)
                dense_mesh = trimesh.Trimesh(vertices=dense_verts, faces=dense_mesh.faces, process=False)
            scale_applied = factor
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="scale",
                    status="ok",
                    detail=f"scaled by {factor:.4f}",
                    seconds=dt,
                )
            )
        except Exception as e:
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="scale",
                    status="failed",
                    detail=str(e),
                    seconds=dt,
                )
            )
            _fill_skipped_stages(stages, ["clean", "decimate", "bake", "export"])
            return BuildResult(
                regime=regime,
                stages=stages,
                glb_path=None,
                faces=None,
                scale_applied=None,
                warnings=warnings,
            )
    else:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="scale",
                status="skipped",
                detail="no dimensions specified",
                seconds=dt,
            )
        )

    # S4 Clean
    t0 = time.monotonic()
    try:
        mesh, clean_report = clean_mesh(mesh)
        cleaned_mesh = mesh
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="clean",
                status="ok",
                detail=f"{clean_report.faces_before} -> {clean_report.faces_after} faces",
                seconds=dt,
            )
        )
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="clean",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["decimate", "bake", "export"])
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=None,
            scale_applied=scale_applied,
            warnings=warnings,
        )

    # S6a Decimate
    t0 = time.monotonic()
    source_faces = len(mesh.faces)
    faces: int | None = None
    try:
        mesh, dec_res = decimate_to_budget(mesh, spec.target_faces)
        faces = len(mesh.faces)
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="decimate",
                status="ok",
                detail=f"{dec_res.faces_before} -> {dec_res.faces_after} faces (target {spec.target_faces})",
                seconds=dt,
            )
        )
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="decimate",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["bake", "export"])
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=None,
            scale_applied=scale_applied,
            warnings=warnings,
        )

    # S6b Bake / Unwrap
    t0 = time.monotonic()
    normal_map = None
    try:
        unwrap_res = unwrap_uv(mesh)
        mesh = unwrap_res.mesh
        if bake:
            normal_map = bake_object_space_normals(dense_mesh, mesh, resolution=normal_res)
            detail = f"uv unwrapped, baked normal map ({normal_res}x{normal_res})"
        else:
            detail = "uv unwrapped (bake disabled)"
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="bake",
                status="ok",
                detail=detail,
                seconds=dt,
            )
        )
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="bake",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        _fill_skipped_stages(stages, ["export"])
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=faces,
            scale_applied=scale_applied,
            warnings=warnings,
        )

    # Export
    t0 = time.monotonic()
    glb_path: str | None = None
    try:
        out_dir = ws.root / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{spec.name}.glb"

        extras = {
            "regime": regime.value,
            "source_faces": source_faces,
            "target_faces": spec.target_faces,
            "scale_applied": scale_applied,
            "scale_source": spec.scale_source.value if hasattr(spec.scale_source, "value") else str(spec.scale_source),
        }

        exported_path = export_glb(
            mesh=mesh,
            out_path=out_path,
            normal_map=normal_map,
            extras=extras,
        )
        glb_path = str(exported_path)
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="export",
                status="ok",
                detail=f"exported to {glb_path}",
                seconds=dt,
            )
        )
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(
                name="export",
                status="failed",
                detail=str(e),
                seconds=dt,
            )
        )
        return BuildResult(
            regime=regime,
            stages=stages,
            glb_path=None,
            faces=faces,
            scale_applied=scale_applied,
            warnings=warnings,
        )

    parts_dir: str | None = None
    parts_info: list[dict] = []
    if split:
        t0 = time.monotonic()
        try:
            parts = split_parts(cleaned_mesh, max_parts=max_parts)
            named_parts = name_parts(parts, cleaned_mesh, object_hint=object_hint)
            parts_out_dir = ws.root / "output" / "parts"
            exported_parts, _ = export_parts(
                named_parts,
                parts_out_dir,
                total_budget=spec.target_faces,
                whole_volume=cleaned_mesh.volume,
                extras={"regime": regime.value},
            )
            parts_dir = str(parts_out_dir)
            parts_info = [
                {
                    "name": p.name,
                    "file": p.file,
                    "faces": p.faces,
                    "target_faces": p.target_faces,
                }
                for p in exported_parts
            ]
            first_names = [p.name for p in exported_parts[:4]]
            detail_str = f"{len(exported_parts)} parts: {', '.join(first_names)}"
            if len(exported_parts) > 4:
                detail_str += ", ..."
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="parts",
                    status="ok",
                    detail=detail_str,
                    seconds=dt,
                )
            )
        except Exception as e:
            dt = time.monotonic() - t0
            stages.append(
                StageOutcome(
                    name="parts",
                    status="failed",
                    detail=str(e),
                    seconds=dt,
                )
            )

    return BuildResult(
        regime=regime,
        stages=stages,
        glb_path=glb_path,
        faces=faces,
        scale_applied=scale_applied,
        warnings=warnings,
        parts_dir=parts_dir,
        parts=parts_info,
    )
