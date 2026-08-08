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


def _generative_defaults(
    options: dict | None, *, spec: ObjectSpec, solidify_generated: bool = False
) -> dict:
    """Fill in worker knobs the job already knows the answer to.

    The TRELLIS worker takes a ``decimation_target`` and, when nobody sets one,
    picks its own default of **2,000,000 faces** -- it has no idea what the job
    asked for. So a build with a 300,000-face budget requested a 1.87M-face
    mesh and got one.

    But it must NOT be set when solidify is going to run, and that cost a whole
    build to learn. Solidify repairs a shattered surface by dilating the
    fragments until they touch and then filling the interior -- so it reads the
    dense mesh even though it replaces it. Decimating first starves it:

        1,872,364 faces in ->  fill 0.018 -> 0.958, one component
          298,336 faces in ->  fill 0.025 -> 0.047, 77 components

    Same object, same dilation, same grid. At a sixth of the density the gaps
    between fragments are wider than the dilation can bridge, the interior
    drains straight out, and the result is a hollow blob. Decimation belongs
    *after* the repair, which is exactly where our own decimate stage already
    sits -- so on the solidify path the worker is asked for everything it has
    and the budget is enforced locally.

    Unknown keys are dropped by each backend's own passthrough filter, so
    setting a TRELLIS knob costs nothing on the Hunyuan path.

    ``PIXIECAD_TRELLIS_MESH_PROFILE`` selects the worker's ``hd`` (its default:
    geometry at 1024, textures at 4096) or ``game_ready`` (512 and 2048, and a
    hard 300,000-face cap) profile. Left unset the worker keeps choosing ``hd``,
    which is what every run so far has used -- worth knowing when comparing
    output against a hosted demo, which need not be on the same profile.
    """
    import os

    from .generative.trellis_remote import TRELLIS_MESH_PROFILES

    out = dict(options or {})
    if spec.target_faces and not solidify_generated:
        out.setdefault("decimation_target", int(spec.target_faces))
    profile = os.environ.get("PIXIECAD_TRELLIS_MESH_PROFILE", "").strip().lower()
    if profile:
        # Validate rather than forward. The worker silently falls back to its
        # own default on an unrecognised profile, so a typo would look like it
        # took effect and quietly give back hd -- the exact confusion this knob
        # exists to resolve. It also keeps the value shell-safe: options are
        # interpolated into a curl command as -F "key=value" with no quoting of
        # the value itself.
        if profile not in TRELLIS_MESH_PROFILES:
            raise ValueError(
                f"PIXIECAD_TRELLIS_MESH_PROFILE={profile!r} is not a profile the "
                f"worker knows; use one of {sorted(TRELLIS_MESH_PROFILES)}"
            )
        out.setdefault("mesh_profile", profile)
    return out


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
    bake_location: str = "local",
    solidify_generated: bool = True,
    split: bool = False,
    max_parts: int = 8,
    object_hint: str | None = None,
    executor: Executor | None = None,
    generative_options: dict | None = None,
) -> BuildResult:
    """Regimes with too few photos to triangulate: invent the geometry instead.

    Records the photogrammetry stages as skipped (they genuinely never ran) and
    substitutes S3. The generated mesh is its own bake source — there is no
    higher-detail original, so the normal map only captures decimation loss.
    """
    from .generative import GenerateRequest, GenerativeError, run_generate

    if executor is not None:
        # Registering here is what makes "our own GPU first" the default:
        # auto-selection prefers trellis-remote over fal, and its availability
        # is decided by probing this executor's host at generate time.
        from .generative import (
            make_hunyuan_backend,
            make_trellis_backend,
            register_backend,
        )

        register_backend(lambda: make_hunyuan_backend(executor), "hunyuan-remote")
        register_backend(lambda: make_trellis_backend(executor), "trellis-remote")

    for name in ("sparse", "dense"):
        stages.append(
            StageOutcome(name, "skipped", f"not applicable in {regime.value} regime", 0.0)
        )

    t0 = time.monotonic()
    all_images = [Path(p.working_path) for p in report.photos if p.working_path]
    images = _select_conditioning_views(all_images)
    gen_options = _generative_defaults(
        generative_options, spec=spec, solidify_generated=solidify_generated
    )
    try:
        result = run_generate(
            GenerateRequest(images=images, options=gen_options),
            ws,
            backend=backend,
        )
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

    # Some backends return a surface sample rather than a body. A TRELLIS mesh
    # measured here: 40,341 disconnected patches, 609,689 open edges, enclosing
    # 1.8% of its own convex hull -- the right silhouette made of confetti,
    # with no interior. It reads as "a hollow mesh of polygons", and decimating
    # it only rearranges the fragments, so this has to happen before the mesh
    # tail rather than after.
    #
    # Only when the mesh is actually shattered: Hunyuan already encloses
    # 0.42-0.88 of its hull and is left untouched, verified against real output
    # from both backends.
    if solidify_generated:
        t_solid = time.monotonic()
        try:
            from .meshops.solidify import FILL_FLOOR as SOLIDIFY_FILL_FLOOR
            from .meshops.solidify import MAX_DIVISIONS as SOLIDIFY_MAX_DIVISIONS
            from .meshops.solidify import solidify

            # No target_faces. The grid used to follow the face budget so the
            # output could not undershoot it -- but every error solidify makes
            # is a fixed number of VOXELS, so that made dimensional accuracy a
            # function of how many triangles the user asked for. A 20,000-face
            # job got a 64^3 grid and came back 8.5% too thick on its thinnest
            # axis. It now runs at full resolution and decimate takes the count
            # down, which is what decimate is for.
            outcome = solidify(mesh)
            if outcome.applied:
                mesh = outcome.mesh
                if outcome.repaired:
                    warnings.append(
                        f"generated surface was not a solid "
                        f"({outcome.components_before:,} disconnected pieces "
                        f"enclosing {outcome.fill_before:.2f} of its hull); "
                        f"rebuilt as a watertight body ({outcome.fill_after:.2f})"
                    )
                else:
                    # This must not read as a success. The output of a failed
                    # repair is watertight, single-component and exactly on
                    # budget -- it passes every downstream check and is a blob.
                    # Reported green once, and the user spent twenty minutes
                    # texturing it before finding out.
                    warnings.append(
                        f"THE MODEL IS NOT USABLE: the generated surface could "
                        f"not be closed (encloses {outcome.fill_after:.2f} of "
                        f"its hull, healthy is above {SOLIDIFY_FILL_FLOOR:.2f}). "
                        f"Generation is stochastic -- re-running the same photos "
                        f"usually gives a good mesh."
                    )
                # The grid is a ceiling decimation cannot raise, and dropping
                # target_faces removed the retry that used to guard it. At
                # ~9.5 faces per division squared a 256 grid yields about
                # 620,000, so only a very large budget can undershoot -- and
                # when it does the user should hear it from here rather than
                # comparing face counts and guessing.
                if spec.target_faces and len(mesh.faces) < spec.target_faces * 0.8:
                    warnings.append(
                        f"the rebuilt surface has {len(mesh.faces):,} faces, "
                        f"short of the {spec.target_faces:,} asked for: the voxel "
                        f"grid is capped at {SOLIDIFY_MAX_DIVISIONS}^3 and "
                        "decimation can only remove faces, not add them"
                    )
            # Three outcomes, not two. A mesh that never needed repairing is
            # the BEST case -- Hunyuan output encloses 0.97 of its hull and is
            # deliberately left alone -- and reporting that as [FAILED] tells
            # the user their good model is broken. Only a repair that was
            # attempted and did not close the surface is a failure.
            if outcome.applied and outcome.repaired:
                solid_status = "ok"
            elif outcome.repaired:
                solid_status = "skipped"
            else:
                solid_status = "failed"
            stages.append(
                StageOutcome(
                    "solidify",
                    solid_status,
                    outcome.reason,
                    time.monotonic() - t_solid,
                )
            )
        except Exception as e:
            # Never fatal: an unrepaired mesh is still a mesh, and the sanity
            # check below will say what is wrong with it.
            stages.append(
                StageOutcome("solidify", "skipped", f"solidify failed: {e}",
                             time.monotonic() - t_solid)
            )

    # Structural sanity: a failed generation still yields a valid, correctly
    # budgeted, UV-mapped glTF -- it is just not an object. Every numeric check
    # passed on a mesh that turned out to be a cube of noise, so shape is
    # checked explicitly and loudly rather than assumed.
    from .meshops.sanity import check_mesh

    report = check_mesh(mesh)
    stages.append(
        StageOutcome(
            "sanity",
            "ok" if report.ok else "failed",
            report.summary(),
            0.0,
        )
    )
    for w in report.warnings:
        warnings.append(f"Generated geometry looks wrong: {w}")
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
        bake_location=bake_location,
        executor=executor,
        split=split,
        max_parts=max_parts,
        object_hint=object_hint,
    )


def _select_conditioning_views(images: list[Path], limit: int = 4) -> list[Path]:
    """Pick which photos a generative backend gets, best-first.

    Backends cap conditioning images (four, typically), so this decides which
    four. Naive truncation took them in filename order, which for a normal
    8-view capture meant front, front-left, left, rear-left -- two obliques
    that multi-view models discard, silently halving the real coverage and
    leaving the entire back and right of the object unseen.

    Cardinal views go first because they are the ones a multi-view model can
    actually key on; anything left over fills the remaining slots so
    single-view backends still receive a sensible primary image.
    """
    from .generative.hunyuan_remote import map_views

    cardinals = map_views(images)
    ordered = [cardinals[t] for t in ("front", "left", "back", "right") if t in cardinals]
    for img in images:
        if img not in ordered:
            ordered.append(img)
    return ordered[:limit]


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
    bake_location: str = "local",
    solidify_generated: bool = True,
    generative_backend: str | None = None,
    split: bool = False,
    max_parts: int = 8,
    object_hint: str | None = None,
    generative_options: dict | None = None,
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
            bake_location=bake_location,
            solidify_generated=solidify_generated,
            split=split,
            max_parts=max_parts,
            object_hint=object_hint,
            executor=executor,
            generative_options=generative_options,
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
        bake_location=bake_location,
        executor=executor,
        split=split,
        max_parts=max_parts,
        object_hint=object_hint,
    )


def _first_line(text: str) -> str:
    """The actionable line of a traceback-laden error, for a one-line detail.

    A remote failure arrives as an entire remote traceback. The last line is
    the exception itself, which is the part that says what to fix.
    """
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1][:160] if lines else str(text)[:160]


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
    bake_location: str = "local",
    executor: Executor | None = None,
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

    # S6b Unwrap, then bake.
    #
    # Deliberately two steps with different consequences. Unwrapping produces
    # the UVs the export depends on, so losing it is fatal. Baking only adds a
    # normal map -- detail the renderer uses and a 3D printer ignores entirely
    # -- so losing it must NOT be.
    #
    # They used to share one try block, and a bake failure discarded the whole
    # build. That cost a real job 333 s of GPU generation over a missing
    # module in the worker image: the mesh was finished, correct, and sitting
    # in memory, and the pipeline threw it away because an optional cosmetic
    # stage raised. Never again.
    t0 = time.monotonic()
    try:
        unwrap_res = unwrap_uv(mesh)
        mesh = unwrap_res.mesh
    except Exception as e:
        dt = time.monotonic() - t0
        stages.append(
            StageOutcome(name="bake", status="failed", detail=f"uv unwrap failed: {e}", seconds=dt)
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

    normal_map = None
    bake_error: str | None = None
    if bake:
        try:
            if bake_location == "host" and executor is not None:
                # No local fallback on purpose: choosing the host is a
                # statement about the local machine's memory, so silently
                # baking there anyway would defeat the point.
                from .meshops.bake_remote import bake_object_space_normals_remote

                normal_map = bake_object_space_normals_remote(
                    executor, dense_mesh, mesh, resolution=normal_res
                )
                detail = f"uv unwrapped, baked normal map on host ({normal_res}x{normal_res})"
            else:
                normal_map = bake_object_space_normals(
                    dense_mesh, mesh, resolution=normal_res
                )
                detail = f"uv unwrapped, baked normal map ({normal_res}x{normal_res})"
        except Exception as e:
            bake_error = str(e)
            detail = f"uv unwrapped; normal map skipped ({_first_line(bake_error)})"
    else:
        detail = "uv unwrapped (bake disabled)"

    dt = time.monotonic() - t0
    stages.append(
        StageOutcome(
            name="bake",
            # "skipped", not "failed": the build is intact and the model is
            # about to be exported. Reporting failure here would say the job
            # died when it did not.
            status="skipped" if bake_error else "ok",
            detail=detail,
            seconds=dt,
        )
    )
    if bake_error:
        warnings.append(
            f"normal map not baked, model exported without it: {_first_line(bake_error)}"
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
