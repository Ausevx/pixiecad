"""Post-build finishing: smooth, texture, segment, and size for the web.

Lives outside ``app.py`` on purpose. These stages are the ones a user actually
chooses between in the dashboard, and they involve a remote GPU, so they need
to be testable without standing up FastAPI or a VM. The web layer only
collects options and reports progress; every decision is here.

Order is not arbitrary and is the main thing to get right:

1. smooth   -- geometry only, and free: vertex and face counts are unchanged
2. texture  -- must come after any decimation, because the simplifier discards
               UVs, so texturing first and decimating second silently strips
               the model back to grey
3. segment  -- after texturing, so parts inherit the texture. Safe because the
               paint pipeline preserves face order exactly (verified: maximum
               centroid drift 1e-6 over 20,000 faces)
4. web size -- last, since it only rewrites image resolution and touches
               neither geometry nor UVs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import trimesh


@dataclass
class FinishOptions:
    """What the dashboard lets a user ask for after the mesh exists."""

    smooth_iterations: int = 0
    texture: bool = False
    segmentation: str = "auto"  # "auto" (geometric) | "semantic" (SAM)
    web_export: bool = False
    texture_size: int = 1024
    max_parts: int = 8
    total_budget: int = 20000
    # On by default: recovering surface detail as a texture beats carrying the
    # polygons through the face budget. But it is not free -- this stage runs
    # on the user's own machine, not the GPU host, and is by far the heaviest
    # thing the local pipeline does. Measured against a 1.31M face mesh, the
    # scale a generative backend actually returns:
    #
    #   1024x1024   13.6 s   3.4 GB peak RSS
    #   2048x2048   52.9 s   4.1 GB peak RSS
    #
    # against ~250 MB for every other local stage combined. That is why
    # create_job clamps normal_res to 2048 rather than run_build's 4096
    # ceiling: 4096 would put a background dashboard job within reach of the
    # 16 GB dev machine's limit while its owner is still using it.
    bake_normals: bool = True
    normal_res: int = 1024
    # "auto" resolves to the host whenever one is configured, because the host
    # is already provisioned and already billed for the job, while this machine
    # is the one its owner is using. "local" and "host" are explicit overrides.
    bake_location: str = "auto"  # "auto" | "local" | "host"
    # Needed only by the stages that run on a GPU; absent means those stages
    # are skipped with an explanation rather than failing the whole job.
    gpu_host: str | None = None

    @property
    def needs_gpu(self) -> bool:
        return self.texture or self.segmentation == "semantic"


@dataclass
class FinishReport:
    model_path: Path
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parts: list[dict[str, Any]] = field(default_factory=list)
    parts_dir: Path | None = None
    textured: bool = False
    web_bytes: int | None = None


def _executor(host: str):
    from ..executors.ssh import SSHExecutor

    return SSHExecutor(host)


def finish_model(
    glb_path: str | Path,
    out_dir: str | Path,
    options: FinishOptions,
    *,
    conditioning_image: str | Path | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> FinishReport:
    """Apply the requested finishing stages to an existing model.

    Every stage is individually skippable and failures degrade rather than
    abort: a texture step that cannot reach its GPU host leaves a usable
    untextured model instead of losing the whole build.
    """
    glb_path, out_dir = Path(glb_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = FinishReport(model_path=glb_path)

    mesh = trimesh.load(glb_path, force="mesh", process=False)

    # Always, and first: costs nothing and is the single biggest visual win.
    # Measured on the F1, 90% of 4,800 UV-seam vertices had their shading
    # normals split by more than 10 degrees (mean 56, max 180), drawn by every
    # renderer as a hard crease. That reads as "the polygons are showing" and
    # sends people reaching for a higher face budget, which does not fix it.
    from ..meshops.smooth import weld_normals

    mesh = weld_normals(mesh)
    mesh.export(glb_path)
    report.steps.append("weld_normals")

    if options.smooth_iterations > 0:
        from ..meshops.smooth import smooth_mesh

        mesh, sr = smooth_mesh(mesh, iterations=options.smooth_iterations)
        mesh.export(glb_path)
        log(
            f"Smoothed {options.smooth_iterations} iterations "
            f"(volume {sr.volume_change:+.2%}, face count unchanged)"
        )
        report.steps.append("smooth")

    if options.needs_gpu and not options.gpu_host:
        report.warnings.append(
            "texturing and semantic segmentation need a GPU host; "
            "set one to enable them"
        )
        log(f"WARNING: {report.warnings[-1]}")

    if options.texture and options.gpu_host:
        if conditioning_image is None:
            report.warnings.append("no conditioning photo available for texturing")
            log(f"WARNING: {report.warnings[-1]}")
        else:
            try:
                from ..generative.hunyuan_remote import texture_mesh

                log("Texturing on GPU host (about 2 minutes)...")
                before_faces = len(mesh.faces)
                textured = texture_mesh(
                    _executor(options.gpu_host),
                    glb_path,
                    conditioning_image,
                    out_dir / "texture",
                    # Without this the paint stage remeshes to its hardcoded
                    # 40,000 faces and we publish that, so enabling texturing
                    # silently discarded the face budget the user chose.
                    simplify_target=options.total_budget or None,
                )
                # Replace the published model in place so downstream stages and
                # the download endpoint need no special-casing.
                mesh = trimesh.load(textured, force="mesh", process=False)
                after_faces = len(mesh.faces)
                if after_faces < before_faces * 0.9:
                    # Say it rather than let someone compare face counts and
                    # wonder. The paint stage owns this number, so we report
                    # the loss instead of pretending it did not happen.
                    report.warnings.append(
                        f"texturing remeshed {before_faces:,} faces down to "
                        f"{after_faces:,}"
                    )
                    log(f"WARNING: {report.warnings[-1]}")
                mesh.export(glb_path)
                report.textured = True
                report.steps.append("texture")
                log("Texture applied.")
            except Exception as exc:
                report.warnings.append(f"texturing failed: {exc}")
                log(f"WARNING: {report.warnings[-1]}")

    parts_objs = _segment(mesh, options, report, log)

    if parts_objs:
        from ..parts.export import export_parts

        parts_dir = out_dir / "parts"
        exported, _ = export_parts(
            parts_objs,
            parts_dir,
            total_budget=options.total_budget,
            whole_volume=abs(mesh.volume) or None,
        )
        report.parts_dir = parts_dir
        report.parts = [{"file": e.file, "faces": e.faces, "name": e.name} for e in exported]
        log(f"Exported {len(exported)} parts.")

        if options.web_export:
            _shrink_parts(parts_dir, report, options, log)

    if options.web_export:
        from ..meshops.webexport import export_for_web

        web = export_for_web(mesh, out_dir / "model_web.glb", texture_size=options.texture_size)
        report.web_bytes = web.bytes_after
        report.steps.append("web_export")
        log(
            f"Web model: {web.bytes_after / 1e6:.2f} MB "
            f"({web.ratio:.0%} of {web.bytes_before / 1e6:.2f} MB), "
            f"texture {web.texture_after}"
        )

    return report


def run_semantic_segmentation(
    mesh,
    executor,
    max_parts: int,
    *,
    cache_dir: "Path | None" = None,
    log: "Callable[[str], None]" = lambda _m: None,
) -> "tuple[list, bool]":
    """Render views, run SAM on the GPU host, and fuse labels into parts.

    Returns ``(parts_list, used_cache)``.  Labels are persisted under
    ``cache_dir/sam-labels-<mesh_hash>/`` (keyed on vertex+face bytes) so a
    re-run with the same geometry skips the ~125 s SAM pass.
    ``PrecomputedSegmenter`` is used on the cache-hit path — no parallel
    caching mechanism alongside it.

    Raises on any failure so callers can decide on the fallback-with-warning
    style that fits their context (dashboard log vs. CLI echo).
    """
    import hashlib

    import numpy as np

    from ..parts.render import render_mesh
    from ..parts.sam_remote import PrecomputedSegmenter, RemoteSAMSegmenter
    from ..parts.segment import parts_from_labels
    from ..parts.semantic import segment_semantic

    # Key on vertex+face bytes: any geometry change invalidates the cache, and
    # nothing else (resolution, max_parts) changes the raw label maps.
    h = hashlib.sha256()
    h.update(mesh.vertices.tobytes())
    h.update(mesh.faces.tobytes())
    mesh_hash = h.hexdigest()[:16]

    renders: list = []
    labels: "dict[str, np.ndarray] | None" = None

    if cache_dir is not None:
        label_dir = Path(cache_dir) / f"sam-labels-{mesh_hash}"
        if (label_dir / "_done").exists():
            # render_mesh is cheap; redo it so Render objects' array identities
            # match what PrecomputedSegmenter will be handed below.
            log("SAM labels found in cache — skipping GPU inference.")
            renders = render_mesh(mesh, resolution=768)
            labels = {}
            for r in renders:
                npy = label_dir / f"{r.camera.name}.npy"
                if npy.exists():
                    labels[r.camera.name] = np.load(npy)
    else:
        label_dir = None

    used_cache = labels is not None
    if labels is None:
        log("Rendering 14 views for semantic segmentation...")
        renders = render_mesh(mesh, resolution=768)
        log("Running SAM on GPU host (about 2 minutes)...")
        labels = RemoteSAMSegmenter(executor).segment_renders(renders)

        if label_dir is not None:
            label_dir.mkdir(parents=True, exist_ok=True)
            for name, arr in labels.items():
                np.save(label_dir / f"{name}.npy", arr)
            (label_dir / "_done").write_text("ok\n")

    result = segment_semantic(
        mesh,
        PrecomputedSegmenter(renders, labels),
        renders=renders,
        max_parts=max_parts,
    )
    parts = parts_from_labels(mesh, result.face_labels, max_parts=max_parts)
    return parts, used_cache


def _segment(mesh, options: FinishOptions, report: FinishReport, log) -> list:
    """Split the mesh, semantically if asked for and possible."""
    from ..parts.segment import split_parts

    if options.segmentation == "semantic" and options.gpu_host:
        try:
            parts, _ = run_semantic_segmentation(
                mesh,
                _executor(options.gpu_host),
                options.max_parts,
                log=log,
            )
            log(f"Semantic segmentation produced {len(parts)} parts.")
            report.steps.append("segment:semantic")
            return parts
        except Exception as exc:
            report.warnings.append(f"semantic segmentation failed, using geometric: {exc}")
            log(f"WARNING: {report.warnings[-1]}")

    report.steps.append("segment:geometric")
    return split_parts(mesh, method="auto", max_parts=options.max_parts)


def _shrink_parts(parts_dir: Path, report: FinishReport, options: FinishOptions, log) -> None:
    """Re-export each part with a texture sized to its share of the model.

    Without this a 273-face wing element ships the same 2048x2048 atlas as the
    whole body: nine parts came to 62 MB for ~20k triangles. Scaling per part
    brings the same set to about 2 MB.
    """
    from ..meshops.webexport import export_for_web, part_texture_size

    files = sorted(parts_dir.glob("*.glb"))
    loaded = [(f, trimesh.load(f, force="mesh", process=False)) for f in files]
    total_faces = sum(len(m.faces) for _f, m in loaded) or 1

    before = sum(f.stat().st_size for f, _m in loaded)
    after = 0
    for path, part in loaded:
        size = part_texture_size(len(part.faces), total_faces, base=options.texture_size)
        after += export_for_web(part, path, texture_size=size).bytes_after
    if before:
        log(f"Parts sized for web: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB")
    report.steps.append("web_parts")
