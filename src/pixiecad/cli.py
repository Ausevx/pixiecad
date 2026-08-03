"""PixieCAD CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from .ingest import run_ingest
from .spec import Dimensions, ObjectSpec
from .workspace import Workspace


def _latest_stage_dir(ws: Workspace, prefix: str) -> Path:
    """Most recent finished stage dir whose key starts with prefix."""
    done = [
        d for d in ws.stages_dir.iterdir()
        if d.name.startswith(prefix) and (d / "_done").exists()
    ]
    if not done:
        raise typer.BadParameter(
            f"No finished '{prefix}' stage in {ws.root}; run the earlier stage first."
        )
    return max(done, key=lambda d: d.stat().st_mtime)

app = typer.Typer(help="Photos in, spec-accurate budgeted-polygon 3D models out.")


@app.command()
def init(
    workspace: Path = typer.Argument(..., help="Workspace directory to create"),
    name: str = typer.Option("object", help="Object name"),
    length: str = typer.Option(None, help="Real length, e.g. '4.5m'"),
    width: str = typer.Option(None, help="Real width, e.g. '1.8m'"),
    height: str = typer.Option(None, help="Real height, e.g. '1.4m'"),
    target_faces: int = typer.Option(20_000, help="Total triangle budget"),
):
    """Create a workspace with an ObjectSpec."""
    spec = ObjectSpec(
        name=name,
        dimensions=Dimensions.parse(length, width, height),
        target_faces=target_faces,
    )
    Workspace.create(workspace, spec)
    typer.echo(f"Workspace ready at {workspace} (scale source: {spec.scale_source.value})")


@app.command()
def ingest(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Photo directory"),
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace directory"),
    long_edge: int = typer.Option(1600, help="Working resolution (long edge, px)"),
):
    """Stage 0: score, dedupe, and optimize input photos."""
    ws = Workspace.open(workspace)
    report = run_ingest(photos, ws, working_long_edge=long_edge)
    typer.echo(
        f"accepted={report.accepted} duplicates={report.duplicates} "
        f"rejected={report.rejected} unreadable={report.unreadable}"
    )
    for rec in report.photos:
        if rec.status == "rejected":
            typer.echo(f"  ✗ {Path(rec.source).name}: {'; '.join(rec.reject_reasons)}")
        elif rec.warnings:
            typer.echo(f"  ⚠ {Path(rec.source).name}: {'; '.join(rec.warnings)}")
    for line in report.advice:
        typer.echo(f"→ {line}")


@app.command()
def sparse(
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace directory"),
    images: Path = typer.Option(
        None, help="Image directory (default: latest ingest stage output)"
    ),
    matcher: str = typer.Option("exhaustive", help="exhaustive | sequential"),
):
    """Stage 2a: sparse SfM — camera poses + sparse point cloud (pycolmap)."""
    from .geometry import SparseFailure, run_sparse

    ws = Workspace.open(workspace)
    images_dir = images or _latest_stage_dir(ws, "s0-ingest") / "images"
    try:
        res = run_sparse(images_dir, ws, matcher=matcher)
    except SparseFailure as e:
        typer.echo(f"reconstruction failed: {e}", err=True)
        raise typer.Exit(1)
    cached = " (cached)" if res.cached else ""
    typer.echo(
        f"registered {res.n_registered}/{res.n_images_in} images, "
        f"{res.n_points3d} points, reproj err {res.mean_reproj_error:.2f}px{cached}"
    )
    typer.echo(f"model: {res.model_dir}")


@app.command()
def probe(
    host: str = typer.Option(
        None, help="SSH host to probe (default: probe the local machine)"
    ),
):
    """Report a machine's compute capabilities (GPU name, VRAM, compute cap)."""
    from .executors import LocalExecutor, SSHExecutor

    ex = SSHExecutor(host) if host else LocalExecutor()
    caps = ex.probe()
    if not caps.reachable:
        typer.echo(f"{caps.hostname}: unreachable", err=True)
        raise typer.Exit(1)
    gpu = (
        f"{caps.gpu.name} | {caps.gpu.vram_mb} MB VRAM | compute {caps.gpu.compute_cap}"
        if caps.gpu
        else "no NVIDIA GPU"
    )
    typer.echo(f"{caps.hostname}: {gpu}")


@app.command()
def optimize(
    mesh_file: Path = typer.Argument(..., exists=True, help="Dense mesh (.ply/.obj/.glb/.stl)"),
    out: Path = typer.Option("model.glb", "--out", "-o", help="Output .glb path"),
    target_faces: int = typer.Option(20_000, "--target-faces", "-t", help="Exact triangle budget"),
    bake: bool = typer.Option(True, help="Bake an object-space normal map from the dense mesh"),
    resolution: int = typer.Option(1024, help="Normal map resolution (px)"),
):
    """S4→S6: cleanup → exact-budget decimation → UV unwrap → normal bake → .glb."""
    import trimesh

    from .export import export_glb
    from .meshops import (
        bake_object_space_normals,
        clean_mesh,
        decimate_to_budget,
        unwrap_uv,
    )

    loaded = trimesh.load(mesh_file, force="mesh", process=False)
    typer.echo(f"loaded: {len(loaded.faces)} faces")

    cleaned, crep = clean_mesh(loaded)
    typer.echo(
        f"cleanup: {crep.faces_before} → {crep.faces_after} faces, "
        f"{crep.components_removed} floater(s) removed, watertight={crep.watertight}"
    )

    low, drep = decimate_to_budget(cleaned, target_faces)
    exact = "exact" if drep.achieved_exact else "closest achievable"
    typer.echo(f"decimate: {drep.faces_after} faces (target {target_faces}, {exact})")

    unwrapped = unwrap_uv(low)
    normal_map = None
    if bake:
        normal_map = bake_object_space_normals(
            cleaned, unwrapped.mesh, resolution=resolution
        )
        typer.echo(f"baked {resolution}x{resolution} object-space normal map")

    path = export_glb(
        unwrapped.mesh,
        out,
        normal_map=normal_map,
        extras={"generator": "pixiecad", "source_faces": crep.faces_before,
                "target_faces": target_faces},
    )
    typer.echo(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


@app.command()
def build(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Photo directory"),
    workspace: Path = typer.Option(..., "--workspace", "-w", help="Workspace directory"),
    host: str = typer.Option(None, help="SSH host for the GPU dense stage (omit to skip dense)"),
    bake: bool = typer.Option(True, help="Bake a normal map from the dense mesh"),
    normal_res: int = typer.Option(1024, help="Normal map resolution (px)"),
    backend: str = typer.Option(
        None, help="Generative backend for <16 photos (fal | trellis-remote | fake)"
    ),
    split: bool = typer.Option(False, "--split", help="Also export named parts"),
    max_parts: int = typer.Option(8, help="Maximum number of parts when splitting"),
    object_hint: str = typer.Option(
        None, "--object", help="What the object is (improves part naming), e.g. 'an F1 car'"
    ),
):
    """Run the whole pipeline: photos in, budgeted .glb out.

    With 16+ usable photos of a real object this measures it (photogrammetry).
    With fewer, it falls back to a generative backend, which invents geometry.
    --split additionally exports each component as its own .glb.
    """
    from .executors import SSHExecutor
    from .pipeline import run_build

    executor = SSHExecutor(host) if host else None
    result = run_build(
        photos, workspace, executor=executor, dense=bool(host),
        bake=bake, normal_res=normal_res, generative_backend=backend,
        split=split, max_parts=max_parts, object_hint=object_hint,
    )

    typer.echo(f"regime: {result.regime.value}")
    for line in result.summary_lines():
        typer.echo("  " + line)
    for w in result.warnings:
        typer.echo(f"⚠ {w}")
    if result.scale_applied:
        typer.echo(f"scale: ×{result.scale_applied:.4f} (metric)")
    if result.glb_path:
        typer.echo(f"✓ {result.glb_path} ({result.faces} faces)")
        for p in result.parts:
            typer.echo(f"    part: {p['name']:<24} {p['faces']:>6} faces  {p['file']}")
    else:
        typer.echo("no model produced — see stage statuses above", err=True)
        raise typer.Exit(1)


@app.command()
def run(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Photo directory"),
    runs_root: Path = typer.Option(
        "runs", "--runs", "-r", help="Root directory holding one folder per run"
    ),
    label: str = typer.Option(None, "--label", "-l", help="Name for this run (used in the folder name)"),
    target_faces: int = typer.Option(20_000, "--faces", "-t", help="Total triangle budget"),
    length: str = typer.Option(None, help="Real length, e.g. '5.6m'"),
    width: str = typer.Option(None, help="Real width, e.g. '2.0m'"),
    height: str = typer.Option(None, help="Real height, e.g. '0.95m'"),
    host: str = typer.Option(None, help="SSH host for GPU stages (dense and/or generative)"),
    backend: str = typer.Option(None, help="Generative backend (trellis-remote | fal | fake)"),
    bake: bool = typer.Option(True, help="Bake a normal map"),
    normal_res: int = typer.Option(1024, help="Normal map resolution (px)"),
    split: bool = typer.Option(True, "--split/--no-split", help="Also export named parts"),
    max_parts: int = typer.Option(8, help="Maximum number of parts"),
    object_hint: str = typer.Option(None, "--object", help="What the object is, e.g. 'an F1 car'"),
    drawings: bool = typer.Option(False, "--drawings", help="Also write ortho SVG drawings"),
    fast: bool = typer.Option(False, "--fast", help="Fast generative profile (minutes, not tens of minutes)"),
    multiview: bool = typer.Option(
        False, "--multiview",
        help="Condition on up to 4 cardinal views (older 2.0-line model; usually worse)",
    ),
    segmentation: str = typer.Option(
        "auto",
        "--segmentation",
        help="Part segmentation method: auto (geometric, default) | semantic (SAM on GPU host)",
    ),
):
    """Headless end-to-end run: photos in, .glb (+ parts) out, in a fresh session folder.

    Creates ``<runs>/<timestamp>-<label>/`` holding ``input/``, ``work/`` and
    ``output/``. Nothing outside that folder is touched, so runs never collide
    and an old run stays reproducible.
    """
    from .executors import SSHExecutor
    from .pipeline import run_build
    from .session import new_session

    # Validate --segmentation before any expensive work begins so the user
    # sees a clear message rather than a confusing mid-build traceback.
    if segmentation not in ("auto", "semantic"):
        typer.echo(
            f"⚠ unknown --segmentation value '{segmentation}'; "
            "valid choices are 'auto' and 'semantic'. Falling back to auto.",
        )
        segmentation = "auto"

    if segmentation == "semantic" and not split:
        typer.echo(
            "⚠ --segmentation semantic has no effect when --no-split is active; "
            "using auto.",
        )
        segmentation = "auto"

    if segmentation == "semantic" and not host:
        # Print a one-line explanation and keep going; silently downgrading is
        # just as confusing as hard-failing when the user intended semantic.
        typer.echo(
            "⚠ --segmentation semantic needs a GPU host (--host); "
            "using geometric segmentation instead.",
        )
        segmentation = "auto"

    session = new_session(runs_root, label=label, source=photos)
    staged = session.stage_inputs(photos)
    typer.echo(f"session: {session.root}")
    typer.echo(f"input:   {len(staged)} image(s) → {session.input_dir}")

    spec = ObjectSpec(
        name=slugify_name(label) or "model",
        dimensions=Dimensions.parse(length, width, height),
        target_faces=target_faces,
    )
    ws = Workspace.create(session.work_dir / "ws", spec)

    executor = SSHExecutor(host) if host else None
    if multiview:
        import os

        os.environ["PIXIECAD_HUNYUAN_MULTIVIEW"] = "1"
    # The HD default (1024_cascade) exceeded 27 minutes on an L4 and timed out.
    # game_ready at 512 is the profile that actually finishes on this hardware.
    gen_options = (
        {
            "mesh_profile": "game_ready",
            "geometry_resolution": 512,
            "texture_generation_mode": "fast_512",
        }
        if fast
        else None
    )
    result = run_build(
        session.input_dir, ws, executor=executor, dense=bool(host),
        bake=bake, normal_res=normal_res, generative_backend=backend,
        split=split, max_parts=max_parts, object_hint=object_hint,
        generative_options=gen_options,
    )

    typer.echo(f"regime:  {result.regime.value}")
    for line in result.summary_lines():
        typer.echo("  " + line)
    for w in result.warnings:
        typer.echo(f"⚠ {w}")

    # If semantic segmentation was requested and the build produced a mesh with
    # parts enabled, replace the geometric split with a SAM-fused one.
    # pipeline.py is untouched; run_semantic_segmentation (shared with the
    # dashboard) owns the render→SAM→fuse logic for both callers.
    if segmentation == "semantic" and result.glb_path and split:
        import trimesh

        from .parts import export_parts
        from .vision.triage import name_parts
        from .web.finishing import run_semantic_segmentation

        try:
            mesh = trimesh.load(result.glb_path, force="mesh", process=False)
            typer.echo("semantic segmentation: rendering views and calling SAM...")
            # Labels are cached in the workspace stages directory; a re-run of
            # the same session (same mesh bytes) skips the ~125 s GPU pass.
            parts_list, used_cache = run_semantic_segmentation(
                mesh,
                executor,
                max_parts,
                cache_dir=ws.stages_dir,
                log=lambda m: typer.echo(f"  {m}"),
            )
            if used_cache:
                typer.echo("  (labels loaded from cache)")
            named = name_parts(parts_list, mesh, object_hint=object_hint)
            # Export into the workspace output dir so collect_outputs can copy
            # to session.output_dir/parts the same way the geometric path does.
            parts_out = ws.root / "output" / "parts"
            exported, _ = export_parts(
                named,
                parts_out,
                total_budget=target_faces,
                whole_volume=abs(mesh.volume) or None,
            )
            result.parts[:] = [
                {
                    "name": e.name,
                    "file": e.file,
                    "faces": e.faces,
                    "target_faces": e.target_faces,
                }
                for e in exported
            ]
            result.parts_dir = str(parts_out)
        except Exception as exc:
            typer.echo(f"⚠ semantic segmentation failed, using geometric parts: {exc}")

    collected = collect_outputs(result, session, drawings=drawings, spec=spec)
    session.write_meta(
        photos_source=str(Path(photos).resolve()),
        regime=result.regime.value,
        target_faces=target_faces,
        backend=backend,
        host=host,
        glb=collected.get("model"),
        parts=[p["name"] for p in result.parts],
    )

    if not result.glb_path:
        typer.echo(f"no model produced — inputs kept at {session.input_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\noutput:  {session.output_dir}")
    typer.echo(f"  model.glb              {result.faces} faces")
    for p in result.parts:
        typer.echo(f"  parts/{Path(p['file']).name:<22} {p['faces']:>6} faces  ({p['name']})")
    for extra in collected.get("drawings", []):
        typer.echo(f"  {extra}")


def slugify_name(label: str | None) -> str:
    """Spec name from a user label (empty string when there is no label)."""
    from .session import slugify

    return slugify(label) if label else ""


def collect_outputs(result, session, *, drawings: bool, spec) -> dict:
    """Copy a build's deliverables out of the workspace into ``output/``.

    The workspace is scratch — it holds the stage cache and every intermediate.
    ``output/`` is what the user actually wants, under stable names, so a
    downstream script never has to guess at a spec-derived filename.
    """
    import json
    import shutil

    collected: dict = {}
    if result.glb_path:
        dest = session.output_dir / "model.glb"
        shutil.copy2(result.glb_path, dest)
        collected["model"] = str(dest)

    if result.parts:
        parts_dir = session.output_dir / "parts"
        parts_dir.mkdir(exist_ok=True)
        missing = []
        for p in result.parts:
            # ``file`` is a bare filename, not a path: resolving it relative to
            # the process CWD made every copy silently skip, and the user got
            # an output/parts holding nothing but a manifest.
            name = Path(p["file"]).name
            src = Path(result.parts_dir) / name if result.parts_dir else Path(name)
            if src.is_file():
                shutil.copy2(src, parts_dir / name)
            else:
                missing.append(name)
        if missing:
            # Loud, because a quietly empty parts folder looks like "this
            # object has no parts" rather than "the copy step is broken".
            raise FileNotFoundError(
                f"{len(missing)} part file(s) missing from {result.parts_dir}: "
                f"{', '.join(missing[:5])}"
            )
        src_manifest = Path(result.parts_dir) / "manifest.json" if result.parts_dir else None
        if src_manifest and src_manifest.exists():
            shutil.copy2(src_manifest, parts_dir / "manifest.json")

    if drawings and result.glb_path:
        import trimesh

        from .meshops import render_all, save_views

        mesh = trimesh.load(result.glb_path, force="mesh", process=False)
        views = render_all(mesh, dimensions=spec.dimensions)
        collected["drawings"] = [
            str(p.relative_to(session.output_dir))
            for p in save_views(views, session.output_dir / "drawings")
        ]

    build_json = session.output_dir / "build.json"
    build_json.write_text(
        json.dumps(
            {
                "session": session.session_id,
                "regime": result.regime.value,
                "faces": result.faces,
                "scale_applied": result.scale_applied,
                "warnings": result.warnings,
                "parts": result.parts,
                "stages": [
                    {"name": s.name, "status": s.status, "detail": s.detail,
                     "seconds": round(s.seconds, 3)}
                    for s in result.stages
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return collected


@app.command()
def drawings(
    mesh_file: Path = typer.Argument(..., exists=True, help="Mesh file (.glb/.ply/.obj/.stl)"),
    out_dir: Path = typer.Option("drawings", "--out", "-o", help="Output directory"),
    workspace: Path = typer.Option(None, "--workspace", "-w", help="Workspace, for dimensions"),
    size: int = typer.Option(800, help="Drawing size (px)"),
):
    """S7: orthographic front/side/top SVG drawings, annotated with dimensions."""
    import trimesh

    from .meshops import render_all, save_views

    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    dims = Workspace.open(workspace).spec().dimensions if workspace else None
    views = render_all(mesh, dimensions=dims, size_px=size)
    for path in save_views(views, out_dir):
        typer.echo(f"wrote {path}")


@app.command()
def convert(
    mesh_file: Path = typer.Argument(..., help="Input .glb / .obj / .stl / .ply"),
    out: Path = typer.Argument(..., help="Output path; format comes from the suffix"),
    size_mm: float = typer.Option(
        None, "--size-mm", help="Scale so the longest edge is this many millimetres"
    ),
    repair: bool = typer.Option(False, "--repair", help="Attempt to close holes first"),
):
    """Convert a model for CAD or 3D printing, and report whether it is a solid."""
    import trimesh

    from .meshops.cadexport import export_cad, inspect_solid

    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    typer.echo("input:")
    for line in inspect_solid(mesh).summary():
        typer.echo(f"  {line}")

    path, report, actions = export_cad(mesh, out, longest_mm=size_mm, repair=repair)
    for action in actions:
        typer.echo(f"  - {action}")
    typer.echo(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    typer.echo(
        "  ready for a slicer" if report.printable
        else "  NOT a closed solid -- try --repair, or fix in Meshmixer/Blender"
    )


@app.command()
def serve(
    root: Path = typer.Option("jobs", "--root", "-r", help="Directory for job workspaces"),
    port: int = typer.Option(8000, help="Port"),
    host_addr: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
):
    """Launch the local web dashboard."""
    import uvicorn

    from .web import create_app

    root.mkdir(parents=True, exist_ok=True)
    typer.echo(f"pixiecad dashboard → http://{host_addr}:{port}")
    uvicorn.run(create_app(root), host=host_addr, port=port, log_level="warning")


@app.command()
def parts(
    mesh_file: Path = typer.Argument(..., exists=True, help="Mesh to split (.glb/.ply/.obj)"),
    out_dir: Path = typer.Option("parts", "--out", "-o", help="Output directory"),
    total_faces: int = typer.Option(20_000, "--total-faces", "-t", help="Budget across all parts"),
    method: str = typer.Option("auto", help="auto | connected | cluster"),
    max_parts: int = typer.Option(8, help="Maximum number of parts"),
    object_hint: str = typer.Option(None, "--object", help="What the object is, e.g. 'an F1 car'"),
    name: bool = typer.Option(True, help="Name parts with a vision model when one is configured"),
):
    """S5: split a mesh into separately budgeted, separately exported parts."""
    import trimesh

    from .parts import export_parts, split_parts

    mesh = trimesh.load(mesh_file, force="mesh", process=False)
    found = split_parts(mesh, method=method, max_parts=max_parts)
    typer.echo(f"{len(found)} part(s) via {found[0].method}")

    if name:
        from .vision import name_parts

        found = name_parts(found, mesh, object_hint=object_hint)
        source = found[0].metadata.get("named_by", "geometry")
        typer.echo(f"named by: {source}")

    exported, manifest = export_parts(
        found, out_dir, total_budget=total_faces, whole_volume=mesh.volume
    )
    for e in exported:
        typer.echo(f"  {e.name:<24} {e.faces:>6} faces  {e.file}")
    typer.echo(f"manifest: {manifest}")


@app.command()
def triage(
    photos: Path = typer.Argument(..., exists=True, file_okay=False, help="Photo directory"),
    provider: str = typer.Option(None, help="anthropic | gemini (default: first configured)"),
):
    """S0.5: check photos before reconstructing — coverage, viewpoints, problems."""
    from .vision import triage_photos

    images = sorted(
        p for p in photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        typer.echo(f"no images in {photos}", err=True)
        raise typer.Exit(1)

    report = triage_photos(images, provider=provider)
    typer.echo(report.summary())
    typer.echo(f"visual distinctness: {report.distinctness:.3f}")
    for note in report.photos:
        flags = ("  ⚠ " + "; ".join(note.issues)) if note.issues else ""
        typer.echo(f"  {note.image:<28} {note.viewpoint}{flags}")
    if report.missing:
        typer.echo(f"missing viewpoints: {', '.join(report.missing)}")
    for a in report.advice:
        typer.echo(f"→ {a}")


@app.command()
def hfcheck():
    """Verify a HuggingFace token can reach TRELLIS.2's gated dependency.

    Worth one HTTP request before provisioning: without access the GPU worker
    downloads 19 GB and loads for minutes before failing with a 401.
    """
    import urllib.error
    import urllib.request

    from .generative.trellis_remote import GATED_REPO, resolve_hf_token

    token = resolve_hf_token()
    if not token:
        typer.echo("no token found (checked HF_TOKEN, HUGGING_FACE_HUB_TOKEN, ~/.cache/huggingface/token)", err=True)
        typer.echo("fix:  huggingface-cli login    (or export HF_TOKEN=hf_...)")
        raise typer.Exit(1)

    typer.echo(f"token found ({len(token)} chars)")
    url = f"https://huggingface.co/{GATED_REPO}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read(1)
        typer.echo(f"✓ {GATED_REPO} is accessible — the generative backend can run")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            typer.echo(f"✗ {GATED_REPO} returned {e.code}", err=True)
            typer.echo(f"  accept the licence at https://huggingface.co/{GATED_REPO}")
            typer.echo("  and make sure the token has read access to gated repos")
        else:
            typer.echo(f"✗ unexpected HTTP {e.code}: {e.reason}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        # Never let a token reach the terminal via an exception message.
        typer.echo(f"✗ could not reach HuggingFace: {type(e).__name__}", err=True)
        raise typer.Exit(1)


@app.command()
def backends():
    """List generative backends and whether each is usable right now."""
    from .generative import available_backends
    from .generative.base import _REGISTRY  # noqa: PLC2701 — introspection only

    usable = set(available_backends())
    for name in sorted(_REGISTRY):
        mark = "✓" if name in usable else "✗"
        note = "" if name in usable else "  (not configured)"
        typer.echo(f"  {mark} {name}{note}")
    typer.echo("\nremote backends need a GPU host:  pixiecad probe --host <alias>")


@app.command()
def cloud(
    watch: bool = typer.Option(False, help="Refresh every 10s until interrupted"),
):
    """Show gcloud status, running GPU instances, and estimated session spend."""
    import time as _time

    from .cloud import snapshot

    while True:
        snap = snapshot()
        g = snap.gcloud
        if not g.installed:
            typer.echo("gcloud: not installed", err=True)
            raise typer.Exit(1)
        typer.echo(f"gcloud {g.version or '?'} | {g.account or 'not authenticated'} | project {g.project or '-'}")
        if not snap.instances:
            typer.echo("no instances running")
        for i in snap.instances:
            cost = f"${i.estimated_cost_usd:.2f}" if i.estimated_cost_usd is not None else "—"
            up = f"{i.uptime_hours:.2f}h" if i.uptime_hours is not None else "—"
            kind = "spot" if i.preemptible else "on-demand"
            typer.echo(f"  {i.name} [{i.status}] {i.machine_type} {i.accelerator or 'no gpu'} {kind} up {up} ≈{cost}")
        typer.echo(f"estimated spend this session: ${snap.total_estimated_cost_usd:.2f}")
        typer.echo(f"credit balance is console-only: {snap.console_billing_url}")
        if not watch:
            break
        _time.sleep(10)


if __name__ == "__main__":
    app()
