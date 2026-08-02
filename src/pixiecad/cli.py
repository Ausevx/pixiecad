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


if __name__ == "__main__":
    app()
