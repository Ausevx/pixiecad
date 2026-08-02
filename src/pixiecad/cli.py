"""PixieCAD CLI."""

from __future__ import annotations

from pathlib import Path

import typer

from .ingest import run_ingest
from .spec import Dimensions, ObjectSpec
from .workspace import Workspace

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


if __name__ == "__main__":
    app()
