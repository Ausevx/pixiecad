"""Tests for CAD / 3D-printing export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pixiecad.meshops.cadexport import (
    export_cad,
    inspect_solid,
    repair_for_print,
    scale_to_size,
)


def _seamed_sphere() -> trimesh.Trimesh:
    """A closed sphere whose vertices are duplicated as a UV seam would.

    This is the case that matters: the geometry is a perfect solid, but the
    index buffer splits vertices, and a naive check calls it full of holes.
    """
    mesh = trimesh.creation.icosphere(subdivisions=2)
    extra = mesh.faces[0]
    verts = np.vstack([mesh.vertices, mesh.vertices[extra]])
    faces = mesh.faces.copy()
    faces[0] = [len(mesh.vertices) + i for i in range(3)]
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


class TestInspectSolid:
    def test_closed_mesh_reads_as_a_solid(self):
        report = inspect_solid(trimesh.creation.box())
        assert report.watertight and report.printable
        assert report.euler_number == 2

    def test_uv_seam_duplicates_do_not_read_as_holes(self):
        """The bug this module exists to avoid.

        trimesh's merge_vertices() keeps seams split, which reported the
        tetrapod as 'not watertight, Euler 210, 7096 boundary faces' when it
        is in fact a perfect closed solid.
        """
        report = inspect_solid(_seamed_sphere())
        assert report.watertight, "seam duplicates were mistaken for holes"
        assert report.euler_number == 2
        assert report.boundary_faces == 0

    def test_open_mesh_is_reported_as_not_printable(self):
        box = trimesh.creation.box()
        open_shell = trimesh.Trimesh(
            vertices=box.vertices, faces=box.faces[:6], process=False
        )
        report = inspect_solid(open_shell)
        assert not report.watertight
        assert not report.printable
        assert any("NOT a closed solid" in line for line in report.summary())

    def test_volume_only_reported_when_meaningful(self):
        box = trimesh.creation.box()
        open_shell = trimesh.Trimesh(
            vertices=box.vertices, faces=box.faces[:6], process=False
        )
        assert inspect_solid(box).volume is not None
        assert inspect_solid(open_shell).volume is None


class TestScaleToSize:
    def test_sets_the_longest_edge(self):
        scaled = scale_to_size(trimesh.creation.box(extents=[1, 2, 4]), 60.0)
        assert np.isclose(max(scaled.extents), 60.0)

    def test_proportions_are_preserved(self):
        mesh = trimesh.creation.box(extents=[1, 2, 4])
        scaled = scale_to_size(mesh, 60.0)
        assert np.allclose(scaled.extents / max(scaled.extents),
                           mesh.extents / max(mesh.extents))

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="longest_mm"):
            scale_to_size(trimesh.creation.box(), 0)


class TestExportCad:
    @pytest.mark.parametrize("ext", ["stl", "obj", "ply", "glb"])
    def test_writes_each_supported_format(self, tmp_path: Path, ext: str):
        path, report, _ = export_cad(trimesh.creation.box(), tmp_path / f"m.{ext}")
        assert path.is_file() and path.stat().st_size > 0
        assert report.printable

    def test_rejects_unknown_format(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unsupported format"):
            export_cad(trimesh.creation.box(), tmp_path / "m.step")

    def test_scaling_is_applied_to_the_written_file(self, tmp_path: Path):
        path, _, actions = export_cad(
            trimesh.creation.box(extents=[1, 1, 2]), tmp_path / "m.stl", longest_mm=50
        )
        assert np.isclose(max(trimesh.load(path).extents), 50.0)
        assert any("50" in a for a in actions)

    def test_stl_warns_that_texture_is_dropped(self, tmp_path: Path):
        mesh = trimesh.creation.box()
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2))
        )
        _, _, actions = export_cad(mesh, tmp_path / "m.stl")
        assert any("STL stores geometry only" in a for a in actions)


class TestRepair:
    def test_reports_when_nothing_needed(self):
        _, actions = repair_for_print(trimesh.creation.box())
        assert actions

    def test_does_not_break_an_already_solid_mesh(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        repaired, _ = repair_for_print(mesh)
        assert inspect_solid(repaired).watertight
