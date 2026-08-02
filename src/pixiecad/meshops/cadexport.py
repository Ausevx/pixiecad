"""Export a generated mesh to formats CAD and slicer tools accept.

The format conversion itself is trivial -- GLB, STL and OBJ are all triangle
meshes, and trimesh writes any of them. What is *not* trivial, and what
actually decides whether a file is usable downstream, is whether the geometry
is a closed solid.

Measuring that correctly is itself a trap. trimesh's ``merge_vertices()``
keeps UV seams split, so on a textured mesh it reports thousands of phantom
holes: the tetrapod came back "not watertight, Euler 210, 7,096 boundary
faces" when welded that way, and "watertight, Euler 2, zero boundary faces"
when welded by position. Only the second is true. This module always welds by
position before judging.

What each format keeps:

- **STL**  geometry only. No colour, no UV, no texture, no units. The
  universal 3D-printing format and what Tinkercad likes best.
- **OBJ**  geometry + UVs + a .mtl referencing the texture image. Use this
  when you want the model to still look like something.
- **GLB**  what we produce natively: geometry, UVs, PBR texture, in one file.

And the limitation worth knowing before you plan around it: none of these
carry CAD *features*. A mesh is a bag of triangles, so tools like Fusion 360
can import it but cannot give you back editable sketches, fillets or
parametric history. Converting mesh to a real solid (BRep) is a separate,
lossy operation with practical face-count limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

# Formats a slicer or CAD tool will accept from us.
CAD_FORMATS = ("stl", "obj", "ply", "glb")


@dataclass
class SolidReport:
    """Whether a mesh is a closed solid, and what stands in the way."""

    watertight: bool
    winding_consistent: bool
    euler_number: int
    boundary_faces: int
    n_faces: int
    volume: float | None

    @property
    def printable(self) -> bool:
        """Safe to hand to a slicer without repair."""
        return self.watertight and self.winding_consistent

    def summary(self) -> list[str]:
        lines = [
            f"faces: {self.n_faces}",
            f"watertight: {self.watertight}",
            f"winding consistent: {self.winding_consistent}",
            f"euler number: {self.euler_number} (2 = one closed shell)",
        ]
        if self.boundary_faces:
            lines.append(
                f"faces on a hole boundary: {self.boundary_faces} "
                f"({self.boundary_faces / max(self.n_faces, 1):.0%})"
            )
        if self.volume is not None:
            lines.append(f"volume: {self.volume:.6g} (cubic model units)")
        if not self.printable:
            lines.append(
                "NOT a closed solid: fine for rendering, but a slicer cannot "
                "tell inside from outside. Repair before printing."
            )
        return lines


def _positional_weld(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Weld vertices that share a position, ignoring UV and normal splits.

    This is the difference between a correct answer and a badly wrong one.
    trimesh's ``merge_vertices()`` deliberately keeps UV seams separate, so on
    a textured mesh it leaves every seam vertex duplicated and the topology
    reads as full of holes. Measured on the tetrapod: the default said "not
    watertight, Euler 210, 7,096 boundary faces"; welding by position says
    watertight, Euler 2, zero boundary faces -- and the second is the truth.
    """
    from trimesh import grouping

    unique, inverse = grouping.unique_rows(mesh.vertices, digits=6)
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices)[unique],
        faces=inverse[np.asarray(mesh.faces)],
        process=False,
    )


def inspect_solid(mesh: trimesh.Trimesh) -> SolidReport:
    """Report the properties that decide whether a mesh is CAD/print ready."""
    welded = _positional_weld(mesh)
    try:
        boundary = len(trimesh.repair.broken_faces(welded))
    except Exception:  # pragma: no cover - trimesh version differences
        boundary = 0
    return SolidReport(
        watertight=bool(welded.is_watertight),
        winding_consistent=bool(welded.is_winding_consistent),
        euler_number=int(welded.euler_number),
        boundary_faces=int(boundary),
        n_faces=int(len(mesh.faces)),
        volume=float(abs(mesh.volume)) if welded.is_watertight else None,
    )


def scale_to_size(mesh: trimesh.Trimesh, longest_mm: float) -> trimesh.Trimesh:
    """Scale so the longest bounding-box edge is ``longest_mm`` millimetres.

    Generated meshes carry no units -- ours measures about 1.9 across, which
    means nothing. STL has no unit field either, so slicers assume
    millimetres. Measuring one real edge and scaling to it is the whole of
    getting a correctly-sized print.
    """
    if longest_mm <= 0:
        raise ValueError(f"longest_mm must be > 0, got {longest_mm}")
    out = mesh.copy()
    current = float(np.max(out.extents))
    if current <= 0:
        raise ValueError("mesh has no extent to scale")
    out.apply_scale(longest_mm / current)
    return out


def repair_for_print(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, list[str]]:
    """Best-effort repair toward a closed solid. Returns (mesh, what it did).

    Deliberately conservative: it welds, drops degenerate and duplicate faces,
    fixes winding and normals, and fills holes. It does *not* attempt
    voxel remeshing, which would make any mesh watertight at the cost of
    replacing the surface -- if that is what you want it should be a
    deliberate choice, not something a "repair" quietly did to your model.
    """
    out = mesh.copy()
    actions: list[str] = []

    before_v = len(out.vertices)
    # merge_tex/merge_norm so UV seams are welded too; without them a textured
    # mesh keeps thousands of coincident duplicates and never closes.
    out.merge_vertices(merge_tex=True, merge_norm=True)
    if len(out.vertices) != before_v:
        actions.append(f"welded {before_v - len(out.vertices)} duplicate vertices")

    before_f = len(out.faces)
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    if len(out.faces) != before_f:
        actions.append(f"removed {before_f - len(out.faces)} degenerate/duplicate faces")

    if not out.is_winding_consistent:
        trimesh.repair.fix_winding(out)
        actions.append("fixed face winding")
    trimesh.repair.fix_normals(out)
    trimesh.repair.fix_inversion(out)

    if not out.is_watertight:
        try:
            trimesh.repair.fill_holes(out)
            actions.append(
                "filled holes -> watertight" if out.is_watertight else "filled some holes"
            )
        except Exception:  # pragma: no cover
            pass

    if not actions:
        actions.append("nothing to repair")
    return out, actions


def export_cad(
    mesh: trimesh.Trimesh,
    path: str | Path,
    *,
    longest_mm: float | None = None,
    repair: bool = False,
) -> tuple[Path, SolidReport, list[str]]:
    """Write ``mesh`` in the format implied by ``path``'s suffix."""
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in CAD_FORMATS:
        raise ValueError(f"unsupported format '{fmt}'; use one of {', '.join(CAD_FORMATS)}")

    out = mesh
    actions: list[str] = []
    if repair:
        out, actions = repair_for_print(out)
    if longest_mm is not None:
        out = scale_to_size(out, longest_mm)
        actions.append(f"scaled longest edge to {longest_mm} mm")

    path.parent.mkdir(parents=True, exist_ok=True)
    # STL cannot carry UVs or texture, so say so rather than let a user
    # discover their texture vanished.
    if fmt == "stl" and getattr(getattr(out, "visual", None), "uv", None) is not None:
        actions.append("note: STL stores geometry only -- texture and UVs are dropped")

    out.export(path)
    return path, inspect_solid(out), actions
