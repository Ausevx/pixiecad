"""Mesh decimation tools using pyfqmr."""

from __future__ import annotations

from dataclasses import dataclass
import pyfqmr
import trimesh

__all__ = ["DecimateResult", "decimate_to_budget", "budget_for_parts"]


@dataclass
class DecimateResult:
    faces_before: int
    faces_after: int
    target_faces: int
    achieved_exact: bool


def decimate_to_budget(
    mesh: trimesh.Trimesh,
    target_faces: int,
    *,
    aggressiveness: float = 7.0,
) -> tuple[trimesh.Trimesh, DecimateResult]:
    """Decimate a mesh to a target face count budget using Fast Quadric Mesh Reduction.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to decimate. Never mutated.
    target_faces : int
        Target triangle count budget (> 0).
    aggressiveness : float, optional
        pyfqmr aggressiveness parameter, by default 7.0.

    Returns
    -------
    tuple[trimesh.Trimesh, DecimateResult]
        Decimated mesh (or copy of original if already within budget) and result metadata.
    """
    if target_faces <= 0:
        raise ValueError(f"target_faces must be positive (> 0), got {target_faces}")

    faces_before = len(mesh.faces)
    if faces_before <= target_faces:
        return mesh.copy(), DecimateResult(
            faces_before=faces_before,
            faces_after=faces_before,
            target_faces=target_faces,
            achieved_exact=(faces_before == target_faces),
        )

    # First simplification pass
    s1 = pyfqmr.Simplify()
    s1.setMesh(mesh.vertices, mesh.faces)
    s1.simplify_mesh(
        target_count=target_faces,
        aggressiveness=aggressiveness,
        preserve_border=True,
        verbose=0,
    )
    v1, f1, _ = s1.getMesh()
    count1 = len(f1)

    best_v, best_f = v1, f1
    best_count = count1

    # Overshoot compensation if first pass produced more faces than target
    if count1 > target_faces:
        compensated_target = target_faces - (count1 - target_faces)
        if compensated_target > 0:
            s2 = pyfqmr.Simplify()
            s2.setMesh(mesh.vertices, mesh.faces)
            s2.simplify_mesh(
                target_count=compensated_target,
                aggressiveness=aggressiveness,
                preserve_border=True,
                verbose=0,
            )
            v2, f2, _ = s2.getMesh()
            count2 = len(f2)
            if abs(count2 - target_faces) < abs(count1 - target_faces):
                best_v, best_f = v2, f2
                best_count = count2

    result_mesh = trimesh.Trimesh(vertices=best_v, faces=best_f, process=False)
    result_info = DecimateResult(
        faces_before=faces_before,
        faces_after=best_count,
        target_faces=target_faces,
        achieved_exact=(best_count == target_faces),
    )
    return result_mesh, result_info


def budget_for_parts(
    total_budget: int,
    part_face_counts: dict[str, int],
) -> dict[str, int]:
    """Split a total budget across named parts proportionally to current face counts.

    Enforces a minimum of 32 faces per part. Fixes rounding drift on the largest part.
    Raises ValueError if total_budget < 32 * len(part_face_counts).
    """
    num_parts = len(part_face_counts)
    if total_budget < 32 * num_parts:
        raise ValueError(
            f"total_budget ({total_budget}) must be at least {32 * num_parts} "
            f"for {num_parts} parts (minimum 32 per part)"
        )

    if not part_face_counts:
        return {}

    largest_part = max(part_face_counts.keys(), key=lambda k: part_face_counts[k])

    allocated: dict[str, int] = {}
    unconstrained = set(part_face_counts.keys())
    remaining_budget = total_budget

    while unconstrained:
        total_current = sum(part_face_counts[k] for k in unconstrained)
        newly_clamped = set()

        for k in unconstrained:
            if total_current > 0:
                raw_share = remaining_budget * (part_face_counts[k] / total_current)
            else:
                raw_share = remaining_budget / len(unconstrained)

            if raw_share < 32:
                allocated[k] = 32
                newly_clamped.add(k)

        if newly_clamped:
            for k in newly_clamped:
                unconstrained.remove(k)
                remaining_budget -= 32
        else:
            for k in unconstrained:
                if total_current > 0:
                    raw_share = remaining_budget * (part_face_counts[k] / total_current)
                else:
                    raw_share = remaining_budget / len(unconstrained)
                allocated[k] = int(round(raw_share))
            break

    drift = total_budget - sum(allocated.values())
    allocated[largest_part] += drift

    return allocated
