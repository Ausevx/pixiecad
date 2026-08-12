"""Tests for mesh decimation and polygon budget allocation."""

import numpy as np
import pytest
import trimesh

from pixiecad.meshops.decimate import DecimateResult, budget_for_parts, decimate_to_budget


def test_decimate_icosphere_exact_target():
    mesh = trimesh.creation.icosphere(subdivisions=4)
    assert len(mesh.faces) == 5120

    decimated, res = decimate_to_budget(mesh, 800)
    assert isinstance(res, DecimateResult)
    assert res.faces_before == 5120
    assert res.faces_after == 800
    assert res.target_faces == 800
    assert res.achieved_exact is True
    assert len(decimated.faces) == 800

    # Ensure no NaN vertices and non-zero volume for sphere case
    assert not np.isnan(decimated.vertices).any()
    assert decimated.volume > 0.0


def test_decimate_already_below_budget():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    initial_faces = len(mesh.faces)  # 320 faces
    target = 500

    decimated, res = decimate_to_budget(mesh, target)
    assert res.faces_before == initial_faces
    assert res.faces_after == initial_faces
    assert res.target_faces == target
    assert res.achieved_exact is False
    assert len(decimated.faces) == initial_faces

    # Must return a copy, not the exact same object reference
    assert decimated is not mesh


def test_decimate_already_at_budget():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    initial_faces = len(mesh.faces)  # 320 faces

    decimated, res = decimate_to_budget(mesh, initial_faces)
    assert res.faces_before == initial_faces
    assert res.faces_after == initial_faces
    assert res.achieved_exact is True
    assert decimated is not mesh


def test_decimate_invalid_target_faces():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    with pytest.raises(ValueError, match="target_faces must be positive"):
        decimate_to_budget(mesh, 0)

    with pytest.raises(ValueError, match="target_faces must be positive"):
        decimate_to_budget(mesh, -50)


def test_decimate_input_mesh_unmutated():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    orig_v = mesh.vertices.copy()
    orig_f = mesh.faces.copy()

    _decimated, _res = decimate_to_budget(mesh, 300)
    np.testing.assert_array_equal(mesh.vertices, orig_v)
    np.testing.assert_array_equal(mesh.faces, orig_f)


def test_decimate_preserves_texture():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    uvs = np.random.rand(len(mesh.vertices), 2)
    mat = trimesh.visual.material.PBRMaterial()
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)

    decimated, _ = decimate_to_budget(mesh, 400)
    assert isinstance(decimated.visual, trimesh.visual.TextureVisuals)
    assert decimated.visual.uv is not None
    assert len(decimated.visual.uv) == len(decimated.vertices)
    assert decimated.visual.material is mat


def test_decimate_already_below_budget_preserves_texture():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    uvs = np.random.rand(len(mesh.vertices), 2)
    mat = trimesh.visual.material.PBRMaterial()
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=mat)

    decimated, _ = decimate_to_budget(mesh, 500)
    assert isinstance(decimated.visual, trimesh.visual.TextureVisuals)
    assert decimated.visual.uv is not None
    assert len(decimated.visual.uv) == len(decimated.vertices)


def test_decimate_preserves_vertex_colors():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    colors = np.random.randint(0, 255, (len(mesh.vertices), 4))
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)

    decimated, _ = decimate_to_budget(mesh, 400)
    assert decimated.visual.kind == "vertex"
    assert decimated.visual.vertex_colors is not None
    assert len(decimated.visual.vertex_colors) == len(decimated.vertices)


def test_decimate_untextured_remains_untextured():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    decimated, _ = decimate_to_budget(mesh, 400)

    assert not isinstance(decimated.visual, trimesh.visual.TextureVisuals)
    assert not getattr(decimated.visual, "defined", False)


def test_decimate_input_mesh_unmutated_visuals():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    orig_v_count = len(mesh.vertices)
    orig_uvs = np.random.rand(orig_v_count, 2)
    mesh.visual = trimesh.visual.TextureVisuals(uv=orig_uvs)

    orig_uv_id = id(mesh.visual.uv)

    decimate_to_budget(mesh, 400)

    assert len(mesh.vertices) == orig_v_count
    assert id(mesh.visual.uv) == orig_uv_id
    np.testing.assert_array_equal(mesh.visual.uv, orig_uvs)


def test_budget_for_parts_proportional():
    parts = {"chassis": 50000, "wheels": 10000}
    budget = 10000
    res = budget_for_parts(budget, parts)

    assert sum(res.values()) == budget
    # Chassis should get approximately 5/6 of 10000 (~8333)
    assert pytest.approx(res["chassis"], abs=5) == 8333
    assert pytest.approx(res["wheels"], abs=5) == 1667


def test_budget_for_parts_minimum_enforcement():
    # tiny part gets bumped to 32 minimum
    parts = {"chassis": 100000, "tiny_detail": 1}
    budget = 1000
    res = budget_for_parts(budget, parts)

    assert sum(res.values()) == budget
    assert res["tiny_detail"] >= 32
    assert res["chassis"] == budget - res["tiny_detail"]


def test_budget_for_parts_infeasible():
    parts = {"part1": 10, "part2": 10, "part3": 10}

    # 3 parts * 32 = 96 minimum required. 90 is infeasible.
    with pytest.raises(ValueError, match="must be at least 96"):
        budget_for_parts(90, parts)


def test_budget_for_parts_empty():
    assert budget_for_parts(100, {}) == {}
