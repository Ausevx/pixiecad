import numpy as np
import trimesh

from pixiecad.meshops import CleanupReport, clean_mesh


def test_floater_removal():
    sphere = trimesh.creation.icosphere()
    cube = trimesh.creation.box()
    cube.apply_scale(0.01)
    cube.apply_translation([10.0, 0.0, 0.0])
    composite = trimesh.util.concatenate([sphere, cube])

    cleaned, report = clean_mesh(composite)

    assert isinstance(report, CleanupReport)
    assert report.components_removed == 1
    assert report.faces_after == len(sphere.faces)
    assert len(cleaned.faces) == len(sphere.faces)


def test_duplicate_vertices_cleanup():
    sphere = trimesh.creation.icosphere()
    dup_vertices = np.vstack([sphere.vertices, sphere.vertices])
    dup_mesh = trimesh.Trimesh(vertices=dup_vertices, faces=sphere.faces)

    cleaned, report = clean_mesh(dup_mesh)

    assert report.vertices_after <= report.vertices_before
    assert len(cleaned.vertices) <= len(dup_mesh.vertices)


def test_input_mesh_not_mutated():
    sphere = trimesh.creation.icosphere()
    cube = trimesh.creation.box()
    cube.apply_scale(0.01)
    cube.apply_translation([10.0, 0.0, 0.0])
    composite = trimesh.util.concatenate([sphere, cube])

    faces_before = len(composite.faces)
    vertices_before = len(composite.vertices)

    cleaned, report = clean_mesh(composite)

    assert len(composite.faces) == faces_before
    assert len(composite.vertices) == vertices_before
    assert composite is not cleaned


def test_clean_icosphere_pass_through():
    sphere = trimesh.creation.icosphere()

    cleaned, report = clean_mesh(sphere)

    assert report.faces_after == report.faces_before
    assert report.watertight is True
    assert len(cleaned.faces) == len(sphere.faces)
