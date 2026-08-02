"""S5 part splitting: segmentation, budgets, export."""

import json

import pytest
import trimesh

from pixiecad.parts import export_parts, slugify, split_parts


def _car_like():
    """A body with four detached 'wheels' — the connected-components case."""
    body = trimesh.creation.box(extents=[4.0, 1.8, 1.0])
    parts = [body]
    for x in (-1.4, 1.4):
        for y in (-0.9, 0.9):
            w = trimesh.creation.cylinder(radius=0.4, height=0.3)
            w.apply_translation([x, y, -0.6])
            parts.append(w)
    return trimesh.util.concatenate(parts)


def test_connected_splits_detached_components():
    parts = split_parts(_car_like(), method="connected")
    assert len(parts) == 5  # body + 4 wheels
    assert all(p.method == "connected" for p in parts)


def test_parts_are_ordered_by_size_not_tessellation():
    """The body must come first even though a wheel has 10x its face count."""
    parts = split_parts(_car_like(), method="connected")
    areas = [p.area for p in parts]
    assert areas == sorted(areas, reverse=True)
    body, wheel = parts[0], parts[1]
    assert body.area > wheel.area
    assert body.n_faces < wheel.n_faces  # coarse box vs fine cylinder


def test_cluster_splits_a_single_shell():
    """Connectivity can't split a watertight sphere; clustering must."""
    sphere = trimesh.creation.icosphere(subdivisions=3)
    assert len(sphere.split(only_watertight=False)) == 1

    parts = split_parts(sphere, method="cluster", max_parts=4)
    assert 2 <= len(parts) <= 4
    assert sum(p.n_faces for p in parts) <= len(sphere.faces)
    assert all(p.method == "cluster" for p in parts)


def test_auto_falls_back_to_cluster():
    parts = split_parts(trimesh.creation.icosphere(subdivisions=3), method="auto", max_parts=3)
    assert len(parts) > 1
    assert parts[0].method == "cluster"


def test_auto_prefers_connectivity_when_available():
    parts = split_parts(_car_like(), method="auto")
    assert parts[0].method == "connected"


def test_deterministic():
    m = trimesh.creation.icosphere(subdivisions=3)
    a = [p.n_faces for p in split_parts(m, method="cluster", max_parts=4, seed=7)]
    b = [p.n_faces for p in split_parts(m, method="cluster", max_parts=4, seed=7)]
    assert a == b


def test_max_parts_and_validation():
    parts = split_parts(_car_like(), method="connected", max_parts=2)
    assert len(parts) == 2
    with pytest.raises(ValueError, match="max_parts"):
        split_parts(_car_like(), max_parts=0)
    with pytest.raises(ValueError, match="unknown method"):
        split_parts(_car_like(), method="nonsense")


def test_debris_is_dropped():
    big = trimesh.creation.box(extents=[4, 2, 1])
    speck = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
    speck.apply_translation([9, 0, 0])
    parts = split_parts(trimesh.util.concatenate([big, speck]), method="connected")
    # The speck has as many faces as the body (both are boxes) — only an
    # area-based threshold can tell them apart.
    assert len(parts) == 1


def test_describe_is_scale_free_and_positioned():
    whole = _car_like()
    parts = split_parts(whole, method="connected")
    descs = [p.describe(whole) for p in parts]
    for d in descs:
        for v in d["position"].values():
            assert 0.0 <= v <= 1.0
        assert d["volume_fraction"] >= 0.0
    # The body dominates the volume.
    assert descs[0]["volume_fraction"] > max(d["volume_fraction"] for d in descs[1:])


def test_export_parts_writes_files_and_manifest(tmp_path):
    whole = _car_like()
    parts = split_parts(whole, method="connected")
    for p in parts:
        p.name = "body" if p.index == 0 else "wheel"

    exported, manifest = export_parts(parts, tmp_path, total_budget=2000, whole_volume=whole.volume)

    assert len(exported) == len(parts)
    assert sum(e.target_faces for e in exported) == 2000
    for e in exported:
        assert (tmp_path / e.file).exists()
        assert (tmp_path / e.file).stat().st_size > 0

    data = json.loads(manifest.read_text())
    assert data["n_parts"] == len(parts)
    assert data["total_budget"] == 2000
    assert data["named"] is True
    # Duplicate names must not collide on disk.
    assert len({e.file for e in exported}) == len(exported)


def test_export_requires_parts(tmp_path):
    with pytest.raises(ValueError, match="no parts"):
        export_parts([], tmp_path, total_budget=1000)


def test_slugify():
    assert slugify("Front Left Wheel") == "front-left-wheel"
    assert slugify("  ") == "part"
    assert slugify("A/B\\C") == "a-b-c"
