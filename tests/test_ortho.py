"""Tests for orthographic drawing export (pixiecad.meshops.ortho)."""

import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
import numpy as np
import trimesh

from pixiecad.meshops.ortho import (
    OrthoView,
    feature_edges,
    project,
    render_all,
    render_view,
    save_views,
    silhouette_edges,
)
from pixiecad.spec import Dimensions


def test_unit_box_views_valid_xml():
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    views = render_all(box)
    assert set(views.keys()) == {"front", "side", "top"}

    for name, view in views.items():
        assert isinstance(view, OrthoView)
        assert view.name == name
        assert view.svg != ""
        assert "<line" in view.svg
        root = ET.fromstring(view.svg)
        assert root.tag.endswith("svg")


def test_projection_correctness_scaled_box():
    box = trimesh.creation.box(extents=(2.0, 1.0, 4.0))

    front = render_view(box, "front")
    assert pytest.approx(front.width_units, abs=1e-5) == 2.0
    assert pytest.approx(front.height_units, abs=1e-5) == 4.0
    assert pytest.approx(front.width_units / front.height_units, abs=1e-5) == 2.0 / 4.0

    top = render_view(box, "top")
    assert pytest.approx(top.width_units, abs=1e-5) == 2.0
    assert pytest.approx(top.height_units, abs=1e-5) == 1.0
    assert pytest.approx(top.width_units / top.height_units, abs=1e-5) == 2.0 / 1.0

    side = render_view(box, "side")
    assert pytest.approx(side.width_units, abs=1e-5) == 1.0
    assert pytest.approx(side.height_units, abs=1e-5) == 4.0


def test_feature_edges():
    box = trimesh.creation.box()
    box_fe = feature_edges(box, angle_deg=25.0)
    assert len(box_fe) == 12

    ico = trimesh.creation.icosphere(subdivisions=3)
    ico_fe = feature_edges(ico, angle_deg=25.0)
    total_ico_edges = len(ico.edges_sorted)
    assert len(ico_fe) < total_ico_edges / 10


def test_dimension_annotations():
    box = trimesh.creation.box()

    view_with_dims = render_view(box, "front", dimensions=Dimensions(length_m=4.0))
    assert "<text" in view_with_dims.svg
    assert "m" in view_with_dims.svg

    view_no_dims = render_view(box, "front", dimensions=None)
    assert "<text" not in view_no_dims.svg

    view_empty_dims = render_view(box, "front", dimensions=Dimensions())
    assert "<text" not in view_empty_dims.svg


def test_save_views(tmp_path: Path):
    box = trimesh.creation.box()
    views = render_all(box)

    saved = save_views(views, tmp_path)
    assert len(saved) == 3

    for p in saved:
        assert p.exists()
        assert p.stat().st_size > 0

    expected_names = {tmp_path / "front.svg", tmp_path / "side.svg", tmp_path / "top.svg"}
    assert set(saved) == expected_names
