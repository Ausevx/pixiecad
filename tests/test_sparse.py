import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pixiecad.geometry import SparseFailure, SparseResult, run_sparse
from pixiecad.spec import ObjectSpec
from pixiecad.workspace import Workspace
from conftest import checkerboard, save_jpg


@pytest.fixture
def workspace(tmp_path):
    spec = ObjectSpec(name="test_obj")
    return Workspace.create(tmp_path / "ws", spec)


@pytest.fixture
def sample_images(tmp_path):
    img_dir = tmp_path / "input_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = checkerboard(100, 10)
    for i in range(3):
        save_jpg(img_dir / f"img_{i:02d}.jpg", img)
    return img_dir


@pytest.fixture
def mock_pycolmap(monkeypatch):
    mock_mod = MagicMock()

    rec = MagicMock()
    rec.num_reg_images.return_value = 5
    rec.num_points3D.return_value = 100
    rec.compute_mean_reprojection_error.return_value = 0.25

    mock_mod.incremental_mapping.return_value = {0: rec}

    monkeypatch.setitem(sys.modules, "pycolmap", mock_mod)
    return mock_mod


def test_lazy_import_without_pycolmap(monkeypatch):
    """Assert pycolmap is not imported merely by importing pixiecad.geometry."""
    monkeypatch.setitem(sys.modules, "pycolmap", None)
    import pixiecad.geometry

    assert hasattr(pixiecad.geometry, "run_sparse")


def test_fewer_than_3_images_error(tmp_path, workspace):
    img_dir = tmp_path / "few_images"
    img_dir.mkdir()
    img = checkerboard(100, 10)
    save_jpg(img_dir / "img_01.jpg", img)
    save_jpg(img_dir / "img_02.jpg", img)

    with pytest.raises(ValueError, match="At least 3 images required"):
        run_sparse(img_dir, workspace)


def test_run_sparse_exhaustive_matching(sample_images, workspace, mock_pycolmap):
    res = run_sparse(sample_images, workspace, matcher="exhaustive")

    assert isinstance(res, SparseResult)
    assert res.n_images_in == 3
    assert res.n_registered == 5
    assert res.n_points3d == 100
    assert res.mean_reproj_error == 0.25
    assert res.cached is False

    mock_pycolmap.extract_features.assert_called_once()
    mock_pycolmap.match_exhaustive.assert_called_once()
    mock_pycolmap.match_sequential.assert_not_called()
    mock_pycolmap.incremental_mapping.assert_called_once()


def test_run_sparse_sequential_matching(sample_images, workspace, mock_pycolmap):
    res = run_sparse(sample_images, workspace, matcher="sequential")

    assert res.n_images_in == 3
    mock_pycolmap.extract_features.assert_called_once()
    mock_pycolmap.match_sequential.assert_called_once()
    mock_pycolmap.match_exhaustive.assert_not_called()
    mock_pycolmap.incremental_mapping.assert_called_once()


def test_run_sparse_unknown_matcher(sample_images, workspace, mock_pycolmap):
    with pytest.raises(ValueError, match="Unknown matcher"):
        run_sparse(sample_images, workspace, matcher="invalid_matcher")


def test_run_sparse_cache_hit(sample_images, workspace, mock_pycolmap):
    res1 = run_sparse(sample_images, workspace, matcher="exhaustive")
    assert res1.cached is False
    initial_calls = mock_pycolmap.extract_features.call_count

    # Second call with same inputs & workspace must return cached=True without calling pycolmap
    res2 = run_sparse(sample_images, workspace, matcher="exhaustive")
    assert res2.cached is True
    assert res2.n_images_in == res1.n_images_in
    assert res2.n_registered == res1.n_registered
    assert res2.n_points3d == res1.n_points3d
    assert res2.mean_reproj_error == res1.mean_reproj_error
    assert mock_pycolmap.extract_features.call_count == initial_calls


def test_run_sparse_empty_maps_failure(sample_images, workspace, mock_pycolmap):
    mock_pycolmap.incremental_mapping.return_value = {}

    with pytest.raises(SparseFailure, match="Sparse reconstruction failed"):
        run_sparse(sample_images, workspace)


def test_run_sparse_picks_best_reconstruction(sample_images, workspace, mock_pycolmap):
    rec1 = MagicMock()
    rec1.num_reg_images.return_value = 3
    rec1.num_points3D.return_value = 40
    rec1.compute_mean_reprojection_error.return_value = 0.8

    rec2 = MagicMock()
    rec2.num_reg_images.return_value = 10
    rec2.num_points3D.return_value = 500
    rec2.compute_mean_reprojection_error.return_value = 0.15

    mock_pycolmap.incremental_mapping.return_value = {0: rec1, 1: rec2}

    res = run_sparse(sample_images, workspace)
    assert res.n_registered == 10
    assert res.n_points3d == 500
    assert res.mean_reproj_error == 0.15


def test_run_sparse_reproj_error_attribute_error_fallback(sample_images, workspace, mock_pycolmap):
    rec = MagicMock()
    rec.num_reg_images.return_value = 4
    rec.num_points3D.return_value = 80
    del rec.compute_mean_reprojection_error

    mock_pycolmap.incremental_mapping.return_value = {0: rec}

    res = run_sparse(sample_images, workspace)
    assert res.mean_reproj_error == 0.0
