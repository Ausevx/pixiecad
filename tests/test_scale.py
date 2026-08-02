"""Tests for ArUco marker detection and dimension-based scaling."""

import cv2
import numpy as np
import pytest
from pixiecad.scale import (
    MarkerDetection,
    apply_scale,
    detect_markers,
    scale_factor_from_dimensions,
    scale_factor_from_marker,
    scan_images,
)
from pixiecad.spec import Dimensions


def test_aruco_round_trip():
    dict_name = "4X4_50"
    dict_obj = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_id = 7
    marker_img = cv2.aruco.generateImageMarker(dict_obj, marker_id, 200)

    # Pad with white border >= 50px quiet zone
    padded = cv2.copyMakeBorder(
        marker_img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255
    )
    bgr_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    detections = detect_markers(bgr_img, image_name="test_marker.jpg", dictionary=dict_name)
    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, MarkerDetection)
    assert det.image == "test_marker.jpg"
    assert det.marker_id == marker_id
    assert len(det.corners_px) == 4
    for pt in det.corners_px:
        assert len(pt) == 2
        # Check corner coordinates fall within image bounds (320x320)
        assert 0 <= pt[0] <= 320
        assert 0 <= pt[1] <= 320


def test_no_marker():
    # Plain white image
    blank_bgr = np.full((300, 300, 3), 255, dtype=np.uint8)
    detections = detect_markers(blank_bgr, image_name="blank.jpg")
    assert detections == []


def test_scan_images(tmp_path):
    dict_obj = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(dict_obj, 3, 200)
    padded = cv2.copyMakeBorder(
        marker_img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255
    )
    bgr_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    cv2.imwrite(str(img_dir / "frame_01.jpg"), bgr_img)
    cv2.imwrite(str(img_dir / "frame_02.png"), bgr_img)
    # Write a plain image with no marker
    blank_bgr = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img_dir / "frame_03.jpg"), blank_bgr)
    # Write non-image file
    (img_dir / "notes.txt").write_text("hello")

    detections = scan_images(img_dir, dictionary="4X4_50")
    assert len(detections) == 2
    det_names = {d.image for d in detections}
    assert det_names == {"frame_01.jpg", "frame_02.png"}
    assert all(d.marker_id == 3 for d in detections)


def test_scale_factor_from_dimensions_single_dim():
    bbox = (2.0, 1.0, 0.5)
    dims = Dimensions(length_m=4.0)
    factor = scale_factor_from_dimensions(bbox, dims)
    assert pytest.approx(factor) == 2.0


def test_scale_factor_from_dimensions_two_dims():
    bbox = (2.0, 1.0, 0.5)
    # Longest declared = 4.0 matches longest bbox extent (2.0) -> ratio 2.0
    # Second declared = 1.5 matches second bbox extent (1.0) -> ratio 1.5
    # Average ratio = 1.75
    dims = Dimensions(length_m=4.0, width_m=1.5)
    factor = scale_factor_from_dimensions(bbox, dims)
    assert pytest.approx(factor) == 1.75


def test_scale_factor_from_dimensions_no_dims_raises():
    bbox = (2.0, 1.0, 0.5)
    dims = Dimensions()
    with pytest.raises(ValueError, match="No dimensions declared"):
        scale_factor_from_dimensions(bbox, dims)


def test_scale_factor_from_dimensions_zero_extent_raises():
    dims = Dimensions(length_m=4.0)
    with pytest.raises(ValueError, match="positive non-zero values"):
        scale_factor_from_dimensions((2.0, 0.0, 0.5), dims)


def test_scale_factor_from_marker():
    factor = scale_factor_from_marker(measured_side_m=0.1, true_side_m=0.2)
    assert pytest.approx(factor) == 2.0

    with pytest.raises(ValueError, match="must be positive"):
        scale_factor_from_marker(measured_side_m=0.0, true_side_m=0.2)

    with pytest.raises(ValueError, match="must be positive"):
        scale_factor_from_marker(measured_side_m=0.1, true_side_m=-0.5)


def test_apply_scale():
    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    vertices_orig = vertices.copy()
    scaled = apply_scale(vertices, factor=2.0)

    # Assert scaled array is correct
    np.testing.assert_allclose(
        scaled, np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=np.float32)
    )

    # Assert input array was not mutated
    np.testing.assert_array_equal(vertices, vertices_orig)
    assert scaled is not vertices


def test_apply_scale_invalid_raises():
    vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="Scale factor must be positive"):
        apply_scale(vertices, factor=-1.0)

    invalid_shape = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must have shape"):
        apply_scale(invalid_shape, factor=2.0)
