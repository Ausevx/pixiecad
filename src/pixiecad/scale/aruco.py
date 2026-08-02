"""ArUco marker detection in single images and photo sets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DICT_MAP: dict[str, int] = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass
class MarkerDetection:
    image: str
    marker_id: int
    corners_px: list[list[float]]  # 4 [x, y] float pairs


def detect_markers(
    image_bgr: np.ndarray,
    image_name: str,
    dictionary: str = "4X4_50",
) -> list[MarkerDetection]:
    """Detect ArUco markers in a BGR image."""
    if dictionary not in DICT_MAP:
        raise ValueError(
            f"Unsupported ArUco dictionary {dictionary!r}; expected one of {list(DICT_MAP.keys())}"
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dictionary])
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids, _ = detector.detectMarkers(image_bgr)

    if ids is None or len(ids) == 0:
        return []

    detections: list[MarkerDetection] = []
    for corner_arr, marker_id in zip(corners, ids.flatten()):
        pts = corner_arr.reshape(4, 2)
        corners_px = [[float(pt[0]), float(pt[1])] for pt in pts]
        detections.append(
            MarkerDetection(
                image=image_name,
                marker_id=int(marker_id),
                corners_px=corners_px,
            )
        )
    return detections


def scan_images(
    images_dir: Path,
    dictionary: str = "4X4_50",
) -> list[MarkerDetection]:
    """Scan a directory for image files (*.jpg, *.png) and detect markers."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise ValueError(f"Directory not found: {images_dir}")

    image_paths = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
    )

    all_detections: list[MarkerDetection] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        detections = detect_markers(img, image_name=path.name, dictionary=dictionary)
        all_detections.extend(detections)

    return all_detections
