from pathlib import Path

import cv2
import numpy as np
import pytest

RNG = np.random.default_rng(42)


def checkerboard(size: int = 800, tile: int = 25) -> np.ndarray:
    """Sharp, high-contrast synthetic photo stand-in (BGR)."""
    # Midtone tiles (60/190) so a "good photo" isn't histogram-clipped.
    row = (np.arange(size) // tile) % 2
    board = (row[:, None] ^ row[None, :]) * 130 + 60
    noise = RNG.integers(0, 20, (size, size), dtype=np.uint8)
    gray = np.clip(board + noise, 0, 249).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def save_jpg(path: Path, img: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sharp():
    return checkerboard()


@pytest.fixture
def blurry(sharp):
    return cv2.GaussianBlur(sharp, (51, 51), 20)
