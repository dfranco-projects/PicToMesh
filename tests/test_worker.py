"""
Worker task tests
=================
Covers image loading with the resolution cap. Pipeline execution itself is
exercised by the pipeline unit and E2E tests.
"""

from pathlib import Path

import cv2
import numpy as np

from backend.worker.tasks import _MAX_SIDE, _load_image


def _write_image(path: Path, w: int, h: int) -> str:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return str(path)


def test_load_image_downscales_to_max_side(tmp_path):
    p = _write_image(tmp_path / "big.png", 2048, 1024)
    img = _load_image(p)
    assert img is not None
    assert img.shape[1] == _MAX_SIDE
    assert img.shape[0] == _MAX_SIDE // 2  # aspect ratio preserved


def test_load_image_keeps_small_images(tmp_path):
    p = _write_image(tmp_path / "small.png", 100, 80)
    img = _load_image(p)
    assert img is not None
    assert img.shape[:2] == (80, 100)


def test_load_image_unreadable_returns_none(tmp_path):
    bad = tmp_path / "not_an_image.png"
    bad.write_bytes(b"nope")
    assert _load_image(str(bad)) is None
