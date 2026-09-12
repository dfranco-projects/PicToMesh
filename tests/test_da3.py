"""
Depth Anything 3 reconstructor tests
====================================
Covers everything around the model: view preparation (sizes DA3 leaves unchanged, masks
kept aligned), lifting depth maps into one cloud with camera-facing normals, dropping
unreliable pixels and views, and scale normalisation. The model itself runs in the slow
mesh tests (test_multiview_meshes.py).
"""

import numpy as np
import open3d as o3d
import pytest

from pictomesh.reconstruction.da3 import (
    PATCH_SIZE,
    PROCESS_RES,
    _agreed_by_other_views,
    _trusted_views,
    lift_views,
    normalise_scale,
    prepare_views,
)

# ── prepare_views ─────────────────────────────────────────────────────────────


def test_views_get_one_shared_size_da3_leaves_alone():
    images = [np.zeros((1600, 1200, 3), np.uint8), np.zeros((1533, 1024, 3), np.uint8)]
    rgbs, subjects = prepare_views(images)

    shapes = {r.shape[:2] for r in rgbs}
    assert len(shapes) == 1
    h, w = shapes.pop()
    assert max(h, w) <= PROCESS_RES
    assert h % PATCH_SIZE == 0 and w % PATCH_SIZE == 0
    assert all(s.shape == (h, w) and s.all() for s in subjects)


def test_masks_stay_aligned_with_mixed_orientations():
    """A landscape and a portrait photo are cropped to one size; masks must crop identically."""
    images, masks = [], []
    for h, w in ((1200, 1600), (1600, 1200)):
        img = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), bool)
        img[h // 2 - 100 : h // 2 + 100, w // 2 - 100 : w // 2 + 100] = 255
        mask[h // 2 - 100 : h // 2 + 100, w // 2 - 100 : w // 2 + 100] = True
        images.append(img)
        masks.append(mask)

    rgbs, subjects = prepare_views(images, masks)

    for rgb, subject in zip(rgbs, subjects):
        white = rgb[..., 0] > 127
        assert (white == subject).mean() > 0.99


# ── lift_views ────────────────────────────────────────────────────────────────


def _planar_views(n_views: int = 2, size: int = 56):
    """n cameras side by side looking at a fronto-parallel square at depth 2."""
    k = np.array([[50.0, 0, size / 2], [0, 50.0, size / 2], [0, 0, 1]])
    depth = np.full((n_views, size, size), 2.0, np.float32)
    conf = np.full((n_views, size, size), 5.0, np.float32)
    w2c = np.stack([np.hstack([np.eye(3), [[-0.1 * i], [0.0], [0.0]]]) for i in range(n_views)])
    subjects = [np.pad(np.ones((size - 20, size - 20), bool), 10) for _ in range(n_views)]
    rgbs = [np.full((size, size, 3), 200, np.uint8) for _ in range(n_views)]
    return depth, conf, np.stack([k] * n_views), w2c, subjects, rgbs


def test_lifted_points_face_the_cameras():
    depth, conf, k, w2c, subjects, rgbs = _planar_views()
    pcd = lift_views(depth, conf, k, w2c, subjects, rgbs)

    points, normals = np.asarray(pcd.points), np.asarray(pcd.normals)
    front = points[:, 2] < points[:, 2].min() + 0.05  # the photographed plane
    assert front.sum() > 100
    # the cameras sit at negative z relative to the plane: its normals must point back at them
    assert (normals[front, 2] < 0).mean() > 0.95


def test_lifted_cloud_has_colours_and_normals_for_every_point():
    pcd = lift_views(*_planar_views())
    n = len(pcd.points)
    assert len(pcd.colors) == n and len(pcd.normals) == n
    assert np.asarray(pcd.colors)[:, 0].max() == pytest.approx(200 / 255, abs=1e-3)


def test_low_confidence_pixels_are_dropped():
    depth, conf, k, w2c, subjects, rgbs = _planar_views()
    depth[:, 20:24, 20:24] = 50.0  # far-off junk...
    conf[:, 20:24, 20:24] = 1.0  # ...that the model is unsure about
    pcd = lift_views(depth, conf, k, w2c, subjects, rgbs)
    z = np.asarray(pcd.points)[:, 2]
    # scaling is about the origin, so depth ratios survive: the plane sits at 2, junk at 50
    assert z.max() < 5 * z.min()


def test_views_far_less_confident_than_the_rest_are_not_trusted():
    conf = np.stack([np.full((4, 4), c, np.float32) for c in (10.0, 9.0, 1.5)])
    subjects = [np.ones((4, 4), bool)] * 3
    depth = np.ones((3, 4, 4), np.float32)
    assert _trusted_views(conf, subjects, depth) == [True, True, False]


def test_points_another_view_contradicts_are_dropped():
    """Ghost copies land where another view saw background or empty space; hidden points stay"""
    depth, _, k, w2c, subjects, _ = _planar_views()
    points = np.array(
        [
            [0.0, 0.0, 2.0],  # on the plane: the other view measured it there
            [0.9, 0.0, 2.0],  # beside the plane: the other view saw background
            [0.0, 0.0, 3.0],  # behind the plane: hidden from the other view, no evidence
            [0.0, 0.0, 1.5],  # in front of the plane: the other view saw through it
        ]
    )
    owner = np.zeros(len(points), dtype=int)
    measured = [np.where(s, d, 0.0) for d, s in zip(depth, subjects)]  # as lift_views passes them

    agreed = _agreed_by_other_views(points, owner, [0, 1], measured, subjects, k, w2c)

    assert agreed.tolist() == [True, False, True, False]


def test_two_views_are_always_trusted():
    conf = np.stack([np.full((4, 4), c, np.float32) for c in (10.0, 1.0)])
    assert _trusted_views(conf, [np.ones((4, 4), bool)] * 2, np.ones((2, 4, 4))) == [True, True]


# ── normalise_scale ───────────────────────────────────────────────────────────


def test_normalise_scale_makes_largest_extent_one():
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[0, 0, 0], [4.0, 2, 1]])))
    extent = np.ptp(np.asarray(normalise_scale(pcd).points), axis=0)
    assert extent.max() == pytest.approx(1.0)
