"""
Depth Anything 3 reconstructor tests
====================================
View preparation, lifting depth maps into one cloud, dropping unreliable pixels, views and
ghost copies, and scale normalisation. The model itself runs in test_multiview_meshes.py
"""

import numpy as np
import open3d as o3d
import pytest

from pictomesh.reconstruction.da3 import (
    PATCH_SIZE,
    PROCESS_RES,
    _agreed_by_other_views,
    _align,
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
    """A landscape and a portrait photo are cropped to one size; masks must crop identically"""
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
    """n cameras side by side looking at a fronto-parallel square at depth 2"""
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


def _plane_before_wall():
    """Full-photo depth as DA3 predicts it: the square at depth 2, a wall at 6 around it"""
    depth, _, k, w2c, subjects, _ = _planar_views()
    return np.where(subjects, depth, 6.0), k, w2c


def test_points_another_view_contradicts_are_dropped():
    """Ghost copies sit where another view saw through to something farther; hidden points stay"""
    depth, k, w2c = _plane_before_wall()
    points = np.array(
        [
            [0.0, 0.0, 2.0],  # on the plane: the other view measured it there
            [0.9, 0.0, 2.0],  # beside the plane: the other view saw the wall behind it
            [0.0, 0.0, 3.0],  # behind the plane: hidden from the other view, no evidence
            [0.0, 0.0, 1.5],  # in front of the plane: the other view saw through it
        ]
    )
    owner = np.zeros(len(points), dtype=int)

    agreed = _agreed_by_other_views(points, owner, [0, 1], depth, k, w2c)

    assert agreed.tolist() == [True, False, True, False]


def test_thin_parts_that_reproject_a_pixel_off_are_kept():
    depth, k, w2c = _plane_before_wall()
    depth[1] = 6.0
    depth[1, :, 30:32] = 2.0  # the other view sees a 2 px rod at depth 2...
    # ...and view 0's point on it lands 2 px beside it, on the wall
    u = 33
    point = np.array([[(u + 0.5 - k[1, 0, 2]) * 2.0 / k[1, 0, 0] + 0.1, 0.0, 2.0]])

    agreed = _agreed_by_other_views(point, np.zeros(1, dtype=int), [0, 1], depth, k, w2c)

    assert agreed.tolist() == [True]


def test_parts_another_views_mask_misses_are_kept(monkeypatch):
    """rembg can miss thin dark parts in some photos; DA3's depth still sees them there"""
    no_fill = (np.empty((0, 3)),) * 3  # only what the photos saw
    monkeypatch.setattr(
        "pictomesh.reconstruction.da3.fill_unobserved", lambda *args, **kwargs: no_fill
    )
    depth, conf, k, w2c, subjects, rgbs = _planar_views()
    subjects[1][:, 30:] = False  # view 1's mask misses the square's right side...
    rgbs[0][:, 30:] = (255, 0, 0)  # ...which view 0 saw, in red

    pcd = lift_views(depth, conf, k, w2c, subjects, rgbs)

    red = np.asarray(pcd.points)[np.asarray(pcd.colors)[:, 1] < 0.5]
    width, height = np.ptp(red[:, :2], axis=0)
    assert width > 0.3 * height  # the 15 by 34 px strip, not a sliver along view 1's mask edge


def test_views_a_pose_error_shifted_are_pulled_back_together():
    rng = np.random.default_rng(0)
    shift = np.array([0.006, -0.008, 0.008])  # under 1% of the surface, as DA3 leaves views
    views = []
    for offset in (np.zeros(3), shift):
        xy = rng.uniform(-1, 1, (20_000, 2))
        bumps = 0.2 * np.sin(3 * xy[:, 0]) * np.cos(2 * xy[:, 1])  # something for ICP to grip
        view = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.column_stack([xy, bumps])))
        view.translate(offset)
        view.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
        views.append(view)
    before = np.asarray(views[0].points).copy()

    moves = _align(views)

    relative = np.linalg.inv(moves[1]) @ moves[0]  # view 1's remaining offset from view 0
    np.testing.assert_allclose(relative[:3, 3], shift, atol=2e-3)
    np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1e-2)
    # lift_views moves each camera by its view's move, so the move must match the points
    moved = before @ moves[0][:3, :3].T + moves[0][:3, 3]
    np.testing.assert_allclose(np.asarray(views[0].points), moved, atol=1e-9)


def test_two_views_are_always_trusted():
    conf = np.stack([np.full((4, 4), c, np.float32) for c in (10.0, 1.0)])
    assert _trusted_views(conf, [np.ones((4, 4), bool)] * 2, np.ones((2, 4, 4))) == [True, True]


# ── normalise_scale ───────────────────────────────────────────────────────────


def test_normalise_scale_makes_largest_extent_one():
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array([[0, 0, 0], [4.0, 2, 1]])))
    extent = np.ptp(np.asarray(normalise_scale(pcd).points), axis=0)
    assert extent.max() == pytest.approx(1.0)
