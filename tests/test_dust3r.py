"""
DUSt3R reconstructor tests
==========================
Unit tests cover the photo preprocessing (mirrors upstream load_images) and the merge of
per-view point maps, neither of which needs the model. The E2E tests (marked `slow`) run
the real model on the Statue of Liberty views.
"""

from pathlib import Path

import numpy as np
import pytest

from pictomesh.reconstruction.dust3r import (
    IMAGE_SIZE,
    PATCH_SIZE,
    merge_views,
    normalise_scale,
    prepare_view,
)

ASSETS = Path(__file__).parent / "assets"


# ── prepare_view ──────────────────────────────────────────────────────────────


class TestPrepareView:
    def test_long_side_resized_and_patch_aligned(self):
        view, colors, subject = prepare_view(np.zeros((900, 600, 3), dtype=np.uint8), None, 0)
        h, w = colors.shape[:2]
        assert max(h, w) <= IMAGE_SIZE
        assert h % PATCH_SIZE == 0 and w % PATCH_SIZE == 0
        assert view["img"].shape == (1, 3, h, w)
        assert tuple(view["true_shape"][0]) == (h, w)
        assert subject.shape == (h, w) and subject.all()

    def test_square_photo_cropped_to_4_3(self):
        _, colors, _ = prepare_view(np.zeros((800, 800, 3), dtype=np.uint8), None, 0)
        assert colors.shape[:2] == (384, 512)

    def test_mask_follows_the_crop(self):
        bgr = np.zeros((600, 900, 3), dtype=np.uint8)
        mask = np.zeros((600, 900), dtype=bool)
        mask[:, :450] = True  # left half is subject
        _, colors, subject = prepare_view(bgr, mask, 0)
        w = subject.shape[1]
        assert subject.shape == colors.shape[:2]
        assert subject[:, : w // 2 - 4].all()
        assert not subject[:, w // 2 + 4 :].any()

    def test_normalisation_and_index(self):
        view, colors, _ = prepare_view(np.full((64, 64, 3), 255, dtype=np.uint8), None, 3)
        assert view["idx"] == 3 and view["instance"] == "3"
        assert float(view["img"].max()) == pytest.approx(1.0)  # (1.0 - 0.5) / 0.5
        assert float(colors.max()) == pytest.approx(1.0)


# ── merge_views ───────────────────────────────────────────────────────────────


class TestMergeViews:
    def test_keeps_only_confident_subject_points(self):
        pts = [np.arange(12, dtype=float).reshape(2, 2, 3)]
        conf = [np.array([[True, False], [True, True]])]
        subject = [np.array([[True, True], [False, True]])]
        colors = [np.full((2, 2, 3), 0.5)]
        pcd = merge_views(pts, conf, subject, colors, np.eye(4))
        np.testing.assert_allclose(np.asarray(pcd.points), [[0, 1, 2], [9, 10, 11]])
        assert np.asarray(pcd.colors).shape == (2, 3)

    def test_points_expressed_in_first_camera_frame(self):
        pts = [np.array([[[1.0, 2.0, 3.0]]])]
        ones = [np.array([[True]])]
        colors = [np.zeros((1, 1, 3))]
        world_to_camera0 = np.eye(4)
        world_to_camera0[:3, 3] = [-1.0, -2.0, -3.0]
        pcd = merge_views(pts, ones, ones, colors, world_to_camera0)
        np.testing.assert_allclose(np.asarray(pcd.points), [[0.0, 0.0, 0.0]])

    def test_concatenates_views(self):
        pts = [np.zeros((1, 1, 3)), np.ones((1, 1, 3))]
        ones = [np.ones((1, 1), dtype=bool)] * 2
        colors = [np.zeros((1, 1, 3))] * 2
        pcd = merge_views(pts, ones, ones, colors, np.eye(4))
        assert len(pcd.points) == 2


# ── normalise_scale ───────────────────────────────────────────────────────────


class TestNormaliseScale:
    def test_largest_extent_becomes_one(self):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [0.5, 0.0, 4.0]]))
        normalise_scale(pcd)
        np.testing.assert_allclose(np.asarray(pcd.points)[1], [0.125, 0.0, 1.0])

    def test_empty_cloud_untouched(self):
        import open3d as o3d

        assert len(normalise_scale(o3d.geometry.PointCloud()).points) == 0


# ── E2E tests (slow) ──────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDust3rE2E:
    @pytest.fixture(scope="class")
    def reconstructor(self):
        pytest.importorskip("roma")
        from pictomesh.reconstruction.dust3r import Dust3rReconstructor

        return Dust3rReconstructor(iterations=100)

    @pytest.fixture(scope="class")
    def liberty(self):
        import cv2

        from pictomesh.segmentation.service import RembgSegmentor, SegmentationService

        paths = sorted((ASSETS / "liberty").glob("liberty_*.jpeg"))[:3]
        images = [cv2.imread(str(p)) for p in paths]
        rgba = SegmentationService(RembgSegmentor()).process_batch(images)
        return images, [a[:, :, 3] > 0 for a in rgba]

    def test_three_views_give_a_coloured_cloud_with_depth(self, reconstructor, liberty):
        images, masks = liberty
        pcd = reconstructor.reconstruct(images, masks)
        pts = np.asarray(pcd.points)
        assert len(pts) > 10_000
        assert len(pcd.colors) == len(pts)
        extents = pts.max(axis=0) - pts.min(axis=0)
        assert extents.max() == pytest.approx(1.0)
        assert extents.min() / extents.max() > 0.1  # a volume, not a sheet

    def test_two_views_use_pair_mode(self, reconstructor, liberty):
        images, masks = liberty
        pcd = reconstructor.reconstruct(images[:2], masks[:2])
        assert len(pcd.points) > 1_000

    def test_single_image_rejected(self, reconstructor, liberty):
        images, masks = liberty
        with pytest.raises(ValueError, match="at least two"):
            reconstructor.reconstruct(images[:1], masks[:1])
