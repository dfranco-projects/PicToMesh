import numpy as np
import open3d as o3d
import pytest

from pictomesh.reconstruction.service import (
    CameraIntrinsics,
    DepthAnythingEstimator,
    FlatDepthEstimator,
    ReconstructionService,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def service() -> ReconstructionService:
    return ReconstructionService()


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)


@pytest.fixture
def flat_depth() -> np.ndarray:
    """100×100 depth map — all pixels at z=1.0 m."""
    return np.ones((100, 100), dtype=np.float32)


@pytest.fixture
def color_image() -> np.ndarray:
    """100×100 solid red BGR image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # red channel in BGR
    return img


# ── CameraIntrinsics ──────────────────────────────────────────────────────────


class TestCameraIntrinsics:
    def test_to_open3d_type(self, intrinsics):
        result = intrinsics.to_open3d()
        assert isinstance(result, o3d.camera.PinholeCameraIntrinsic)

    def test_to_open3d_preserves_values(self, intrinsics):
        o3d_intr = intrinsics.to_open3d()
        assert o3d_intr.width == intrinsics.width
        assert o3d_intr.height == intrinsics.height
        assert o3d_intr.intrinsic_matrix[0, 0] == pytest.approx(intrinsics.fx)
        assert o3d_intr.intrinsic_matrix[1, 1] == pytest.approx(intrinsics.fy)

    def test_estimate_sets_principal_point_to_centre(self):
        intr = CameraIntrinsics.estimate(width=200, height=100)
        assert intr.cx == pytest.approx(100.0)
        assert intr.cy == pytest.approx(50.0)

    def test_estimate_focal_length_equals_width(self):
        intr = CameraIntrinsics.estimate(width=640, height=480)
        assert intr.fx == pytest.approx(640.0)
        assert intr.fy == pytest.approx(640.0)

    def test_estimate_dimensions_preserved(self):
        intr = CameraIntrinsics.estimate(width=320, height=240)
        assert intr.width == 320
        assert intr.height == 240

    def test_frozen_dataclass_is_immutable(self, intrinsics):
        with pytest.raises(Exception):
            intrinsics.fx = 999.0  # type: ignore[misc]


# ── from_depth ────────────────────────────────────────────────────────────────


class TestFromDepth:
    def test_returns_point_cloud(self, service, flat_depth, intrinsics):
        pcd = service.from_depth(flat_depth, intrinsics)
        assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_has_points(self, service, flat_depth, intrinsics):
        pcd = service.from_depth(flat_depth, intrinsics)
        assert len(pcd.points) > 0

    def test_zero_depth_pixels_excluded(self, service, intrinsics):
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[50, 50] = 1.0  # only one valid pixel
        pcd = service.from_depth(depth, intrinsics)
        assert len(pcd.points) == 1

    def test_all_zero_depth_gives_empty_cloud(self, service, intrinsics):
        depth = np.zeros((100, 100), dtype=np.float32)
        pcd = service.from_depth(depth, intrinsics)
        assert len(pcd.points) == 0

    def test_depth_trunc_clips_far_points(self, service, intrinsics):
        depth_near = np.ones((100, 100), dtype=np.float32) * 1.0
        depth_far = np.ones((100, 100), dtype=np.float32) * 5.0
        pcd_near = service.from_depth(depth_near, intrinsics, depth_trunc=3.0)
        pcd_far = service.from_depth(depth_far, intrinsics, depth_trunc=3.0)
        assert len(pcd_near.points) > 0
        assert len(pcd_far.points) == 0

    def test_points_at_expected_depth(self, service, intrinsics):
        depth = np.zeros((100, 100), dtype=np.float32)
        depth[50, 50] = 2.0  # centre pixel at 2 m
        pcd = service.from_depth(depth, intrinsics)
        pts = np.asarray(pcd.points)
        assert pts.shape == (1, 3)
        assert pts[0, 2] == pytest.approx(2.0, abs=1e-3)  # z = depth

    def test_accepts_float32_input(self, service, intrinsics):
        depth = np.ones((50, 50), dtype=np.float32)
        pcd = service.from_depth(depth, intrinsics=CameraIntrinsics.estimate(50, 50))
        assert len(pcd.points) > 0


# ── from_rgbd ─────────────────────────────────────────────────────────────────


class TestFromRgbd:
    def test_returns_point_cloud(self, service, color_image, flat_depth, intrinsics):
        pcd = service.from_rgbd(color_image, flat_depth, intrinsics)
        assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_has_points(self, service, color_image, flat_depth, intrinsics):
        pcd = service.from_rgbd(color_image, flat_depth, intrinsics)
        assert len(pcd.points) > 0

    def test_has_colors(self, service, color_image, flat_depth, intrinsics):
        pcd = service.from_rgbd(color_image, flat_depth, intrinsics)
        assert pcd.has_colors()

    def test_color_count_matches_point_count(self, service, color_image, flat_depth, intrinsics):
        pcd = service.from_rgbd(color_image, flat_depth, intrinsics)
        assert len(pcd.colors) == len(pcd.points)

    def test_zero_depth_excluded(self, service, color_image, intrinsics):
        depth = np.zeros((100, 100), dtype=np.float32)
        pcd = service.from_rgbd(color_image, depth, intrinsics)
        assert len(pcd.points) == 0

    def test_depth_trunc_clips_far_points(self, service, color_image, intrinsics):
        deep = np.ones((100, 100), dtype=np.float32) * 20.0
        pcd = service.from_rgbd(color_image, deep, intrinsics, depth_trunc=5.0)
        assert len(pcd.points) == 0


# ── from_images ───────────────────────────────────────────────────────────────


class TestFromImages:
    @pytest.fixture
    def mock_reconstructor(self, sphere_pcd):
        """Minimal object satisfying MultiViewReconstructor protocol."""
        class _Mock:
            def reconstruct(self, images):
                return sphere_pcd
        return _Mock()

    def test_returns_point_cloud(self, service, mock_reconstructor):
        images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        pcd = service.from_images(images, mock_reconstructor)
        assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_passes_images_to_reconstructor(self, service):
        received = []
        class _Spy:
            def reconstruct(self, images):
                received.extend(images)
                return o3d.geometry.PointCloud()
        images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        service.from_images(images, _Spy())
        assert len(received) == 3

    def test_raises_with_single_image(self, service, mock_reconstructor):
        with pytest.raises(ValueError, match="at least 2"):
            service.from_images([np.zeros((64, 64, 3), dtype=np.uint8)], mock_reconstructor)

    def test_raises_with_empty_list(self, service, mock_reconstructor):
        with pytest.raises(ValueError):
            service.from_images([], mock_reconstructor)


# ── FlatDepthEstimator ────────────────────────────────────────────────────────


class TestFlatDepthEstimator:
    def test_shape_matches_image(self):
        est = FlatDepthEstimator()
        depth = est.estimate(np.zeros((48, 64, 3), dtype=np.uint8))
        assert depth.shape == (48, 64)

    def test_dtype_is_float32(self):
        est = FlatDepthEstimator()
        depth = est.estimate(np.zeros((32, 32, 3), dtype=np.uint8))
        assert depth.dtype == np.float32

    def test_constant_value(self):
        est = FlatDepthEstimator(depth=2.5)
        depth = est.estimate(np.zeros((32, 32, 3), dtype=np.uint8))
        assert np.all(depth == pytest.approx(2.5))


# ── DepthAnythingEstimator ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def depth_anything_estimator():
    pytest.importorskip("transformers")
    return DepthAnythingEstimator()


class TestDepthAnythingEstimator:
    @pytest.mark.slow
    def test_output_shape_matches_input(self, depth_anything_estimator):
        image = np.zeros((128, 96, 3), dtype=np.uint8)
        depth = depth_anything_estimator.estimate(image)
        assert depth.shape == (128, 96)

    @pytest.mark.slow
    def test_dtype_is_float32(self, depth_anything_estimator):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        depth = depth_anything_estimator.estimate(image)
        assert depth.dtype == np.float32

    @pytest.mark.slow
    def test_depth_in_metre_range(self, depth_anything_estimator):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        depth = depth_anything_estimator.estimate(image)
        assert depth.min() >= 0.49
        assert depth.max() <= 5.01
