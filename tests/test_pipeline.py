"""
Pipeline tests
==============
Unit tests use lightweight doubles for every service — no model weights, fast.

The E2E test (marked `slow`) exercises the full chain with real services:
  real images → rembg segmentation → flat depth → BPA mesh → GLB on disk.
It validates service compatibility and I/O, not geometric accuracy.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
import trimesh

from pictomesh.filtering.service import FilteringService
from pictomesh.mesh.service import MeshService
from pictomesh.pipeline import Pipeline
from pictomesh.reconstruction.service import (
    ReconstructionService,
)
from pictomesh.segmentation.service import RembgSegmentor, SegmentationService

ASSETS = Path(__file__).parent / "assets"


# ── Test doubles ──────────────────────────────────────────────────────────────


def _bgr(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _rgba(h: int = 8, w: int = 8) -> np.ndarray:
    return np.zeros((h, w, 4), dtype=np.uint8)


def _empty_pcd() -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud()


def _tiny_pcd() -> o3d.geometry.PointCloud:
    """3-point cloud — enough for BPA to produce a degenerate-but-valid mesh."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    )
    return pcd


class _Segmentation:
    """Always returns RGBA images; records inputs."""

    def __init__(self) -> None:
        self.received: list[list[np.ndarray]] = []

    def process_batch(self, images):
        self.received.append(images)
        return [_rgba(img.shape[0], img.shape[1]) for img in images]


class _Filtering:
    """Returns the indices it was configured with."""

    def __init__(self, indices: list[int] | None = None) -> None:
        self._indices = indices
        self.received: list[list[np.ndarray]] = []

    def filter(self, images):
        self.received.append(images)
        return self._indices if self._indices is not None else list(range(len(images)))


class _Reconstruction:
    """Records which method was called; returns a tiny point cloud."""

    def __init__(self) -> None:
        self.depth_calls = 0
        self.multi_calls = 0

    def from_depth(self, depth, intrinsics, **kwargs):
        self.depth_calls += 1
        return _tiny_pcd()

    def from_images(self, images, reconstructor):
        self.multi_calls += 1
        return _tiny_pcd()


class _Mesh:
    """Records which algorithm was used; returns an empty trimesh."""

    def __init__(self) -> None:
        self.bpa_calls = 0
        self.poisson_calls = 0

    def ball_pivoting(self, pcd):
        self.bpa_calls += 1
        return trimesh.Trimesh()

    def poisson(self, pcd, depth=9):
        self.poisson_calls += 1
        return trimesh.Trimesh()

    def export(self, mesh, path, fmt="glb"):
        out = path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
        return out


class _DepthEstimator:
    def estimate(self, image):
        return np.ones(image.shape[:2], dtype=np.float32)


class _Reconstructor:
    def reconstruct(self, images):
        return _tiny_pcd()


def _make_pipeline(
    segmentation=None,
    filtering=None,
    reconstruction=None,
    mesh=None,
    depth_estimator=None,
    reconstructor=None,
    image_threshold=5,
) -> tuple[Pipeline, _Segmentation, _Filtering, _Reconstruction, _Mesh]:
    seg = segmentation or _Segmentation()
    flt = filtering or _Filtering()
    rec = reconstruction or _Reconstruction()
    msh = mesh or _Mesh()
    dep = depth_estimator or _DepthEstimator()
    p = Pipeline(seg, flt, rec, msh, dep, reconstructor, image_threshold)
    return p, seg, flt, rec, msh


# ── Unit tests ────────────────────────────────────────────────────────────────


class TestPipelineRouting:
    def test_empty_images_raises(self, tmp_path):
        p, *_ = _make_pipeline()
        with pytest.raises(ValueError, match="[Aa]t least one"):
            p.run([], tmp_path / "out")

    def test_filter_reduces_image_list(self, tmp_path):
        flt = _Filtering(indices=[0, 2])
        p, seg, *_ = _make_pipeline(filtering=flt)
        images = [_bgr() for _ in range(4)]
        p.run(images, tmp_path / "out")
        passed_to_seg = seg.received[0]
        assert len(passed_to_seg) == 2

    def test_below_threshold_uses_bpa(self, tmp_path):
        p, _, _, rec, msh = _make_pipeline(image_threshold=5)
        p.run([_bgr() for _ in range(3)], tmp_path / "out")
        assert msh.bpa_calls == 1
        assert msh.poisson_calls == 0

    def test_below_threshold_uses_depth_not_multiview(self, tmp_path):
        p, _, _, rec, _ = _make_pipeline(image_threshold=5)
        p.run([_bgr() for _ in range(3)], tmp_path / "out")
        assert rec.depth_calls == 3
        assert rec.multi_calls == 0

    def test_above_threshold_with_reconstructor_uses_multiview(self, tmp_path):
        p, _, _, rec, msh = _make_pipeline(reconstructor=_Reconstructor(), image_threshold=3)
        p.run([_bgr() for _ in range(5)], tmp_path / "out")
        assert rec.multi_calls == 1
        assert rec.depth_calls == 0

    def test_above_threshold_with_reconstructor_uses_poisson(self, tmp_path):
        p, _, _, rec, msh = _make_pipeline(reconstructor=_Reconstructor(), image_threshold=3)
        p.run([_bgr() for _ in range(5)], tmp_path / "out")
        assert msh.poisson_calls == 1
        assert msh.bpa_calls == 0

    def test_above_threshold_without_reconstructor_falls_back_to_depth(self, tmp_path):
        p, _, _, rec, msh = _make_pipeline(reconstructor=None, image_threshold=3)
        p.run([_bgr() for _ in range(5)], tmp_path / "out")
        assert rec.depth_calls == 5
        assert rec.multi_calls == 0
        assert msh.bpa_calls == 1

    def test_one_image_uses_bpa(self, tmp_path):
        p, _, _, _, msh = _make_pipeline()
        p.run([_bgr()], tmp_path / "out")
        assert msh.bpa_calls == 1

    def test_output_file_created(self, tmp_path):
        p, *_ = _make_pipeline()
        out = p.run([_bgr()], tmp_path / "mesh", fmt="glb")
        assert out.exists()

    def test_output_format_respected(self, tmp_path):
        for fmt in ("glb", "obj", "stl"):
            p, *_ = _make_pipeline()
            out = p.run([_bgr()], tmp_path / f"mesh_{fmt}", fmt=fmt)
            assert out.suffix == f".{fmt}"

    def test_depth_called_once_per_image(self, tmp_path):
        class _CountingDepth:
            calls = 0

            def estimate(self, img):
                _CountingDepth.calls += 1
                return np.ones(img.shape[:2], dtype=np.float32)

        dep = _CountingDepth()
        p, *_ = _make_pipeline(depth_estimator=dep, image_threshold=10)
        p.run([_bgr() for _ in range(4)], tmp_path / "out")
        assert dep.calls == 4

    def test_segmentation_receives_filtered_images(self, tmp_path):
        flt = _Filtering(indices=[1, 3])
        seg = _Segmentation()
        p, *_ = _make_pipeline(segmentation=seg, filtering=flt)
        images = [_bgr() for _ in range(5)]
        p.run(images, tmp_path / "out")
        assert len(seg.received[0]) == 2


class TestOrientation:
    class _PointReconstruction:
        """Returns a single known camera-frame point (x right, y down, z forward)."""

        def from_depth(self, depth, intrinsics, **kwargs):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array([[0.1, 0.2, 2.0]]))
            return pcd

        def from_images(self, images, reconstructor):
            return self.from_depth(None, None)

    class _RecordingMesh(_Mesh):
        def __init__(self) -> None:
            super().__init__()
            self.clouds: list[np.ndarray] = []

        def ball_pivoting(self, pcd):
            self.clouds.append(np.asarray(pcd.points).copy())
            return super().ball_pivoting(pcd)

    def test_camera_frame_rotated_to_gltf(self, tmp_path):
        msh = self._RecordingMesh()
        p, *_ = _make_pipeline(reconstruction=self._PointReconstruction(), mesh=msh)
        p.run([_bgr()], tmp_path / "out")
        np.testing.assert_allclose(msh.clouds[0], [[0.1, -0.2, -2.0]])


class TestDepthFitting:
    class _DepthRecordingReconstruction:
        """Records the depth maps passed to from_depth."""

        def __init__(self) -> None:
            self.depths: list[np.ndarray] = []

        def from_depth(self, depth, intrinsics, **kwargs):
            self.depths.append(depth)
            return _tiny_pcd()

        def from_images(self, images, reconstructor):
            return _tiny_pcd()

    class _HalfMaskSegmentation:
        """Marks the left half of each image as foreground."""

        def process_batch(self, images):
            out = []
            for img in images:
                rgba = _rgba(img.shape[0], img.shape[1])
                rgba[:, : img.shape[1] // 2, 3] = 255
                out.append(rgba)
            return out

    def test_background_depth_zeroed(self, tmp_path):
        rec = self._DepthRecordingReconstruction()
        p, *_ = _make_pipeline(segmentation=self._HalfMaskSegmentation(), reconstruction=rec)
        p.run([_bgr()], tmp_path / "out")
        depth = rec.depths[0]
        assert (depth[:, :4] == 1.0).all()
        assert (depth[:, 4:] == 0.0).all()

    def test_empty_mask_keeps_full_depth(self, tmp_path):
        rec = self._DepthRecordingReconstruction()
        p, *_ = _make_pipeline(reconstruction=rec)  # default double: alpha all zero
        p.run([_bgr()], tmp_path / "out")
        assert (rec.depths[0] == 1.0).all()

    class _GradientDepth:
        """Depth increasing left to right, 2 m → 4 m, as a whole-frame estimator would."""

        def estimate(self, image):
            h, w = image.shape[:2]
            return np.tile(np.linspace(2.0, 4.0, w, dtype=np.float32), (h, 1))

    def test_subject_depth_rescaled_to_relief(self, tmp_path):
        rec = self._DepthRecordingReconstruction()
        p, *_ = _make_pipeline(
            segmentation=self._HalfMaskSegmentation(),
            reconstruction=rec,
            depth_estimator=self._GradientDepth(),
        )
        p.run([_bgr()], tmp_path / "out")
        depth = rec.depths[0]
        # 8 px frame → fx = 8; 4 px subject → 0.5 m wide at 1 m → relief = 0.25 × 0.5 m
        assert depth[:, 0] == pytest.approx(1.0)
        assert depth[:, 3] == pytest.approx(1.125)
        assert (depth[:, 4:] == 0.0).all()

    def test_depth_estimated_on_original_image(self, tmp_path):
        class _RecordingDepth:
            def __init__(self) -> None:
                self.images: list[np.ndarray] = []

            def estimate(self, image):
                self.images.append(image)
                return np.ones(image.shape[:2], dtype=np.float32)

        dep = _RecordingDepth()
        p, *_ = _make_pipeline(depth_estimator=dep)
        original = np.full((8, 8, 3), 200, dtype=np.uint8)
        p.run([original], tmp_path / "out")
        assert (dep.images[0] == 200).all()  # the cutout double would be all zeros


# ── E2E test (slow) ───────────────────────────────────────────────────────────


@pytest.mark.slow
class TestPipelineE2E:
    """Full chain with real services, no mocks.

    Uses a FlatDepthEstimator (constant depth plane) in place of Depth Anything
    v2, so the output mesh is geometrically flat — but every service boundary
    is exercised with real data.
    """

    @pytest.fixture(scope="class")
    def pipeline(self) -> Pipeline:
        seg_svc = SegmentationService(RembgSegmentor())

        # Use a pass-through encoder so filtering is fast (no CLIP download)
        class _IdentityEncoder:
            def encode(self, images):
                n = len(images)
                return np.eye(n, dtype=np.float32)

        flt_svc = FilteringService(_IdentityEncoder(), similarity_threshold=0.5)
        rec_svc = ReconstructionService()
        mesh_svc = MeshService()

        # Vary depth per call so the merged cloud has unique point positions.
        class _VaryingDepth:
            def __init__(self):
                self._d = 0.5

            def estimate(self, image):
                self._d += 0.5
                return np.full(image.shape[:2], self._d, dtype=np.float32)

        return Pipeline(seg_svc, flt_svc, rec_svc, mesh_svc, _VaryingDepth())

    @pytest.fixture(scope="class")
    def chair_images(self) -> list[np.ndarray]:
        import cv2

        paths = sorted((ASSETS).glob("chair_*.jpeg"))
        imgs = [cv2.imread(str(p)) for p in paths]
        # Downscale to keep BPA fast during tests
        return [cv2.resize(img, (64, 64)) for img in imgs]

    def test_produces_glb_file(self, pipeline, chair_images, tmp_path):
        out = pipeline.run(chair_images, tmp_path / "chair", fmt="glb")
        assert out.exists()
        assert out.suffix == ".glb"
        assert out.stat().st_size > 0

    def test_produces_obj_file(self, pipeline, chair_images, tmp_path):
        out = pipeline.run(chair_images, tmp_path / "chair", fmt="obj")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_image_runs_without_error(self, pipeline, chair_images, tmp_path):
        out = pipeline.run(chair_images[:1], tmp_path / "single", fmt="glb")
        assert out.exists()
