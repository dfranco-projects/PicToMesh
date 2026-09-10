from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import open3d as o3d

# ── Camera intrinsics ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def to_open3d(self) -> o3d.camera.PinholeCameraIntrinsic:
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )

    @staticmethod
    def estimate(width: int, height: int) -> CameraIntrinsics:
        """Estimate intrinsics assuming a ~60° horizontal FoV (no EXIF available)."""
        fx = fy = float(width)
        return CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
            width=width,
            height=height,
        )


# ── Protocols ─────────────────────────────────────────────────────────────────


class DepthEstimator(Protocol):
    """Implemented by Depth Anything v2 or any monocular depth backend."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Return an H×W float32 depth map in metres.

        Args:
            image: H×W×3 BGR uint8 image.
        """
        ...


class MultiViewReconstructor(Protocol):
    """Implemented by DUSt3R (or any future neural SfM backend)."""

    def reconstruct(
        self, images: list[np.ndarray], masks: list[np.ndarray] | None = None
    ) -> o3d.geometry.PointCloud:
        """Return a dense point cloud from a set of unposed BGR images.

        Args:
            images: H×W×3 BGR uint8 photos of the same object from different viewpoints.
            masks:  Optional H×W boolean subject masks, one per image; background points
                    are dropped when given.
        """
        ...


# ── Concrete depth estimators ─────────────────────────────────────────────────


class FlatDepthEstimator:
    """Returns a constant depth plane — useful for testing and CPU fallback."""

    def __init__(self, depth: float = 1.0) -> None:
        self._depth = depth

    def estimate(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.full((h, w), self._depth, dtype=np.float32)


class DepthAnythingEstimator:
    """Monocular depth via Depth Anything V2, normalized to metres.

    Requires: uv sync --extra depth
    """

    def __init__(self, model: str = "depth-anything/Depth-Anything-V2-Small-hf") -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise ImportError(
                "transformers is required for DepthAnythingEstimator. "
                "Install with: uv sync --extra depth"
            ) from e
        self._processor = AutoImageProcessor.from_pretrained(model)
        self._model = AutoModelForDepthEstimation.from_pretrained(model)
        self._model.eval()

    def estimate(self, image: np.ndarray) -> np.ndarray:
        """Return an H×W float32 depth map normalized to [0.5, 5.0] metres.

        Depth Anything V2 outputs relative inverse depth (larger values are
        closer), so the mapping is reversed: nearest pixel lands at 0.5 m,
        farthest at 5.0 m.

        Args:
            image: H×W×3 BGR uint8 image.
        """
        import torch
        from PIL import Image as PILImage

        h, w = image.shape[:2]
        pil = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=pil, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        depth = (
            torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth = 0.5 + (d_max - depth) / (d_max - d_min) * 4.5
        else:
            depth = np.full_like(depth, 1.0)

        return depth.astype(np.float32)


# ── Service ───────────────────────────────────────────────────────────────────


class ReconstructionService:
    """Converts images / depth maps into Open3D point clouds."""

    def from_depth(
        self,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 1.0,
        depth_trunc: float = 10.0,
    ) -> o3d.geometry.PointCloud:
        """Lift a single-channel depth map to a point cloud.

        Args:
            depth:       H×W float32 array with depth values in metres.
            intrinsics:  Pinhole camera intrinsics.
            depth_scale: Multiplier applied to depth values before processing.
                         Use 1000.0 if depth is in millimetres.
            depth_trunc: Points beyond this distance (metres) are discarded.
        """
        # Open3D silently ignores depth_trunc for float32 images (only works for
        # uint16). Mask manually: set out-of-range pixels to 0 (treated as invalid).
        depth = depth.astype(np.float32).copy()
        depth[depth > depth_trunc] = 0.0

        depth_img = o3d.geometry.Image(depth)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_img,
            intrinsics.to_open3d(),
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
        )
        return pcd

    def from_rgbd(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale: float = 1.0,
        depth_trunc: float = 10.0,
    ) -> o3d.geometry.PointCloud:
        """Create a coloured point cloud from an RGB image and a depth map.

        Args:
            color:  H×W×3 uint8 BGR image (OpenCV convention).
            depth:  H×W float32 depth map in metres.
            intrinsics, depth_scale, depth_trunc: same as from_depth().
        """
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color_img = o3d.geometry.Image(rgb.astype(np.uint8))
        depth = depth.astype(np.float32).copy()
        depth[depth > depth_trunc] = 0.0
        depth_img = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics.to_open3d())
        return pcd

    def from_images(
        self,
        images: list[np.ndarray],
        reconstructor: MultiViewReconstructor,
        masks: list[np.ndarray] | None = None,
    ) -> o3d.geometry.PointCloud:
        """Multi-view reconstruction via an injected reconstructor (e.g. DUSt3R).

        Args:
            images:        List of H×W×3 BGR images from different viewpoints.
            reconstructor: Any object satisfying MultiViewReconstructor protocol.
            masks:         Optional per-image boolean subject masks, passed through.
        """
        if len(images) < 2:
            raise ValueError(
                f"Multi-view reconstruction requires at least 2 images, got {len(images)}."
            )
        return reconstructor.reconstruct(images, masks)
