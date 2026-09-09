from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from pictomesh.filtering.service import FilteringService
from pictomesh.mesh.service import ExportFormat, MeshService
from pictomesh.reconstruction.service import (
    CameraIntrinsics,
    DepthEstimator,
    MultiViewReconstructor,
    ReconstructionService,
)
from pictomesh.segmentation.service import SegmentationService

# Reconstructed clouds are in the OpenCV camera frame (x right, y down, z forward).
# glTF / three.js are y-up with the camera looking down -z: rotate 180° about x.
CAMERA_TO_GLTF = np.diag([1.0, -1.0, -1.0, 1.0])

# Monocular depth is only relative, so the lifted relief needs a scale: the nearest
# subject pixel sits at BASE_DEPTH_M, the farthest RELIEF_RATIO × subject width behind it.
BASE_DEPTH_M = 1.0
RELIEF_RATIO = 0.25


class Pipeline:
    """Top-level orchestrator: BGR images → exported mesh file.

    Routing logic:
      < image_threshold images   →  depth-lift each image → BPA mesh
      >= image_threshold images  →  MASt3R multi-view (if reconstructor provided) → Poisson mesh
                                     else depth-lift each image → BPA mesh
    """

    def __init__(
        self,
        segmentation: SegmentationService,
        filtering: FilteringService,
        reconstruction: ReconstructionService,
        mesh: MeshService,
        depth_estimator: DepthEstimator,
        reconstructor: MultiViewReconstructor | None = None,
        image_threshold: int = 5,
    ) -> None:
        self._segmentation = segmentation
        self._filtering = filtering
        self._reconstruction = reconstruction
        self._mesh = mesh
        self._depth_estimator = depth_estimator
        self._reconstructor = reconstructor
        self._image_threshold = image_threshold

    def run(
        self,
        images: list[np.ndarray],
        output_path: Path,
        fmt: ExportFormat = "glb",
    ) -> Path:
        """Process *images* and write the mesh to *output_path*.<fmt>.

        Args:
            images:      Non-empty list of H×W×3 BGR uint8 arrays.
            output_path: Destination path without extension.
            fmt:         Output format — "glb", "obj", or "stl".

        Returns:
            Path to the written mesh file.

        Raises:
            ValueError: If *images* is empty.
        """
        if not images:
            raise ValueError("At least one image is required.")

        # 1. Filter outliers (passthrough for < 3 images)
        keep = self._filtering.filter(images)
        images = [images[i] for i in keep]

        # 2. Remove backgrounds → RGBA
        segmented = self._segmentation.process_batch(images)

        # 3. Reconstruct → Open3D point cloud, re-oriented for glTF
        pcd = self._reconstruct(images, segmented)
        pcd.transform(CAMERA_TO_GLTF)

        # 4. Mesh + export
        use_multi = len(images) >= self._image_threshold and self._reconstructor is not None
        trimesh_mesh = self._mesh.poisson(pcd) if use_multi else self._mesh.ball_pivoting(pcd)
        return self._mesh.export(trimesh_mesh, output_path, fmt)

    # ── private ───────────────────────────────────────────────────────────────

    def _reconstruct(
        self, images: list[np.ndarray], segmented: list[np.ndarray]
    ) -> o3d.geometry.PointCloud:
        if len(images) >= self._image_threshold and self._reconstructor is not None:
            bgr = [self._rgba_to_bgr(img) for img in segmented]
            return self._reconstruction.from_images(bgr, self._reconstructor)

        clouds = []
        for img, rgba in zip(images, segmented):
            h, w = img.shape[:2]
            intr = CameraIntrinsics.estimate(w, h)
            # Estimate on the original photo: the cutout's black background skews the model.
            depth = self._depth_estimator.estimate(img)
            depth = self._fit_depth_to_subject(depth, rgba, intr)
            clouds.append(self._reconstruction.from_depth(depth, intr))
        return self._merge_clouds(clouds)

    @staticmethod
    def _rgba_to_bgr(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

    @staticmethod
    def _fit_depth_to_subject(
        depth: np.ndarray, rgba: np.ndarray, intr: CameraIntrinsics
    ) -> np.ndarray:
        """Zero the background and rescale subject depth to a relief of known size.

        The estimator normalises over the whole frame, so the subject usually spans a
        thin slice of its range. Remap the subject's own [min, max] to
        [BASE_DEPTH_M, BASE_DEPTH_M + RELIEF_RATIO × subject width] (metres at that depth).
        Background pixels become 0 so from_depth() drops them.

        If the alpha mask is entirely empty (segmentation found no subject),
        the depth map is returned unchanged instead of producing an empty cloud.
        """
        mask = rgba[:, :, 3] > 0
        if not mask.any():
            return depth
        subject = depth[mask]
        d_min, d_max = float(subject.min()), float(subject.max())
        cols = np.flatnonzero(mask.any(axis=0))
        width_m = (cols[-1] - cols[0] + 1) / intr.fx * BASE_DEPTH_M
        fitted = np.zeros_like(depth)
        if d_max > d_min:
            fitted[mask] = (
                BASE_DEPTH_M + (subject - d_min) / (d_max - d_min) * RELIEF_RATIO * width_m
            )
        else:
            fitted[mask] = BASE_DEPTH_M
        return fitted

    @staticmethod
    def _merge_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        merged = o3d.geometry.PointCloud()
        for pcd in clouds:
            merged += pcd
        return merged
