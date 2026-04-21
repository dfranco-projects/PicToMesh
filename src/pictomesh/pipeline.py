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


class Pipeline:
    """Top-level orchestrator: BGR images → exported mesh file.

    Routing logic:
      < image_threshold images   →  depth-lift each image → BPA mesh
      >= image_threshold images  →  MASt3R multi-view (if reconstructor provided)
                                     else depth-lift each image → Poisson mesh
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

        # 3. Reconstruct → Open3D point cloud
        pcd = self._reconstruct(segmented)

        # 4. Mesh + export
        use_multi = len(images) >= self._image_threshold and self._reconstructor is not None
        trimesh_mesh = self._mesh.poisson(pcd) if use_multi else self._mesh.ball_pivoting(pcd)
        return self._mesh.export(trimesh_mesh, output_path, fmt)

    # ── private ───────────────────────────────────────────────────────────────

    def _reconstruct(self, segmented: list[np.ndarray]) -> o3d.geometry.PointCloud:
        bgr = [self._rgba_to_bgr(img) for img in segmented]

        if len(bgr) >= self._image_threshold and self._reconstructor is not None:
            return self._reconstruction.from_images(bgr, self._reconstructor)

        clouds = []
        for img in bgr:
            h, w = img.shape[:2]
            depth = self._depth_estimator.estimate(img)
            intr = CameraIntrinsics.estimate(w, h)
            clouds.append(self._reconstruction.from_depth(depth, intr))
        return self._merge_clouds(clouds)

    @staticmethod
    def _rgba_to_bgr(img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)

    @staticmethod
    def _merge_clouds(clouds: list[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        merged = o3d.geometry.PointCloud()
        for pcd in clouds:
            merged += pcd
        return merged
