from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from pictomesh.filtering.service import FilteringService
from pictomesh.mesh.service import ExportFormat, MeshService
from pictomesh.mesh.triposr import SingleImageReconstructor
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
      < image_threshold images   →  single-image model (TripoSR) on the first image, if provided
                                     else depth-lift each image → BPA mesh
      >= image_threshold images  →  multi-view model (DUSt3R) on the photos → Poisson mesh
                                     else same as below threshold
    """

    def __init__(
        self,
        segmentation: SegmentationService,
        filtering: FilteringService,
        reconstruction: ReconstructionService,
        mesh: MeshService,
        depth_estimator: DepthEstimator,
        reconstructor: MultiViewReconstructor | None = None,
        image_threshold: int = 2,
        single_image_reconstructor: SingleImageReconstructor | None = None,
    ) -> None:
        self._segmentation = segmentation
        self._filtering = filtering
        self._reconstruction = reconstruction
        self._mesh = mesh
        self._depth_estimator = depth_estimator
        self._reconstructor = reconstructor
        self._image_threshold = image_threshold
        self._single_image = single_image_reconstructor

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

        # 2. Pick the route before segmenting: the single-image model consumes only
        #    the first image that survived filtering, so segmenting the rest is waste.
        use_multi = len(images) >= self._image_threshold and self._reconstructor is not None
        use_single = not use_multi and self._single_image is not None

        # 3. Remove backgrounds → RGBA
        segmented = self._segmentation.process_batch(images[:1] if use_single else images)

        # 4. Reconstruct → mesh
        if use_single:
            trimesh_mesh = self._single_image.reconstruct(segmented[0])
        else:
            pcd = self._reconstruct(images, segmented)
            pcd.transform(CAMERA_TO_GLTF)
            trimesh_mesh = self._mesh.poisson(pcd) if use_multi else self._mesh.ball_pivoting(pcd)

        # 5. Export
        return self._mesh.export(trimesh_mesh, output_path, fmt)

    # ── private ───────────────────────────────────────────────────────────────

    def _reconstruct(
        self, images: list[np.ndarray], segmented: list[np.ndarray]
    ) -> o3d.geometry.PointCloud:
        if len(images) >= self._image_threshold and self._reconstructor is not None:
            # The model needs the photo context; the cutouts only supply the subject masks.
            masks = [rgba[:, :, 3] > 0 for rgba in segmented]
            return self._reconstruction.from_images(images, self._reconstructor, masks)

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
