"""Multi-view reconstruction with DUSt3R (vendored under pictomesh._vendor.dust3r)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import open3d as o3d

from pictomesh.device import default_device

MODEL = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
IMAGE_SIZE = 512
PATCH_SIZE = 16
MIN_CONFIDENCE = 3.0  # DUSt3R's default cut on its confidence maps (exp space, >= 1)
ALIGNMENT_ITERATIONS = 300


def prepare_view(
    bgr: np.ndarray, mask: np.ndarray | None, index: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Resize and crop one photo the way upstream load_images() does.

    Long side to IMAGE_SIZE, centre crop to a multiple of PATCH_SIZE (4:3 when the photo
    is square). Returns the DUSt3R view dict, the cropped RGB image in [0, 1] and the
    cropped boolean subject mask (all True when *mask* is None).
    """
    import torch

    h, w = bgr.shape[:2]
    scale = IMAGE_SIZE / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    rgb = cv2.cvtColor(cv2.resize(bgr, (new_w, new_h), interpolation=interp), cv2.COLOR_BGR2RGB)
    if mask is None:
        subject = np.ones((new_h, new_w), dtype=bool)
    else:
        subject = cv2.resize(
            mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    cx, cy = new_w // 2, new_h // 2
    half_w = ((2 * cx) // PATCH_SIZE) * PATCH_SIZE // 2
    half_h = ((2 * cy) // PATCH_SIZE) * PATCH_SIZE // 2
    if new_w == new_h:
        half_h = 3 * half_w // 4
    rgb = rgb[cy - half_h : cy + half_h, cx - half_w : cx + half_w]
    subject = subject[cy - half_h : cy + half_h, cx - half_w : cx + half_w]

    colors = rgb.astype(np.float32) / 255.0
    tensor = (torch.from_numpy(colors).permute(2, 0, 1) - 0.5) / 0.5
    view = {
        "img": tensor[None],
        "true_shape": np.int32([rgb.shape[:2]]),
        "idx": index,
        "instance": str(index),
    }
    return view, colors, subject


def merge_views(
    points: list[np.ndarray],
    confident: list[np.ndarray],
    subjects: list[np.ndarray],
    colors: list[np.ndarray],
    world_to_camera0: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Keep confident subject pixels from every view and express them in the first camera's frame.

    That frame follows the OpenCV convention (x right, y down, z forward), which is what
    the pipeline's glTF rotation expects.
    """
    xyz, rgb = [], []
    for pts, conf, subject, col in zip(points, confident, subjects, colors):
        keep = conf & subject
        xyz.append(pts[keep])
        rgb.append(col[keep])
    xyz = np.concatenate(xyz) @ world_to_camera0[:3, :3].T + world_to_camera0[:3, 3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(rgb).astype(np.float64))
    return pcd


def normalise_scale(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Scale the cloud about the origin so its largest extent is 1.

    DUSt3R's scale is arbitrary, and the viewer frames unit-sized objects.
    """
    extent = float(np.max(pcd.get_max_bound() - pcd.get_min_bound())) if len(pcd.points) else 0.0
    if extent > 0:
        pcd.scale(1.0 / extent, center=np.zeros(3))
    return pcd


class Dust3rReconstructor:
    """Unposed photos → coloured point cloud via DUSt3R pairwise point maps + global alignment.

    Requires: uv sync --extra multi-view. Weights (~2.4 GB, CC BY-NC-SA 4.0, non-commercial)
    are downloaded from Hugging Face on first use.
    """

    def __init__(
        self,
        model: str = MODEL,
        device: str | None = None,
        iterations: int = ALIGNMENT_ITERATIONS,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        try:
            from pictomesh._vendor.dust3r.model import AsymmetricCroCo3DStereo
        except ImportError as e:
            raise ImportError(
                "DUSt3R dependencies are missing. Install with: uv sync --extra multi-view"
            ) from e

        self._device = device or default_device()
        self._model = AsymmetricCroCo3DStereo.from_pretrained(model).to(self._device).eval()
        self._iterations = iterations
        self._min_confidence = min_confidence

    def reconstruct(
        self, images: list[np.ndarray], masks: list[np.ndarray] | None = None
    ) -> o3d.geometry.PointCloud:
        """Reconstruct from two or more H×W×3 BGR photos; *masks* are optional H×W subject masks."""
        import torch

        from pictomesh._vendor.dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from pictomesh._vendor.dust3r.image_pairs import make_pairs
        from pictomesh._vendor.dust3r.inference import inference

        if len(images) < 2:
            raise ValueError("DUSt3R needs at least two images.")
        masks = masks or [None] * len(images)
        prepared = [prepare_view(img, m, i) for i, (img, m) in enumerate(zip(images, masks))]
        views = [p[0] for p in prepared]

        pairs = make_pairs(views, scene_graph="complete", prefilter=None, symmetrize=True)
        with torch.no_grad():
            output = inference(pairs, self._model, self._device, batch_size=1, verbose=False)

        # Pose initialisation uses SVD, which MPS does not implement: align on the CPU there.
        align_device = "cpu" if self._device == "mps" else self._device
        if len(images) == 2:
            scene = global_aligner(
                output,
                device=align_device,
                mode=GlobalAlignerMode.PairViewer,
                min_conf_thr=self._min_confidence,
                verbose=False,
            )
        else:
            scene = global_aligner(
                output,
                device=align_device,
                mode=GlobalAlignerMode.PointCloudOptimizer,
                min_conf_thr=self._min_confidence,
                verbose=False,
            )
            scene.compute_global_alignment(
                init="mst", niter=self._iterations, schedule="cosine", lr=0.01
            )

        points = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
        confident = [c.detach().cpu().numpy() for c in scene.get_masks()]
        world_to_camera0 = np.linalg.inv(scene.get_im_poses()[0].detach().cpu().numpy())
        pcd = merge_views(
            points, confident, [p[2] for p in prepared], [p[1] for p in prepared], world_to_camera0
        )
        return normalise_scale(pcd)
