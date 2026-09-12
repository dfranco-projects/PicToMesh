"""Multi-view reconstruction with Depth Anything 3 (vendored under pictomesh._vendor.depth_anything_3)."""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d

from pictomesh.device import default_device
from pictomesh.reconstruction.hull import fill_unobserved, project

MODEL = "depth-anything/DA3-LARGE-1.1"
PROCESS_RES = 504  # DA3's working resolution, on the long side
PATCH_SIZE = 14
# Subject pixels below this share of their view's median confidence are dropped: mostly
# smeared depth along silhouettes. A percentile cut was tried and removed thin parts
# instead (the statue's arm and head sit at ~0.6× the median; the 30th percentile, ~0.9×).
MIN_CONFIDENCE_RATIO = 0.3
# A view whose median subject confidence is this far below the other views' is dropped:
# on the liberty set, a far shot where the statue fills 3% of the frame scored 1.8 against
# 5.6-16.5 and landed a fifth of the object's size away from the other views.
LOW_CONFIDENCE_RATIO = 0.3
# Silhouette pixels mix subject and background (sky-blue fringes on the liberty statue), so
# points come from masks shrunk by this many pixels; the hull still uses the full masks.
MASK_EROSION_PX = 1
NORMAL_NEIGHBOURS = 30
# Stray depth at silhouettes and thin parts lands far in front of or behind the object
# (the chair cloud reached 2.6 units deep for 1.6 across); Poisson then wraps a dome
# around it. The confidence cut can't catch it in scenes DA3 is unsure of throughout.
OUTLIER_NEIGHBOURS = 20
OUTLIER_STD_RATIO = 2.0
CONSISTENCY_TOLERANCE = 0.02  # relative depth within which another view confirms a point
CONSISTENCY_MASK_DILATION = 0.01  # of the image diagonal


def prepare_views(
    images: list[np.ndarray], masks: list[np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Resize photos and masks to one shared size that DA3 processes unchanged.

    Long side to PROCESS_RES and both sides to the nearest multiple of PATCH_SIZE, then a
    centre crop to the smallest height and width so every view matches. DA3 leaves inputs
    of that shape alone, so the masks, which get the identical transform, stay aligned
    with its depth maps.

    Returns RGB uint8 images and boolean masks (all True when *masks* is None).
    """
    rgbs, subjects = [], []
    for i, bgr in enumerate(images):
        h, w = bgr.shape[:2]
        scale = PROCESS_RES / max(h, w)
        size = (_nearest_multiple(w * scale), _nearest_multiple(h * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rgbs.append(cv2.cvtColor(cv2.resize(bgr, size, interpolation=interp), cv2.COLOR_BGR2RGB))
        mask = np.ones((h, w), dtype=bool) if masks is None else masks[i]
        subjects.append(
            cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        )
    h = min(r.shape[0] for r in rgbs)
    w = min(r.shape[1] for r in rgbs)
    return [_centre_crop(r, h, w) for r in rgbs], [_centre_crop(m, h, w) for m in subjects]


def lift_views(
    depth: np.ndarray,
    conf: np.ndarray,
    intrinsics: np.ndarray,
    world_to_cameras: np.ndarray,
    subjects: list[np.ndarray],
    rgbs: list[np.ndarray],
) -> o3d.geometry.PointCloud:
    """Merge every view's confident subject pixels into one cloud in the first camera's frame.

    That frame follows the OpenCV convention (x right, y down, z forward), which is what the
    pipeline's glTF rotation expects. Normals are estimated per view and turned towards the
    camera that saw them, which Poisson needs to tell inside from outside. Views the model
    is much less sure of than the rest are left out entirely (see LOW_CONFIDENCE_RATIO),
    since their cameras are suspect too. Visual-hull samples then fill the regions no
    trusted view covered.
    """
    to_first = np.linalg.inv(_homogeneous(world_to_cameras[0]))
    # camera 0 frame → camera i frame, so hull carving works in the output frame too
    cameras = np.stack([_homogeneous(w2c) @ to_first for w2c in world_to_cameras])
    trusted = _trusted_views(conf, subjects, depth)

    points, normals, colors, owners, observed, used = [], [], [], [], [], []
    for i, (d, c, k, cam, subject, rgb) in enumerate(
        zip(depth, conf, intrinsics, cameras, subjects, rgbs)
    ):
        core = cv2.erode(
            subject.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=MASK_EROSION_PX
        )
        valid = core.astype(bool) & np.isfinite(d) & (d > 0)
        if not trusted[i] or not valid.any():
            continue
        valid &= c >= MIN_CONFIDENCE_RATIO * np.median(c[valid])
        observed.append(np.where(valid, d, 0.0))
        used.append(i)
        v, u = np.nonzero(valid)
        z = d[v, u]
        in_camera = np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=1)
        camera_to_first = np.linalg.inv(cam)
        xyz = in_camera @ camera_to_first[:3, :3].T + camera_to_first[:3, 3]

        view = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        view.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(NORMAL_NEIGHBOURS))
        view.orient_normals_towards_camera_location(camera_to_first[:3, 3])
        points.append(xyz)
        normals.append(np.asarray(view.normals))
        colors.append(rgb[v, u] / 255.0)
        owners.append(np.full(len(xyz), i))

    if not points:
        raise ValueError("No confident subject pixels in any view.")
    points, normals, colors, owner = map(np.concatenate, (points, normals, colors, owners))
    agreed = _agreed_by_other_views(
        points,
        owner,
        used,
        observed,
        [subjects[i] for i in used],
        intrinsics[used],
        cameras[used],
    )
    points, normals, colors = points[agreed], normals[agreed], colors[agreed]
    merged = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    _, inliers = merged.remove_statistical_outlier(OUTLIER_NEIGHBOURS, OUTLIER_STD_RATIO)
    points, normals, colors = points[inliers], normals[inliers], colors[inliers]
    fill = fill_unobserved(
        points,
        colors,
        [subjects[i] for i in used],
        intrinsics[used],
        cameras[used],
        depths=observed,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([points, fill[0]]))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate([normals, fill[1]]))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors, fill[2]]))
    return normalise_scale(pcd)


def normalise_scale(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Scale the cloud about the origin so its largest extent is 1.

    Multi-view depth has an arbitrary scale, and the viewer frames unit-sized objects.
    """
    extent = float(np.max(pcd.get_max_bound() - pcd.get_min_bound())) if len(pcd.points) else 0.0
    if extent > 0:
        pcd.scale(1.0 / extent, center=np.zeros(3))
    return pcd


class Da3Reconstructor:
    """Unposed photos → coloured cloud with oriented normals, via Depth Anything 3.

    One forward pass predicts every view's depth, confidence and camera; see lift_views for
    how they become a cloud. Requires: uv sync --extra multi-view. Weights (~1.6 GB,
    CC BY-NC 4.0, non-commercial) are downloaded from Hugging Face on first use.
    """

    def __init__(self, model: str = MODEL, device: str | None = None) -> None:
        try:
            from pictomesh._vendor.depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise ImportError(
                "Depth Anything 3 dependencies are missing. Install with: uv sync --extra multi-view"
            ) from e

        self._model = DepthAnything3.from_pretrained(model).to(device or default_device()).eval()

    def reconstruct(
        self, images: list[np.ndarray], masks: list[np.ndarray] | None = None
    ) -> o3d.geometry.PointCloud:
        """Reconstruct from two or more H×W×3 BGR photos; *masks* are optional H×W subject masks."""
        if len(images) < 2:
            raise ValueError("Multi-view reconstruction needs at least two images.")
        rgbs, subjects = prepare_views(images, masks)
        prediction = self._model.inference(rgbs)
        return lift_views(
            prediction.depth,
            prediction.conf,
            prediction.intrinsics,
            prediction.extrinsics,
            subjects,
            rgbs,
        )


def _trusted_views(conf: np.ndarray, subjects: list[np.ndarray], depth: np.ndarray) -> list[bool]:
    """False for views whose median subject confidence is far below the others'.

    With two views there is nothing to compare against, so both are kept.
    """
    medians = np.array(
        [
            float(np.median(c[s & (d > 0)])) if (s & (d > 0)).any() else 0.0
            for c, s, d in zip(conf, subjects, depth)
        ]
    )
    if len(medians) < 3:
        return [True] * len(medians)
    return list(medians >= LOW_CONFIDENCE_RATIO * np.median(medians))


def _agreed_by_other_views(
    points: np.ndarray,
    owner: np.ndarray,
    views: list[int],
    depths: list[np.ndarray],
    masks: list[np.ndarray],
    intrinsics: np.ndarray,
    cameras: np.ndarray,
) -> np.ndarray:
    """False for points more of the other views contradict than confirm (ghost copies)

    A view confirms a point it measured at the same depth and contradicts one it sees
    against background or in front of its surface; a point hidden behind that surface is
    no evidence either way
    """
    support = np.zeros(len(points), dtype=int)
    conflict = np.zeros(len(points), dtype=int)
    for j, depth, mask, k, cam in zip(views, depths, masks, intrinsics, cameras):
        other = np.flatnonzero(owner != j)
        h, w = mask.shape
        radius = max(1, round(CONSISTENCY_MASK_DILATION * np.hypot(h, w)))
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((2 * radius + 1,) * 2, np.uint8))
        in_camera = points[other] @ cam[:3, :3].T + cam[:3, 3]
        u, v, inside = project(in_camera, k, w, h)
        z = in_camera[:, 2]
        measured = np.zeros(len(other))
        measured[inside] = depth[v[inside], u[inside]]
        background = np.zeros(len(other), dtype=bool)
        background[inside] = dilated[v[inside], u[inside]] == 0
        known = measured > 0
        support[other] += known & (np.abs(z - measured) <= CONSISTENCY_TOLERANCE * measured)
        conflict[other] += background | (known & (z < measured * (1 - CONSISTENCY_TOLERANCE)))
    return conflict <= support


def _nearest_multiple(x: float) -> int:
    return max(PATCH_SIZE, round(x / PATCH_SIZE) * PATCH_SIZE)


def _centre_crop(a: np.ndarray, h: int, w: int) -> np.ndarray:
    top, left = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
    return a[top : top + h, left : left + w]


def _homogeneous(transform: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[: transform.shape[0], :4] = transform
    return out
