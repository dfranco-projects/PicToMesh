"""Visual hull samples that close the parts of a multi-view cloud no photo saw

The hull is carved from a voxel grid by silhouettes and by the empty space depth maps
observe; Poisson follows its boundary where the cloud has no points
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HULL_RESOLUTION = 128  # voxels along the longest side of the hull's bounding box
SEARCH_RESOLUTION = 48  # coarse pass that finds the hull's extent
SEARCH_MARGIN = 0.25  # of the cloud's largest extent: how far unseen parts may reach
MAX_THICKNESS = 0.2  # of the cloud's largest extent: how deep an unseen back may go
MASK_DILATION_PX = 1  # absorbs pixel rounding at the silhouette
DEPTH_EXTENSION_PX = 4  # covers the silhouette edge the shrunk-mask depth maps miss
EMPTY_TOLERANCE_VOXELS = 1.0  # space this far in front of an observed surface is empty
FILL_RADIUS_VOXELS = 3.0  # hull samples closer than this to a real point are not needed
MIN_VIEWS = 2  # one view can't rule out anything along its line of sight


def carve(
    masks: list[np.ndarray],
    intrinsics: np.ndarray,
    world_to_cameras: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    resolution: int = HULL_RESOLUTION,
    depths: list[np.ndarray | None] | None = None,
    max_thickness: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Carve a voxel grid down to the volume the views allow for the object

    A voxel goes when a view sees it outside the mask or in front of its depth map, when
    fewer than MIN_VIEWS views see it, or (with max_thickness) when it lies that far behind
    every depth-mapped view's surface; cavities that leaves inside are filled back in.
    depths are per-view maps (0 where unknown, None for no map); world_to_cameras are
    OpenCV N×3×4 transforms.

    Returns:
        (occupancy, origin, voxel_size): boolean X×Y×Z grid, centre of voxel [0, 0, 0],
        voxel edge length
    """
    extent = bounds_max - bounds_min
    voxel = float(extent.max()) / resolution
    dims = np.maximum(np.ceil(extent / voxel).astype(int), 1)
    origin = bounds_min + voxel / 2
    axes = [origin[i] + voxel * np.arange(dims[i]) for i in range(3)]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    occupied = np.ones(len(centres), dtype=bool)
    seen_by = np.zeros(len(centres), dtype=int)
    has_depth = np.zeros(len(centres), dtype=bool)
    far_behind_all = np.ones(len(centres), dtype=bool)
    kernel = np.ones((2 * MASK_DILATION_PX + 1,) * 2, np.uint8)

    for i, (mask, k, w2c) in enumerate(zip(masks, intrinsics, world_to_cameras)):
        h, w = mask.shape
        dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        cam = centres @ w2c[:3, :3].T + w2c[:3, 3]
        z = cam[:, 2]
        u, v, seen = project(cam, k, w, h)
        in_mask = np.zeros(len(cam), dtype=bool)
        in_mask[seen] = dilated[v[seen], u[seen]]
        occupied &= ~seen | in_mask
        depth = depths[i] if depths else None
        if depth is not None:
            depth = _extend_depth(depth, dilated)
            observed = np.zeros(len(cam))
            observed[seen] = depth[v[seen], u[seen]]
            known = seen & (observed > 0)
            occupied &= ~(known & (z < observed - EMPTY_TOLERANCE_VOXELS * voxel))
            if max_thickness is not None:
                far_behind_all &= ~known | (z > observed + max_thickness)
                has_depth |= known
        seen_by += seen

    constrained = seen_by >= min(MIN_VIEWS, len(masks))
    grid = (occupied & constrained).reshape(tuple(dims))
    if max_thickness is None:
        return grid, origin, voxel
    thin = grid & ~(has_depth & far_behind_all).reshape(tuple(dims))
    return ndimage.binary_fill_holes(thin) & grid, origin, voxel


def surface_samples(
    occupancy: np.ndarray, origin: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Centres of the hull's boundary voxels, with outward normals from the smoothed occupancy"""
    if min(occupancy.shape) < 2:  # too thin for a gradient: no usable surface
        return np.empty((0, 3)), np.empty((0, 3))
    boundary = occupancy & ~ndimage.binary_erosion(occupancy)
    smooth = ndimage.gaussian_filter(occupancy.astype(np.float32), sigma=1.5)
    gradient = np.stack(np.gradient(smooth), axis=-1)
    idx = np.argwhere(boundary)
    normals = -gradient[boundary]
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-6
    return origin + idx[keep] * voxel, normals[keep] / lengths[keep, None]


def fill_unobserved(
    points: np.ndarray,
    colors: np.ndarray,
    masks: list[np.ndarray],
    intrinsics: np.ndarray,
    world_to_cameras: np.ndarray,
    depths: list[np.ndarray | None] | None = None,
    resolution: int = HULL_RESOLUTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hull samples (points, normals, colours) where the real cloud has no points

    Only hull pieces holding real points count: detached pockets are space a few views failed
    to rule out. Each sample takes the colour of its nearest real point
    """
    empty = np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    thickness = MAX_THICKNESS * float(np.ptp(points, axis=0).max()) if depths else None
    lo, hi = _hull_bounds(points, masks, intrinsics, world_to_cameras, depths, thickness)
    occupancy, origin, voxel = carve(
        masks, intrinsics, world_to_cameras, lo, hi, resolution, depths, thickness
    )
    occupancy = _pieces_holding(occupancy, points, origin, voxel)
    samples, normals = surface_samples(occupancy, origin, voxel)
    if len(samples) == 0:
        return empty
    distance, nearest = cKDTree(points).query(samples)
    keep = distance > FILL_RADIUS_VOXELS * voxel
    if not keep.any():
        return empty
    return samples[keep], normals[keep], colors[nearest[keep]]


def _hull_bounds(
    points: np.ndarray,
    masks: list[np.ndarray],
    intrinsics: np.ndarray,
    world_to_cameras: np.ndarray,
    depths: list[np.ndarray | None] | None,
    thickness: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounding box of the hull, from a coarse carve: unseen parts can reach past the cloud"""
    lo, hi = points.min(axis=0), points.max(axis=0)
    margin = float((hi - lo).max()) * SEARCH_MARGIN
    coarse, origin, voxel = carve(
        masks,
        intrinsics,
        world_to_cameras,
        lo - margin,
        hi + margin,
        SEARCH_RESOLUTION,
        depths,
        thickness,
    )
    occupied = np.argwhere(coarse)
    if len(occupied) == 0:
        return lo, hi
    return (
        np.minimum(origin + (occupied.min(axis=0) - 1) * voxel, lo),
        np.maximum(origin + (occupied.max(axis=0) + 1) * voxel, hi),
    )


def _pieces_holding(
    occupancy: np.ndarray, points: np.ndarray, origin: np.ndarray, voxel: float
) -> np.ndarray:
    """The connected parts of *occupancy* that contain at least one of *points*"""
    labels, _ = ndimage.label(occupancy)
    idx = np.round((points - origin) / voxel).astype(int)
    inside = np.all((idx >= 0) & (idx < occupancy.shape), axis=1)
    held = np.unique(labels[tuple(idx[inside].T)])
    return np.isin(labels, held[held > 0])


def _extend_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy the nearest known depth into unknown mask pixels up to DEPTH_EXTENSION_PX away"""
    known = depth > 0
    if not known.any():
        return depth
    distance, (rows, cols) = ndimage.distance_transform_edt(~known, return_indices=True)
    extend = mask & ~known & (distance <= DEPTH_EXTENSION_PX)
    out = depth.copy()
    out[extend] = depth[rows[extend], cols[extend]]
    return out


def project(
    cam: np.ndarray, k: np.ndarray, w: int, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates of camera-frame points, and which of them land in the image"""
    z = cam[:, 2]
    front = z > 1e-9
    u = np.full(len(cam), -1)
    v = np.full(len(cam), -1)
    u[front] = np.floor(k[0, 0] * cam[front, 0] / z[front] + k[0, 2]).astype(int)
    v[front] = np.floor(k[1, 1] * cam[front, 1] / z[front] + k[1, 2]).astype(int)
    return u, v, front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
