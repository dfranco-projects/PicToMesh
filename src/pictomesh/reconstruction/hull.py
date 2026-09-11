"""Visual hull from subject masks, used to close the parts of a multi-view cloud no photo saw.

Poisson meshing of a partial cloud either leaves holes where no photo overlaps or bulges out
across them. The volume the photos allow for the object bounds what the unseen surface can
be: sampling its boundary where the cloud has no points gives Poisson a watertight envelope
to follow there, while photographed surfaces keep their real detail.

That volume is carved from a voxel grid: whatever a view saw as background (outside its
mask) or as empty space in front of a surface (from its depth map) goes. With silhouettes
alone, few photos leave a hull that bulges between camera directions; the depth maps cut
those bulges back wherever a photo saw past them.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HULL_RESOLUTION = 128  # voxels along the longest side of the hull's bounding box
SEARCH_RESOLUTION = 48  # coarse pass that finds the hull's extent
SEARCH_MARGIN = 0.25  # of the cloud's largest extent: how far unseen parts may reach
# Behind an object photographed from one side, nothing rules out the long cone of its
# silhouettes (blobs behind the chair). Space further than this behind the surface every
# depth-mapped view reports is assumed empty; enclosed interiors are filled back in, so
# thick objects photographed all round stay solid.
MAX_THICKNESS = 0.2  # of the cloud's largest extent
MASK_DILATION_PX = 1  # absorbs pixel rounding at the silhouette
# Depth maps miss the silhouette's own edge (points come from shrunk masks). Space in front
# of those pixels would survive as a thin sleeve from each camera to the object, so each
# map is extended this far (nearest known depth) before carving.
DEPTH_EXTENSION_PX = 4
EMPTY_TOLERANCE_VOXELS = 1.0  # space this far in front of an observed surface is empty
FILL_RADIUS_VOXELS = 3.0  # hull samples closer than this to a real point are not needed
# One view can't rule out anything along its line of sight, and casual photos are tightly
# framed, so space only one view sees would stay occupied as long wedges: require two.
MIN_VIEWS = 2


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
    """Carve a voxel grid down to the volume the views allow for the object.

    Args:
        masks:            H×W boolean subject masks, one per view.
        intrinsics:       N×3×3 pinhole intrinsics, in pixels of the masks.
        world_to_cameras: N×3×4 (or N×4×4) OpenCV world→camera transforms.
        bounds_min/max:   Corners of the region to carve, in world coordinates.
        depths:           Optional per-view depth maps (0 where unknown, None for no map).
        max_thickness:    With depths, also carve what lies further than this behind the
                          surface in every view that has depth for it (world units).

    A voxel is carved when a view sees it outside the subject mask, or in front of the
    surface its depth map reports. Views that don't see the voxel, because it falls outside
    the frame or behind the camera, leave it alone; a voxel fewer than MIN_VIEWS views see
    (or not all of them, when there are fewer) is dropped as unconstrained. Cavities the
    thickness rule leaves inside the object are filled back in.

    Returns:
        (occupancy, origin, voxel_size): a boolean X×Y×Z grid, the world position of
        voxel [0, 0, 0]'s centre, and the voxel edge length.
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
        u, v, seen = _project(cam, k, w, h)
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
    """Centres of the hull's boundary voxels, with outward unit normals.

    Normals follow the gradient of the smoothed occupancy, from inside (1) to outside (0).
    """
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
    """Hull samples, with normals and colours, for the regions the cloud does not cover.

    Args:
        points, colors: The real cloud and its colours, in the cameras' world frame.
        masks, intrinsics, world_to_cameras, depths: The views, as for carve().

    With depth maps, unseen backs are assumed no deeper than MAX_THICKNESS. Only hull
    pieces that contain real points count: a separate pocket is space a few, often
    near-parallel, views failed to rule out (a detached sheet beside the liberty statue),
    not part of the object. Samples closer than FILL_RADIUS_VOXELS to a real point are
    dropped, so photographed surfaces are left to the real points. Each kept sample takes
    the colour of its nearest real point.

    Returns:
        (points, normals, colors) of the fill samples; empty arrays when nothing is missing.
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
    """Bounding box of the hull, found with a coarse carve of a generous box around the cloud.

    Unseen parts can reach well past the photographed points (the far side of an object
    photographed from the front), so the box can't come from the cloud alone.
    """
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
    """The connected parts of *occupancy* that contain at least one of *points*."""
    labels, _ = ndimage.label(occupancy)
    idx = np.round((points - origin) / voxel).astype(int)
    inside = np.all((idx >= 0) & (idx < occupancy.shape), axis=1)
    held = np.unique(labels[tuple(idx[inside].T)])
    return np.isin(labels, held[held > 0])


def _extend_depth(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy the nearest known depth into unknown mask pixels up to DEPTH_EXTENSION_PX away."""
    known = depth > 0
    if not known.any():
        return depth
    distance, (rows, cols) = ndimage.distance_transform_edt(~known, return_indices=True)
    extend = mask & ~known & (distance <= DEPTH_EXTENSION_PX)
    out = depth.copy()
    out[extend] = depth[rows[extend], cols[extend]]
    return out


def _project(
    cam: np.ndarray, k: np.ndarray, w: int, h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel coordinates of camera-frame points, and which of them land in the image."""
    z = cam[:, 2]
    front = z > 1e-9
    u = np.full(len(cam), -1)
    v = np.full(len(cam), -1)
    u[front] = np.floor(k[0, 0] * cam[front, 0] / z[front] + k[0, 2]).astype(int)
    v[front] = np.floor(k[1, 1] * cam[front, 1] / z[front] + k[1, 2]).astype(int)
    return u, v, front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
