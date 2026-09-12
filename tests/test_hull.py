"""
Visual hull tests
=================
A sphere seen by four cameras on a ring: silhouettes are discs, so carving, boundary
samples and gap filling can be checked against known geometry
"""

import numpy as np
import open3d as o3d
import pytest

from pictomesh.mesh.service import MeshService
from pictomesh.reconstruction.hull import (
    _pieces_holding,
    carve,
    fill_unobserved,
    surface_samples,
)

RADIUS = 0.5
DISTANCE = 3.0
FOCAL = 200.0
SIZE = 200


def _look_at_origin(angle: float) -> np.ndarray:
    """OpenCV world→camera transform for a camera on a ring around the y axis"""
    centre = DISTANCE * np.array([np.sin(angle), 0.0, -np.cos(angle)])
    forward = -centre / np.linalg.norm(centre)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return np.hstack([rotation, (-rotation @ centre)[:, None]])


@pytest.fixture(scope="module")
def cameras():
    w2c = np.stack([_look_at_origin(a) for a in np.radians([0, 90, 180, 270])])
    k = np.array([[FOCAL, 0, SIZE / 2], [0, FOCAL, SIZE / 2], [0, 0, 1]])
    intrinsics = np.stack([k] * len(w2c))
    # A sphere's silhouette from distance D is a disc of radius f·R / √(D² − R²)
    disc = FOCAL * RADIUS / np.sqrt(DISTANCE**2 - RADIUS**2)
    v, u = np.mgrid[:SIZE, :SIZE]
    mask = np.hypot(u - SIZE / 2, v - SIZE / 2) <= disc
    return [mask] * len(w2c), intrinsics, w2c


def _sphere_depth(w2c: np.ndarray) -> np.ndarray:
    """Depth map of the sphere from one camera (0 where the ray misses it)"""
    v, u = np.mgrid[:SIZE, :SIZE]
    rays = np.stack([(u - SIZE / 2) / FOCAL, (v - SIZE / 2) / FOCAL, np.ones_like(u, float)], -1)
    rotation, centre = w2c[:, :3], -w2c[:, :3].T @ w2c[:, 3]
    d = rays @ rotation  # camera-frame ray (z = 1) expressed in world axes
    # |centre + t·d| = RADIUS; with z_cam = 1 per unit t, t is the depth itself
    a = np.einsum("...i,...i", d, d)
    b = 2 * d @ centre
    disc = b**2 - 4 * a * (centre @ centre - RADIUS**2)
    hit = disc >= 0
    depth = np.zeros((SIZE, SIZE))
    depth[hit] = (-b[hit] - np.sqrt(disc[hit])) / (2 * a[hit])
    return depth


def _sphere_points(n: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals * RADIUS, normals


def test_carve_keeps_the_inside_and_carves_outside_every_silhouette(cameras):
    masks, intrinsics, w2c = cameras
    occupancy, origin, voxel = carve(masks, intrinsics, w2c, -np.ones(3), np.ones(3), 64)

    def occupied(point):
        return occupancy[tuple(np.round((np.array(point) - origin) / voxel).astype(int))]

    assert occupied([0, 0, 0])
    assert not occupied([0.8, 0, 0])  # outside the side silhouettes
    assert not occupied([0, 0.8, 0])  # above the sphere in every view


def test_voxels_no_camera_sees_are_not_kept():
    """A box reaching outside every frame must not keep those corners as phantom volume"""
    mask = np.ones((SIZE, SIZE), dtype=bool)
    k = np.array([[FOCAL, 0, SIZE / 2], [0, FOCAL, SIZE / 2], [0, 0, 1]])
    occupancy, _, _ = carve(
        [mask], k[None], _look_at_origin(0.0)[None], -np.full(3, 5.0), np.full(3, 5.0), 32
    )
    assert 0 < occupancy.mean() < 0.5


def test_surface_samples_have_outward_normals(cameras):
    masks, intrinsics, w2c = cameras
    occupancy, origin, voxel = carve(masks, intrinsics, w2c, -np.ones(3), np.ones(3), 64)
    samples, normals = surface_samples(occupancy, origin, voxel)
    assert len(samples) > 100
    outward = np.einsum("ij,ij->i", samples, normals) > 0
    assert outward.mean() > 0.95


def _sphere_without_cap(cameras):
    """Sphere points minus the cap facing +x, and the three other views with their depth maps"""
    masks, intrinsics, w2c = cameras
    points, normals = _sphere_points()
    seen = points[:, 0] < 0.3
    views = [0, 2, 3]
    depths = [_sphere_depth(w2c[i]) for i in views]
    return (
        points[seen],
        normals[seen],
        ([masks[i] for i in views], intrinsics[views], w2c[views], depths),
    )


def test_fill_goes_into_the_gap(cameras):
    points, _, (masks, intrinsics, w2c, depths) = _sphere_without_cap(cameras)
    colors = np.full((len(points), 3), 0.5)

    fill, fill_normals, fill_colors = fill_unobserved(
        points, colors, masks, intrinsics, w2c, depths=depths
    )

    assert len(fill) > 0
    assert (fill[:, 0] > 0.25).mean() > 0.9  # in the missing cap
    assert fill_normals.shape == fill.shape
    np.testing.assert_allclose(fill_colors, 0.5)


def test_hull_pockets_without_real_points_are_ignored():
    """A pocket the views failed to rule out, away from the object, is not part of it"""
    occupancy = np.zeros((20, 20, 20), dtype=bool)
    occupancy[2:8, 2:8, 2:8] = True  # the object's piece
    occupancy[12:18, 12:18, 12:18] = True  # a separate pocket
    points = np.array([[5.0, 5.0, 5.0]])  # voxel units: origin 0, size 1

    kept = _pieces_holding(occupancy, points, np.zeros(3), 1.0)

    assert kept[5, 5, 5] and not kept[15, 15, 15]


def test_hull_samples_in_space_a_view_saw_empty_are_dropped(cameras):
    """With four views the hull bulges past the sphere; the front view's depth rules the bulge out"""
    masks, intrinsics, w2c = cameras
    points, _ = _sphere_points()
    front = points[points[:, 2] < 0]
    colors = np.full((len(front), 3), 0.5)

    depth = _sphere_depth(w2c[0])
    blind, _, _ = fill_unobserved(front, colors, masks, intrinsics, w2c)
    seeing, _, _ = fill_unobserved(
        front, colors, masks, intrinsics, w2c, depths=[depth, None, None, None]
    )

    def in_front_of_the_surface(samples):
        cam = samples @ w2c[0][:, :3].T + w2c[0][:, 3]
        u = np.floor(FOCAL * cam[:, 0] / cam[:, 2] + SIZE / 2).astype(int)
        v = np.floor(FOCAL * cam[:, 1] / cam[:, 2] + SIZE / 2).astype(int)
        observed = depth[np.clip(v, 0, SIZE - 1), np.clip(u, 0, SIZE - 1)]
        return int(((observed > 0) & (cam[:, 2] < observed - 0.02)).sum())

    assert in_front_of_the_surface(blind) > 100  # the bulge the silhouettes allow...
    assert in_front_of_the_surface(seeing) < 0.05 * in_front_of_the_surface(blind)  # ...is gone


def test_cloud_with_a_gap_plus_fill_meshes_into_one_closed_body(cameras):
    points, normals, (masks, intrinsics, w2c, depths) = _sphere_without_cap(cameras)
    fill, fill_normals, _ = fill_unobserved(
        points, np.full((len(points), 3), 0.5), masks, intrinsics, w2c, depths=depths
    )
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([points, fill])))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate([normals, fill_normals]))

    mesh = MeshService().poisson(pcd, depth=7)

    assert mesh.body_count == 1
    assert open_edges(mesh) == 0  # no holes (Poisson may leave a few non-manifold edges)
    # the missing cap is closed, not cut flat at the edge of the photographed points
    assert mesh.vertices[:, 0].max() > 0.85 * RADIUS


def open_edges(mesh) -> int:
    """Edges used by a single face: the rims of holes"""
    _, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    return int((counts == 1).sum())
