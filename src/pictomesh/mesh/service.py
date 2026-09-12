from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d
import trimesh

ExportFormat = Literal["glb", "obj", "stl"]

# One object per photo set: pieces under this share of the largest are stray blobs or shells
POISSON_MIN_PIECE = 0.2
# Working cube vs the cloud's bounds (Open3D default 1.1): room for gap closures to bulge
POISSON_SCALE = 1.3


class MeshService:
    """Converts Open3D point clouds into 3D meshes and exports them."""

    def poisson(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
    ) -> trimesh.Trimesh:
        """Poisson surface reconstruction.

        Needs consistently oriented normals (kept when the cloud has them). Higher depth →
        finer detail but slower. Fill gaps before this; the result is watertight.

        Runs single-threaded: Open3D's parallel octree build races and segfaults.
        Measured on a 5k-point box cloud, one call per fresh process — n_threads=-1
        (the default, all cores) crashed 8/25, n_threads=2 crashed 3/25, n_threads=1
        crashed 0/25. Costs ~1.5x runtime (1.9s → 3.0s at depth 9 on 50k points).
        """
        pcd = self._ensure_normals(pcd)
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, scale=POISSON_SCALE, n_threads=1
        )
        self._drop_fragments(mesh_o3d)
        return self._to_trimesh(mesh_o3d)

    def ball_pivoting(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> trimesh.Trimesh:
        """Ball pivoting algorithm (BPA).

        Radii are auto-computed from the mean nearest-neighbour distance so the
        algorithm adapts to the scale of the point cloud without manual tuning.
        """
        pcd = self._ensure_normals(pcd)
        radii = self._compute_radii(pcd)
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        return self._to_trimesh(mesh_o3d)

    def export(
        self,
        mesh: trimesh.Trimesh,
        path: Path,
        fmt: ExportFormat = "glb",
    ) -> Path:
        """Export mesh to *path* with the given format extension.

        Parent directories are created automatically. Returns the written path.
        """
        out = path.with_suffix(f".{fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "glb":
            mesh.export(str(out), include_normals=True)  # viewers shade normal-less glTF flat
        else:
            mesh.export(str(out))
        return out

    # ── private ──────────────────────────────────────────────────────────────

    def _ensure_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Return a copy of *pcd* with normals, never mutating the original."""
        pcd = o3d.geometry.PointCloud(pcd)
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        return pcd

    def _compute_radii(self, pcd: o3d.geometry.PointCloud) -> list[float]:
        dists = np.asarray(pcd.compute_nearest_neighbor_distance())
        dists = dists[dists > 0]  # exclude duplicates (distance == 0)
        if len(dists) == 0:
            raise ValueError(
                "Cannot compute BPA radii: all points are duplicates. "
                "Ensure the point cloud has unique positions."
            )
        r = float(np.mean(dists))
        return [r, r * 2, r * 4, r * 8]

    @staticmethod
    def _drop_fragments(mesh_o3d: o3d.geometry.TriangleMesh) -> None:
        """Remove connected pieces smaller than POISSON_MIN_PIECE of the largest one"""
        cluster_ids, cluster_sizes, _ = mesh_o3d.cluster_connected_triangles()
        cluster_sizes = np.asarray(cluster_sizes)
        if len(cluster_sizes) == 0:
            return
        sizes = cluster_sizes[np.asarray(cluster_ids)]
        mesh_o3d.remove_triangles_by_mask(sizes < POISSON_MIN_PIECE * cluster_sizes.max())
        mesh_o3d.remove_unreferenced_vertices()

    def _to_trimesh(self, mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        colors = np.asarray(mesh_o3d.vertex_colors)
        return trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors if len(colors) == len(vertices) else None,
            process=False,
        )
