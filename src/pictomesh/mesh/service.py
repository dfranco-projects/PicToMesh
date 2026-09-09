from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d
import trimesh

ExportFormat = Literal["glb", "obj", "stl"]


class MeshService:
    """Converts Open3D point clouds into 3D meshes and exports them."""

    def poisson(
        self,
        pcd: o3d.geometry.PointCloud,
        depth: int = 9,
    ) -> trimesh.Trimesh:
        """Poisson surface reconstruction.

        Best for dense, uniformly sampled point clouds. Higher depth → finer
        detail but slower. Produces a watertight mesh.
        """
        pcd = self._ensure_normals(pcd)
        mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
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

    def _to_trimesh(self, mesh_o3d: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
        vertices = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
