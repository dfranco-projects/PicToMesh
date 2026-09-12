import json

import numpy as np
import open3d as o3d
import pytest
import trimesh

from pictomesh.mesh.service import MeshService


@pytest.fixture
def service() -> MeshService:
    return MeshService()


# ── Poisson ───────────────────────────────────────────────────────────────────


def test_poisson_returns_trimesh(service, sphere_pcd):
    mesh = service.poisson(sphere_pcd, depth=6)
    assert isinstance(mesh, trimesh.Trimesh)


def test_poisson_has_geometry(service, sphere_pcd):
    mesh = service.poisson(sphere_pcd, depth=6)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_poisson_box_input(service, box_pcd):
    mesh = service.poisson(box_pcd, depth=6)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_poisson_estimates_normals_when_missing(service):
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_uniformly(3_000)
    pcd.normals = o3d.utility.Vector3dVector([])
    mesh = service.poisson(pcd, depth=6)
    assert len(mesh.vertices) > 0


def test_poisson_higher_depth_more_detail(service, sphere_pcd):
    coarse = service.poisson(sphere_pcd, depth=5)
    fine = service.poisson(sphere_pcd, depth=7)
    assert len(fine.vertices) >= len(coarse.vertices)


# ── Ball Pivoting ─────────────────────────────────────────────────────────────


class TestBallPivoting:
    def test_returns_trimesh(self, service, sphere_pcd):
        mesh = service.ball_pivoting(sphere_pcd)
        assert isinstance(mesh, trimesh.Trimesh)

    def test_has_geometry(self, service, sphere_pcd):
        mesh = service.ball_pivoting(sphere_pcd)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_box_input(self, service, box_pcd):
        mesh = service.ball_pivoting(box_pcd)
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_estimates_normals_when_missing(self, service):
        pcd = o3d.geometry.TriangleMesh.create_sphere(radius=1.0).sample_points_uniformly(3_000)
        pcd.normals = o3d.utility.Vector3dVector([])
        mesh = service.ball_pivoting(pcd)
        assert len(mesh.vertices) > 0


# ── Export ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bpa_sphere_mesh(sphere_pcd) -> trimesh.Trimesh:
    """Pre-built mesh shared across export tests (uses BPA — stable across calls)."""
    return MeshService().ball_pivoting(sphere_pcd)


class TestExport:
    @pytest.mark.parametrize("fmt", ["glb", "obj", "stl"])
    def test_creates_non_empty_file(self, service, bpa_sphere_mesh, tmp_path, fmt):
        out = service.export(bpa_sphere_mesh, tmp_path / "mesh", fmt=fmt)
        assert out.exists()
        assert out.stat().st_size > 0

    @pytest.mark.parametrize("fmt", ["glb", "obj", "stl"])
    def test_correct_extension(self, service, bpa_sphere_mesh, tmp_path, fmt):
        out = service.export(bpa_sphere_mesh, tmp_path / "mesh", fmt=fmt)
        assert out.suffix == f".{fmt}"

    def test_returns_correct_path(self, service, bpa_sphere_mesh, tmp_path):
        out = service.export(bpa_sphere_mesh, tmp_path / "mesh", fmt="obj")
        assert out == tmp_path / "mesh.obj"

    def test_creates_parent_directories(self, service, bpa_sphere_mesh, tmp_path):
        out = service.export(bpa_sphere_mesh, tmp_path / "a" / "b" / "mesh", fmt="glb")
        assert out.exists()

    def test_default_format_is_glb(self, service, bpa_sphere_mesh, tmp_path):
        out = service.export(bpa_sphere_mesh, tmp_path / "mesh")
        assert out.suffix == ".glb"

    def test_glb_carries_normals_and_colours(self, service, tmp_path):
        """Viewers shade glTF meshes without normals flat, one facet per triangle"""
        mesh = trimesh.creation.icosphere()
        mesh.visual.vertex_colors = [200, 50, 50, 255]
        out = service.export(mesh, tmp_path / "mesh", fmt="glb")

        data = out.read_bytes()
        gltf = json.loads(data[20 : 20 + int.from_bytes(data[12:16], "little")])
        assert {"NORMAL", "COLOR_0"} <= gltf["meshes"][0]["primitives"][0]["attributes"].keys()
        loaded = trimesh.load(out, force="mesh", process=False)
        assert (loaded.visual.vertex_colors == [200, 50, 50, 255]).all()


def test_poisson_keeps_vertex_colours(service, sphere_pcd):
    coloured = o3d.geometry.PointCloud(sphere_pcd)
    coloured.colors = o3d.utility.Vector3dVector(
        np.tile([0.2, 0.6, 0.9], (len(coloured.points), 1))
    )
    mesh = service.poisson(coloured, depth=6)
    colors = np.asarray(mesh.visual.vertex_colors)
    assert colors.shape == (len(mesh.vertices), 4)
    np.testing.assert_allclose(colors[:, :3].mean(axis=0), [51, 153, 229.5], atol=4)


def test_poisson_drops_debris(service, sphere_pcd):
    debris = o3d.geometry.PointCloud(sphere_pcd).random_down_sample(0.01)
    debris.translate((8.0, 0.0, 0.0))
    mesh = service.poisson(sphere_pcd + debris, depth=6)
    assert mesh.body_count == 1
