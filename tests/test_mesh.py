import json
import subprocess
import sys

import open3d as o3d
import pytest
import trimesh

from pictomesh.mesh.service import MeshService


@pytest.fixture
def service() -> MeshService:
    return MeshService()


# ── Poisson ───────────────────────────────────────────────────────────────────
# Open3D's Poisson leaves the process in a state where a later call segfaults, so
# each call needs its own process. Forking is not enough: by the time this module
# runs, the pytest process has torch/transformers/sklearn loaded with live OpenMP
# pools, and forking that aborts (signal 6). A spawned interpreter inherits none
# of it, so every call below runs in a fresh one.

_POISSON_SCRIPT = """
import json, sys
import open3d as o3d
from pictomesh.mesh.service import MeshService

shape, n_points, depth, drop_normals = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
geom = (
    o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    if shape == "sphere"
    else o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
)
pcd = geom.sample_points_uniformly(number_of_points=n_points)
if drop_normals == "1":
    pcd.normals = o3d.utility.Vector3dVector([])
mesh = MeshService().poisson(pcd, depth=depth)
print(json.dumps({
    "vertices": len(mesh.vertices),
    "faces": len(mesh.faces),
    "is_trimesh": isinstance(mesh, __import__("trimesh").Trimesh),
}))
"""


def poisson_out_of_process(
    shape: str = "sphere",
    n_points: int = 5_000,
    depth: int = 6,
    drop_normals: bool = False,
) -> dict:
    """Run one Poisson reconstruction in a fresh interpreter, return its stats."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _POISSON_SCRIPT,
            shape,
            str(n_points),
            str(depth),
            "1" if drop_normals else "0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"poisson subprocess failed ({result.returncode}): {result.stderr[-2000:]}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_poisson_returns_trimesh():
    assert poisson_out_of_process()["is_trimesh"]


def test_poisson_has_geometry():
    stats = poisson_out_of_process()
    assert stats["vertices"] > 0
    assert stats["faces"] > 0


def test_poisson_box_input():
    stats = poisson_out_of_process(shape="box")
    assert stats["vertices"] > 0
    assert stats["faces"] > 0


def test_poisson_estimates_normals_when_missing():
    stats = poisson_out_of_process(n_points=3_000, drop_normals=True)
    assert stats["vertices"] > 0


def test_poisson_higher_depth_more_detail():
    coarse = poisson_out_of_process(depth=5)
    fine = poisson_out_of_process(depth=7)
    assert fine["vertices"] >= coarse["vertices"]


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
