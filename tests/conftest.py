import open3d as o3d
import pytest


@pytest.fixture(scope="session")
def sphere_pcd() -> o3d.geometry.PointCloud:
    """Dense unit-sphere point cloud — reliable input for both Poisson and BPA."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    return mesh.sample_points_uniformly(number_of_points=5_000)


@pytest.fixture(scope="session")
def box_pcd() -> o3d.geometry.PointCloud:
    """Dense unit-box point cloud."""
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    return mesh.sample_points_uniformly(number_of_points=5_000)
