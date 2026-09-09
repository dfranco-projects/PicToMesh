"""
TripoSR tests
=============
Unit tests cover input preprocessing, the z-up → y-up transform and the vendored
marching-cubes helper (PyMCubes replaces torchmcubes upstream). The E2E test
(marked `slow`) runs the real model on a chair photo.
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pictomesh.mesh.triposr import BACKGROUND_GREY, TRIPOSR_TO_GLTF, prepare_image

ASSETS = Path(__file__).parent / "assets"


# ── prepare_image ─────────────────────────────────────────────────────────────


class TestPrepareImage:
    @pytest.fixture
    def rgba(self) -> np.ndarray:
        """40×60 frame with a 20×30 red subject offset from the centre."""
        img = np.zeros((40, 60, 4), dtype=np.uint8)
        img[10:30, 20:50] = [200, 50, 50, 255]
        return img

    def test_output_is_square_rgb(self, rgba):
        out = prepare_image(rgba)
        assert out.dtype == np.uint8
        assert out.shape[0] == out.shape[1]
        assert out.shape[2] == 3

    def test_subject_scaled_to_ratio_of_side(self, rgba):
        out = prepare_image(rgba, ratio=0.85)
        assert out.shape[0] == 36  # ceil(30 / 0.85)
        red = (out == [200, 50, 50]).all(axis=-1)
        ys, xs = np.nonzero(red)
        assert xs.max() - xs.min() + 1 == 30
        assert ys.max() - ys.min() + 1 == 20

    def test_subject_centred(self, rgba):
        out = prepare_image(rgba)
        red = (out == [200, 50, 50]).all(axis=-1)
        ys, xs = np.nonzero(red)
        centre = (out.shape[0] - 1) / 2
        assert abs(xs.mean() - centre) <= 1
        assert abs(ys.mean() - centre) <= 1

    def test_background_is_grey(self, rgba):
        out = prepare_image(rgba)
        assert (out[0, 0] == BACKGROUND_GREY).all()
        assert (out[-1, -1] == BACKGROUND_GREY).all()

    def test_transparent_pixels_inside_bbox_become_grey(self, rgba):
        rgba[15, 25] = [255, 255, 255, 0]
        out = prepare_image(rgba)
        ys, xs = np.nonzero((out == [200, 50, 50]).all(axis=-1))
        assert (out[ys.min() + 5, xs.min() + 5] == BACKGROUND_GREY).all()

    def test_empty_alpha_uses_full_frame(self):
        rgba = np.zeros((40, 60, 4), dtype=np.uint8)
        out = prepare_image(rgba, ratio=0.85)
        assert out.shape[0] == 71  # ceil(60 / 0.85)
        assert (out == BACKGROUND_GREY).all()


# ── orientation ───────────────────────────────────────────────────────────────


class TestOrientation:
    def test_z_up_becomes_y_up(self):
        up = TRIPOSR_TO_GLTF @ np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(up[:3], [0.0, 1.0, 0.0], atol=1e-12)

    def test_input_view_faces_viewer(self):
        # Model-frame +x (towards the input camera) must end up towards the glTF viewer on +z.
        front = TRIPOSR_TO_GLTF @ np.array([1.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(front[:3], [0.0, 0.0, 1.0], atol=1e-12)

    def test_is_proper_rotation(self):
        r = TRIPOSR_TO_GLTF[:3, :3]
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)


# ── vendored marching cubes ───────────────────────────────────────────────────


class TestMarchingCubes:
    def test_sphere_position_and_outward_faces(self):
        pytest.importorskip("mcubes")
        import torch

        from pictomesh._vendor.tsr.models.isosurface import MarchingCubeHelper

        helper = MarchingCubeHelper(resolution=32)
        centre = torch.tensor([0.3, 0.5, 0.7])
        inside = 0.2**2 - ((helper.grid_vertices - centre) ** 2).sum(-1)  # > 0 inside
        # extract_mesh() passes -(density - threshold); the helper negates it back.
        v, f = helper(-inside)
        mesh = trimesh.Trimesh(v.numpy(), f.numpy(), process=False)

        np.testing.assert_allclose(mesh.vertices.mean(axis=0), centre.numpy(), atol=0.02)
        outward = np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles_center - centre.numpy())
        assert (outward > 0).mean() > 0.95


# ── E2E test (slow) ───────────────────────────────────────────────────────────


@pytest.mark.slow
class TestTripoSRE2E:
    def test_chair_mesh_has_volume_and_colour(self):
        pytest.importorskip("mcubes")
        pytest.importorskip("omegaconf")
        import cv2

        from pictomesh.mesh.triposr import TripoSRReconstructor
        from pictomesh.segmentation.service import RembgSegmentor, SegmentationService

        img = cv2.imread(str(ASSETS / "chair_1.jpeg"))
        rgba = SegmentationService(RembgSegmentor()).process_batch([img])[0]
        mesh = TripoSRReconstructor(mc_resolution=128).reconstruct(rgba)

        assert len(mesh.faces) > 1000
        assert mesh.extents.min() / mesh.extents.max() > 0.2  # a solid, not a sheet
        assert mesh.visual.vertex_colors.shape[0] == len(mesh.vertices)
