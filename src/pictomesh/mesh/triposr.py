"""Single-image reconstruction with TripoSR (vendored under pictomesh._vendor.tsr)."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import trimesh

from pictomesh.device import default_device

# TripoSR meshes are z-up with the input view looking along -x (camera on +x).
# glTF is y-up and the viewer camera sits on +z, so the pictured side must face +z:
# rotate -90° about x, then -90° about y.
TRIPOSR_TO_GLTF = trimesh.transformations.rotation_matrix(
    -np.pi / 2, [0, 1, 0]
) @ trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])

# Subject size relative to the square crop the model sees, as in upstream run.py.
FOREGROUND_RATIO = 0.85
BACKGROUND_GREY = 128


class SingleImageReconstructor(Protocol):
    """Implemented by TripoSR or any feed-forward image → mesh model."""

    def reconstruct(self, rgba: np.ndarray) -> trimesh.Trimesh:
        """Return a closed, y-up mesh from one H×W×4 RGBA uint8 image (alpha = subject)."""
        ...


def prepare_image(rgba: np.ndarray, ratio: float = FOREGROUND_RATIO) -> np.ndarray:
    """Crop to the subject, centre it in a square at *ratio* of the side, composite on grey.

    Mirrors the preprocessing in upstream run.py so the model sees the input
    distribution it was trained on. Returns an S×S×3 uint8 RGB image.
    """
    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(ys):
        rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    h, w = rgba.shape[:2]
    side = int(np.ceil(max(h, w) / ratio))
    top, left = (side - h) // 2, (side - w) // 2

    canvas = np.zeros((side, side, 4), dtype=np.float32)
    canvas[top : top + h, left : left + w] = rgba
    a = canvas[:, :, 3:4] / 255.0
    rgb = canvas[:, :, :3] * a + BACKGROUND_GREY * (1.0 - a)
    return rgb.round().astype(np.uint8)


class TripoSRReconstructor:
    """Feed-forward single image → vertex-coloured mesh via TripoSR.

    Requires: uv sync --extra single-image. Weights (~1.7 GB) are downloaded
    from Hugging Face on first use.
    """

    def __init__(
        self,
        model: str = "stabilityai/TripoSR",
        device: str | None = None,
        mc_resolution: int = 256,
        chunk_size: int = 8192,
    ) -> None:
        try:
            from pictomesh._vendor.tsr.system import TSR
        except ImportError as e:
            raise ImportError(
                "TripoSR dependencies are missing. Install with: uv sync --extra single-image"
            ) from e

        self._device = device or default_device()
        self._model = TSR.from_pretrained(
            model, config_name="config.yaml", weight_name="model.ckpt"
        )
        self._model.renderer.set_chunk_size(chunk_size)
        self._model.to(self._device).eval()
        self._resolution = mc_resolution

    def reconstruct(self, rgba: np.ndarray) -> trimesh.Trimesh:
        import torch

        image = prepare_image(rgba)
        with torch.no_grad():
            scene_codes = self._model([image], device=self._device)
            mesh = self._model.extract_mesh(
                scene_codes, has_vertex_color=True, resolution=self._resolution
            )[0]
        mesh.apply_transform(TRIPOSR_TO_GLTF)
        return mesh
