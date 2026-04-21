from __future__ import annotations

import cv2
import numpy as np


class CLIPEncoder:
    """CLIP image encoder satisfying the filtering.service.Encoder protocol.

    Model weights are loaded lazily on the first call to encode() to avoid
    startup overhead and to allow importing without a GPU or CLIP installed.

    Default model: ViT-B-16-plus-240 / laion400m_e32 (open-clip-torch).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._device: str | None = None

    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Encode a list of BGR images to L2-normalised CLIP feature vectors.

        Args:
            images: List of H×W×3 BGR uint8 arrays.

        Returns:
            (N, D) float32 array of unit-length feature vectors.
        """
        self._ensure_loaded()

        import torch
        from PIL import Image

        features = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self._preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._model.encode_image(tensor).float()
                feat /= feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy()[0])

        return np.stack(features).astype(np.float32)

    # ── private ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
            import torch
        except ImportError as e:
            raise ImportError(
                "open-clip-torch and torch are required for CLIPEncoder."
            ) from e

        self._device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model.to(self._device)
        self._model.eval()
