from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


# ── Protocol ──────────────────────────────────────────────────────────────────


class Segmentor(Protocol):
    """Implemented by SAM2, rembg, or any future segmentation backend."""

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Return a boolean foreground mask of shape (H, W).

        Args:
            image: H×W×3 BGR uint8 image.

        Returns:
            H×W bool array — True where the subject is, False for background.
        """
        ...


# ── Service ───────────────────────────────────────────────────────────────────


class SegmentationService:
    """Removes image backgrounds using an injected segmentor backend."""

    def __init__(self, segmentor: Segmentor) -> None:
        self._segmentor = segmentor

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """Return an RGBA image with background pixels made transparent.

        Args:
            image: H×W×3 BGR uint8 image.

        Returns:
            H×W×4 uint8 RGBA image. Alpha=255 for foreground, 0 for background.
        """
        self._validate_image(image)
        mask = self._segmentor.segment(image)
        self._validate_mask(mask, image)
        return self._apply_mask(image, mask)

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        """Return the foreground mask without compositing.

        Args:
            image: H×W×3 BGR uint8 image.

        Returns:
            H×W bool array.
        """
        self._validate_image(image)
        mask = self._segmentor.segment(image)
        self._validate_mask(mask, image)
        return mask.astype(bool)

    def process_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply remove_background to every image in the list."""
        return [self.remove_background(img) for img in images]

    # ── private ───────────────────────────────────────────────────────────────

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        alpha = (mask.astype(bool) * 255).astype(np.uint8)
        return np.dstack([rgb, alpha])

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected H×W×3 BGR image, got shape {image.shape}."
            )

    @staticmethod
    def _validate_mask(mask: np.ndarray, image: np.ndarray) -> None:
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image spatial dims {image.shape[:2]}."
            )


# ── Concrete backends ─────────────────────────────────────────────────────────


class RembgSegmentor:
    """Lightweight CPU-friendly segmentor backed by rembg (u2net model).

    Requires: uv sync --extra single-image
    """

    def __init__(self) -> None:
        try:
            from rembg import new_session
            self._session = new_session()
        except ImportError as e:
            raise ImportError(
                "rembg is required for RembgSegmentor. "
                "Install it with: uv sync --extra single-image"
            ) from e

    def segment(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image
        from rembg import remove

        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        result = remove(pil, session=self._session)          # RGBA PIL image
        alpha = np.array(result)[:, :, 3]                   # extract alpha
        return alpha > 128                                   # bool mask
