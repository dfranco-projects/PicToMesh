from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


class ImageManager:
    """Loads and preprocesses images from a folder."""

    def __init__(
        self,
        folder_path: Path,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
    ) -> None:
        if not folder_path.exists():
            raise FileNotFoundError(f"Image folder not found: {folder_path}")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")

        self.folder_path = folder_path
        self.extensions = extensions
        self.image_paths: list[Path] = self._discover_paths()
        self.image_names: list[str] = [p.stem for p in self.image_paths]
        self.images: list[np.ndarray] = []
        self.metadata: list[dict] = []

    def load_images(self, resize: tuple[int, int] | None = None) -> None:
        """Load all images from the folder, optionally resizing them.

        Calling this multiple times resets the previously loaded data.

        Args:
            resize: Target (width, height) for resizing. None keeps original size.
        """
        self.images = []
        self.metadata = []

        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue
            if resize is not None:
                img = cv2.resize(img, resize)
            self.images.append(img)
            self.metadata.append({"path": str(path), "shape": img.shape})

    def get_images(self) -> list[np.ndarray]:
        """Return loaded image arrays. Call load_images() first."""
        return self.images

    def get_metadata(self) -> list[dict]:
        """Return metadata dicts with 'path' and 'shape' for each loaded image."""
        return self.metadata

    # ── private ──────────────────────────────────────────────────────────────

    def _discover_paths(self) -> list[Path]:
        return sorted(
            p for p in self.folder_path.iterdir()
            if p.suffix.lower() in self.extensions
        )
