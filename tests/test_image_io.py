from pathlib import Path

import numpy as np
import pytest

from pictomesh.image_io.manager import ImageManager

ASSETS = Path(__file__).parent / "assets"


@pytest.fixture
def manager() -> ImageManager:
    return ImageManager(ASSETS)


# ── Initialisation ────────────────────────────────────────────────────────────


class TestInit:
    def test_discovers_images(self, manager):
        assert len(manager.image_paths) == 5

    def test_paths_are_sorted(self, manager):
        names = [p.name for p in manager.image_paths]
        assert names == sorted(names)

    def test_image_names_match_paths(self, manager):
        assert manager.image_names == [p.stem for p in manager.image_paths]

    def test_raises_on_missing_folder(self):
        with pytest.raises(FileNotFoundError):
            ImageManager(Path("/nonexistent/folder"))

    def test_raises_on_file_instead_of_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(NotADirectoryError):
            ImageManager(f)

    def test_custom_extensions_filter(self, tmp_path):
        (tmp_path / "a.jpg").touch()
        (tmp_path / "b.png").touch()
        (tmp_path / "c.bmp").touch()
        mgr = ImageManager(tmp_path, extensions=(".jpg",))
        assert len(mgr.image_paths) == 1
        assert mgr.image_paths[0].suffix == ".jpg"

    def test_empty_folder_gives_empty_paths(self, tmp_path):
        mgr = ImageManager(tmp_path)
        assert mgr.image_paths == []
        assert mgr.image_names == []


# ── load_images ───────────────────────────────────────────────────────────────


class TestLoadImages:
    def test_loads_all_images(self, manager):
        manager.load_images()
        assert len(manager.get_images()) == 5

    def test_images_are_numpy_arrays(self, manager):
        manager.load_images()
        for img in manager.get_images():
            assert isinstance(img, np.ndarray)

    def test_images_are_3d_bgr(self, manager):
        manager.load_images()
        for img in manager.get_images():
            assert img.ndim == 3
            assert img.shape[2] == 3

    def test_metadata_has_required_keys(self, manager):
        manager.load_images()
        for meta in manager.get_metadata():
            assert "path" in meta
            assert "shape" in meta

    def test_metadata_path_matches_image_path(self, manager):
        manager.load_images()
        loaded_paths = {m["path"] for m in manager.get_metadata()}
        expected_paths = {str(p) for p in manager.image_paths}
        assert loaded_paths == expected_paths

    def test_metadata_shape_matches_image(self, manager):
        manager.load_images()
        for img, meta in zip(manager.get_images(), manager.get_metadata()):
            assert meta["shape"] == img.shape

    def test_resize_changes_dimensions(self, manager):
        target = (64, 64)
        manager.load_images(resize=target)
        for img in manager.get_images():
            assert img.shape[1] == target[0]  # width
            assert img.shape[0] == target[1]  # height

    def test_calling_twice_resets_data(self, manager):
        manager.load_images()
        manager.load_images()
        assert len(manager.get_images()) == 5

    def test_images_empty_before_load(self):
        mgr = ImageManager(ASSETS)
        assert mgr.get_images() == []
        assert mgr.get_metadata() == []
