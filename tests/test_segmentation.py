import numpy as np
import pytest

from pictomesh.segmentation.service import RembgSegmentor, SegmentationService


# ── Helpers & fixtures ────────────────────────────────────────────────────────


def make_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Solid green BGR image."""
    return np.full((h, w, 3), (0, 200, 0), dtype=np.uint8)


def make_mask(h: int = 64, w: int = 64, foreground: bool = True) -> np.ndarray:
    """Uniform bool mask."""
    return np.full((h, w), foreground, dtype=bool)


def make_half_mask(h: int = 64, w: int = 64) -> np.ndarray:
    """Top half = foreground, bottom half = background."""
    mask = np.zeros((h, w), dtype=bool)
    mask[: h // 2, :] = True
    return mask


class _ConstantMaskSegmentor:
    """Test double — always returns the mask it was initialised with."""

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    def segment(self, image: np.ndarray) -> np.ndarray:
        return self._mask


class _SpySegmentor:
    """Test double — records calls and returns an all-foreground mask."""

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def segment(self, image: np.ndarray) -> np.ndarray:
        self.calls.append(image)
        return np.ones(image.shape[:2], dtype=bool)


@pytest.fixture
def image() -> np.ndarray:
    return make_image()


@pytest.fixture
def all_fg_service(image) -> SegmentationService:
    return SegmentationService(_ConstantMaskSegmentor(make_mask(foreground=True)))


@pytest.fixture
def all_bg_service(image) -> SegmentationService:
    return SegmentationService(_ConstantMaskSegmentor(make_mask(foreground=False)))


@pytest.fixture
def half_service() -> SegmentationService:
    return SegmentationService(_ConstantMaskSegmentor(make_half_mask()))


# ── remove_background ─────────────────────────────────────────────────────────


class TestRemoveBackground:
    def test_returns_rgba(self, all_fg_service, image):
        out = all_fg_service.remove_background(image)
        assert out.ndim == 3
        assert out.shape[2] == 4

    def test_spatial_dims_preserved(self, all_fg_service, image):
        out = all_fg_service.remove_background(image)
        assert out.shape[:2] == image.shape[:2]

    def test_all_foreground_alpha_is_255(self, all_fg_service, image):
        out = all_fg_service.remove_background(image)
        assert np.all(out[:, :, 3] == 255)

    def test_all_background_alpha_is_0(self, all_bg_service, image):
        out = all_bg_service.remove_background(image)
        assert np.all(out[:, :, 3] == 0)

    def test_half_mask_splits_alpha(self, half_service, image):
        out = half_service.remove_background(image)
        h = image.shape[0]
        assert np.all(out[: h // 2, :, 3] == 255)   # foreground rows
        assert np.all(out[h // 2 :, :, 3] == 0)     # background rows

    def test_bgr_converted_to_rgb(self):
        # Pure blue BGR = (255, 0, 0) → RGB R=0, G=0, B=255
        blue_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        blue_bgr[:, :, 0] = 255
        svc = SegmentationService(_ConstantMaskSegmentor(make_mask(10, 10, foreground=True)))
        out = svc.remove_background(blue_bgr)
        assert np.all(out[:, :, 0] == 0)    # R
        assert np.all(out[:, :, 2] == 255)  # B

    def test_raises_on_non_3channel_input(self, all_fg_service):
        with pytest.raises(ValueError, match="H×W×3"):
            all_fg_service.remove_background(np.zeros((64, 64), dtype=np.uint8))

    def test_raises_on_4channel_input(self, all_fg_service):
        with pytest.raises(ValueError, match="H×W×3"):
            all_fg_service.remove_background(np.zeros((64, 64, 4), dtype=np.uint8))

    def test_raises_on_mask_shape_mismatch(self):
        bad_mask = np.ones((32, 32), dtype=bool)  # wrong size
        svc = SegmentationService(_ConstantMaskSegmentor(bad_mask))
        with pytest.raises(ValueError, match="Mask shape"):
            svc.remove_background(make_image(64, 64))


# ── get_mask ──────────────────────────────────────────────────────────────────


class TestGetMask:
    def test_returns_2d_bool_array(self, all_fg_service, image):
        mask = all_fg_service.get_mask(image)
        assert mask.ndim == 2
        assert mask.dtype == bool

    def test_shape_matches_image(self, all_fg_service, image):
        mask = all_fg_service.get_mask(image)
        assert mask.shape == image.shape[:2]

    def test_all_foreground(self, all_fg_service, image):
        mask = all_fg_service.get_mask(image)
        assert np.all(mask)

    def test_all_background(self, all_bg_service, image):
        mask = all_bg_service.get_mask(image)
        assert not np.any(mask)

    def test_raises_on_invalid_input(self, all_fg_service):
        with pytest.raises(ValueError):
            all_fg_service.get_mask(np.zeros((64, 64), dtype=np.uint8))


# ── process_batch ─────────────────────────────────────────────────────────────


class TestProcessBatch:
    def test_returns_list_of_same_length(self, all_fg_service):
        images = [make_image() for _ in range(4)]
        results = all_fg_service.process_batch(images)
        assert len(results) == 4

    def test_each_result_is_rgba(self, all_fg_service):
        images = [make_image() for _ in range(3)]
        for out in all_fg_service.process_batch(images):
            assert out.shape[2] == 4

    def test_segmentor_called_once_per_image(self):
        spy = _SpySegmentor()
        svc = SegmentationService(spy)
        images = [make_image() for _ in range(5)]
        svc.process_batch(images)
        assert len(spy.calls) == 5

    def test_empty_batch_returns_empty_list(self, all_fg_service):
        assert all_fg_service.process_batch([]) == []


# ── RembgSegmentor integration ────────────────────────────────────────────────


class TestRembgSegmentor:
    def test_segment_returns_bool_mask(self):
        seg = RembgSegmentor()
        image = make_image()
        mask = seg.segment(image)
        assert mask.dtype == bool
        assert mask.shape == image.shape[:2]

    def test_service_integration(self):
        svc = SegmentationService(RembgSegmentor())
        image = make_image()
        out = svc.remove_background(image)
        assert out.shape == (64, 64, 4)
