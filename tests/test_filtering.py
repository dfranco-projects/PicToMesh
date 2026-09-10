from pathlib import Path

import numpy as np
import pytest

from pictomesh.filtering.service import FilteringService

ASSETS = Path(__file__).parent / "assets"

# ── Test doubles & helpers ────────────────────────────────────────────────────

DIM = 512  # match CLIP's output dimensionality


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def make_centroid(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal(DIM).astype(np.float32))


def orthogonal_centroid(reference: np.ndarray, seed: int) -> np.ndarray:
    """Return a unit vector orthogonal to *reference*."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    v -= np.dot(v, reference) * reference  # Gram-Schmidt
    return unit(v)


def noisy_cluster(centroid: np.ndarray, n: int, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """Return *n* unit vectors near *centroid*."""
    rng = np.random.default_rng(seed)
    vecs = centroid + rng.standard_normal((n, DIM)).astype(np.float32) * noise
    return np.stack([unit(v) for v in vecs])


class _FixedEncoder:
    """Returns a pre-built feature matrix regardless of input images."""

    def __init__(self, features: np.ndarray) -> None:
        self._features = features

    def encode(self, images: list) -> np.ndarray:
        return self._features


class _SpyEncoder:
    """Records calls and returns identity features (one-hot per image)."""

    def __init__(self) -> None:
        self.call_count = 0

    def encode(self, images: list) -> np.ndarray:
        self.call_count += 1
        n = len(images)
        feats = np.eye(n, dtype=np.float32)  # orthogonal — sim = 0 for i≠j
        return feats


def make_images(n: int) -> list[np.ndarray]:
    """Dummy BGR images — content irrelevant when using FixedEncoder."""
    return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]


# ── FilteringService.filter ───────────────────────────────────────────────────


class TestFilter:
    def test_single_image_returns_index_zero(self):
        assert FilteringService(_SpyEncoder()).filter(make_images(1)) == [0]

    def test_two_images_returned_unchanged(self):
        svc = FilteringService(_SpyEncoder())
        assert svc.filter(make_images(2)) == [0, 1]

    def test_all_similar_returns_all_indices(self):
        c = make_centroid(0)
        features = noisy_cluster(c, n=6)
        svc = FilteringService(_FixedEncoder(features), similarity_threshold=0.7)
        result = svc.filter(make_images(6))
        assert result == list(range(6))

    def test_two_clusters_returns_larger(self):
        c0 = make_centroid(0)
        c1 = orthogonal_centroid(c0, seed=99)
        large = noisy_cluster(c0, n=6, seed=1)  # indices 0-5
        small = noisy_cluster(c1, n=3, seed=2)  # indices 6-8
        features = np.vstack([large, small])
        svc = FilteringService(_FixedEncoder(features), similarity_threshold=0.7)
        result = svc.filter(make_images(9))
        assert result == list(range(6))

    def test_two_equal_clusters_returns_one_complete_cluster(self):
        c0 = make_centroid(0)
        c1 = orthogonal_centroid(c0, seed=99)
        f0 = noisy_cluster(c0, n=4, seed=1)  # indices 0-3
        f1 = noisy_cluster(c1, n=4, seed=2)  # indices 4-7
        features = np.vstack([f0, f1])
        svc = FilteringService(_FixedEncoder(features), similarity_threshold=0.7)
        result = svc.filter(make_images(8))
        assert len(result) == 4
        assert set(result) in ({0, 1, 2, 3}, {4, 5, 6, 7})

    def test_no_edges_returns_all_indices(self):
        # Identity features → all pairwise similarities = 0 → no edges at threshold=0.7
        features = np.eye(5, dtype=np.float32)
        svc = FilteringService(_FixedEncoder(features), similarity_threshold=0.7)
        result = svc.filter(make_images(5))
        assert result == list(range(5))

    def test_result_is_sorted(self):
        c0 = make_centroid(0)
        c1 = orthogonal_centroid(c0, seed=99)
        # Interleave the clusters: 0,2,4 → c0; 1,3,5,6,7 → c1
        rows = []
        for i in range(8):
            c = c0 if i % 2 == 0 else c1
            rng = np.random.default_rng(i)
            v = c + rng.standard_normal(DIM).astype(np.float32) * 0.02
            rows.append(unit(v))
        features = np.stack(rows)
        svc = FilteringService(_FixedEncoder(features), similarity_threshold=0.7)
        result = svc.filter(make_images(8))
        assert result == sorted(result)

    def test_encoder_called_once_per_filter(self):
        spy = _SpyEncoder()
        svc = FilteringService(spy)
        svc.filter(make_images(5))
        assert spy.call_count == 1


# ── FilteringService.similarity_matrix ───────────────────────────────────────


class TestSimilarityMatrix:
    def test_shape(self):
        c = make_centroid(0)
        features = noisy_cluster(c, n=4)
        svc = FilteringService(_FixedEncoder(features))
        mat = svc.similarity_matrix(make_images(4))
        assert mat.shape == (4, 4)

    def test_diagonal_is_one(self):
        c = make_centroid(0)
        features = noisy_cluster(c, n=5)
        svc = FilteringService(_FixedEncoder(features))
        mat = svc.similarity_matrix(make_images(5))
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_symmetric(self):
        features = noisy_cluster(make_centroid(0), n=6)
        svc = FilteringService(_FixedEncoder(features))
        mat = svc.similarity_matrix(make_images(6))
        np.testing.assert_allclose(mat, mat.T, atol=1e-6)

    def test_orthogonal_vectors_have_zero_similarity(self):
        c0 = make_centroid(0)
        c1 = orthogonal_centroid(c0, seed=99)
        features = np.stack([c0, c1])
        svc = FilteringService(_FixedEncoder(features))
        mat = svc.similarity_matrix(make_images(2))
        assert abs(mat[0, 1]) < 1e-5

    def test_identical_vectors_have_unit_similarity(self):
        v = make_centroid(0)
        features = np.stack([v, v])
        svc = FilteringService(_FixedEncoder(features))
        mat = svc.similarity_matrix(make_images(2))
        assert mat[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_values_bounded_between_minus_one_and_one(self):
        features = noisy_cluster(make_centroid(0), n=8, noise=1.0)
        svc = FilteringService(_FixedEncoder(features))
        mat = svc.similarity_matrix(make_images(8))
        assert np.all(mat >= -1.0)
        assert np.all(mat <= 1.0)


# ── E2E test (slow) ───────────────────────────────────────────────────────────


@pytest.mark.slow
class TestFilterE2E:
    """Real CLIP features: five views of one monument keep each other, a cat is dropped."""

    def test_outlier_removed_from_liberty_set(self):
        import cv2

        from pictomesh.filtering.clip_encoder import CLIPEncoder

        paths = sorted((ASSETS / "liberty").glob("liberty_*.jpeg")) + [ASSETS / "cat_1.jpeg"]
        images = [cv2.resize(cv2.imread(str(p)), (224, 224)) for p in paths]
        keep = FilteringService(CLIPEncoder()).filter(images)
        assert keep == [0, 1, 2, 3, 4]
