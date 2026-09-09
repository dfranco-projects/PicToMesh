from __future__ import annotations

from typing import Protocol

import networkx as nx
import numpy as np

# ── Protocol ──────────────────────────────────────────────────────────────────


class Encoder(Protocol):
    def encode(self, images: list[np.ndarray]) -> np.ndarray:
        """Return an (N, D) float32 feature matrix, L2-normalised.

        Args:
            images: List of H×W×3 BGR uint8 arrays.

        Returns:
            Array of shape (N, D) where each row is a unit-length feature vector.
        """
        ...


# ── Service ───────────────────────────────────────────────────────────────────


class FilteringService:
    """Filters a set of images to the most visually consistent cluster.

    Pipeline:
      1. Encode images → (N, D) feature matrix.
      2. Compute pairwise cosine similarity → (N, N) matrix.
      3. Build a graph: edge (i, j) exists if similarity > threshold.
      4. Run Louvain community detection.
      5. Return indices of the largest community.
    """

    def __init__(
        self,
        encoder: Encoder,
        similarity_threshold: float = 0.7,
        seed: int = 42,
    ) -> None:
        self._encoder = encoder
        self._threshold = similarity_threshold
        self._seed = seed

    def filter(self, images: list[np.ndarray]) -> list[int]:
        """Return indices of images in the dominant consistent cluster.

        Falls back to returning all indices when:
        - fewer than 3 images (clustering is not meaningful)
        - no image pair exceeds the similarity threshold (graph has no edges)
        """
        n = len(images)
        if n < 3:
            return list(range(n))

        features = self._encoder.encode(images)
        sim = self._compute_similarity(features)
        graph = self._build_graph(sim, n)

        if graph.number_of_edges() == 0:
            return list(range(n))

        communities = nx.community.louvain_communities(graph, seed=self._seed)
        dominant = max(communities, key=len)
        return sorted(dominant)

    def similarity_matrix(self, images: list[np.ndarray]) -> np.ndarray:
        """Return the (N, N) pairwise cosine similarity matrix for *images*."""
        features = self._encoder.encode(images)
        return self._compute_similarity(features)

    # ── private ───────────────────────────────────────────────────────────────

    def _compute_similarity(self, features: np.ndarray) -> np.ndarray:
        """Cosine similarity via dot product (features are L2-normalised)."""
        return (features @ features.T).clip(-1.0, 1.0)

    def _build_graph(self, similarity: np.ndarray, n: int) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] > self._threshold:
                    graph.add_edge(i, j, weight=float(similarity[i, j]))
        return graph
