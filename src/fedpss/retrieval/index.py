from __future__ import annotations

import numpy as np


class SimilarityIndex:
    def __init__(self, embeddings: np.ndarray, ids: list[str], labels: list[int] | None = None):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        self.embeddings = embeddings / norms
        self.ids = ids
        self.labels = labels

    def search(self, query_embedding: np.ndarray, top_k: int = 10, exclude_id: str | None = None):
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        scores = self.embeddings @ q
        order = np.argsort(-scores)
        results = []
        for idx in order:
            if exclude_id is not None and self.ids[idx] == exclude_id:
                continue
            results.append((self.ids[idx], float(scores[idx]), int(idx)))
            if len(results) >= top_k:
                break
        return results
