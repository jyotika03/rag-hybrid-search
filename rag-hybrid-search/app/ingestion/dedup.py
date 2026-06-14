"""Near-duplicate detection. Skips chunks whose cosine similarity against an
already-seen chunk exceeds the threshold (default 0.95), preventing the
retriever from wasting context-window slots on redundant content."""
from __future__ import annotations

import numpy as np


class Deduplicator:
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self._vectors: list[np.ndarray] = []

    def _norm(self, v):
        v = np.array(v, dtype=float)
        return v / (np.linalg.norm(v) + 1e-9)

    def is_duplicate(self, embedding: list[float]) -> bool:
        nv = self._norm(embedding)
        for existing in self._vectors:
            if float(np.dot(nv, existing)) > self.threshold:
                return True
        return False

    def add(self, embedding: list[float]) -> None:
        self._vectors.append(self._norm(embedding))
