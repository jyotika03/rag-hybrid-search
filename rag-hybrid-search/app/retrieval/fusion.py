"""Reciprocal Rank Fusion (RRF).

Combines dense and sparse ranked lists into one. RRF scores by rank position:
    score(d) = sum_over_lists( weight / (rrf_k + rank) )
Weighting (default 0.7 dense / 0.3 sparse) is configurable per use case.
"""
from __future__ import annotations

from app.config import get_settings


def reciprocal_rank_fusion(dense: list[dict], sparse: list[dict],
                           rrf_k: int | None = None,
                           dense_weight: float | None = None,
                           sparse_weight: float | None = None) -> list[dict]:
    s = get_settings()
    rrf_k = rrf_k if rrf_k is not None else s.rrf_k
    dense_weight = dense_weight if dense_weight is not None else s.dense_weight
    sparse_weight = sparse_weight if sparse_weight is not None else s.sparse_weight

    fused: dict[str, dict] = {}

    def add(results, weight, key):
        for rank, r in enumerate(results):
            cid = r["id"]
            contribution = weight / (rrf_k + rank + 1)
            if cid not in fused:
                fused[cid] = {"id": cid, "source": r["source"], "fused_score": 0.0,
                              "dense_score": None, "sparse_score": None}
            fused[cid]["fused_score"] += contribution
            fused[cid][key] = r["score"]

    add(dense, dense_weight, "dense_score")
    add(sparse, sparse_weight, "sparse_score")

    return sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)
