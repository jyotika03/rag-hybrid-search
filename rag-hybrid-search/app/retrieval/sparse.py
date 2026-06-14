"""Sparse retrieval.

Default path uses OpenSearch BM25 (the managed store already indexes the text
field with the english analyzer). A pure in-memory rank_bm25 fallback is kept
for offline unit testing without a running cluster.
"""
from __future__ import annotations

from app.config import get_settings


def sparse_search(question: str, k: int | None = None, client=None) -> list[dict]:
    s = get_settings()
    k = k or s.sparse_k
    from app.store.opensearch_client import get_client  # lazy import
    client = client or get_client()
    body = {
        "size": k,
        "query": {"match": {"text": {"query": question}}},
        "_source": {"excludes": ["embedding"]},
    }
    res = client.search(index=s.opensearch_index, body=body)
    hits = res["hits"]["hits"]
    return [{"id": h["_id"], "score": h["_score"], "source": h["_source"]} for h in hits]


class BM25Index:
    """In-memory BM25 over a chunk corpus (offline / test fallback)."""

    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self._tokenized = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized)

    def search(self, question: str, k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(question.lower().split())
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{"id": c["id"], "score": float(sc), "source": c} for c, sc in ranked]
