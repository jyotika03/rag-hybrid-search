"""Dense retrieval via OpenSearch k-NN over the embedded question."""
from __future__ import annotations

from app.config import get_settings
from app.ingestion.embeddings import embed_query


def dense_search(question: str, k: int | None = None, client=None) -> list[dict]:
    s = get_settings()
    k = k or s.dense_k
    from app.store.opensearch_client import get_client  # lazy import
    client = client or get_client()
    vec = embed_query(question)
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": vec, "k": k}}},
        "_source": {"excludes": ["embedding"]},
    }
    res = client.search(index=s.opensearch_index, body=body)
    hits = res["hits"]["hits"]
    return [{"id": h["_id"], "score": h["_score"], "source": h["_source"]} for h in hits]
