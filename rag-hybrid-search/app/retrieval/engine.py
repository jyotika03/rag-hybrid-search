"""Hybrid retrieval orchestrator: dense + sparse -> RRF -> rerank.

Supports a dense-only mode so the API can serve a side-by-side comparison
(hybrid vs dense-only) for the dashboard.
"""
from __future__ import annotations

from app.config import get_settings
from app.models import Chunk, RetrievedChunk
from app.retrieval.dense import dense_search
from app.retrieval.sparse import sparse_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.rerank import rerank


def _to_retrieved(item: dict) -> RetrievedChunk:
    src = item["source"]
    chunk = Chunk(
        id=item["id"],
        text=src["text"],
        source_document=src["source_document"],
        chunk_index=src.get("chunk_index", 0),
        section_heading=src.get("section_heading"),
        page_number=src.get("page_number"),
        chunking_strategy=src.get("chunking_strategy", "unknown"),
        char_count=src.get("char_count", len(src["text"])),
    )
    return RetrievedChunk(
        chunk=chunk,
        dense_score=item.get("dense_score"),
        sparse_score=item.get("sparse_score"),
        fused_score=item.get("fused_score"),
        rerank_score=item.get("rerank_score"),
    )


def retrieve(question: str, hybrid: bool = True, client=None) -> list[RetrievedChunk]:
    s = get_settings()
    dense = dense_search(question, client=client)
    if not hybrid:
        ranked = rerank(question, [
            {"id": d["id"], "source": d["source"], "dense_score": d["score"],
             "fused_score": d["score"]} for d in dense
        ])
        return [_to_retrieved(r) for r in ranked]

    sparse = sparse_search(question, client=client)
    fused = reciprocal_rank_fusion(dense, sparse)
    ranked = rerank(question, fused)
    return [_to_retrieved(r) for r in ranked]
