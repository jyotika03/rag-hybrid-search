"""Ingestion orchestrator: load -> chunk -> embed -> dedup -> index.

Indexes into a single OpenSearch index that backs BOTH dense (knn_vector) and
sparse (BM25 over the analyzed text field) retrieval, so the two stay in sync.
"""
from __future__ import annotations

import hashlib

from app.config import get_settings
from app.ingestion.loaders import LoadedDocument, load_document, load_directory
from app.ingestion.chunking import chunk_document, RawChunk
from app.ingestion.embeddings import embed_texts
from app.ingestion.dedup import Deduplicator
from app.store.opensearch_client import ensure_index, bulk_index, get_client


def _chunk_id(c: RawChunk) -> str:
    h = hashlib.sha1(
        f"{c.source_document}:{c.chunking_strategy}:{c.chunk_index}:{c.text[:64]}".encode()
    ).hexdigest()[:16]
    return f"{c.source_document}::{c.chunking_strategy}::{c.chunk_index}::{h}"


def index_documents(docs: list[LoadedDocument], strategy: str = "recursive",
                    recreate: bool = False, dedup_threshold: float = 0.95) -> dict:
    get_settings()
    client = get_client()
    ensure_index(client, recreate=recreate)

    raw_chunks: list[RawChunk] = []
    for doc in docs:
        raw_chunks.extend(
            chunk_document(doc, strategy, embed_fn=embed_texts)
        )

    texts = [c.text for c in raw_chunks]
    embeddings = embed_texts(texts) if texts else []

    dedup = Deduplicator(dedup_threshold)
    payload, skipped = [], 0
    for c, emb in zip(raw_chunks, embeddings):
        if dedup.is_duplicate(emb):
            skipped += 1
            continue
        dedup.add(emb)
        payload.append({
            "id": _chunk_id(c),
            "text": c.text,
            "source_document": c.source_document,
            "chunk_index": c.chunk_index,
            "section_heading": c.section_heading,
            "page_number": c.page_number,
            "chunking_strategy": c.chunking_strategy,
            "char_count": c.char_count,
            "embedding": emb,
        })

    indexed = bulk_index(payload, client) if payload else 0
    return {"indexed": indexed, "skipped_duplicates": skipped,
            "documents": len(docs), "strategy": strategy}


def index_path(path: str, strategy: str = "recursive", recreate: bool = False) -> dict:
    import os
    docs = load_directory(path) if os.path.isdir(path) else [load_document(path)]
    return index_documents(docs, strategy=strategy, recreate=recreate)
