"""FastAPI service for the hybrid-search RAG pipeline."""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import (AskRequest, AskResponse, IngestRequest, DocumentInfo)
from app.retrieval.engine import retrieve
from app.generation.generator import generate_answer
from app.generation import citations as cit
from app.generation import confidence as conf
from app.ingestion.indexer import index_path
from app.ingestion.loaders import LOADERS
from app.store.opensearch_client import get_client, ensure_index

app = FastAPI(title="Hybrid RAG API", version="1.0.0",
              description="RAG with dense+sparse hybrid retrieval, RRF, reranking, "
                          "citation verification and confidence scoring.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    try:
        get_client().info()
        ok = True
    except Exception:
        ok = False
    return {"status": "ok", "opensearch": ok}


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest):
    s = get_settings()
    chunks = retrieve(req.question, hybrid=req.hybrid)
    if not chunks:
        return AskResponse(
            answer=conf.low_confidence_response(req.question, []),
            answered=False, citations=[],
            confidence=conf.ConfidenceBreakdown(0, 0, 0, 0),
            retrieved=[], notes="No chunks retrieved. Is the index populated?",
        )

    rc = conf.retrieval_confidence(chunks)
    if rc < s.confidence_threshold:
        return AskResponse(
            answer=conf.low_confidence_response(req.question, chunks),
            answered=False, citations=[],
            confidence=conf.ConfidenceBreakdown(round(rc, 4), 0, 0, round(rc * 0.4, 4)),
            retrieved=chunks,
            notes="Retrieval confidence below threshold; declined to answer.",
        )

    answer = generate_answer(req.question, chunks)
    parsed = cit.parse_citations(answer, chunks)
    if req.verify_citations and parsed:
        parsed = cit.verify_citations(parsed, chunks)
    confidence = conf.score(req.question, answer, chunks, parsed)
    return AskResponse(answer=answer, answered=True, citations=parsed,
                       confidence=confidence, retrieved=chunks)


@app.get("/v1/documents", response_model=list[DocumentInfo])
def documents():
    s = get_settings()
    client = get_client()
    ensure_index(client)
    body = {
        "size": 0,
        "aggs": {
            "docs": {
                "terms": {"field": "source_document", "size": 1000},
                "aggs": {"strategies": {"terms": {"field": "chunking_strategy"}}},
            }
        },
    }
    res = client.search(index=s.opensearch_index, body=body)
    out = []
    for b in res["aggregations"]["docs"]["buckets"]:
        out.append(DocumentInfo(
            source_document=b["key"],
            chunk_count=b["doc_count"],
            strategies=[x["key"] for x in b["strategies"]["buckets"]],
        ))
    return out


@app.post("/v1/ingest")
async def ingest(file: UploadFile = File(...),
                 strategy: str = "recursive", reindex: bool = False):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in LOADERS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    # preserve original filename for metadata
    final = os.path.join(os.path.dirname(path), file.filename)
    os.rename(path, final)
    try:
        result = index_path(final, strategy=strategy, recreate=reindex)
    finally:
        os.remove(final)
    return result
