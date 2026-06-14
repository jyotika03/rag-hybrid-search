"""Second-pass reranking of fused candidates.

Two backends:
- cross-encoder: a small cross-encoder scores (question, chunk) pairs.
- llm:           LLM-as-judge scores relevance 0-10.
Keeps the top FINAL_TOP_K. This precision pass is the main quality lever.
"""
from __future__ import annotations

from functools import lru_cache
from app.config import get_settings


@lru_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(get_settings().cross_encoder_model)


def _rerank_cross_encoder(question: str, candidates: list[dict]) -> list[dict]:
    pairs = [(question, c["source"]["text"]) for c in candidates]
    scores = _cross_encoder().predict(pairs)
    for c, sc in zip(candidates, scores):
        c["rerank_score"] = float(sc)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


def _rerank_llm(question: str, candidates: list[dict]) -> list[dict]:
    from openai import OpenAI
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key)
    for c in candidates:
        prompt = (
            "Rate from 0 to 10 how relevant the passage is to answering the "
            f"question. Reply with only a number.\n\nQuestion: {question}\n\n"
            f"Passage: {c['source']['text'][:1500]}"
        )
        resp = client.chat.completions.create(
            model=s.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        )
        try:
            c["rerank_score"] = float(resp.choices[0].message.content.strip().split()[0])
        except (ValueError, IndexError):
            c["rerank_score"] = 0.0
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


def rerank(question: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
    s = get_settings()
    top_k = top_k or s.final_top_k
    candidates = candidates[: s.rerank_top_n]
    if not candidates:
        return []
    if s.rerank_backend == "llm":
        ranked = _rerank_llm(question, candidates)
    else:
        ranked = _rerank_cross_encoder(question, candidates)
    return ranked[:top_k]
