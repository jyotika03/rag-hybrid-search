"""Composite answer confidence scoring + graceful 'I don't know' handling."""
from __future__ import annotations

from functools import lru_cache
from app.config import get_settings
from app.models import RetrievedChunk, Citation, ConfidenceBreakdown


def retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    """Normalized mean rerank score of the final chunks (0-1)."""
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if not scores:
        return 0.0
    top = scores[: min(3, len(scores))]
    mean = sum(top) / len(top)
    # cross-encoder logits roughly in [-12, 12]; squash to 0-1
    import math
    return 1 / (1 + math.exp(-mean))


def citation_coverage(citations: list[Citation], answer: str) -> float:
    if not citations:
        return 0.0
    supported = [c for c in citations if c.supported]
    # coverage relative to total citations issued
    return len(supported) / len(citations)


@lru_cache
def _client():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key)


def answer_completeness(question: str, answer: str) -> float:
    s = get_settings()
    prompt = (
        "Rate 0-10 how completely the ANSWER addresses every part of the "
        "QUESTION. Reply with only a number.\n\n"
        f"QUESTION: {question}\n\nANSWER: {answer}"
    )
    try:
        resp = _client().chat.completions.create(
            model=s.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=4,
        )
        return min(1.0, float(resp.choices[0].message.content.strip().split()[0]) / 10)
    except Exception:
        return 0.5


def score(question: str, answer: str, chunks: list[RetrievedChunk],
          citations: list[Citation], skip_completeness: bool = False) -> ConfidenceBreakdown:
    rc = retrieval_confidence(chunks)
    cc = citation_coverage(citations, answer)
    ac = 0.5 if skip_completeness else answer_completeness(question, answer)
    composite = round(0.4 * rc + 0.35 * cc + 0.25 * ac, 4)
    return ConfidenceBreakdown(
        retrieval_confidence=round(rc, 4),
        citation_coverage=round(cc, 4),
        answer_completeness=round(ac, 4),
        composite=composite,
    )


def low_confidence_response(question: str, chunks: list[RetrievedChunk]) -> str:
    docs = sorted({c.chunk.source_document for c in chunks})
    found = "; ".join(
        f"'{c.chunk.section_heading or c.chunk.source_document}'" for c in chunks[:3]
    ) or "no closely matching passages"
    return (
        "I don't have enough grounded context to answer this confidently. "
        f"The closest material I found was: {found}. "
        f"You may want to manually check these documents: {', '.join(docs) or 'n/a'}."
    )
