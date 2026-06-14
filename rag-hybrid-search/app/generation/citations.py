"""Citation parsing + verification.

Parses [n] markers from the answer, maps each to its retrieved chunk, then asks
an LLM-as-judge whether the cited chunk actually supports the surrounding claim.
Unsupported citations are flagged.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.config import get_settings
from app.models import RetrievedChunk, Citation


@lru_cache
def _client():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key)


def _split_into_claims(answer: str) -> list[str]:
    # one "claim" per sentence that carries at least one citation marker
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if re.search(r"\[\d+\]", s)]


def parse_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for claim in _split_into_claims(answer):
        for marker in {int(m) for m in re.findall(r"\[(\d+)\]", claim)}:
            if 1 <= marker <= len(chunks):
                rc = chunks[marker - 1].chunk
                citations.append(Citation(
                    marker=marker,
                    chunk_id=rc.id,
                    source_document=rc.source_document,
                    claim=re.sub(r"\s*\[\d+\]", "", claim).strip(),
                ))
    return citations


def verify_citations(citations: list[Citation],
                     chunks: list[RetrievedChunk]) -> list[Citation]:
    s = get_settings()
    by_marker = {i + 1: rc.chunk.text for i, rc in enumerate(chunks)}
    for cit in citations:
        context = by_marker.get(cit.marker, "")
        prompt = (
            "Does the SOURCE text support the CLAIM? Answer strictly as "
            "'YES: <reason>' or 'NO: <reason>'.\n\n"
            f"CLAIM: {cit.claim}\n\nSOURCE: {context[:1500]}"
        )
        resp = _client().chat.completions.create(
            model=s.generation_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=120,
        )
        out = resp.choices[0].message.content.strip()
        cit.supported = out.upper().startswith("YES")
        cit.verifier_reasoning = out
    return citations
