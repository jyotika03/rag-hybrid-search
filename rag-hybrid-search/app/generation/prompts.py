"""Grounded generation prompt construction."""
from __future__ import annotations

from app.models import RetrievedChunk

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY \
the provided numbered context blocks.

Rules:
1. Answer strictly from the context. Do not use outside knowledge.
2. Cite every factual claim with bracketed references matching the context \
block numbers, e.g. [1], [2]. A claim may cite multiple blocks.
3. If the context does not contain enough information to answer, say so \
explicitly and state what is missing. Do not fabricate.
4. Be concise and directly address every part of the question."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        c = rc.chunk
        head = f" | section: {c.section_heading}" if c.section_heading else ""
        lines.append(f"[{i}] (source: {c.source_document}{head})\n{c.text}")
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        f"Context blocks:\n\n{build_context_block(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with bracketed citations."
    )
