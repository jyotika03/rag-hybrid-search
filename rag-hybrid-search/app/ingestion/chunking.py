"""Three switchable chunking strategies.

- fixed:     fixed-size windows with overlap (baseline)
- recursive: structure-aware split on section headers then size
- semantic:  split on topic boundaries using embedding similarity

Each emitted chunk records which strategy produced it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.loaders import LoadedDocument, Section


@dataclass
class RawChunk:
    text: str
    source_document: str
    chunk_index: int
    section_heading: str | None
    page_number: int | None
    chunking_strategy: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def fixed_size(doc: LoadedDocument, size: int = 1000, overlap: int = 150) -> list[RawChunk]:
    text = doc.plaintext
    chunks, idx, start = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(RawChunk(piece, doc.source_document, idx, None, None, "fixed"))
            idx += 1
        if end == len(text):
            break
        start = end - overlap
    return chunks


def recursive(doc: LoadedDocument, size: int = 1000, overlap: int = 150) -> list[RawChunk]:
    chunks, idx = [], 0
    for sec in doc.sections:
        if len(sec.text) <= size:
            if sec.text.strip():
                chunks.append(
                    RawChunk(sec.text, doc.source_document, idx, sec.section_heading,
                             sec.page_number, "recursive")
                )
                idx += 1
            continue
        # section too big: window within section, preserving heading
        start = 0
        while start < len(sec.text):
            end = min(start + size, len(sec.text))
            piece = sec.text[start:end].strip()
            if piece:
                chunks.append(
                    RawChunk(piece, doc.source_document, idx, sec.section_heading,
                             sec.page_number, "recursive")
                )
                idx += 1
            if end == len(sec.text):
                break
            start = end - overlap
    return chunks


def semantic(doc: LoadedDocument, embed_fn, threshold: float = 0.55,
             max_chars: int = 1400) -> list[RawChunk]:
    """Greedy semantic chunking. Adjacent sentences are merged while their
    embeddings stay similar; a drop below threshold starts a new chunk."""
    import numpy as np

    chunks, idx = [], 0
    for sec in doc.sections:
        sentences = _split_sentences(sec.text)
        if not sentences:
            continue
        embeddings = embed_fn(sentences)
        cur = [sentences[0]]
        cur_vec = np.array(embeddings[0], dtype=float)
        for i in range(1, len(sentences)):
            v = np.array(embeddings[i], dtype=float)
            sim = float(
                np.dot(cur_vec, v)
                / ((np.linalg.norm(cur_vec) * np.linalg.norm(v)) + 1e-9)
            )
            joined = " ".join(cur)
            if sim < threshold or len(joined) > max_chars:
                chunks.append(
                    RawChunk(joined, doc.source_document, idx, sec.section_heading,
                             sec.page_number, "semantic")
                )
                idx += 1
                cur = [sentences[i]]
                cur_vec = v
            else:
                cur.append(sentences[i])
                cur_vec = (cur_vec * len(cur) + v) / (len(cur) + 1)
        if cur:
            chunks.append(
                RawChunk(" ".join(cur), doc.source_document, idx, sec.section_heading,
                         sec.page_number, "semantic")
            )
            idx += 1
    return chunks


def chunk_document(doc: LoadedDocument, strategy: str, embed_fn=None, **kw) -> list[RawChunk]:
    if strategy == "fixed":
        return fixed_size(doc, **kw)
    if strategy == "recursive":
        return recursive(doc, **kw)
    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking requires embed_fn")
        return semantic(doc, embed_fn, **kw)
    raise ValueError(f"Unknown chunking strategy: {strategy}")
