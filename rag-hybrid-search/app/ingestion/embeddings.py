"""Embedding generation via OpenAI text-embedding-3-small (batched)."""
from __future__ import annotations

from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings


@lru_cache
def _client():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    s = get_settings()
    resp = _client().embeddings.create(model=s.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
