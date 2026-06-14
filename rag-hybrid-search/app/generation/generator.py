"""Answer generation against the retrieved, reranked context."""
from __future__ import annotations

from functools import lru_cache
from app.config import get_settings
from app.models import RetrievedChunk
from app.generation.prompts import SYSTEM_PROMPT, build_user_prompt


@lru_cache
def _client():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    s = get_settings()
    resp = _client().chat.completions.create(
        model=s.generation_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()
