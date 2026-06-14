"""LLM-as-judge eval metrics: correctness, faithfulness, retrieval relevance,
citation accuracy."""
from __future__ import annotations

from functools import lru_cache
from app.config import get_settings
from app.models import AskResponse


@lru_cache
def _client():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key)


def _judge(prompt: str) -> float:
    s = get_settings()
    resp = _client().chat.completions.create(
        model=s.generation_model,
        messages=[{"role": "user", "content": prompt + "\nReply with only a 0-10 number."}],
        temperature=0, max_tokens=4,
    )
    try:
        return min(1.0, float(resp.choices[0].message.content.strip().split()[0]) / 10)
    except (ValueError, IndexError):
        return 0.0


def correctness(question: str, answer: str, expected: str) -> float:
    return _judge(f"Rate how well the ANSWER matches the GOLDEN answer.\n"
                  f"Q: {question}\nANSWER: {answer}\nGOLDEN: {expected}")


def faithfulness(answer: str, contexts: list[str]) -> float:
    joined = "\n---\n".join(contexts)[:6000]
    return _judge(f"Rate how fully every claim in the ANSWER is grounded in CONTEXT.\n"
                  f"ANSWER: {answer}\nCONTEXT: {joined}")


def retrieval_relevance(question: str, contexts: list[str]) -> float:
    joined = "\n---\n".join(contexts)[:6000]
    return _judge(f"Rate how relevant the retrieved CONTEXT is to the QUESTION.\n"
                  f"QUESTION: {question}\nCONTEXT: {joined}")


def citation_accuracy(resp: AskResponse) -> float:
    if not resp.citations:
        return 0.0
    supported = sum(1 for c in resp.citations if c.supported)
    return supported / len(resp.citations)


def evaluate_case(case: dict, resp: AskResponse) -> dict:
    contexts = [r.chunk.text for r in resp.retrieved]
    return {
        "id": case["id"],
        "type": case.get("type"),
        "correctness": round(correctness(case["question"], resp.answer, case["expected"]), 3),
        "faithfulness": round(faithfulness(resp.answer, contexts), 3),
        "retrieval_relevance": round(retrieval_relevance(case["question"], contexts), 3),
        "citation_accuracy": round(citation_accuracy(resp), 3),
        "composite_confidence": resp.confidence.composite,
        "answered": resp.answered,
    }
