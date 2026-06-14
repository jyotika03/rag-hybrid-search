"""Run the golden eval suite, optionally across all three chunking strategies,
and emit a comparison report."""
from __future__ import annotations

import json
import os
import statistics
from datetime import datetime

from app.models import AskRequest
from app.ingestion.indexer import index_path
from app.retrieval.engine import retrieve
from app.generation.generator import generate_answer
from app.generation import citations as cit
from app.generation import confidence as conf
from app.models import AskResponse, ConfidenceBreakdown
from app.eval.metrics import evaluate_case

HERE = os.path.dirname(__file__)


def _answer(question: str, hybrid: bool = True) -> AskResponse:
    chunks = retrieve(question, hybrid=hybrid)
    if not chunks:
        return AskResponse(answer="", answered=False, citations=[],
                           confidence=ConfidenceBreakdown(0, 0, 0, 0), retrieved=[])
    answer = generate_answer(question, chunks)
    parsed = cit.verify_citations(cit.parse_citations(answer, chunks), chunks)
    c = conf.score(question, answer, chunks, parsed)
    return AskResponse(answer=answer, answered=True, citations=parsed,
                       confidence=c, retrieved=chunks)


def run_suite(dataset_path: str, hybrid: bool = True) -> list[dict]:
    cases = json.load(open(dataset_path))
    results = []
    for case in cases:
        resp = _answer(case["question"], hybrid=hybrid)
        results.append(evaluate_case(case, resp))
    return results


def aggregate(results: list[dict]) -> dict:
    keys = ["correctness", "faithfulness", "retrieval_relevance", "citation_accuracy"]
    return {k: round(statistics.mean(r[k] for r in results), 3) for k in keys}


def compare_chunking(corpus_path: str, dataset_path: str) -> dict:
    report = {}
    for strategy in ["fixed", "recursive", "semantic"]:
        index_path(corpus_path, strategy=strategy, recreate=True)
        results = run_suite(dataset_path, hybrid=True)
        report[strategy] = aggregate(results)
    return report


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/sample_docs")
    p.add_argument("--dataset", default=os.path.join(HERE, "golden_dataset.json"))
    p.add_argument("--compare-chunking", action="store_true")
    args = p.parse_args()

    if args.compare_chunking:
        out = compare_chunking(args.corpus, args.dataset)
    else:
        res = run_suite(args.dataset)
        out = {"per_case": res, "aggregate": aggregate(res)}

    out["generated_at"] = datetime.utcnow().isoformat()
    print(json.dumps(out, indent=2))
