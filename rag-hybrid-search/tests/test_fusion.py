from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_merges_and_ranks():
    dense = [{"id": "a", "score": 0.9, "source": {"text": "x"}},
             {"id": "b", "score": 0.8, "source": {"text": "y"}}]
    sparse = [{"id": "b", "score": 5.0, "source": {"text": "y"}},
              {"id": "c", "score": 4.0, "source": {"text": "z"}}]
    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60,
                                   dense_weight=0.7, sparse_weight=0.3)
    ids = [f["id"] for f in fused]
    assert set(ids) == {"a", "b", "c"}
    # b appears in both lists -> should rank at or near the top
    assert ids[0] == "b"
    assert fused[0]["fused_score"] >= fused[1]["fused_score"]


def test_bm25_fallback():
    from app.retrieval.sparse import BM25Index
    chunks = [{"id": "1", "text": "error code E42 timeout"},
              {"id": "2", "text": "general overview of the system"}]
    idx = BM25Index(chunks)
    res = idx.search("E42 timeout", k=1)
    assert res[0]["id"] == "1"
