from __future__ import annotations

from nyayarag.retrieval.hybrid import rrf_fuse
from nyayarag.schema import StatuteChunk


def test_rrf_fuse_combines_dense_and_sparse_ranks() -> None:
    a = StatuteChunk(statute_id="a", text="Article 14 equality")
    b = StatuteChunk(statute_id="b", text="Article 21 liberty")
    results = rrf_fuse(dense=[(a, 0.9, 1)], sparse=[(b, 3.0, 1), (a, 2.0, 2)], top_k=2, rrf_k=60)
    assert [item.statute_id for item in results] == ["a", "b"]
    assert results[0].dense_rank == 1
    assert results[0].bm25_rank == 2
