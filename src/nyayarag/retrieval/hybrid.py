from __future__ import annotations

from pathlib import Path

from nyayarag.config import Settings
from nyayarag.retrieval.bm25 import BM25Store
from nyayarag.retrieval.qdrant_store import QdrantStatuteStore
from nyayarag.schema import RetrievedStatute, StatuteChunk


class RetrievalNotReadyError(RuntimeError):
    pass


class HybridRetriever:
    def __init__(self, settings: Settings, bm25: BM25Store, qdrant: QdrantStatuteStore):
        from sentence_transformers import SentenceTransformer

        self.settings = settings
        self.bm25 = bm25
        self.qdrant = qdrant
        self.encoder = SentenceTransformer(settings.embedding_model)

    @classmethod
    def from_settings(cls, settings: Settings) -> HybridRetriever:
        if not settings.bm25_path.exists():
            raise RetrievalNotReadyError(f"BM25 artifact not found: {settings.bm25_path}")
        if not Path(settings.qdrant_path).exists():
            raise RetrievalNotReadyError(
                f"Qdrant artifact directory not found: {settings.qdrant_path}"
            )
        bm25 = BM25Store.load(settings.bm25_path)
        qdrant = QdrantStatuteStore(settings.qdrant_path, settings.qdrant_collection)
        return cls(settings, bm25, qdrant)

    def retrieve(
        self,
        case_text: str,
        known_sections: str = "",
        top_k: int | None = None,
        candidate_pool: int | None = None,
    ) -> list[RetrievedStatute]:
        top_k = top_k or self.settings.top_k
        candidate_pool = candidate_pool or self.settings.candidate_pool
        query = build_statute_query(case_text, known_sections)
        query_vector = self.encoder.encode(
            [query], prompt_name="query", normalize_embeddings=True
        )[0]
        dense = self.qdrant.search(query_vector.tolist(), candidate_pool)
        sparse = self.bm25.search(query, candidate_pool)
        return rrf_fuse(dense, sparse, top_k, self.settings.rrf_k)


def build_statute_query(case_text: str, known_sections: str = "") -> str:
    parts = [
        "Instruct: Retrieve Indian legal statutory provisions relevant to deciding this "
        "Supreme Court case.",
    ]
    if known_sections.strip():
        parts.append(f"Known statutes or provisions: {known_sections.strip()}")
    parts.append(f"Case text: {case_text.strip()[:12000]}")
    return "\n\n".join(parts)


def rrf_fuse(
    dense: list[tuple[StatuteChunk, float, int]],
    sparse: list[tuple[StatuteChunk, float, int]],
    top_k: int,
    rrf_k: int,
) -> list[RetrievedStatute]:
    scores: dict[str, float] = {}
    best: dict[str, RetrievedStatute] = {}

    for chunk, score, rank in dense:
        scores[chunk.statute_id] = scores.get(chunk.statute_id, 0.0) + 1.0 / (rrf_k + rank)
        item = best.get(chunk.statute_id) or RetrievedStatute(**chunk.model_dump(), rank=0)
        item.dense_rank = rank
        item.dense_score = score
        best[chunk.statute_id] = item

    for chunk, score, rank in sparse:
        scores[chunk.statute_id] = scores.get(chunk.statute_id, 0.0) + 1.0 / (rrf_k + rank)
        item = best.get(chunk.statute_id) or RetrievedStatute(**chunk.model_dump(), rank=0)
        item.bm25_rank = rank
        item.bm25_score = score
        best[chunk.statute_id] = item

    ranked_ids = sorted(scores, key=lambda statute_id: scores[statute_id], reverse=True)[:top_k]
    out: list[RetrievedStatute] = []
    for rank, statute_id in enumerate(ranked_ids, start=1):
        item = best[statute_id]
        item.rank = rank
        item.rrf_score = scores[statute_id]
        out.append(item)
    return out
