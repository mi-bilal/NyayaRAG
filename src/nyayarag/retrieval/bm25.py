from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from nyayarag.schema import StatuteChunk


def tokenize_for_bm25(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [token for token in tokens if len(token) >= 2 or token.isdigit()]


class BM25Store:
    def __init__(self, bm25: BM25Okapi, ids: list[str], chunks: dict[str, StatuteChunk]):
        self.bm25 = bm25
        self.ids = ids
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[StatuteChunk]) -> BM25Store:
        docs = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        ids = [chunk.statute_id for chunk in chunks]
        return cls(BM25Okapi(docs), ids, {chunk.statute_id: chunk for chunk in chunks})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"bm25": self.bm25, "ids": self.ids, "chunks": self.chunks}, fh)

    @classmethod
    def load(cls, path: Path) -> BM25Store:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return cls(payload["bm25"], payload["ids"], payload["chunks"])

    def search(self, query: str, top_k: int) -> list[tuple[StatuteChunk, float, int]]:
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        out: list[tuple[StatuteChunk, float, int]] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score <= 0:
                continue
            chunk = self.chunks[self.ids[idx]]
            out.append((chunk, float(score), rank))
        return out
