from __future__ import annotations

from pathlib import Path

import orjson

from nyayarag.config import get_settings
from nyayarag.retrieval.bm25 import BM25Store
from nyayarag.retrieval.qdrant_store import QdrantStatuteStore
from nyayarag.schema import StatuteChunk


def load_chunks(path: Path) -> list[StatuteChunk]:
    return [
        StatuteChunk.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
    ]


def main() -> None:
    settings = get_settings()
    chunks = load_chunks(Path("data/samples/statute_chunks.jsonl"))
    vectors = []
    for index, _chunk in enumerate(chunks):
        vector = [0.0] * settings.embedding_dim
        vector[index % settings.embedding_dim] = 1.0
        vectors.append(vector)

    qdrant = QdrantStatuteStore(settings.qdrant_path, settings.qdrant_collection)
    qdrant.recreate(settings.embedding_dim)
    qdrant.upsert(chunks, vectors)
    BM25Store.build(chunks).save(settings.bm25_path)
    print("Built sample Qdrant and BM25 artifacts.")


if __name__ == "__main__":
    main()
