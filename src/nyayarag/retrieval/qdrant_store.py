from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from nyayarag.schema import StatuteChunk


class QdrantStatuteStore:
    def __init__(self, path: Path, collection: str):
        self.path = path
        self.collection = collection
        self.client = QdrantClient(path=str(path))

    def recreate(self, vector_size: int) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(
        self, chunks: list[StatuteChunk], vectors: list[list[float]], batch_size: int = 128
    ) -> None:
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            points = [
                PointStruct(id=start + offset, vector=vector, payload=chunk.model_dump())
                for offset, (chunk, vector) in enumerate(
                    zip(batch_chunks, batch_vectors, strict=True)
                )
            ]
            self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int) -> list[tuple[StatuteChunk, float, int]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        out: list[tuple[StatuteChunk, float, int]] = []
        for rank, hit in enumerate(response.points, start=1):
            out.append((StatuteChunk.model_validate(hit.payload or {}), float(hit.score), rank))
        return out
