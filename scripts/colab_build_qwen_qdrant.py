from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import orjson
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser(
        description="Build Qdrant + BM25 artifacts with Qwen3 embeddings."
    )
    parser.add_argument("--chunks", default=Path("data/processed/statute_chunks.jsonl"), type=Path)
    parser.add_argument(
        "--drive-output", default=Path("/content/drive/MyDrive/nyayarag_artifacts"), type=Path
    )
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()

    settings = get_settings()
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    chunks = load_chunks(args.chunks)
    model = SentenceTransformer(
        settings.embedding_model,
        model_kwargs={"device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"},
    )

    texts = [chunk.text for chunk in chunks]
    vectors = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embedding statutes"):
        batch = texts[start : start + args.batch_size]
        encoded = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(encoded.tolist())

    qdrant = QdrantStatuteStore(settings.qdrant_path, settings.qdrant_collection)
    qdrant.recreate(settings.embedding_dim)
    qdrant.upsert(chunks, vectors)

    bm25 = BM25Store.build(chunks)
    bm25.save(settings.bm25_path)

    manifest_dir = settings.artifact_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "qdrant_collection": settings.qdrant_collection,
        "chunks": len(chunks),
    }
    (manifest_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    args.drive_output.mkdir(parents=True, exist_ok=True)
    zip_base = args.drive_output / "nyayarag_artifacts"
    shutil.make_archive(str(zip_base), "zip", settings.artifact_dir)
    print(f"Saved artifact zip to {zip_base}.zip")


if __name__ == "__main__":
    main()
