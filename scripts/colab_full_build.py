from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import orjson
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from nyayarag.config import get_settings
from nyayarag.embeddings import load_embedding_model
from nyayarag.preprocessing import extract_statute_chunks
from nyayarag.retrieval.bm25 import BM25Store
from nyayarag.retrieval.qdrant_store import QdrantStatuteStore
from nyayarag.schema import StatuteChunk


def load_chunks(path: Path) -> list[StatuteChunk]:
    return [
        StatuteChunk.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
    ]


def write_chunks(chunks: list[StatuteChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for chunk in chunks:
            fh.write(orjson.dumps(chunk.model_dump()))
            fh.write(b"\n")


def download_corpus(repo_id: str, filename: str, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    preferred = ["updated_SCI_56k_single.json", "SCI_judgements_56k_summarized.json"]
    for name in preferred:
        matches = list(extract_dir.rglob(name))
        if matches:
            return matches[0]

    jsons = sorted(extract_dir.rglob("*.json"), key=lambda p: p.stat().st_size, reverse=True)
    if not jsons:
        raise RuntimeError(f"No JSON files found after extracting {filename}")
    return jsons[0]


def build_vectors(chunks: list[StatuteChunk], batch_size: int) -> list[list[float]]:
    settings = get_settings()
    model = load_embedding_model(settings)
    vectors: list[list[float]] = []
    texts = [chunk.text for chunk in chunks]
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding statutes"):
        batch = texts[start : start + batch_size]
        encoded = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(encoded.tolist())
    return vectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab one-shot build: download corpus, extract statutes, build Qdrant/BM25."
    )
    parser.add_argument("--repo-id", default="L-NLProc/NyayaRAG")
    parser.add_argument("--zip-name", default="1.Base Dataset.zip")
    parser.add_argument("--work-dir", default=Path("/content/nyayarag_build"), type=Path)
    parser.add_argument(
        "--drive-output", default=Path("/content/drive/MyDrive/nyayarag_artifacts"), type=Path
    )
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    settings = get_settings()
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    raw_dir = args.work_dir / "raw"
    processed_dir = args.work_dir / "processed"
    corpus_json = download_corpus(args.repo_id, args.zip_name, raw_dir)
    records = json.loads(corpus_json.read_text(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    chunks_path = processed_dir / "statute_chunks.jsonl"
    chunks = extract_statute_chunks(records)
    write_chunks(chunks, chunks_path)
    chunks = load_chunks(chunks_path)

    vectors = build_vectors(chunks, args.batch_size)

    qdrant = QdrantStatuteStore(settings.qdrant_path, settings.qdrant_collection)
    qdrant.recreate(settings.embedding_dim)
    qdrant.upsert(chunks, vectors)

    BM25Store.build(chunks).save(settings.bm25_path)

    manifest_dir = settings.artifact_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "corpus_json": str(corpus_json),
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "qdrant_collection": settings.qdrant_collection,
        "statute_chunks": len(chunks),
    }
    (manifest_dir / "index_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    args.drive_output.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(args.drive_output / "nyayarag_artifacts"), "zip", settings.artifact_dir)
    print(f"Artifact zip: {args.drive_output / 'nyayarag_artifacts.zip'}")
    print(
        "Download/unzip this into your local project root so artifacts/qdrant "
        "and artifacts/bm25 exist."
    )


if __name__ == "__main__":
    main()
