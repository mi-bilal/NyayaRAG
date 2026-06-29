from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import orjson
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nyayarag.chunking import token_chunks
from nyayarag.config import get_settings
from nyayarag.embeddings import load_embedding_model
from nyayarag.preprocessing import extract_statute_chunks
from nyayarag.retrieval.bm25 import BM25Store
from nyayarag.retrieval.qdrant_store import QdrantStatuteStore
from nyayarag.schema import StatuteChunk


def log(message: str) -> None:
    print(message, flush=True)


def show_gpu_status() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            log(f"[gpu] {result.stdout.strip()}")
        else:
            log("[gpu] nvidia-smi returned no GPU info")
    except FileNotFoundError:
        log("[gpu] nvidia-smi not found; continuing without GPU status")


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
    log(f"[2/9] Downloading corpus archive: {repo_id}/{filename}")
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    log(f"[3/9] Extracting corpus archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    preferred = ["updated_SCI_56k_single.json", "SCI_judgements_56k_summarized.json"]
    for name in preferred:
        matches = list(extract_dir.rglob(name))
        if matches:
            log(f"[3/9] Found corpus JSON: {matches[0]}")
            return matches[0]

    jsons = sorted(extract_dir.rglob("*.json"), key=lambda p: p.stat().st_size, reverse=True)
    if not jsons:
        raise RuntimeError(f"No JSON files found after extracting {filename}")
    log(f"[3/9] Using largest JSON fallback: {jsons[0]}")
    return jsons[0]


def split_long_chunks(
    chunks: list[StatuteChunk], max_tokens: int, overlap_tokens: int
) -> list[StatuteChunk]:
    settings = get_settings()
    log(f"[6/9] Loading tokenizer for token splitting: {settings.embedding_model}")
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    split: list[StatuteChunk] = []
    oversized = 0
    for chunk in tqdm(chunks, desc="Token-splitting statute chunks"):
        if len(encode(chunk.text)) > max_tokens:
            oversized += 1
        pieces = token_chunks(
            chunk.text,
            encode=encode,
            decode=decode,
            chunk_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_chunk_tokens=16,
        ) or [chunk.text]
        for piece_index, piece in enumerate(pieces):
            suffix = f"__part_{piece_index:03d}" if len(pieces) > 1 else ""
            payload = chunk.model_dump()
            payload["statute_id"] = f"{chunk.statute_id}{suffix}"
            payload["text"] = piece
            if suffix:
                payload["title"] = f"{chunk.title} (part {piece_index + 1}/{len(pieces)})"
            split.append(StatuteChunk.model_validate(payload))
    log(
        f"[6/9] Token split complete: {len(chunks):,} raw chunks -> "
        f"{len(split):,} embedding chunks ({oversized:,} oversized raw chunks split)"
    )
    return split


def build_qdrant_streaming(
    chunks: list[StatuteChunk], batch_size: int, max_seq_length: int, log_every: int
) -> None:
    settings = get_settings()
    log(f"[7/9] Loading embedding model on GPU/CPU: {settings.embedding_model}")
    model = load_embedding_model(settings, max_seq_length=max_seq_length)
    log(f"[7/9] Embedding model loaded with max_seq_length={model.max_seq_length}")
    show_gpu_status()

    log(f"[8/9] Creating Qdrant collection at {settings.qdrant_path}")
    qdrant = QdrantStatuteStore(settings.qdrant_path, settings.qdrant_collection)
    qdrant.recreate(settings.embedding_dim)

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    log(
        f"[8/9] Embedding/upserting {len(chunks):,} chunks "
        f"in {total_batches:,} batches of {batch_size}"
    )
    for batch_index, start in enumerate(
        tqdm(range(0, len(chunks), batch_size), desc="Embedding/upserting statutes"), start=1
    ):
        batch_chunks = chunks[start : start + batch_size]
        batch_texts = [chunk.text for chunk in batch_chunks]
        try:
            encoded = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=len(batch_texts),
            )
        except torch.OutOfMemoryError:
            log(
                f"[oom] Batch {batch_index}/{total_batches} OOM at size {len(batch_chunks)}. "
                "Retrying item-by-item."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch_chunks) == 1:
                raise
            for offset, chunk in enumerate(batch_chunks):
                encoded = model.encode(
                    [chunk.text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=1,
                )
                qdrant.upsert([chunk], encoded.tolist(), point_id_offset=start + offset)
                del encoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue

        qdrant.upsert(batch_chunks, encoded.tolist(), point_id_offset=start)
        del encoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if batch_index == 1 or batch_index % log_every == 0 or batch_index == total_batches:
            log(f"[8/9] Completed batch {batch_index:,}/{total_batches:,}")
            show_gpu_status()


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
    parser.add_argument("--embed-max-tokens", default=512, type=int)
    parser.add_argument("--embed-overlap-tokens", default=64, type=int)
    parser.add_argument("--log-every", default=10, type=int)
    parser.add_argument("--limit", default=None, type=int)
    args = parser.parse_args()

    log("[1/9] Starting NyayaRAG Colab artifact build")
    log(f"[1/9] Python: {sys.version.split()[0]}")
    show_gpu_status()

    settings = get_settings()
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token
    else:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it to Colab secrets or write it into .env before building."
        )

    raw_dir = args.work_dir / "raw"
    processed_dir = args.work_dir / "processed"
    corpus_json = download_corpus(args.repo_id, args.zip_name, raw_dir)
    log(f"[4/9] Reading corpus JSON: {corpus_json}")
    records = json.loads(corpus_json.read_text(encoding="utf-8"))
    log(f"[4/9] Loaded {len(records):,} records")
    if args.limit:
        records = records[: args.limit]
        log(f"[4/9] Applied limit: {len(records):,} records")

    chunks_path = processed_dir / "statute_chunks.jsonl"
    log("[5/9] Extracting statute-like chunks from records")
    chunks = extract_statute_chunks(records)
    log(f"[5/9] Extracted {len(chunks):,} raw statute chunks")
    chunks = split_long_chunks(chunks, args.embed_max_tokens, args.embed_overlap_tokens)
    log(f"[6/9] Writing processed chunks: {chunks_path}")
    write_chunks(chunks, chunks_path)
    chunks = load_chunks(chunks_path)

    build_qdrant_streaming(chunks, args.batch_size, args.embed_max_tokens, args.log_every)

    log(f"[9/9] Building BM25 artifact: {settings.bm25_path}")
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
    log(f"[9/9] Wrote manifest: {manifest_dir / 'index_manifest.json'}")

    args.drive_output.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(args.drive_output / "nyayarag_artifacts"), "zip", settings.artifact_dir)
    log(f"[done] Artifact zip: {args.drive_output / 'nyayarag_artifacts.zip'}")
    log(
        "Download/unzip this into your local project root so artifacts/qdrant "
        "and artifacts/bm25 exist."
    )


if __name__ == "__main__":
    main()
