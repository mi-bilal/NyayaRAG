from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import orjson
import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nyayarag.chunking import token_chunks
from nyayarag.config import get_settings
from nyayarag.embeddings import load_embedding_model
from nyayarag.preprocessing import extract_statute_chunks_with_stats
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


# ---------------------------------------------------------------------------
# Checkpoint / cache helpers
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "checkpoint.json"


def _checkpoint_path(cache_dir: Path) -> Path:
    return cache_dir / CHECKPOINT_FILE


def load_checkpoint(cache_dir: Path) -> dict:
    cp_path = _checkpoint_path(cache_dir)
    if cp_path.exists():
        try:
            return json.loads(cp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_checkpoint(cache_dir: Path, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp_path = _checkpoint_path(cache_dir)
    cp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def chunks_content_hash(chunks: list[StatuteChunk]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.statute_id.encode())
        h.update(chunk.text_hash.encode())
    return h.hexdigest()


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


def encode_vectors_to_memmap(
    chunks: list[StatuteChunk],
    batch_size: int,
    max_seq_length: int,
    vector_cache: Path,
    log_every: int,
    cache_dir: Path,
) -> np.memmap:
    settings = get_settings()

    cp = load_checkpoint(cache_dir)
    ch_hash = chunks_content_hash(chunks)
    start_row = 0

    if cp.get("chunks_hash") == ch_hash and cp.get("completed_up_to", 0) > 0:
        start_row = cp["completed_up_to"]
        if start_row >= len(chunks):
            log(f"[8A/9] All {len(chunks):,} vectors already encoded — skipping")
            return np.memmap(vector_cache, dtype="float32", mode="r", shape=(len(chunks), settings.embedding_dim))
        log(f"[8A/9] Resuming embedding from row {start_row:,}/{len(chunks):,}")

    log(f"[7/9] Loading embedding model on GPU/CPU: {settings.embedding_model}")
    model = load_embedding_model(settings, max_seq_length=max_seq_length)
    log(f"[7/9] Embedding model loaded with max_seq_length={model.max_seq_length}")
    show_gpu_status()

    vector_cache.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if start_row > 0 and vector_cache.exists() else "w+"
    vectors = np.memmap(
        vector_cache,
        dtype="float32",
        mode=mode,
        shape=(len(chunks), settings.embedding_dim),
    )

    total_batches = (len(chunks) - start_row + batch_size - 1) // batch_size
    log(
        f"[8A/9] Encoding {len(chunks) - start_row:,} remaining chunks into {vector_cache} "
        f"in {total_batches:,} batches of {batch_size}"
    )
    batch_counter = 0
    for batch_index, start in enumerate(
        tqdm(range(start_row, len(chunks), batch_size), desc="Encoding statute vectors"), start=1
    ):
        batch_counter += 1
        batch_chunks = chunks[start : start + batch_size]
        batch_texts = [chunk.text for chunk in batch_chunks]
        end = start + len(batch_chunks)
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
                vectors[start + offset : start + offset + 1] = encoded.astype("float32")
                del encoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue

        vectors[start:end] = encoded.astype("float32")
        del encoded
        gc.collect()
        if batch_counter == 1 or batch_counter % log_every == 0 or end >= len(chunks):
            vectors.flush()
            save_checkpoint(cache_dir, {
                **load_checkpoint(cache_dir),
                "chunks_hash": ch_hash,
                "total_rows": len(chunks),
                "completed_up_to": end,
            })
            log(f"[8A/9] Encoded batch {batch_counter:,}/{total_batches:,} — checkpoint saved at row {end:,}")
            show_gpu_status()
    vectors.flush()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors


def bulk_upsert_qdrant(
    chunks: list[StatuteChunk],
    vectors: np.memmap,
    upsert_batch_size: int,
    log_every: int,
    cache_dir: Path,
) -> None:
    settings = get_settings()
    cp = load_checkpoint(cache_dir)
    ch_hash = chunks_content_hash(chunks)
    qdrant_offset = 0

    qdrant = QdrantStatuteStore(settings.qdrant_path, settings.qdrant_collection)

    already_upserted = cp.get("qdrant_upserted", 0)
    if (
        cp.get("chunks_hash") == ch_hash
        and already_upserted > 0
        and qdrant.client.collection_exists(settings.qdrant_collection)
    ):
        qdrant_offset = already_upserted
        log(f"[8B/9] Resuming Qdrant upsert from offset {qdrant_offset:,}/{len(chunks):,}")
    else:
        log(f"[8B/9] Creating Qdrant collection at {settings.qdrant_path}")
        qdrant.recreate(settings.embedding_dim)

    total_batches = (len(chunks) - qdrant_offset + upsert_batch_size - 1) // upsert_batch_size
    log(
        f"[8B/9] Bulk upserting {len(chunks) - qdrant_offset:,} vectors "
        f"in {total_batches:,} batches of {upsert_batch_size}"
    )
    batch_counter = 0
    for batch_index, start in enumerate(
        tqdm(range(qdrant_offset, len(chunks), upsert_batch_size), desc="Bulk upserting Qdrant"),
        start=1,
    ):
        batch_counter += 1
        end = min(start + upsert_batch_size, len(chunks))
        qdrant.upsert(
            chunks[start:end],
            vectors[start:end].tolist(),
            batch_size=upsert_batch_size,
            point_id_offset=start,
        )
        if batch_counter == 1 or batch_counter % log_every == 0 or end >= len(chunks):
            save_checkpoint(cache_dir, {
                **cp,
                "chunks_hash": ch_hash,
                "total_rows": len(chunks),
                "qdrant_upserted": end,
            })
            cp = load_checkpoint(cache_dir)
            log(f"[8B/9] Upserted batch {batch_counter:,}/{total_batches:,} — checkpoint saved at offset {end:,}")


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
    parser.add_argument("--upsert-batch-size", default=1024, type=int)
    parser.add_argument("--embed-max-tokens", default=512, type=int)
    parser.add_argument("--embed-overlap-tokens", default=64, type=int)
    parser.add_argument(
        "--vector-cache", default=Path("artifacts/vector_cache/statute_vectors.npy"), type=Path
    )
    parser.add_argument("--log-every", default=10, type=int)
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Directory for checkpoint/cache files. Defaults to {work_dir}/cache. "
        "Point this at Google Drive to survive Colab runtime restarts.",
    )
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = args.work_dir / "cache"

    log("[1/9] Starting NyayaRAG Colab artifact build")
    log(f"[1/9] Python: {sys.version.split()[0]}")
    log(f"[1/9] Cache directory: {args.cache_dir}")
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

    # --- Step 2-3: Download corpus (cached) ---
    cp = load_checkpoint(args.cache_dir)
    corpus_meta_key = f"corpus:{args.repo_id}:{args.zip_name}"
    corpus_json: Path | None = None

    if cp.get("corpus_meta") == corpus_meta_key and cp.get("corpus_json"):
        candidate = Path(cp["corpus_json"])
        if candidate.exists():
            corpus_json = candidate
            log(f"[2/9] Corpus cache HIT — reusing {corpus_json}")

    if corpus_json is None:
        corpus_json = download_corpus(args.repo_id, args.zip_name, raw_dir)
        save_checkpoint(args.cache_dir, {
            **cp,
            "corpus_meta": corpus_meta_key,
            "corpus_json": str(corpus_json),
        })

    log(f"[4/9] Reading corpus JSON: {corpus_json}")
    records = json.loads(corpus_json.read_text(encoding="utf-8"))
    log(f"[4/9] Loaded {len(records):,} records")
    if args.limit:
        records = records[: args.limit]
        log(f"[4/9] Applied limit: {len(records):,} records")

    # --- Step 5-6: Extract + split chunks (cached) ---
    corpus_hash = file_sha256(corpus_json)
    chunks_path = processed_dir / "statute_chunks.jsonl"
    cp = load_checkpoint(args.cache_dir)

    if (
        cp.get("chunks_corpus_hash") == corpus_hash
        and cp.get("limit") == args.limit
        and cp.get("cached_chunks_path")
    ):
        cached = Path(cp["cached_chunks_path"])
        if cached.exists():
            chunks = load_chunks(cached)
            log(f"[5/9] Chunks cache HIT — loaded {len(chunks):,} chunks from {cached}")
        else:
            chunks = None
    else:
        chunks = None

    if chunks is None:
        log("[5/9] Extracting statute-like chunks from records")
        raw_chunks, extraction_stats = extract_statute_chunks_with_stats(records)
        log(f"[5/9] Extraction stats: {json.dumps(extraction_stats.as_dict(), indent=2)}")
        chunks = split_long_chunks(raw_chunks, args.embed_max_tokens, args.embed_overlap_tokens)
        log(f"[6/9] Writing processed chunks: {chunks_path}")
        write_chunks(chunks, chunks_path)
        save_checkpoint(args.cache_dir, {
            **load_checkpoint(args.cache_dir),
            "chunks_corpus_hash": corpus_hash,
            "limit": args.limit,
            "cached_chunks_path": str(chunks_path),
        })

    # --- Step 7-8A: Embed vectors (row-level checkpoint) ---
    vectors = encode_vectors_to_memmap(
        chunks,
        args.batch_size,
        args.embed_max_tokens,
        args.vector_cache,
        args.log_every,
        args.cache_dir,
    )

    # --- Step 8B: Qdrant upsert (offset-level checkpoint) ---
    bulk_upsert_qdrant(chunks, vectors, args.upsert_batch_size, args.log_every, args.cache_dir)

    # --- Step 9: BM25 (fast, always rebuild) ---
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
