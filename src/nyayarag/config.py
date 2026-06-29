from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    hf_token: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    qdrant_path: Path = Path("artifacts/qdrant")
    qdrant_collection: str = "nyayarag_statutes"
    bm25_path: Path = Path("artifacts/bm25/statutes_bm25.pkl")
    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    top_k: int = 5
    candidate_pool: int = 80
    rrf_k: int = 60
    max_case_tokens: int = 3500
    max_context_tokens: int = 5000
    chunk_tokens: int = 384
    chunk_overlap_tokens: int = 64
    min_chunk_tokens: int = 48
    sentence_transformer_batch_size: int = Field(default=32, ge=1)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        hf_token=os.getenv("HF_TOKEN", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        embedding_dim=_get_int("EMBEDDING_DIM", 1024),
        qdrant_path=Path(os.getenv("QDRANT_PATH", "artifacts/qdrant")),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "nyayarag_statutes"),
        bm25_path=Path(os.getenv("BM25_PATH", "artifacts/bm25/statutes_bm25.pkl")),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
        artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts")),
        top_k=_get_int("TOP_K", 5),
        candidate_pool=_get_int("CANDIDATE_POOL", 80),
        rrf_k=_get_int("RRF_K", 60),
        max_case_tokens=_get_int("MAX_CASE_TOKENS", 3500),
        max_context_tokens=_get_int("MAX_CONTEXT_TOKENS", 5000),
    )
