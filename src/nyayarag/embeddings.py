from __future__ import annotations

import os

import torch
from sentence_transformers import SentenceTransformer

from nyayarag.config import Settings


def load_embedding_model(
    settings: Settings, max_seq_length: int | None = None
) -> SentenceTransformer:
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(settings.embedding_model, device=device)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    return model
