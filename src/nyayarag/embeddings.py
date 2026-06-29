from __future__ import annotations

import os

import torch
from sentence_transformers import SentenceTransformer

from nyayarag.config import Settings


def load_embedding_model(settings: Settings) -> SentenceTransformer:
    if settings.hf_token:
        os.environ["HF_TOKEN"] = settings.hf_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(settings.embedding_model, device=device)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    return model
