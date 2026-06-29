from __future__ import annotations

from collections.abc import Callable


def token_chunks(
    text: str,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int = 32,
) -> list[str]:
    token_ids = encode(text)
    if len(token_ids) <= chunk_tokens:
        return [text.strip()] if len(token_ids) >= min_chunk_tokens else []
    chunks: list[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_tokens]
        if len(window) < min_chunk_tokens:
            break
        chunks.append(decode(window).strip())
        if start + chunk_tokens >= len(token_ids):
            break
    return [chunk for chunk in chunks if chunk]


def trim_to_tokens(
    text: str,
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    max_tokens: int,
) -> str:
    token_ids = encode(text)
    if len(token_ids) <= max_tokens:
        return text
    return decode(token_ids[:max_tokens]).strip()
