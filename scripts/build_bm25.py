from __future__ import annotations

import argparse
from pathlib import Path

import orjson

from nyayarag.retrieval.bm25 import BM25Store
from nyayarag.schema import StatuteChunk


def load_chunks(path: Path) -> list[StatuteChunk]:
    return [
        StatuteChunk.model_validate(orjson.loads(line))
        for line in path.read_bytes().splitlines()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 artifact for statute chunks.")
    parser.add_argument("--chunks", default=Path("data/processed/statute_chunks.jsonl"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/bm25/statutes_bm25.pkl"), type=Path)
    args = parser.parse_args()
    store = BM25Store.build(load_chunks(args.chunks))
    store.save(args.output)
    print(f"Saved BM25 artifact to {args.output}")


if __name__ == "__main__":
    main()
