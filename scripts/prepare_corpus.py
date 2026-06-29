from __future__ import annotations

import argparse
import json
from pathlib import Path

import orjson

from nyayarag.preprocessing import extract_statute_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract statute chunks from NyayaRAG corpus JSON."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/processed/statute_chunks.jsonl"), type=Path)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    chunks = extract_statute_chunks(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        for chunk in chunks:
            fh.write(orjson.dumps(chunk.model_dump()))
            fh.write(b"\n")
    print(f"Wrote {len(chunks):,} statute chunks to {args.output}")


if __name__ == "__main__":
    main()
