from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

from nyayarag.schema import StatuteChunk

WHITESPACE_RE = re.compile(r"\s+")
PROVISION_RE = re.compile(
    r"(?=(?:^|\n|\$)\s*(Section|Article|Rule|Order|Regulation)\s+([A-Za-z0-9().-]+)\b)",
    flags=re.IGNORECASE,
)


def clean_text(text: object) -> str:
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text).replace("\x00", " ")).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def extract_statute_chunks(records: Iterable[dict]) -> list[StatuteChunk]:
    chunks: list[StatuteChunk] = []
    seen: set[str] = set()
    for rec in records:
        source_id = str(rec.get("document_id") or rec.get("case_id") or "")
        sections = str(rec.get("sections") or "")
        if not sections.strip():
            continue
        parts = [p.strip(" $\n\t") for p in PROVISION_RE.split(sections) if p.strip(" $\n\t")]
        candidates = _rejoin_provision_splits(parts) if len(parts) > 1 else [sections]
        for text in candidates:
            text = clean_text(text)
            if len(text) < 40:
                continue
            digest = stable_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            provision_type, provision_number = infer_provision(text)
            title = infer_title(text, provision_type, provision_number)
            chunks.append(
                StatuteChunk(
                    statute_id=f"statute_{len(chunks):08d}",
                    text=text,
                    title=title,
                    provision_type=provision_type,
                    provision_number=provision_number,
                    source_document_id=source_id,
                    text_hash=digest,
                )
            )
    return chunks


def _rejoin_provision_splits(parts: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(parts):
        if i + 2 < len(parts) and parts[i].lower() in {
            "section",
            "article",
            "rule",
            "order",
            "regulation",
        }:
            out.append(f"{parts[i]} {parts[i + 1]} {parts[i + 2]}")
            i += 3
        else:
            out.append(parts[i])
            i += 1
    return out


def infer_provision(text: str) -> tuple[str, str]:
    match = re.search(r"\b(Section|Article|Rule|Order|Regulation)\s+([A-Za-z0-9().-]+)", text, re.I)
    if not match:
        return "", ""
    return match.group(1).title(), match.group(2)


def infer_title(text: str, provision_type: str, provision_number: str) -> str:
    first = text[:180].strip()
    if provision_type and provision_number:
        return f"{provision_type} {provision_number}: {first}"
    return first
