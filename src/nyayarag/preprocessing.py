from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from nyayarag.schema import StatuteChunk

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
HEADER_RE = re.compile(
    r"^\s*(Section|Article|Rule|Order|Regulation)\s+([A-Za-z0-9().-]+)\s+in\s+(.+)",
    flags=re.IGNORECASE,
)
BAD_PROVISION_NUMBERS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "of",
    "or",
    "passed",
    "shall",
    "the",
    "to",
    "was",
    "were",
    "with",
}
BODY_WORDS_IN_ACT = {
    "accused",
    "appeal",
    "court",
    "imprisonment",
    "offence",
    "provided",
    "punishment",
    "shall",
    "whoever",
}


@dataclass
class StatuteExtractionStats:
    records_seen: int = 0
    records_with_sections: int = 0
    raw_entries: int = 0
    parsed_entries: int = 0
    exact_text_duplicates: int = 0
    provision_duplicates: int = 0
    selected_entries: int = 0
    rejected: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        return {
            "records_seen": self.records_seen,
            "records_with_sections": self.records_with_sections,
            "raw_entries": self.raw_entries,
            "parsed_entries": self.parsed_entries,
            "exact_text_duplicates": self.exact_text_duplicates,
            "provision_duplicates": self.provision_duplicates,
            "selected_entries": self.selected_entries,
            "rejected": dict(self.rejected),
        }


def clean_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text).replace("\x00", " ")
    value = re.sub(r"<system-reminder>.*?</system-reminder>", " ", value, flags=re.I | re.S)
    return WHITESPACE_RE.sub(" ", value).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def extract_statute_chunks(records: Iterable[dict]) -> list[StatuteChunk]:
    chunks, _stats = extract_statute_chunks_with_stats(records)
    return chunks


def extract_statute_chunks_with_stats(
    records: Iterable[dict],
) -> tuple[list[StatuteChunk], StatuteExtractionStats]:
    stats = StatuteExtractionStats()
    seen_text: set[str] = set()
    candidates_by_key: dict[str, StatuteChunk] = {}

    for rec in records:
        stats.records_seen += 1
        source_id = str(rec.get("document_id") or rec.get("case_id") or "")
        sections = str(rec.get("sections") or "")
        if not sections.strip():
            continue
        stats.records_with_sections += 1
        for raw_entry in sections.split("$"):
            stats.raw_entries += 1
            parsed = parse_statute_entry(raw_entry)
            if parsed is None:
                stats.rejected["unparseable_or_invalid"] += 1
                continue
            text, provision_type, provision_number, act_name = parsed
            canonical_text = canonicalize_statute_text(text)
            digest = stable_hash(canonical_text)
            if digest in seen_text:
                stats.exact_text_duplicates += 1
                continue
            seen_text.add(digest)
            stats.parsed_entries += 1

            title = infer_title(text, provision_type, provision_number)
            key = statute_dedupe_key(act_name, provision_type, provision_number, digest)
            chunk = StatuteChunk(
                statute_id="",
                text=text,
                title=title,
                act_name=act_name,
                provision_type=provision_type,
                provision_number=provision_number,
                source_document_id=source_id,
                text_hash=digest,
            )
            current = candidates_by_key.get(key)
            if current is None:
                candidates_by_key[key] = chunk
            else:
                stats.provision_duplicates += 1
                if statute_quality_score(chunk) > statute_quality_score(current):
                    candidates_by_key[key] = chunk

    chunks = list(candidates_by_key.values())
    chunks.sort(
        key=lambda c: (
            c.act_name,
            c.provision_type,
            natural_sort_key(c.provision_number),
            c.title,
        )
    )
    for index, chunk in enumerate(chunks):
        chunk.statute_id = f"statute_{index:08d}"
    stats.selected_entries = len(chunks)
    return chunks, stats


def parse_statute_entry(raw_entry: str) -> tuple[str, str, str, str] | None:
    entry = clean_text(raw_entry.strip(" $\n\t"))
    if len(entry) < 40:
        return None
    match = HEADER_RE.match(entry)
    if not match:
        return None

    provision_type = match.group(1).title()
    provision_number = normalize_provision_number(match.group(2))
    if not valid_provision_number(provision_number):
        return None

    rest = match.group(3).strip()
    act_name = extract_act_name(rest, provision_number)
    if not act_name:
        return None

    return entry, provision_type, provision_number, act_name


def extract_act_name(rest: str, provision_number: str) -> str:
    pattern = re.compile(rf"(.+?)(?:{re.escape(provision_number)}\s*[.)])", flags=re.I)
    match = pattern.match(rest)
    candidate = match.group(1) if match else rest[:120]
    candidate = re.sub(r"\b(18|19|20)\d{2}\b", " ", candidate)
    candidate = clean_text(candidate).strip(" ,.;:-—")
    normalized = normalize_act_name(candidate)
    if not normalized:
        return ""
    if any(word in normalized.split() for word in BODY_WORDS_IN_ACT):
        return ""
    return normalized


def normalize_act_name(raw: str) -> str:
    text = raw.lower().replace("&", " and ")
    text = re.sub(r"\b(the|act|code|rules|regulations)\b", " ", text)
    text = re.sub(r"\b(18|19|20)\d{2}\b", " ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()

    aliases = {
        "ipc": "indian penal",
        "indian penal": "indian penal",
        "constitution india": "constitution of india",
        "constitution of india": "constitution of india",
        "criminal procedure": "criminal procedure",
        "civil procedure": "civil procedure",
        "income tax": "income tax",
        "land acquisition": "land acquisition",
        "transfer property": "transfer of property",
        "prevention money laundering": "prevention of money laundering",
    }
    for needle, replacement in aliases.items():
        if needle in text:
            return replacement
    return text


def normalize_provision_number(text: str) -> str:
    return text.strip().strip(".,;:")


def valid_provision_number(num: str) -> bool:
    value = num.lower().strip()
    if not value or value in BAD_PROVISION_NUMBERS:
        return False
    if re.fullmatch(r"\(\d+\)", value):
        return False
    return bool(re.search(r"\d", value))


def canonicalize_statute_text(text: str) -> str:
    return NON_ALNUM_RE.sub(" ", clean_text(text).lower()).strip()


def statute_dedupe_key(
    act_name: str, provision_type: str, provision_number: str, fallback_digest: str
) -> str:
    if act_name and provision_type and provision_number:
        return "|".join(
            [
                normalize_key(act_name),
                normalize_key(provision_type),
                normalize_key(provision_number),
            ]
        )
    return f"text|{fallback_digest}"


def normalize_key(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.lower())


def statute_quality_score(chunk: StatuteChunk) -> tuple[int, int, int, int, int]:
    length = len(chunk.text)
    sane_length = 1 if 80 <= length <= 20_000 else 0
    has_title = 1 if re.search(r"\d\s*[.)]\s*[A-Z]", chunk.text[:300]) else 0
    act_penalty = -1 if any(word in chunk.act_name.split() for word in BODY_WORDS_IN_ACT) else 0
    preferred_length = -abs(min(length, 20_000) - 2_000)
    return sane_length, has_title, act_penalty, preferred_length, min(length, 20_000)


def natural_sort_key(text: str) -> tuple[str, int, str]:
    match = re.match(r"([A-Za-z]*)(\d+)(.*)", text or "")
    if not match:
        return text or "", -1, ""
    return match.group(1), int(match.group(2)), match.group(3)


def infer_title(text: str, provision_type: str, provision_number: str) -> str:
    first = text[:180].strip()
    if provision_type and provision_number:
        return f"{provision_type} {provision_number}: {first}"
    return first
