from __future__ import annotations

from pydantic import BaseModel, Field


class StatuteChunk(BaseModel):
    statute_id: str
    text: str
    title: str = ""
    act_name: str = ""
    provision_type: str = ""
    provision_number: str = ""
    source_document_id: str = ""
    text_hash: str = ""


class RetrievedStatute(StatuteChunk):
    rank: int
    rrf_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class LegalAnalysis(BaseModel):
    outcome: str = Field(description="allowed, dismissed, partial, remand, or uncertain")
    confidence: float = Field(ge=0, le=1)
    issues: list[str]
    applicable_statutes: list[dict[str, str]]
    application_to_facts: list[str]
    final_reasoning: str
    caveats: list[str]
