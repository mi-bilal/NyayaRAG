from __future__ import annotations

from nyayarag.schema import RetrievedStatute

SYSTEM_PROMPT = """You are NyayaRAG, an Indian Supreme Court legal research assistant.
You analyze case text using only the retrieved statutory context provided by the app.
Do not invent statutes, citations, or facts.
If the retrieved context is insufficient, say so in caveats.
Return valid JSON only."""


def build_analysis_prompt(
    case_text: str, known_sections: str, statutes: list[RetrievedStatute]
) -> str:
    statute_block = "\n\n".join(
        f"[{item.rank}] id={item.statute_id}\nTitle: {item.title}\nText: {item.text}"
        for item in statutes
    )
    return f"""Analyze the likely legal outcome using the CaseText + Statutes pipeline.

Case text:
{case_text}

Known sections or articles supplied by user:
{known_sections or "None"}

Retrieved statutes:
{statute_block}

Return a JSON object with this exact shape:
{{
  "outcome": "allowed | dismissed | partial | remand | uncertain",
  "confidence": 0.0,
  "issues": ["concise issue strings"],
  "applicable_statutes": [{{"citation": "retrieved statute id/title", "why_relevant": "reason"}}],
  "application_to_facts": ["apply statute to fact"],
  "final_reasoning": "concise conclusion",
  "caveats": ["limitations or missing context"]
}}
"""
