from __future__ import annotations

from nyayarag.config import Settings
from nyayarag.generation.parser import parse_legal_analysis
from nyayarag.groq_client import GroqGenerator
from nyayarag.prompts import SYSTEM_PROMPT, build_analysis_prompt
from nyayarag.schema import RetrievedStatute


def generate_legal_analysis(
    generator: GroqGenerator,
    case_text: str,
    known_sections: str,
    retrieved_statutes: list[RetrievedStatute],
    settings: Settings,
) -> dict:
    prompt = build_analysis_prompt(
        case_text=case_text[: settings.max_case_tokens * 6],
        known_sections=known_sections,
        statutes=retrieved_statutes,
    )
    raw = generator.complete_json(SYSTEM_PROMPT, prompt)
    return parse_legal_analysis(raw)
