from __future__ import annotations

import json
import re

from nyayarag.schema import LegalAnalysis


def parse_legal_analysis(raw: str) -> dict:
    text = raw.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    return LegalAnalysis.model_validate(payload).model_dump()
