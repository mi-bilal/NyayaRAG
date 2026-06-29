from __future__ import annotations

from nyayarag.generation.parser import parse_legal_analysis


def test_parse_legal_analysis_validates_shape() -> None:
    parsed = parse_legal_analysis(
        '{"outcome":"dismissed","confidence":0.7,"issues":["x"],'
        '"applicable_statutes":[],"application_to_facts":[],"final_reasoning":"y","caveats":[]}'
    )
    assert parsed["outcome"] == "dismissed"
    assert parsed["confidence"] == 0.7
