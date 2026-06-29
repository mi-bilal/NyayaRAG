from __future__ import annotations

from nyayarag.preprocessing import extract_statute_chunks_with_stats


def test_extract_statute_chunks_splits_on_dollar_and_dedupes_provisions() -> None:
    records = [
        {
            "document_id": "case_1",
            "sections": (
                "Section 302 in The Indian Penal Code, 1860302. Punishment for murder. "
                "Whoever commits murder shall be punished. $ "
                "Section 302 in The Indian Penal Code, 1860302. Punishment for murder. "
                "Whoever commits murder shall be punished with death or imprisonment. $ "
                "Section (1) in Bad Body Text should be rejected"
            ),
        }
    ]

    chunks, stats = extract_statute_chunks_with_stats(records)

    assert len(chunks) == 1
    assert chunks[0].act_name == "indian penal"
    assert chunks[0].provision_type == "Section"
    assert chunks[0].provision_number == "302"
    assert stats.raw_entries == 3
    assert stats.provision_duplicates == 1
    assert stats.rejected["unparseable_or_invalid"] == 1
