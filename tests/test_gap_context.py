from deep_research.compression_runtime import json_bytes
from deep_research.gap_context import expand_gap_headings


def test_heading_expansion_retrieves_following_original_paragraph_with_exact_offsets():
    text = "CAM 植物\n夜间开放气孔吸收二氧化碳，白天进入卡尔文循环。"
    hit = dict(evidence_id="E8", source_id="S1", start=0, end=6, quote=text[:6])
    result = expand_gap_headings(
        [hit], [dict(source_id="S1", text=text)], existing=[], max_bytes=500
    )
    assert result[0]["quote"] == text
    assert result[0]["evidence_id"] == "E8"
    assert hit["end"] == 6


def test_expansion_never_exceeds_byte_budget_or_touches_complete_sentence():
    text = "Heading\nA supporting paragraph with additional evidence."
    hit = dict(evidence_id="E1", source_id="S1", start=0, end=7, quote="Heading")
    assert expand_gap_headings(
        [hit], [dict(source_id="S1", text=text)], existing=[], max_bytes=json_bytes([hit])
    ) == [hit]
    sentence = {**hit, "end": len(text), "quote": text}
    assert expand_gap_headings(
        [sentence], [dict(source_id="S1", text=text)], existing=[], max_bytes=12000
    ) == [sentence]
