import hashlib

import pytest

from deep_research.evidence_coverage import coverage


def source(text, key="S1"):
    return dict(source_id=key, text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


def span(text, left, right, key="S1"):
    return dict(source_id=key, start=left, end=right, quote=text[left:right])


def test_adjacent_fragments_fix_false_negative_without_changing_single_span_metric():
    text = "属性已存在。后续读写优先。"
    result = coverage(
        [span(text, 0, 6), span(text, 6, len(text))], [span(text, 0, len(text))], [source(text)]
    )
    assert result["single_hits"] == 0 and result["union_hits"] == 1
    assert result["labels"][0]["covered_chars"] == len(text)


def test_missing_negation_and_one_character_gap_are_not_ignored():
    text = "It is not safe."
    result = coverage(
        [span(text, 0, 6), span(text, 9, len(text))], [span(text, 0, len(text))], [source(text)]
    )
    assert result["union_hits"] == 0
    assert result["labels"][0]["missing_intervals"] == [[6, 9]]


def test_duplicate_overlap_does_not_double_count_or_create_coverage():
    text = "abcdefghij"
    result = coverage(
        [span(text, 0, 6), span(text, 0, 6), span(text, 2, 8)], [span(text, 0, 10)], [source(text)]
    )
    assert result["labels"][0]["covered_chars"] == 8 and result["union_hits"] == 0


def test_different_sources_cannot_be_stitched():
    text = "abcdefghij"
    result = coverage(
        [span(text, 0, 5), span(text, 5, 10, "S2")],
        [span(text, 0, 10)],
        [source(text), source(text, "S2")],
    )
    assert result["union_hits"] == 0


@pytest.mark.parametrize("damage", ["hash", "quote", "bounds", "duplicate_source"])
def test_invalid_source_or_offsets_reject_scoring(damage):
    text = "abcdefghij"
    sources = [source(text)]
    evidence = [span(text, 0, 10)]
    if damage == "hash":
        sources[0]["sha256"] = "wrong"
    elif damage == "quote":
        evidence[0]["quote"] = "wrong"
    elif damage == "bounds":
        evidence[0]["start"] = -1
    else:
        sources.append(dict(sources[0]))
    with pytest.raises(ValueError):
        coverage(evidence, [span(text, 0, 10)], sources)
