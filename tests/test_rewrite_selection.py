import hashlib

import pytest

from deep_research.compression_runtime import json_bytes
from deep_research.evidence_selection import Selection
from deep_research.rewrite_selection import contains, protect


def evidence(text, left, right, source="S1", eid="E1"):
    return dict(evidence_id=eid, source_id=source, start=left, end=right, quote=text[left:right])


def selection(evidence):
    return Selection(evidence, dict(context_bytes=json_bytes(evidence)))


@pytest.mark.parametrize("limit", ["bytes", "items"])
def test_drifting_rewrite_cannot_evict_original_span(limit):
    text = "The callback runs at most once.\nUnrelated atexit callback background."
    original = evidence(text, 0, text.index("\n"))
    drift = evidence(text, text.index("\n") + 1, len(text))
    before = [dict(original)]
    result = protect(
        selection(before),
        selection([drift]),
        [dict(source_id="S1", text=text)],
        max_bytes=json_bytes(before) if limit == "bytes" else 12000,
        max_items=1 if limit == "items" else 16,
    )
    assert result.evidence == before
    assert before == [original]
    assert result.audit["rewrite_guard"]["decisions"][0]["status"] == (
        "byte_limit" if limit == "bytes" else "item_limit"
    )


def test_translation_adds_new_span_when_capacity_is_available():
    text = "原问已选证据。\nAdditional English API constraint."
    old = evidence(text, 0, text.index("\n"))
    new = evidence(text, text.index("\n") + 1, len(text))
    result = protect(
        selection([old]),
        selection([new]),
        [dict(source_id="S1", text=text)],
        max_bytes=12000,
        max_items=2,
        next_id=9,
    )
    assert [e["evidence_id"] for e in result.evidence] == ["E9", "E10"]
    assert contains(result.evidence[0], old) and contains(result.evidence[1], new)
    assert result.audit["rewrite_guard"]["additions"] == 1


def test_overlapping_translation_unions_exact_source_without_duplicate_overhead():
    text = "Only one writer. Readers do not block writers."
    old = evidence(text, 0, 16)
    new = evidence(text, 8, len(text))
    result = protect(
        selection([old]),
        selection([new]),
        [dict(source_id="S1", text=text)],
        max_bytes=json_bytes([evidence(text, 0, len(text))]),
        max_items=1,
    )
    assert len(result.evidence) == 1 and result.evidence[0]["quote"] == text
    assert contains(result.evidence[0], old) and contains(result.evidence[0], new)
    assert result.audit["rewrite_guard"]["merges"] == 1


def test_existing_context_and_ids_are_immutable_under_multibyte_byte_pressure():
    text = "旧上下文。\n原问重要事实。\n改写补充说明。"
    existing = [evidence(text, 0, 6, eid="E1")]
    old = evidence(text, 7, 14, eid="E9")
    new = evidence(text, 15, len(text))
    before = [dict(e) for e in existing]
    result = protect(
        selection([old]),
        selection([new]),
        [dict(source_id="S1", text=text)],
        max_bytes=json_bytes(existing + [old]),
        max_items=4,
        existing=existing,
        next_id=9,
    )
    assert existing == before and result.evidence == [old]


async def test_real_selector_no_rewrite_matches_original_and_audits_protection():
    from deep_research.evidence_selection_v3 import select_coverage
    from deep_research.research_loop import select_rewritten

    text = "Alpha operation is thread safe.\nBeta operation may run twice."
    sources = [dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())]
    old = await select_coverage(sources, "Alpha? Beta?", max_bytes=400)
    new = await select_rewritten(sources, "Alpha? Beta?", rewrites=[], max_bytes=400)
    assert new.evidence == old.evidence
    assert len(new.audit["rewrite_guard"]["protected_spans"]) == len(old.evidence)


async def test_quota_leaves_room_for_translation_instead_of_protecting_every_background_hit():
    from deep_research.research_loop import select_rewritten

    text = (
        "Alpha operation is documented.\nAlpha history is background.\n"
        "Alpha implementation is background.\nBeta translated exception is important."
    )
    sources = [dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())]
    result = await select_rewritten(
        sources,
        "Alpha? 乙约束？",
        rewrites=[dict(point_id="P2", query="Beta translated exception")],
        max_bytes=2000,
        max_items=2,
    )
    assert len(result.evidence) == 2
    assert any("Beta translated" in e["quote"] for e in result.evidence)
    assert any("Alpha" in e["quote"] for e in result.evidence)
    assert result.audit["rewrite_guard"]["seed_item_limit"] == 1
    assert json_bytes(result.evidence) <= 2000


async def test_full_existing_context_does_not_allocate_an_extra_seed_slot():
    from deep_research.research_loop import select_rewritten

    text = "Alpha already exists."
    sources = [dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())]
    existing = [evidence(text, 0, len(text))]
    result = await select_rewritten(
        sources,
        "Alpha?",
        rewrites=[dict(point_id="P1", query="Alpha")],
        existing=existing,
        next_id=2,
        max_bytes=json_bytes(existing),
        max_items=1,
    )
    assert result.evidence == [] and existing[0]["quote"] == text
