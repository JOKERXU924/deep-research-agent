"""Dependency windows and miss attribution without a live model or labels in selection."""

import hashlib

import pytest

from deep_research.compression_runtime import json_bytes
from deep_research.evidence_selection import select_coverage as old_select
from deep_research.evidence_selection_v3 import baseline, contextual, navigation, prepare, window
from deep_research.retrieval_diagnostics import covers, diagnose


def source(text):
    return dict(source_id="S1", text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


async def test_shadow_baseline_matches_unchanged_v1_exactly():
    sources = [source("A task has a result.\nHowever, a task may fail.\nA socket sends data.")]
    ranked = await prepare(sources, "How can a task fail?")
    assert (
        baseline(ranked, max_bytes=600).evidence
        == (await old_select(sources, "How can a task fail?", max_bytes=600)).evidence
    )


async def test_window_restores_heading_and_exception_without_rewriting():
    text = (
        "Mode conditions\nRead access requires a shared file.\n"
        "However, version 3 supports immutable access."
    )
    ranked = await prepare([source(text)], "What are mode conditions?")
    span, reasons = window(ranked, 0)
    assert span["quote"] == text[span["start"] : span["end"]]
    assert "However" in span["quote"] and reasons
    chosen = contextual(ranked, max_bytes=1000)
    assert len(chosen.evidence) == 1  # Overlapping dependencies do not duplicate the whole quote.
    assert chosen.evidence[0]["quote"] == text


async def test_expansion_falls_back_to_anchor_under_hard_byte_limit():
    text = "Task access requires a file.\nHowever, " + "an exception exists " * 15 + "."
    ranked = await prepare([source(text)], "Task access")
    old = baseline(ranked, max_bytes=200)
    assert old.evidence
    cap = json_bytes(old.evidence)
    result = contextual(ranked, max_bytes=cap)
    assert json_bytes(result.evidence) <= cap
    assert any(e["fallback"] for e in result.audit["windows"])
    for item in result.evidence:
        assert text[item["start"] : item["end"]] == item["quote"]


async def test_archived_evidence_is_never_extended_or_modified():
    text = "Task mode\nTask details are available."
    ranked = await prepare([source(text)], "Task")
    existing = [dict(evidence_id="E90", source_id="S1", start=0, end=9, quote=text[:9])]
    original = [dict(e) for e in existing]
    result = contextual(ranked, max_bytes=1000, existing=existing, next_id=91)
    assert existing == original
    assert all(e["evidence_id"] != "E90" for e in result.evidence)


async def test_existing_factual_seed_spans_are_not_evicted_by_expansion():
    text = (
        "Task result is successful.\nHowever, cancellation can happen.\n"
        "Task failures are exceptions."
    )
    ranked = await prepare([source(text)], "Task result? Task failures?")
    seed = baseline(ranked, max_bytes=350)
    selected = contextual(ranked, max_bytes=350)
    for span in seed.evidence:
        if not navigation(span["quote"]):
            assert covers(selected.evidence, span)
    assert navigation("Python »") and navigation("Previous topic")
    assert not navigation("Previous versions did not support this feature.")


def test_diagnostics_distinguish_boundary_eligibility_and_budget():
    full = dict(source_id="S1", start=0, end=10)
    pieces = [dict(source_id="S1", start=0, end=5), dict(source_id="S1", start=5, end=10)]
    assert covers(pieces, full)
    row = diagnose(
        [full],
        candidates=pieces,
        eligible=pieces,
        selected=pieces[:1],
        wide_bytes=pieces,
        wide_items=pieces,
        orders=[[0, 1]],
    )[0]
    assert row["lost_at"] == "byte_budget" and row["fragmented"]
    assert row["first_full_candidate_rank_by_point"] == [None]
    row = diagnose(
        [full],
        candidates=[full],
        eligible=[],
        selected=[],
        wide_bytes=[],
        wide_items=[],
        orders=[[0]],
    )[0]
    assert row["lost_at"] == "eligibility"


async def test_invalid_source_is_rejected_before_any_ranking():
    value = source("Task result.")
    value["text"] = "Modified"
    with pytest.raises(Exception) as error:
        await prepare([value], "Task")
    assert error.value.code == "retrieval_selection"
