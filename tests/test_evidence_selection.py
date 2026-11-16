"""Meaningful retrieval contracts; no real models or network required."""

import asyncio
import hashlib

import numpy as np
import pytest

from deep_research.compression_runtime import json_bytes
from deep_research.errors import ResearchError
from deep_research.evidence_selection import candidates, select_coverage, targets
from deep_research.retrieval_benchmark import metrics


def source(text, key="S1"):
    return dict(source_id=key, text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


async def test_separate_targets_and_exact_offsets_at_equal_byte_budget():
    s = source(
        "Alpha mechanism consumes energy.\nUnrelated history is long.\n"
        "Beta exception occurs at night."
    )
    result = await select_coverage(
        [s], "Alpha mechanism? Beta exception?", max_items=2, max_bytes=400
    )
    assert len(result.evidence) == 2
    assert all(s["text"][e["start"] : e["end"]] == e["quote"] for e in result.evidence)
    assert json_bytes(result.evidence) <= 400
    assert "Beta" in result.evidence[1]["quote"]


async def test_byte_budget_counts_multibyte_and_never_slices_selected_quote():
    s = source("光合作用提供能量。\n碳固定发生在叶绿体。")
    full = await select_coverage([s], "光合作用；碳固定")
    limit = json_bytes(full.evidence[:1])
    result = await select_coverage([s], "光合作用；碳固定", max_bytes=limit)
    assert result.evidence == full.evidence[:1]
    empty = await select_coverage([s], "光合作用", max_bytes=2)
    assert empty.evidence == []


async def test_duplicate_text_from_different_sources_is_not_repeated():
    text = "Identical observation about cancellation."
    result = await select_coverage([source(text), source(text, "S2")], "cancellation")
    assert len(result.evidence) == 1


async def test_append_only_context_and_ids_remain_unchanged():
    s = source("Energy powers fixation.\nNight opening reduces evaporation.")
    old = (await select_coverage([s], "Energy", max_items=1)).evidence
    before = [dict(e) for e in old]
    result = await select_coverage([s], "Night", existing=old, next_id=9, max_items=2)
    assert old == before
    assert result.evidence[0]["evidence_id"] == "E9"
    assert result.evidence[0]["quote"].startswith("Night")
    assert result.audit["context_bytes"] == json_bytes(old + result.evidence)


def test_long_paragraph_tail_and_sentence_boundaries_preserve_all_nonwhite_text():
    text = "Opening sentence. " + "word " * 170 + "Tail fact。最后一句话。"
    chunks = candidates([source(text)])
    assert "".join(c["quote"] for c in chunks) == text
    assert max(len(c["quote"]) for c in chunks) <= 300
    assert chunks[-1]["quote"].endswith("最后一句话。")


async def test_bad_source_hash_rejected_before_encoding():
    s = source("important fact")
    s["sha256"] = "0" * 64
    with pytest.raises(ResearchError, match="哈希"):
        await select_coverage([s], "fact")


@pytest.mark.parametrize("budget", [True, -1, 12001])
async def test_invalid_budget_rejected(budget):
    with pytest.raises(ResearchError):
        await select_coverage([], "fact", max_bytes=budget)


async def test_existing_context_cannot_be_silently_dropped():
    old = (await select_coverage([source("Fact about light.")], "light")).evidence
    with pytest.raises(ResearchError):
        await select_coverage([], "light", existing=old, max_bytes=2)


async def test_encoder_failure_and_cancellation_do_not_fallback():
    class Broken:
        async def encode_queries(self, texts):
            raise ResearchError("embedding_runtime", "test")

    with pytest.raises(ResearchError, match="test"):
        await select_coverage([source("A fact.")], "fact", encoder=Broken())
    task = asyncio.create_task(select_coverage([source("A fact.")], "fact"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("value", [float("nan"), 0.0])
async def test_invalid_semantic_vectors_are_rejected(value):
    class Broken:
        dimensions = 2
        encoder_id = "test"

        async def encode_queries(self, texts):
            return np.full((len(texts), 2), value)

        async def encode_documents(self, texts):
            return np.ones((len(texts), 2))

    with pytest.raises(ResearchError):
        await select_coverage([source("A fact.")], "fact", encoder=Broken())


def test_output_instructions_do_not_become_coverage_targets():
    assert targets("研究光合作用；每个关键结论引用证据；约600字") == ["研究光合作用"]


def test_scoring_requires_complete_gold_span_not_just_keyword():
    s = source("Alpha needs energy.")
    case = dict(gold=[dict(source_id="S1", start=0, end=len(s["text"]))], relevant_paragraphs=[])
    partial = [dict(source_id="S1", start=0, end=5, quote="Alpha")]
    assert metrics(partial, case, {"S1": s})["recall"] == 0
    assert metrics([], case, {"S1": s})["recall"] == 0
