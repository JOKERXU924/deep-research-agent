"""Behavioral contracts for multilingual retrieval, without network or model downloads."""

import asyncio
import hashlib

import numpy as np
import pytest

from deep_research.compression_runtime import json_bytes
from deep_research.errors import ResearchError
from deep_research.evidence_selection_v2 import (
    _bridge_candidates,
    _queries,
    cross_language_rescue,
    passages,
    rank_support,
    select_coverage,
)


def source(text, key="S1", title=""):
    return dict(
        source_id=key, title=title, text=text, sha256=hashlib.sha256(text.encode()).hexdigest()
    )


class Encoder:
    dimensions = 2
    kind = "semantic"
    encoder_id = "fixture"

    async def encode_queries(self, texts):
        return np.array([[1.0, 0.0]] * len(texts))

    async def encode_documents(self, texts):
        return np.array([[1.0, 0.0] if "energy" in t else [0.0, 1.0] for t in texts])


def test_no_lexical_signal_has_no_positional_vote_and_ties_share_rank():
    result = rank_support(np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]]))
    assert not result[0].any()
    assert result[1, 0] == result[1, 1] > result[1, 2]


def test_quoted_claim_is_one_target_and_planner_keywords_do_not_steal_slots():
    targets, contextual = _queries(
        "请研究植物；解释能量来源；判断“无需光，所以只在夜间”是否正确；每个结论引用证据",
        ["extra", "other"],
    )
    assert targets == ["解释能量来源", "判断“无需光，所以只在夜间”是否正确"]
    assert all(q.startswith("请研究植物;") for q in contextual)


def test_heading_keeps_body_and_long_parent_has_exact_offsets():
    text = "Energy process\n\nLight supplies energy.\n" + "A long observation. " * 80
    chunks = passages([source(text)])
    assert chunks[0]["quote"] == "Energy process\n\nLight supplies energy."
    assert max(len(c["quote"]) for c in chunks) <= 600
    assert all(text[c["start"] : c["end"]] == c["quote"] for c in chunks)
    assert sum(len(c["quote"]) for c in chunks) >= len(text) - 3


async def test_zero_overlap_cross_language_uses_semantics_and_preserves_raw_cosine():
    sources = [
        source("History of an unrelated topic.", "S1"),
        source("Light supplies energy.", "S2", "energy"),
    ]
    result = await select_coverage(sources, "能量来源 energy", encoder=Encoder(), max_items=1)
    assert result.evidence[0]["source_id"] == "S2"
    assert result.audit["selected"][0]["semantic_score"] == pytest.approx(1)
    assert result.audit["lexical_weights"] == [0.05]


async def test_utf8_budget_counts_metadata_and_preserves_existing_ids():
    sources = [source("能量来自光反应。\n夜间开放气孔。")]
    old = (await select_coverage(sources, "能量", max_items=1)).evidence
    before = [dict(e) for e in old]
    full = await select_coverage(sources, "夜间", existing=old, next_id=8)
    limit = json_bytes(old + full.evidence)
    result = await select_coverage(sources, "夜间", existing=old, next_id=8, max_bytes=limit)
    assert old == before
    assert result.evidence == full.evidence
    assert result.evidence[0]["evidence_id"] == "E8"
    assert result.audit["context_bytes"] <= limit
    with pytest.raises(ResearchError):
        await select_coverage(sources, "夜间", existing=old, max_bytes=2)


def test_bridge_is_bounded_and_derived_only_from_first_hop_terms():
    chunks = [
        dict(quote="AAA supplies energy."),
        dict(quote="BBB supplies electrons."),
        dict(quote="AAA and BBB support conversion."),
        dict(quote="ZZZ and YYY elsewhere."),
    ]
    sem = np.array([[0.99, 0.98, 0.96, 0.1]])
    chunks = [dict(source_id="S1", start=0, **p) for p in chunks]
    selected, terms, windows = _bridge_candidates(chunks, [[0, 1, 2, 3]], sem)
    assert 2 in selected
    assert len(selected) <= 2 and len(terms) <= 12
    assert 3 not in selected
    assert set(terms) <= {"AAA", "BBB", "ZZZ", "YYY"}
    assert windows[2]["quote"] == chunks[2]["quote"]


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
async def test_encoder_invalid_values_are_errors_not_lexical_fallback(bad):
    class Broken(Encoder):
        async def encode_queries(self, texts):
            return np.full((len(texts), 2), bad)

    with pytest.raises(ResearchError):
        await select_coverage([source("Light supplies energy.")], "能量", encoder=Broken())


async def test_cancellation_and_source_hash_errors_propagate():
    class Cancelled(Encoder):
        async def encode_queries(self, texts):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await select_coverage([source("Light supplies energy.")], "能量", encoder=Cancelled())
    item = source("Light supplies energy.")
    item["sha256"] = "0" * 64
    with pytest.raises(ResearchError):
        await select_coverage([item], "energy")


async def test_gate_rescue_requires_script_mismatch_and_high_semantic_support():
    assert (
        await cross_language_rescue("能量从哪里来", "Light supplies energy.", "Light", Encoder())
        == 1
    )
    assert (
        await cross_language_rescue("能量从哪里来", "Unrelated history.", "History", Encoder())
        is None
    )
    assert (
        await cross_language_rescue(
            "Where does energy come from", "Light supplies energy.", "Light", Encoder()
        )
        is None
    )
