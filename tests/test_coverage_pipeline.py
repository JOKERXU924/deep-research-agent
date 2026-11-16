"""Production-path regression tests for the new evidence selector."""

import asyncio
import json

import pytest
from test_dag_research import FakeRetriever
from test_retrieval_quality_pipeline import ADDITION, ORIGINAL, GapProvider, execute
from test_review_research import storage as review_storage

from deep_research.compression_runtime import json_bytes
from deep_research.research import ResearchRunError
from deep_research.retrieval import SourceDocument
from deep_research.retrieval_quality import RetrievalQualityOptions

storage = review_storage
POLICY = RetrievalQualityOptions(evidence_selection="coverage")


async def test_new_selector_records_report_context_and_keeps_baseline_archive(tmp_path, storage):
    provider = GapProvider(scenario="passed")
    result = await execute(tmp_path, storage, provider=provider, quality_options=POLICY)
    bundle = storage[0].get_report(result["run_id"])
    audit = bundle["retrieval_quality"]["selection"]
    assert audit["version"] == "coverage-bm25-rrf-v1"
    context = json.loads(provider.calls[1][1].messages[-1].content)["evidence"]
    assert json_bytes(context) <= 12000
    assert bundle["features"]["compression"] is True
    assert [r["evidence_id"] for r in audit["selected"]] == [e["evidence_id"] for e in context]
    assert bundle["compression"]["selected_evidence_ids"] == [e["evidence_id"] for e in context]
    assert set(e["evidence_id"] for e in context) <= set(
        e["evidence_id"] for e in bundle["evidence"]
    )


async def test_gap_uses_saved_fulltext_without_network_and_keeps_context_prefix(
    tmp_path, storage, monkeypatch
):
    # Simulate a limited initial context, with the required fact saved in full text.
    import deep_research.dag_research as module

    real = module.select_coverage
    calls = 0

    async def initially_small(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            kwargs["max_items"] = 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(module, "select_coverage", initially_small)
    original = SourceDocument(
        ORIGINAL.url,
        ORIGINAL.title,
        "光合作用与碳固定有什么关系：光合作用包括碳固定。\n景天酸代谢植物在夜间开放气孔以减少水分损失。",
    )
    provider = GapProvider(["景天酸代谢"])
    retriever = FakeRetriever({"initial": [original]})
    result = await execute(
        tmp_path, storage, provider=provider, retriever=retriever, quality_options=POLICY
    )
    assert result["gap_retrieval"]["search_calls"] == 0
    assert result["gap_retrieval"]["fetch_calls"] == 0
    assert result["gap_retrieval"]["local_added_evidence_ids"]
    old = json.loads(provider.calls[2][1].messages[-1].content)["evidence"]
    new = json.loads(provider.calls[3][1].messages[-1].content)["evidence"]
    assert new[: len(old)] == old
    assert "夜间" in new[-1]["quote"]


async def test_two_gap_queries_each_receive_fetch_slot_and_use_gap_relevance(tmp_path, storage):
    second = SourceDocument(
        "https://example.org/cam", "景天酸代谢", "景天酸代谢植物夜间吸收二氧化碳。"
    )
    distractor = SourceDocument("https://example.org/other", "光反应", "第二份光反应资料。")
    retriever = FakeRetriever(
        {"initial": [ORIGINAL], "光反应": [ADDITION, distractor], "景天酸代谢": [second]}
    )
    result = await execute(
        tmp_path,
        storage,
        provider=GapProvider(["光反应", "景天酸代谢"]),
        retriever=retriever,
        quality_options=POLICY,
    )
    assert result["gap_retrieval"]["search_calls"] == 2
    assert result["gap_retrieval"]["fetch_calls"] == 2
    assert len(result["gap_retrieval"]["added_source_ids"]) == 2
    bundle = storage[0].get_report(result["run_id"])
    assert second.url in [s["url"] for s in bundle["sources"]]


async def test_new_selection_save_failure_stops_before_report_model(tmp_path, storage, monkeypatch):
    import deep_research.dag_research as module
    from deep_research.research_store import ResearchStoreError

    def fail(*args):
        raise ResearchStoreError("research_io", "fixture")

    monkeypatch.setattr(module, "save_compressed_spans", fail)
    provider = GapProvider()
    with pytest.raises(ResearchRunError):
        await execute(tmp_path, storage, provider=provider, quality_options=POLICY)
    assert [s for s, _ in provider.calls] == ["plan"]


async def test_early_global_timer_is_timeout_not_external_cancel(tmp_path, storage, monkeypatch):
    from deep_research.dag_research import DagOptions

    original = asyncio.timeout_at
    # Simulate an early Windows timer deterministically; wall-clock deadline is still in future.
    monkeypatch.setattr(asyncio, "timeout_at", lambda deadline: original(deadline - 0.08))
    provider = GapProvider()
    retriever = FakeRetriever(
        {"initial": [ORIGINAL], "光反应": [ADDITION]}, stalled_urls={ADDITION.url}
    )
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=retriever,
        options=DagOptions(max_replans=0, total_timeout_seconds=0.4),
    )
    assert result["gap_retrieval"]["status"] == "timed_out"
    assert [s for s, _ in provider.calls] == ["plan", "report", "red"]
