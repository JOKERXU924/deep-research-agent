"""Offline acceptance of topic filtering and one budgeted review retrieval pass."""

import asyncio
import json
from pathlib import Path

import pytest
from test_compression_research import SpanCompressor
from test_dag_research import FakeRetriever
from test_review_research import ScriptedProvider
from test_review_research import storage as review_storage

from deep_research.compression_runtime import json_bytes
from deep_research.dag_journal import DagJournalError
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.gateway import ModelGateway
from deep_research.model_budget import ModelBudgetLimits
from deep_research.research import ResearchRunError
from deep_research.retrieval import ResearchRetrievalError, SourceDocument
from deep_research.retrieval_quality import RetrievalQualityOptions
from deep_research.review_models import ReviewOptions

storage = review_storage

QUESTION = "光合作用与碳固定有什么关系？"
ORIGINAL = SourceDocument(
    "https://example.org/photosynthesis",
    "光合作用",
    "光合作用包含碳固定过程。\n这是已有的第一条证据。\n需要更多资料解释光反应。",
)
ADDITION = SourceDocument(
    "https://example.org/light",
    "光合作用的光反应",
    "光合作用的光反应提供 ATP 和 NADPH，为碳固定提供能量及还原力。",
)
UNRELATED = SourceDocument("https://example.org/cartoon", "动漫人物", "虚构人物使用光魔法。")


class GapProvider(ScriptedProvider):
    def __init__(self, queries=None, *, scenario="repair", plans=None):
        super().__init__(scenario)
        self.queries = ["光反应"] if queries is None else queries
        self.plans = plans

    async def complete(self, request):
        response = await super().complete(request)
        stage = self.calls[-1][0]
        context = json.loads(request.messages[-1].content)
        data = json.loads(response.content)
        if stage == "red":
            for finding in data["findings"]:
                finding["search_queries"] = self.queries
        elif stage == "blue":
            item = context["evidence"][-1]
            data["operations"][0]["claim"] = {
                "text": item["quote"],
                "citations": [item["evidence_id"]],
            }
        elif stage == "plan" and self.plans is not None:
            data["queries"] = self.plans["replan" if "previous_queries" in context else "plan"]
        return response.model_copy(update={"content": json.dumps(data, ensure_ascii=False)})


async def execute(tmp_path, storage, *, provider=None, retriever=None, **kwargs):
    store, dag, review, _ = storage
    settings = {
        "options": DagOptions(max_replans=0),
        "quality_options": RetrievalQualityOptions(),
        "review_options": ReviewOptions(),
    }
    return await run_dag_pipeline(
        QUESTION,
        gateway=ModelGateway(provider or GapProvider()),
        retriever=retriever or FakeRetriever({"initial": [ORIGINAL], "光反应": [ADDITION]}),
        store=store,
        journal=dag,
        review_journal=review,
        output_dir=tmp_path / "reports",
        source_mode="local_corpus",
        **(settings | kwargs),
    )


def stages(provider):
    return [stage for stage, _ in provider.calls]


def events(storage, run_id):
    return [item["event"] for item in storage[1].snapshot(run_id)["retrieval_events"]]


async def test_irrelevant_source_never_reaches_evidence_or_model(tmp_path, storage):
    provider = GapProvider(scenario="passed")
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=FakeRetriever({"initial": [ORIGINAL, UNRELATED]}),
    )
    bundle = storage[0].get_report(result["run_id"])
    assert [source["title"] for source in bundle["sources"]] == [ORIGINAL.title]
    assert result["retrieval_quality"]["filtered_count"] == 1
    assert result["retrieval_quality"]["accepted_count"] == 1
    assert result["gap_retrieval"]["status"] == "not_needed"
    assert any(item["status"] == "rejected" for item in events(storage, result["run_id"]))
    for _, request in provider.calls:
        assert UNRELATED.text not in request.model_dump_json()
    assert stages(provider) == ["plan", "report", "red"]


async def test_gap_appends_ids_and_reaches_blue_without_extra_planning(tmp_path, storage):
    provider = GapProvider()
    retriever = FakeRetriever({"initial": [ORIGINAL], "光反应": [ADDITION]})
    result = await execute(tmp_path, storage, provider=provider, retriever=retriever)
    bundle = storage[0].get_report(result["run_id"])
    assert result["review"]["status"] == "passed"
    assert stages(provider) == ["plan", "report", "red", "blue", "red"]
    assert [query for query, _ in retriever.search_calls] == ["initial", "光反应"]
    gap = bundle["gap_retrieval"]
    assert gap["status"] == "completed"
    assert gap["search_calls"] == gap["fetch_calls"] == 1
    assert gap["added_source_ids"] == ["S2"]
    old = json.loads(provider.calls[2][1].messages[-1].content)["evidence"]
    new = json.loads(provider.calls[3][1].messages[-1].content)["evidence"]
    assert new[: len(old)] == old
    assert gap["added_evidence_ids"] == [item["evidence_id"] for item in new[len(old) :]]
    assert bundle["review"]["best_evidence_ids"] == [item["evidence_id"] for item in new]
    assert bundle["report"]["sections"][0]["claims"][0]["citations"] == [new[-1]["evidence_id"]]
    assert [item["event"]["kind"] for item in bundle["review_events"]] == [
        "initial",
        "red",
        "evidence_gap",
        "blue",
        "red",
        "stop",
    ]
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved) == bundle


@pytest.mark.parametrize("queries", [[], ["initial"], ["INITIAL"]])
async def test_missing_or_repeated_queries_skip_retrieval(tmp_path, storage, queries):
    retriever = FakeRetriever({"initial": [ORIGINAL]})
    result = await execute(tmp_path, storage, provider=GapProvider(queries), retriever=retriever)
    assert result["gap_retrieval"]["reason"] == "no_queries"
    assert len(retriever.search_calls) == 1


@pytest.mark.parametrize("mode", ["same_url", "same_hash", "irrelevant", "empty", "error"])
async def test_unusable_supplement_does_not_invent_new_evidence(tmp_path, storage, mode):
    additions = {
        "same_url": [ORIGINAL],
        "same_hash": [SourceDocument("https://example.org/alias", ORIGINAL.title, ORIGINAL.text)],
        "irrelevant": [UNRELATED],
        "empty": [],
        "error": ResearchRetrievalError("retrieval_network", "Offline simulated failure"),
    }
    result = await execute(
        tmp_path,
        storage,
        retriever=FakeRetriever({"initial": [ORIGINAL], "光反应": additions[mode]}),
    )
    assert result["gap_retrieval"]["status"] == "no_new_evidence"
    assert result["gap_retrieval"]["added_evidence_ids"] == []
    assert len(storage[0].get_report(result["run_id"])["sources"]) == 1


async def test_gap_maximum_two_searches_two_fetches_even_with_more_hits(tmp_path, storage):
    extras = [
        SourceDocument(f"https://example.org/{index}", "光合作用", f"光合作用证据 {index}")
        for index in range(3)
    ]
    provider = GapProvider(["光反应", "卡尔文循环"], scenario="round_limit")
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=FakeRetriever(
            {
                "initial": [ORIGINAL],
                "光反应": [],
                "卡尔文循环": extras,
            }
        ),
    )
    assert result["gap_retrieval"]["search_calls"] == 2
    assert result["gap_retrieval"]["fetch_calls"] == 2
    assert len(result["gap_retrieval"]["added_source_ids"]) == 2
    assert stages(provider).count("plan") == 1
    assert stages(provider).count("blue") <= 2


async def test_six_total_searches_exhausted_before_gap(tmp_path, storage):
    provider = GapProvider(plans={"plan": ["a", "b", "c"], "replan": ["d", "e", "f"]})
    retriever = FakeRetriever({"a": [ORIGINAL], "b": [], "c": [], "d": [], "e": [], "f": []})
    # Force one ordinary initial-batch retrieval failure so M2 replans.
    retriever.searches["b"] = ResearchRetrievalError("retrieval_network", "Offline failure")
    retriever.searches["c"] = ResearchRetrievalError("retrieval_network", "Offline failure")
    result = await execute(
        tmp_path, storage, provider=provider, retriever=retriever, options=DagOptions(max_replans=1)
    )
    assert result["gap_retrieval"]["reason"] == "retrieval_budget"
    assert result["gap_retrieval"]["search_calls"] == 0
    assert len(retriever.search_calls) == 6


async def test_low_model_budget_does_not_waste_gap_network_calls(tmp_path, storage):
    retriever = FakeRetriever({"initial": [ORIGINAL]})
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path, storage, retriever=retriever, model_budget=ModelBudgetLimits(max_calls=3)
        )
    assert caught.value.code == "model_budget_exceeded"
    audit = storage[1].snapshot(storage[0].last_run_id)
    assert audit["summary"]["gap_retrieval"]["reason"] == "model_budget"
    assert len(retriever.search_calls) == 1


async def test_gap_subdeadline_cancels_fetch_but_can_finish_existing_review(tmp_path, storage):
    retriever = FakeRetriever(
        {"initial": [ORIGINAL], "光反应": [ADDITION]}, stalled_urls={ADDITION.url}
    )
    provider = GapProvider()
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=retriever,
        quality_options=RetrievalQualityOptions(gap_timeout_seconds=0.3),
    )
    assert retriever.drained.is_set()
    assert result["gap_retrieval"]["status"] == "timed_out"
    assert result["gap_retrieval"]["added_evidence_ids"] == []
    assert stages(provider) == ["plan", "report", "red", "blue", "red"]


async def test_global_deadline_never_sends_blue_after_gap_timeout(tmp_path, storage):
    retriever = FakeRetriever(
        {"initial": [ORIGINAL], "光反应": [ADDITION]}, stalled_urls={ADDITION.url}
    )
    provider = GapProvider()
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=retriever,
        options=DagOptions(max_replans=0, total_timeout_seconds=0.2),
    )
    assert retriever.drained.is_set()
    assert result["degraded"] and "global_timeout" in result["reasons"]
    assert result["gap_retrieval"]["status"] == "timed_out"
    assert stages(provider) == ["plan", "report", "red"]
    assert "研究未完成" in storage[0].get_report(result["run_id"])["report"]["title"]
    assert len(storage[1].snapshot(result["run_id"])["model_calls"]) == 3


async def test_cancel_drains_gap_fetch_and_never_calls_blue(tmp_path, storage):
    retriever = FakeRetriever(
        {"initial": [ORIGINAL], "光反应": [ADDITION]}, stalled_urls={ADDITION.url}
    )
    provider = GapProvider()
    task = asyncio.create_task(execute(tmp_path, storage, provider=provider, retriever=retriever))
    await asyncio.wait_for(retriever.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert retriever.drained.is_set()
    assert stages(provider) == ["plan", "report", "red"]
    assert (
        storage[1].snapshot(storage[0].last_run_id)["summary"]["gap_retrieval"]["status"]
        == "cancelled"
    )


@pytest.mark.parametrize("failed_status", ["started", "completed", "failed"])
async def test_gap_audit_failure_prevents_all_subsequent_io(
    tmp_path, storage, monkeypatch, failed_status
):
    provider = GapProvider(["光反应", "卡尔文循环"])
    failure = ResearchRetrievalError("retrieval_network", "Offline network failure")
    retriever = FakeRetriever(
        {
            "initial": [ORIGINAL],
            "光反应": failure if failed_status == "failed" else [ADDITION],
            "卡尔文循环": [ADDITION],
        }
    )
    original = storage[1].record_retrieval_event

    def fail(run_id, event):
        if event["kind"] == "search" and event["status"] == failed_status:
            raise DagJournalError("dag_audit_io", "Offline simulated disk failure")
        return original(run_id, event)

    monkeypatch.setattr(storage[1], "record_retrieval_event", fail)
    with pytest.raises(ResearchRunError):
        await execute(tmp_path, storage, provider=provider, retriever=retriever)
    assert len(retriever.search_calls) == (1 if failed_status == "started" else 2)
    assert retriever.fetch_calls == [ORIGINAL.url]
    assert stages(provider) == ["plan", "report", "red"]
    assert storage[0].get_report(storage[0].last_run_id) is None


async def test_compressed_context_appends_whole_quotes_without_rewriting_old_ids(tmp_path, storage):
    provider = GapProvider()
    result = await execute(tmp_path, storage, provider=provider, compressor=SpanCompressor(line=0))
    bundle = storage[0].get_report(result["run_id"])
    initial = json.loads(provider.calls[2][1].messages[-1].content)["evidence"]
    blue = json.loads(provider.calls[3][1].messages[-1].content)["evidence"]
    assert blue[: len(initial)] == initial
    assert json_bytes(blue) <= 8192
    assert bundle["compression"]["selected_evidence_ids"] == [item["evidence_id"] for item in blue]
    assert bundle["compression"]["gap_added_source_ids"] == ["S2"]
    for item in bundle["evidence"]:
        source = next(
            source for source in bundle["sources"] if source["source_id"] == item["source_id"]
        )
        assert source["text"][item["start"] : item["end"]] == item["quote"]


async def test_full_context_budget_does_not_archive_unused_gap_source(tmp_path, storage):
    provider = GapProvider()
    result = await execute(
        tmp_path, storage, provider=provider, compressor=SpanCompressor(line=0, budget=200)
    )
    assert result["gap_retrieval"]["added_evidence_ids"] == []
    assert result["gap_retrieval"]["status"] == "no_new_evidence"
    assert any(item.get("reason") == "context_budget" for item in events(storage, result["run_id"]))


async def test_successful_supplement_can_still_be_unresolved(tmp_path, storage):
    result = await execute(tmp_path, storage, provider=GapProvider(scenario="round_limit"))
    assert result["gap_retrieval"]["status"] == "completed"
    assert result["review"]["status"] == "unresolved"
    assert result["degraded"] and "review_unresolved" in result["reasons"]


async def test_shared_memory_admission_with_supplement_uses_reviewed_context(tmp_path, storage):
    result = await execute(tmp_path, storage, memory=storage[3], memory_scope="gap-test")
    assert result["review"]["status"] == "passed"
    assert result["memory_write"]["status"] == "saved"


async def test_partial_gap_evidence_survives_subdeadline_and_is_reviewed(tmp_path, storage):
    slow = SourceDocument("https://example.org/slow", "光合作用", "光合作用的第二条补充资料。")
    retriever = FakeRetriever(
        {"initial": [ORIGINAL], "光反应": [ADDITION, slow]}, stalled_urls={slow.url}
    )
    provider = GapProvider()
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        retriever=retriever,
        quality_options=RetrievalQualityOptions(gap_timeout_seconds=0.5),
    )
    assert retriever.drained.is_set()
    assert result["gap_retrieval"]["status"] == "timed_out"
    assert result["gap_retrieval"]["added_source_ids"] == ["S2"]
    blue = json.loads(provider.calls[3][1].messages[-1].content)
    assert any(item["quote"] == ADDITION.text for item in blue["evidence"])
    assert slow.text not in provider.calls[3][1].model_dump_json()


async def test_cancel_in_completed_audit_prevents_saving_new_source(tmp_path, storage, monkeypatch):
    original = storage[1].record_retrieval_event
    provider = GapProvider()

    def cancel_after_audit(run_id, event):
        original(run_id, event)
        if event["kind"] == "fetch" and event["status"] == "completed":
            asyncio.current_task().cancel()

    monkeypatch.setattr(storage[1], "record_retrieval_event", cancel_after_audit)
    task = asyncio.create_task(execute(tmp_path, storage, provider=provider))
    with pytest.raises(asyncio.CancelledError):
        await task
    audit = storage[1].snapshot(storage[0].last_run_id)
    assert audit["summary"]["gap_retrieval"]["added_source_ids"] == []
    assert stages(provider) == ["plan", "report", "red"]
    assert not any(
        item["event"]["kind"] == "filter" and item["event"]["phase"] == "gap"
        for item in audit["retrieval_events"]
    )


async def test_eight_total_fetches_exhausted_before_gap(tmp_path, storage):
    documents = [
        SourceDocument(
            f"https://example.org/test-{index}",
            "光合作用",
            f"光合作用与碳固定的资料编号为{index}。",
        )
        for index in range(8)
    ]
    retriever = FakeRetriever(
        {"a": documents[:2], "b": documents[2:4], "c": documents[4:6], "d": documents[6:]},
        fetch_failures={item.url for item in documents[:4]},
    )
    provider = GapProvider(plans={"plan": ["a", "b"], "replan": ["c", "d"]})
    result = await execute(
        tmp_path, storage, provider=provider, retriever=retriever, options=DagOptions(max_replans=1)
    )
    assert len(retriever.search_calls) == 4
    assert len(retriever.fetch_calls) == 8
    assert result["gap_retrieval"]["reason"] == "retrieval_budget"
    assert result["gap_retrieval"]["search_calls"] == 0
