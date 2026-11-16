"""M5 end-to-end review with real journals and scripted, offline model responses."""

import asyncio
import json
from pathlib import Path

import pytest
from test_compression_research import DOCUMENT, QUESTION, SpanCompressor
from test_dag_research import FakeRetriever, TrackingStore

from deep_research.dag_journal import DagJournal
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.errors import ResearchError
from deep_research.gateway import ModelGateway
from deep_research.memory_store import MemoryError, MemoryStore
from deep_research.research import ResearchRunError
from deep_research.research_models import ResearchReport
from deep_research.retrieval import ResearchRetrievalError, SourceDocument
from deep_research.review_journal import ReviewJournal, ReviewJournalError
from deep_research.review_models import ReviewOptions, report_fingerprint
from deep_research.schemas import MAX_INPUT_BYTES, ModelResponse, TokenUsage


class ScriptedProvider:
    """Return only fixture data; the real gateway parses every simulated response."""

    def __init__(self, scenario="passed", *, stall_stage=None):
        self.scenario = scenario
        self.stall_stage = stall_stage
        self.calls = []
        self.red_calls = 0
        self.blue_calls = 0
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()

    async def complete(self, request):
        context = json.loads(request.messages[-1].content)
        if "claim_inventory" in context:
            stage = "blue" if "red_review" in context else "red"
        else:
            stage = "report" if "evidence" in context else "plan"
        self.calls.append((stage, request))
        if stage == self.stall_stage:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if stage == "plan":
            data = {"queries": ["revised" if "previous_queries" in context else "initial"]}
        elif stage == "report":
            item = context["evidence"][0]
            data = {
                "title": "Initial report",
                "sections": [
                    {
                        "heading": "Evidence",
                        "claims": [
                            {
                                "text": "Initial: " + item["quote"],
                                "citations": [item["evidence_id"]],
                            }
                        ],
                    }
                ],
                "limitations": ["Offline source fixture only."],
            }
        elif stage == "red":
            self.red_calls += 1
            passed = self.scenario in ("passed", "format") or (
                self.scenario == "repair" and self.red_calls > 1
            )
            evidence_id = context["evidence"][0]["evidence_id"]
            if self.scenario == "invalid_reference":
                evidence_id = "E999"
            data = {
                "report_hash": context["report_hash"],
                "scores": {
                    "evidence_support": 90 if passed else 40,
                    "logical_consistency": 90 if passed else 50,
                    "citation_quality": 90 if passed else 60,
                },
                "findings": []
                if passed
                else [
                    {
                        "id": "F1",
                        "category": "evidence",
                        "severity": "high",
                        "claim_ids": ["C1"],
                        "evidence_ids": [evidence_id],
                        "description": "Tighten the first claim to the supplied original evidence.",
                    }
                ],
            }
        else:
            self.blue_calls += 1
            item = context["evidence"][0]
            claim = {
                "text": (
                    f"Revision {self.blue_calls}: "
                    if self.scenario in ("round_limit", "full_budget")
                    else ""
                )
                + item["quote"],
                "citations": ["E1" if self.scenario == "archived_blue" else item["evidence_id"]],
            }
            data = {
                "report_hash": context["report_hash"],
                "operations": [
                    {
                        "op": "MODIFY",
                        "target_id": "C1",
                        "claim": claim,
                        "rationale": "Use the supplied original quote.",
                    }
                ],
            }
        content = json.dumps(data, ensure_ascii=False)
        if self.scenario in ("format", "full_budget") and stage == "red" and self.red_calls == 1:
            content = "{malformed fixture response"
        return ModelResponse(
            provider="fixture",
            model="offline",
            request_id=f"fixture-{len(self.calls)}",
            content=content,
            latency_ms=0.0,
            usage=TokenUsage(input_tokens=7, output_tokens=3, total_tokens=10),
        )


@pytest.fixture
def storage(tmp_path):
    path = tmp_path / "review.sqlite"
    with TrackingStore(path) as store, DagJournal(path) as dag, ReviewJournal(path) as review:
        with MemoryStore(path) as memory:
            yield store, dag, review, memory


async def execute(
    tmp_path, storage, *, provider=None, retriever=None, enable_review=True, **kwargs
):
    store, dag, review, _ = storage
    settings = (
        {"review_options": ReviewOptions(), "review_journal": review} if enable_review else {}
    )
    return await run_dag_pipeline(
        QUESTION,
        gateway=ModelGateway(provider or ScriptedProvider()),
        retriever=retriever or FakeRetriever({"initial": [DOCUMENT]}),
        store=store,
        journal=dag,
        output_dir=tmp_path / "reports",
        source_mode="local_corpus",
        **(settings | kwargs),
    )


async def test_passed_review_persists_snapshot_events_usage_and_explicit_limitations(
    tmp_path, storage
):
    provider = ScriptedProvider()
    result = await execute(tmp_path, storage, provider=provider)
    store, dag, review, _ = storage
    bundle = store.get_report(result["run_id"])
    assert result["status"] == "completed" and result["degraded"] is False
    assert result["review"]["status"] == "passed"
    assert result["review"]["calls"] == result["review_model_calls"] == 1
    assert result["model_call_count"] == 3
    assert bundle["schema_version"] == 5
    assert bundle["features"] == {"review": True, "compression": False}
    events = review.snapshot(result["run_id"])
    assert bundle["review_events"] == events
    assert [item["event"]["kind"] for item in events] == ["initial", "red", "stop"]
    assert events[0]["event"]["report_hash"] == events[-1]["event"]["report_hash"]
    assert [stage for stage, _ in provider.calls] == ["plan", "report", "red"]
    ledger = dag.snapshot(result["run_id"])["model_calls"]
    assert [call["stage"] for call in ledger] == ["plan", "report", "review.red.0"]
    assert all(call["status"] == "completed" for call in ledger)
    assert all(call["details"]["usage"]["total_tokens"] == 10 for call in ledger)
    assert any("不是独立事实核查" in line for line in bundle["system_limitations"])
    for _, request in provider.calls:
        assert request.response_format == "json_object"
        assert len(request.model_dump_json().encode("utf-8")) <= MAX_INPUT_BYTES
    saved = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(saved) == bundle


async def test_blue_patch_must_receive_a_second_red_review_before_becoming_final(tmp_path, storage):
    provider = ScriptedProvider("repair")
    result = await execute(tmp_path, storage, provider=provider)
    bundle = storage[0].get_report(result["run_id"])
    assert [stage for stage, _ in provider.calls] == ["plan", "report", "red", "blue", "red"]
    assert result["review"]["status"] == "passed" and result["review"]["best_round"] == 1
    assert result["review"]["calls"] == 3
    events = bundle["review_events"]
    assert [item["event"]["kind"] for item in events] == ["initial", "red", "blue", "red", "stop"]
    assert events[0]["event"]["report_hash"] != events[-1]["event"]["report_hash"]
    assert events[2]["event"]["candidate_hash"] == events[3]["event"]["report_hash"]
    assert bundle["report"] == events[-1]["event"]["report"]
    assert bundle["report"]["sections"][0]["claims"][0]["text"] == bundle["evidence"][0]["quote"]


async def test_one_format_repair_retains_failed_call_usage_and_separate_ledger_entry(
    tmp_path, storage
):
    provider = ScriptedProvider("format")
    result = await execute(tmp_path, storage, provider=provider)
    assert result["review"]["status"] == "passed"
    assert result["review"]["format_repairs"] == 1
    assert result["review_model_calls"] == 2 and result["model_call_count"] == 4
    ledger = storage[1].snapshot(result["run_id"])["model_calls"]
    assert [call["stage"] for call in ledger[-2:]] == ["review.red.0", "review.red.0.format"]
    assert (
        ledger[-2]["status"] == "failed" and ledger[-2]["details"]["error_code"] == "invalid_json"
    )
    assert ledger[-2]["details"]["usage"]["total_tokens"] == 10
    assert ledger[-1]["status"] == "completed"
    assert len(storage[0].get_report(result["run_id"])["model_calls"]) == 4


async def test_unresolved_review_is_explicitly_degraded_and_not_published_to_memory(
    tmp_path, storage
):
    result = await execute(
        tmp_path,
        storage,
        provider=ScriptedProvider("low"),
        review_options=ReviewOptions(max_rounds=0, max_format_repairs=0),
        memory=storage[3],
        memory_scope="study",
    )
    assert result["status"] == "completed" and result["degraded"] is True
    assert result["review"]["status"] == "unresolved"
    assert result["review"]["stop_reason"] == "max_rounds"
    assert "review_unresolved" in result["reasons"]
    assert {"stage": "review", "code": "review_unresolved"} in result["warnings"]
    assert result["memory_write"] == {"status": "skipped", "reason": "degraded_run"}
    assert storage[3].stats(scope="study", source_mode="local_corpus")["items"] == 0


async def test_two_blue_rounds_are_bounded_and_extend_only_the_opted_in_model_ledger(
    tmp_path, storage
):
    provider = ScriptedProvider("round_limit")
    result = await execute(tmp_path, storage, provider=provider)
    assert provider.red_calls == 3 and provider.blue_calls == 2
    assert result["review_model_calls"] == 5 and result["model_call_count"] == 7
    assert result["model_call_limit"] == 3 + ReviewOptions().max_calls
    assert result["review"]["stop_reason"] == "max_rounds"
    assert (
        result["review"]["best_round"] == 0
    )  # Equal-quality later versions do not displace the first.
    ledger = storage[1].snapshot(result["run_id"])["model_calls"]
    assert [item["call_id"] for item in ledger] == list(range(1, 8))


async def test_review_disabled_preserves_existing_m2_bundle_and_budget(tmp_path, storage):
    provider = ScriptedProvider()
    result = await execute(tmp_path, storage, provider=provider, enable_review=False)
    assert [stage for stage, _ in provider.calls] == ["plan", "report"]
    assert "review" not in result and "review_model_calls" not in result
    bundle = storage[0].get_report(result["run_id"])
    assert bundle["schema_version"] == 2 and "review_events" not in bundle
    assert storage[2].snapshot(result["run_id"]) == []


@pytest.mark.parametrize("invalid", ["options_only", "journal_only", "wrong_journal"])
async def test_review_pair_is_validated_before_any_run_or_model_call(tmp_path, storage, invalid):
    provider = ScriptedProvider()
    options = {
        "options_only": {"review_options": ReviewOptions()},
        "journal_only": {"review_journal": storage[2]},
        "wrong_journal": {"review_options": ReviewOptions(), "review_journal": object()},
    }[invalid]
    with pytest.raises(ResearchError) as caught:
        await execute(tmp_path, storage, provider=provider, enable_review=False, **options)
    assert caught.value.code == "review_configuration"
    assert provider.calls == []
    assert storage[0].last_run_id is None


async def test_review_and_memory_use_only_the_compressed_sent_subset_not_archive(tmp_path, storage):
    provider = ScriptedProvider()
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        compressor=SpanCompressor(),
        memory=storage[3],
        memory_scope="study",
    )
    bundle = storage[0].get_report(result["run_id"])
    assert [item["evidence_id"] for item in bundle["evidence"]] == ["E1", "E2", "E3"]
    assert (
        result["compression"]["selected_evidence_ids"] == result["review"]["evidence_ids"] == ["E3"]
    )
    for stage, request in provider.calls:
        if stage in ("report", "red", "blue"):
            context = json.loads(request.messages[-1].content)
            assert [item["evidence_id"] for item in context["evidence"]] == ["E3"]
    assert result["memory_write"]["status"] == "saved"
    assert result["memory_write"]["added_items"] == 1
    assert storage[3].stats(scope="study", source_mode="local_corpus")["items"] == 1


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("invalid_reference", "review_validation"),
        ("archived_blue", "review_patch"),
    ],
)
async def test_invalid_red_or_blue_references_fail_without_export(
    tmp_path, storage, scenario, expected
):
    provider = ScriptedProvider(scenario)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider=provider, compressor=SpanCompressor())
    assert caught.value.code == expected
    assert storage[0].get_run(caught.value.run_id)["status"] == "failed"
    assert storage[0].get_report(caught.value.run_id) is None
    assert storage[2].snapshot(caught.value.run_id)[0]["event"]["kind"] == "initial"
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("kind", ["initial", "red", "stop"])
async def test_review_event_failure_stops_further_calls_and_blocks_success(
    tmp_path, storage, monkeypatch, kind
):
    review = storage[2]
    original_append = review.append

    def fail_event(run_id, event):
        if event["kind"] == kind:
            raise ReviewJournalError("review_audit_io", "Audit unavailable.")
        return original_append(run_id, event)

    monkeypatch.setattr(review, "append", fail_event)
    provider = ScriptedProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider=provider)
    assert caught.value.code == "review_audit_io" and caught.value.audit_saved is False
    assert [stage for stage, _ in provider.calls] == (
        ["plan", "report"] if kind == "initial" else ["plan", "report", "red"]
    )
    assert storage[0].get_run(caught.value.run_id)["status"] == "failed"
    assert storage[0].get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


async def test_deadline_drains_red_then_exports_baseline_digest_without_blue_or_more_calls(
    tmp_path, storage
):
    provider = ScriptedProvider(stall_stage="red")
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        compressor=SpanCompressor(),
        options=DagOptions(total_timeout_seconds=0.2),
    )
    assert provider.entered.is_set() and provider.drained.is_set()
    assert [stage for stage, _ in provider.calls] == ["plan", "report", "red"]
    assert result["report_kind"] == "evidence_digest" and result["degraded"] is True
    assert result["review"]["status"] == "interrupted"
    assert result["review"]["stop_reason"] == "global_timeout"
    bundle = storage[0].get_report(result["run_id"])
    assert all(
        claim["text"].startswith("原文摘录：")
        for claim in bundle["report"]["sections"][0]["claims"]
    )
    assert bundle["review_events"][0]["event"]["kind"] == "initial"
    assert bundle["scheduler_audit"]["model_calls"][-1]["status"] == "cancelled"


async def test_user_cancel_during_red_keeps_audit_but_never_exports_or_publishes(tmp_path, storage):
    provider = ScriptedProvider(stall_stage="red")
    task = asyncio.create_task(
        execute(
            tmp_path,
            storage,
            provider=provider,
            memory=storage[3],
            memory_scope="study",
        )
    )
    try:
        await asyncio.wait_for(provider.entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.drained.is_set()
        run_id = storage[0].last_run_id
        assert storage[0].get_run(run_id)["status"] == "cancelled"
        assert storage[0].get_report(run_id) is None
        assert storage[2].snapshot(run_id)[0]["event"]["kind"] == "initial"
        assert storage[1].snapshot(run_id)["model_calls"][-1]["status"] == "cancelled"
        assert storage[3].stats(scope="study", source_mode="local_corpus")["items"] == 0
        assert not (tmp_path / "reports").exists()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_review_preserves_old_memory_identity_and_never_republishes_it_as_fresh(
    tmp_path, storage
):
    old = SourceDocument(
        "https://example.org/historical-review",
        "Previous observation",
        "Evidence source tracking retains historical source observations.",
    )
    seed = await execute(
        tmp_path,
        storage,
        enable_review=False,
        retriever=FakeRetriever({"initial": [old]}),
    )
    store, _, _, memory = storage
    memory.ingest_run(seed["run_id"], scope="study")
    provider = ScriptedProvider()
    result = await execute(
        tmp_path,
        storage,
        provider=provider,
        compressor=SpanCompressor(),
        memory=memory,
        memory_scope="study",
    )
    bundle = store.get_report(result["run_id"])
    context = json.loads(provider.calls[-1][1].messages[-1].content)
    assert [item["evidence_id"] for item in context["evidence"]] == ["E3", "E4"]
    assert context["memory_context"] == bundle["memory"]
    recalled = context["memory_context"]["recalled"][0]
    assert recalled["origin_run_id"] == seed["run_id"]
    assert recalled["origin_evidence_id"] == "E1" and recalled["evidence_id"] == "E4"
    assert recalled["retrieved_at"] == store.get_sources(seed["run_id"])[0]["retrieved_at"]
    assert result["memory_write"]["added_items"] == 1
    assert result["memory_write"]["skipped_recalled"] == 1
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 2
    assert memory.stats(scope="another", source_mode="local_corpus")["items"] == 0
    assert memory.stats(scope="study", source_mode="wikipedia")["items"] == 0


async def test_exhausted_format_repair_budget_keeps_known_usage_without_export(tmp_path, storage):
    provider = ScriptedProvider("format")
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            provider=provider,
            review_options=ReviewOptions(max_format_repairs=0),
        )
    assert caught.value.code == "invalid_json"
    assert [stage for stage, _ in provider.calls] == ["plan", "report", "red"]
    ledger = storage[1].snapshot(caught.value.run_id)["model_calls"]
    assert ledger[-1]["status"] == "failed"
    assert ledger[-1]["details"]["usage"]["total_tokens"] == 10
    assert storage[0].get_run(caught.value.run_id)["status"] == "failed"
    assert storage[0].get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


async def test_replan_plus_two_review_rounds_and_one_format_repair_stop_at_nine_total_calls(
    tmp_path, storage
):
    provider = ScriptedProvider("full_budget")
    retriever = FakeRetriever(
        {
            "initial": ResearchRetrievalError("retrieval_network", "Fixture search failure."),
            "revised": [DOCUMENT],
        }
    )
    result = await execute(tmp_path, storage, provider=provider, retriever=retriever)
    assert result["replan_count"] == 1
    assert result["review_model_calls"] == 6
    assert result["review"]["format_repairs"] == 1
    assert result["review"]["rounds"] == 2 and result["review"]["stop_reason"] == "max_rounds"
    assert result["model_call_count"] == result["model_call_limit"] == len(provider.calls) == 9
    ledger = storage[1].snapshot(result["run_id"])["model_calls"]
    assert [item["call_id"] for item in ledger] == list(range(1, 10))
    assert [item["stage"] for item in ledger] == [
        "plan",
        "replan",
        "report",
        "review.red.0",
        "review.red.0.format",
        "review.blue.1",
        "review.red.1",
        "review.blue.2",
        "review.red.2",
    ]
    assert sum(item["details"]["usage"]["total_tokens"] for item in ledger) == 90


@pytest.mark.parametrize(
    "damage",
    [
        "missing_review",
        "missing_compression",
        "bad_hash",
        "low_score",
        "unselected_citation",
        "broadened_review_context",
        "missing_features",
    ],
)
async def test_schema5_memory_import_rejects_incomplete_or_inconsistent_review_audit(
    tmp_path, storage, damage
):
    result = await execute(tmp_path, storage, compressor=SpanCompressor())
    store, _, _, memory = storage
    bundle = store.get_report(result["run_id"])
    assert bundle["review"]["status"] == "passed"
    assert bundle["compression"]["selected_evidence_ids"] == ["E3"]
    if damage == "missing_review":
        bundle.pop("review")
    elif damage == "missing_compression":
        bundle.pop("compression")
    elif damage == "bad_hash":
        bundle["review"]["best_hash"] = "0" * 64
    elif damage == "low_score":
        bundle["review"]["final_scores"]["evidence_support"] = 79
    elif damage == "unselected_citation":
        bundle["report"]["sections"][0]["claims"][0]["citations"] = ["E1"]
        bundle["review"]["best_hash"] = report_fingerprint(
            ResearchReport.model_validate(bundle["report"])
        )
    elif damage == "broadened_review_context":
        bundle["review"]["evidence_ids"] = ["E1", "E2", "E3"]
    else:
        bundle.pop("features")
    store._db().execute(
        "UPDATE research_runs SET report_json = ? WHERE id = ?",
        (json.dumps(bundle, ensure_ascii=False), result["run_id"]),
    )
    with pytest.raises(MemoryError) as caught:
        memory.ingest_run(result["run_id"], scope="study")
    assert caught.value.code == "memory_validation"
    assert memory.stats(scope="study", source_mode="local_corpus")["items"] == 0
