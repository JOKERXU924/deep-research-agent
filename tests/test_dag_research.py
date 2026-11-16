"""M2 integration contracts with real SQLite and deterministic in-memory workers."""

import asyncio
import json
from pathlib import Path

import pytest

from deep_research import dag_research
from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.errors import ModelError
from deep_research.gateway import StructuredResult
from deep_research.research import ResearchRunError
from deep_research.research_models import (
    CitedClaim,
    ReportSection,
    ResearchPlan,
    ResearchReport,
)
from deep_research.research_store import ResearchStore, ResearchStoreError
from deep_research.retrieval import ResearchRetrievalError, SearchHit, SourceDocument
from deep_research.schemas import ModelResponse


def document(name: str) -> SourceDocument:
    return SourceDocument(
        f"https://example.org/{name}", f"Evidence {name}", f"Original evidence for {name}."
    )


class FakeGateway:
    def __init__(self, plans=None, *, stall_stage=None, fail_stage=None):
        self.plans = plans or [["initial"], ["revised"]]
        self.calls = []
        self.plan_count = 0
        self.stall_stage = stall_stage
        self.fail_stage = fail_stage
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()

    async def generate_structured(self, request, schema):
        stage = "plan" if schema is ResearchPlan and not self.plan_count else "replan"
        if schema is ResearchReport:
            stage = "report"
        self.calls.append((stage, request))
        if stage == self.stall_stage:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if stage == self.fail_stage:
            raise ModelError("rate_limit", "Temporary model limit.", status_code=429)
        if schema is ResearchPlan:
            data = ResearchPlan(queries=self.plans[self.plan_count])
            self.plan_count += 1
        else:
            assert schema is ResearchReport
            context = json.loads(request.messages[-1].content)
            evidence = context["evidence"]
            data = ResearchReport(
                title="Offline evidence report",
                sections=[
                    ReportSection(
                        heading="Saved evidence",
                        claims=[
                            CitedClaim(text=item["quote"], citations=[item["evidence_id"]])
                            for item in evidence[:6]
                        ],
                    )
                ],
                limitations=["Only the provided sources were examined."],
            )
        return StructuredResult(
            data=data,
            response=ModelResponse(
                provider="fake",
                model="offline",
                request_id=f"call-{len(self.calls)}",
                content=data.model_dump_json(),
                latency_ms=0.0,
            ),
        )


class FakeRetriever:
    def __init__(self, searches=None, *, fetch_failures=None, stalled_urls=None):
        self.searches = searches if searches is not None else {"initial": [document("normal")]}
        self.fetch_failures = fetch_failures or set()
        self.stalled_urls = stalled_urls or set()
        self.documents = {
            item.url: item
            for result in self.searches.values()
            if isinstance(result, list)
            for item in result
        }
        self.search_calls = []
        self.fetch_calls = []
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()

    async def search(self, query, limit=3):
        self.search_calls.append((query, limit))
        result = self.searches[query]
        if isinstance(result, Exception):
            raise result
        return [SearchHit(item.url, item.title) for item in result]

    async def fetch(self, hit):
        self.fetch_calls.append(hit.url)
        if hit.url in self.stalled_urls:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if hit.url in self.fetch_failures:
            raise ResearchRetrievalError("retrieval_network", "Source unavailable.")
        return self.documents[hit.url]


class TrackingStore(ResearchStore):
    last_run_id = None

    def create_run(self, question):
        self.last_run_id = super().create_run(question)
        return self.last_run_id


@pytest.fixture
def storage(tmp_path):
    path = tmp_path / "dag.sqlite"
    with TrackingStore(path) as store, DagJournal(path) as journal:
        yield store, journal


async def execute(tmp_path, storage, *, gateway=None, retriever=None, **kwargs):
    store, journal = storage
    return await run_dag_pipeline(
        "How is evidence preserved?",
        gateway=gateway or FakeGateway(),
        retriever=retriever or FakeRetriever(),
        store=store,
        journal=journal,
        output_dir=tmp_path / "reports",
        **kwargs,
    )


def assert_terminal_audit(audit):
    terminal = {"succeeded", "failed", "timed_out", "skipped", "cancelled"}
    for batch in audit["batches"]:
        assert all(task["state"] in terminal for task in batch["tasks"])
        assert [event["seq"] for event in batch["events"]] == list(
            range(1, len(batch["events"]) + 1)
        )


async def test_normal_run_uses_two_model_calls_and_persists_dependency_audit(tmp_path, storage):
    gateway, retriever = FakeGateway(), FakeRetriever()
    result = await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    store, journal = storage
    assert result["status"] == "completed"
    assert result["engine"] == "dag"
    assert result["degraded"] is False
    assert result["report_kind"] == "model_report"
    assert result["replan_count"] == 0
    assert result["counters"] == {"model": 2, "search": 1, "fetch": 1}
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    audit = journal.snapshot(result["run_id"])
    assert_terminal_audit(audit)
    assert len(audit["batches"]) == 1
    batch = audit["batches"][0]
    states = {(event["task_id"], event["state"]): event["seq"] for event in batch["events"]}
    assert states[("search:0", "succeeded")] < states[("fetch:0:0", "running")]
    assert all(call["status"] == "completed" for call in audit["model_calls"])
    assert [call["stage"] for call in audit["model_calls"]] == ["plan", "report"]
    bundle = store.get_report(result["run_id"])
    assert bundle["scheduler_audit"] == audit
    assert bundle["sources"] == store.get_sources(result["run_id"])
    exported = await asyncio.to_thread(Path(result["evidence_path"]).read_text, encoding="utf-8")
    assert json.loads(exported) == bundle


async def test_exact_half_failure_replans_once_and_preserves_previous_evidence(tmp_path, storage):
    old, new = document("old"), document("new")
    gateway = FakeGateway([["oldgood", "oldbad"], ["replacement"]])
    retriever = FakeRetriever(
        {
            "oldgood": [old],
            "oldbad": ResearchRetrievalError("retrieval_network", "Unavailable."),
            "replacement": [old, new],
        }
    )
    result = await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    store, journal = storage
    assert result["batch_statistics"][0]["failure_ratio"] == 0.5
    assert result["batch_statistics"][0]["planned_io_nodes"] == 6
    assert result["replan_count"] == 1
    assert [stage for stage, _ in gateway.calls] == ["plan", "replan", "report"]
    assert retriever.fetch_calls.count(old.url) == 1
    assert set(retriever.fetch_calls) == {old.url, new.url}
    sources = store.get_sources(result["run_id"])
    assert [(source["source_id"], source["url"]) for source in sources] == [
        ("S1", old.url),
        ("S2", new.url),
    ]
    evidence = store.get_evidence(result["run_id"])
    assert [(item["evidence_id"], item["source_id"]) for item in evidence] == [
        ("E1", "S1"),
        ("E2", "S2"),
    ]
    assert result["degraded"] is True
    assert "retrieval_gaps" in result["reasons"]
    assert_terminal_audit(journal.snapshot(result["run_id"]))


async def test_both_batches_share_search_fetch_and_model_call_budgets(tmp_path, storage):
    plans = [["q0", "q1", "q2"], ["q3", "q4", "q5"]]
    searches = {f"q{i}": [document(f"q{i}-{slot}") for slot in range(4)] for i in range(6)}
    initial_urls = {item.url for i in range(3) for item in searches[f"q{i}"]}
    gateway, retriever = FakeGateway(plans), FakeRetriever(searches, fetch_failures=initial_urls)
    result = await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    assert result["counters"] == {"model": 3, "search": 6, "fetch": 8}
    assert len(retriever.search_calls) == 6
    assert all(limit == 3 for _, limit in retriever.search_calls)
    assert len(retriever.fetch_calls) == len(set(retriever.fetch_calls)) == 8
    assert len(gateway.calls) == 3
    assert result["source_count"] == 4
    store, journal = storage
    assert len(store.get_sources(result["run_id"])) == 4
    assert len(journal.snapshot(result["run_id"])["batches"]) == 2


async def test_second_empty_batch_stops_without_a_third_plan_or_report(tmp_path, storage):
    gateway = FakeGateway([["initial"], ["revised"]])
    retriever = FakeRetriever({"initial": [], "revised": []})
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    store, journal = storage
    assert caught.value.code == "no_evidence"
    assert [stage for stage, _ in gateway.calls] == ["plan", "replan"]
    assert retriever.fetch_calls == []
    assert len(retriever.search_calls) == 2
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_report(caught.value.run_id) is None
    assert journal.snapshot(caught.value.run_id)["summary"]["replan_count"] == 1
    assert not (tmp_path / "reports").exists()


async def test_replanning_can_be_disabled_without_losing_successful_branch(tmp_path, storage):
    gateway = FakeGateway([["good", "bad"]])
    retriever = FakeRetriever(
        {
            "good": [document("saved")],
            "bad": ResearchRetrievalError("retrieval_network", "Unavailable."),
        }
    )
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        retriever=retriever,
        options=DagOptions(max_replans=0),
    )
    assert result["replan_count"] == 0
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    assert result["source_count"] == 1
    assert result["degraded"] is True


async def test_unchanged_replan_cannot_repeat_old_queries(tmp_path, storage):
    gateway = FakeGateway([["initial"], ["INITIAL"]])
    retriever = FakeRetriever({"initial": []})
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    assert caught.value.code == "no_evidence"
    assert retriever.search_calls == [("initial", 3)]
    assert len(gateway.calls) == 2


async def test_failed_url_can_be_fetched_again_only_in_the_replanned_batch(tmp_path, storage):
    original = document("temporarily-unavailable")

    class RecoveringRetriever(FakeRetriever):
        async def fetch(self, hit):
            self.fetch_calls.append(hit.url)
            if len(self.fetch_calls) == 1:
                raise ResearchRetrievalError("retrieval_network", "First batch cannot fetch it.")
            return self.documents[hit.url]

    gateway = FakeGateway([["initial"], ["different-angle"]])
    retriever = RecoveringRetriever(
        {
            # Duplicate hits within a batch must not cause automatic repeated attempts.
            "initial": [original, original],
            "different-angle": [original, original],
        }
    )
    result = await execute(tmp_path, storage, gateway=gateway, retriever=retriever)

    assert result["status"] == "completed"
    assert result["replan_count"] == 1
    assert result["counters"] == {"model": 3, "search": 2, "fetch": 2}
    assert retriever.fetch_calls == [original.url, original.url]
    assert [stage for stage, _ in gateway.calls] == ["plan", "replan", "report"]
    store, journal = storage
    sources = store.get_sources(result["run_id"])
    assert len(sources) == 1
    assert sources[0]["url"] == original.url
    assert sources[0]["text"] == original.text
    assert [item["evidence_id"] for item in store.get_evidence(result["run_id"])] == ["E1"]
    audit = journal.snapshot(result["run_id"])
    assert audit["batches"][0]["tasks"][1]["state"] == "failed"
    assert audit["batches"][1]["tasks"][1]["state"] == "succeeded"


async def test_node_timeout_drains_worker_and_completes_from_other_source(tmp_path, storage):
    good, slow = document("good"), document("slow")
    retriever = FakeRetriever({"initial": [good, slow]}, stalled_urls={slow.url})
    gateway = FakeGateway()
    result = await execute(
        tmp_path,
        storage,
        gateway=gateway,
        retriever=retriever,
        options=DagOptions(task_timeout_seconds=0.1, total_timeout_seconds=3.0),
    )
    assert retriever.entered.is_set() and retriever.drained.is_set()
    assert result["source_count"] == 1
    assert result["degraded"] is True
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    audit = storage[1].snapshot(result["run_id"])
    assert any(task["state"] == "timed_out" for task in audit["batches"][0]["tasks"])
    assert_terminal_audit(audit)


@pytest.mark.parametrize("stage", ["fetch", "report"])
async def test_global_deadline_drains_before_evidence_digest_and_never_calls_model_afterward(
    tmp_path, storage, monkeypatch, stage
):
    good, slow = document("good"), document("slow")
    gateway = FakeGateway(stall_stage="report" if stage == "report" else None)
    retriever = FakeRetriever(
        {"initial": [good, slow] if stage == "fetch" else [good]},
        stalled_urls={slow.url} if stage == "fetch" else set(),
    )
    drain = retriever.drained if stage == "fetch" else gateway.drained
    entered = retriever.entered if stage == "fetch" else gateway.entered
    original_export = dag_research._export_bundle
    exported_after_drain = []

    def checked_export(bundle, output_dir):
        assert drain.is_set()
        exported_after_drain.append(True)
        return original_export(bundle, output_dir)

    monkeypatch.setattr(dag_research, "_export_bundle", checked_export)
    async with asyncio.timeout(4):
        result = await execute(
            tmp_path,
            storage,
            gateway=gateway,
            retriever=retriever,
            options=DagOptions(total_timeout_seconds=0.5, task_timeout_seconds=5.0),
        )
    assert entered.is_set() and drain.is_set()
    assert exported_after_drain == [True]
    assert result["status"] == "completed" and result["degraded"] is True
    assert result["report_kind"] == "evidence_digest"
    assert "global_timeout" in result["reasons"]
    expected_stages = ["plan"] if stage == "fetch" else ["plan", "report"]
    assert [name for name, _ in gateway.calls] == expected_stages
    store, journal = storage
    bundle = store.get_report(result["run_id"])
    quotes = {item["evidence_id"]: item["quote"] for item in bundle["evidence"]}
    for section in bundle["report"]["sections"]:
        for claim in section["claims"]:
            assert claim["text"] == "原文摘录：" + quotes[claim["citations"][0]]
    audit = journal.snapshot(result["run_id"])
    assert_terminal_audit(audit)
    if stage == "report":
        assert audit["model_calls"][-1]["stage"] == "report"
        assert audit["model_calls"][-1]["status"] == "cancelled"
        assert "usage" not in audit["model_calls"][-1]["details"]


async def test_deadline_without_evidence_does_not_export(tmp_path, storage):
    gateway = FakeGateway(stall_stage="plan")
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            gateway=gateway,
            options=DagOptions(total_timeout_seconds=0.1),
        )
    assert caught.value.code == "no_evidence"
    assert gateway.drained.is_set()
    assert len(gateway.calls) == 1
    assert storage[0].get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()
    assert storage[1].snapshot(caught.value.run_id)["model_calls"][0]["status"] == "cancelled"


@pytest.mark.parametrize("stage", ["fetch", "report"])
async def test_user_cancellation_drains_and_never_creates_a_digest(tmp_path, storage, stage):
    good, slow = document("good"), document("slow")
    gateway = FakeGateway(stall_stage="report" if stage == "report" else None)
    retriever = FakeRetriever(
        {"initial": [good, slow] if stage == "fetch" else [good]},
        stalled_urls={slow.url} if stage == "fetch" else set(),
    )
    entered = retriever.entered if stage == "fetch" else gateway.entered
    drained = retriever.drained if stage == "fetch" else gateway.drained
    task = asyncio.create_task(execute(tmp_path, storage, gateway=gateway, retriever=retriever))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert drained.is_set()
        store, journal = storage
        assert store.get_run(store.last_run_id)["status"] == "cancelled"
        assert store.get_report(store.last_run_id) is None
        assert not (tmp_path / "reports").exists()
        audit = journal.snapshot(store.last_run_id)
        assert audit["summary"]["stop_reason"] == "cancelled"
        assert_terminal_audit(audit)
        if stage == "report":
            assert audit["model_calls"][-1]["status"] == "cancelled"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("stage", ["plan", "report"])
async def test_model_failure_has_failed_ledger_entry_without_hidden_retry(tmp_path, storage, stage):
    gateway = FakeGateway(fail_stage=stage)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway)
    assert caught.value.code == "rate_limit"
    assert caught.value.status_code == 429
    assert len(gateway.calls) == (1 if stage == "plan" else 2)
    audit = storage[1].snapshot(caught.value.run_id)
    assert audit["model_calls"][-1]["stage"] == stage
    assert audit["model_calls"][-1]["status"] == "failed"
    assert audit["model_calls"][-1]["details"] == {"error_code": "rate_limit"}
    assert storage[0].get_report(caught.value.run_id) is None


@pytest.mark.parametrize("operation", ["start_model_call", "record_event", "save_summary"])
async def test_required_audit_write_failure_stops_without_successful_export(
    tmp_path, storage, monkeypatch, operation
):
    store, journal = storage
    gateway, retriever = FakeGateway(), FakeRetriever()

    def fail_audit(*args, **kwargs):
        raise DagJournalError("dag_audit_io", "Audit unavailable.")

    monkeypatch.setattr(journal, operation, fail_audit)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway, retriever=retriever)
    assert caught.value.code == "dag_audit_io"
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()
    if operation == "start_model_call":
        assert gateway.calls == []
        assert retriever.search_calls == []
    elif operation == "record_event":
        assert [stage for stage, _ in gateway.calls] == ["plan"]
        assert retriever.search_calls == []


async def test_deadline_audit_cleanup_failure_disables_digest_fallback(
    tmp_path, storage, monkeypatch
):
    store, journal = storage
    gateway = FakeGateway(stall_stage="report")
    original_finish = journal.finish_model_call

    def finish(run_id, call_id, status, details):
        if status == "cancelled":
            raise DagJournalError("dag_audit_io", "Cancellation audit unavailable.")
        return original_finish(run_id, call_id, status, details)

    monkeypatch.setattr(journal, "finish_model_call", finish)
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            gateway=gateway,
            options=DagOptions(total_timeout_seconds=0.2),
        )
    assert gateway.drained.is_set()
    assert caught.value.code == "dag_audit_io"
    assert caught.value.audit_saved is False
    assert store.get_sources(caught.value.run_id)
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()


async def test_progress_display_failure_does_not_disable_audit_or_success(tmp_path, storage):
    def broken_progress(message):
        raise BrokenPipeError("Closed display")

    result = await execute(tmp_path, storage, on_progress=broken_progress)
    assert result["status"] == "completed"
    assert storage[0].get_run(result["run_id"])["status"] == "completed"
    assert_terminal_audit(storage[1].snapshot(result["run_id"]))


async def test_audit_reservation_crossing_deadline_never_starts_a_model_request(
    tmp_path, storage, monkeypatch
):
    store, journal = storage
    gateway, retriever = FakeGateway(), FakeRetriever()
    loop = asyncio.get_running_loop()
    original_clock = loop.time
    original_start = journal.start_model_call
    offset = 0.0

    def shifted_clock():
        return original_clock() + offset

    def slow_reservation(run_id, stage):
        nonlocal offset
        call_id = original_start(run_id, stage)
        # Model synchronous SQLite work finishing after the run deadline.
        offset = 2.0
        return call_id

    monkeypatch.setattr(journal, "start_model_call", slow_reservation)
    with monkeypatch.context() as clock_patch:
        clock_patch.setattr(loop, "time", shifted_clock)
        try:
            with pytest.raises(ResearchRunError) as caught:
                await execute(
                    tmp_path,
                    storage,
                    gateway=gateway,
                    retriever=retriever,
                    options=DagOptions(total_timeout_seconds=1.0),
                )
        finally:
            # Restore the clock before control returns to pytest's event loop.
            offset = 0.0

    assert caught.value.code == "no_evidence"
    assert gateway.calls == []
    assert retriever.search_calls == []
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    audit = journal.snapshot(caught.value.run_id)
    assert audit["summary"]["counters"] == {"model": 1, "search": 0, "fetch": 0}
    assert len(audit["model_calls"]) == 1
    assert audit["model_calls"][0]["status"] == "cancelled"
    assert audit["model_calls"][0]["details"] == {"error_code": "global_timeout"}
    assert not (tmp_path / "reports").exists()


async def test_completed_model_call_audit_failure_blocks_export_and_marks_audit_uncertain(
    tmp_path, storage, monkeypatch
):
    store, journal = storage
    gateway = FakeGateway()
    original_finish = journal.finish_model_call

    def fail_completed_report(run_id, call_id, status, details):
        if call_id == 2 and status == "completed":
            raise DagJournalError("dag_audit_io", "Report-call audit could not be saved.")
        return original_finish(run_id, call_id, status, details)

    monkeypatch.setattr(journal, "finish_model_call", fail_completed_report)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, gateway=gateway)
    assert caught.value.code == "dag_audit_io"
    assert caught.value.audit_saved is False
    assert caught.value.status_saved is True
    assert [stage for stage, _ in gateway.calls] == ["plan", "report"]
    assert store.get_sources(caught.value.run_id)
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()
    ledger = journal.snapshot(caught.value.run_id)["model_calls"]
    assert ledger[0]["status"] == "completed"
    assert ledger[1]["stage"] == "report"
    assert ledger[1]["status"] == "in_flight"


async def test_storage_failure_before_global_timeout_cannot_become_successful_digest(
    tmp_path, storage, monkeypatch
):
    store, journal = storage
    saved_one, saved_two = document("saved-one"), document("saved-two")
    failed, slow = document("storage-failure"), document("slow-peer")
    gateway = FakeGateway([["saved", "other"]])
    retriever = FakeRetriever(
        {"saved": [saved_one, saved_two], "other": [failed, slow]},
        stalled_urls={slow.url},
    )
    original_save = store.save_sources
    failed_write = asyncio.Event()

    def save_with_failure(run_id, sources, evidence):
        if sources[0]["url"] == failed.url:
            failed_write.set()
            raise ResearchStoreError("store_io", "Source snapshot could not be saved.")
        return original_save(run_id, sources, evidence)

    monkeypatch.setattr(store, "save_sources", save_with_failure)
    with pytest.raises(ResearchRunError) as caught:
        async with asyncio.timeout(4):
            await execute(
                tmp_path,
                storage,
                gateway=gateway,
                retriever=retriever,
                options=DagOptions(total_timeout_seconds=0.5, task_timeout_seconds=5.0),
            )
    assert failed_write.is_set()
    assert retriever.entered.is_set() and retriever.drained.is_set()
    assert caught.value.code == "store_io"
    assert len(store.get_sources(caught.value.run_id)) == 2
    assert store.get_run(caught.value.run_id)["status"] == "failed"
    assert store.get_report(caught.value.run_id) is None
    assert not (tmp_path / "reports").exists()
    assert [stage for stage, _ in gateway.calls] == ["plan"]
    audit = journal.snapshot(caught.value.run_id)
    assert audit["summary"]["stop_reason"] == "store_io"
    assert_terminal_audit(audit)
