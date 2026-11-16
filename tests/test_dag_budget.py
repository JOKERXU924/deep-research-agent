"""M7 integration uses real SQLite and fake providers; no model quota is consumed."""

import asyncio
import json
import sqlite3

import pytest

from deep_research.dag_demo import DagDemoProvider, DagDemoRetriever
from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.dag_research import DagOptions, run_dag_pipeline
from deep_research.demo import DEMO_QUESTION
from deep_research.errors import ModelError, ResearchError
from deep_research.gateway import ModelGateway
from deep_research.model_budget import ModelBudgetLimits
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore
from deep_research.review_demo import ReviewDemoProvider
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewOptions
from deep_research.schemas import ModelResponse, TokenUsage


class RecordingProvider:
    def __init__(self, *, inner=None, bad_at=None, stall_at=None, error=None, usage=True):
        self.inner = inner or DagDemoProvider()
        self.requests = []
        self.bad_at = bad_at
        self.stall_at = stall_at
        self.error = error
        self.usage = usage
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()

    async def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == self.stall_at:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if self.error is not None:
            raise self.error
        response = await self.inner.complete(request)
        data = response.model_dump()
        if self.usage:
            data["usage"] = TokenUsage(input_tokens=10, output_tokens=3, total_tokens=13)
        if len(self.requests) == self.bad_at:
            data["content"] = "not valid JSON"
        return ModelResponse.model_validate(data)


def profile():
    return {
        "id": "demo",
        "label": "教学演示",
        "provider": "offline_demo",
        "model": "fixture",
        "fingerprint": "b" * 64,
    }


@pytest.fixture
def storage(tmp_path):
    path = tmp_path / "research.sqlite3"
    with ResearchStore(path) as store, DagJournal(path) as journal, ReviewJournal(path) as review:
        yield store, journal, review


async def execute(tmp_path, storage, provider, **kwargs):
    store, journal, _ = storage
    return await run_dag_pipeline(
        DEMO_QUESTION,
        gateway=ModelGateway(provider),
        retriever=DagDemoRetriever(),
        store=store,
        journal=journal,
        output_dir=tmp_path / "outputs",
        source_mode="offline_demo",
        model_budget=kwargs.pop("model_budget", ModelBudgetLimits()),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_success_saves_final_request_reservations_and_immutable_profile(tmp_path, storage):
    provider = RecordingProvider()
    selected = profile()

    def mutate(_):
        selected["model"] = "mutated-after-start"

    result = await execute(tmp_path, storage, provider, model_profile=selected, on_progress=mutate)
    budget = result["model_budget"]
    assert len(provider.requests) == budget["reserved"]["calls"] == 2
    assert budget["reserved"]["input_bytes"] == sum(
        len(item.model_dump_json().encode("utf-8")) for item in provider.requests
    )
    assert budget["reserved"]["output_tokens"] == 768 + 3072
    assert budget["known_usage"] == {"input_tokens": 20, "output_tokens": 6, "total_tokens": 26}
    assert budget["usage_complete"] is True
    assert result["model_profile"]["model"] == "fixture"
    store, journal, _ = storage
    snapshot = journal.snapshot(result["run_id"])
    assert snapshot["model_budget"] == budget == snapshot["summary"]["model_budget"]
    bundle = store.get_report(result["run_id"])
    assert bundle["model_profile"] == result["model_profile"]
    assert bundle["model_budget"] == budget
    for call, request in zip(snapshot["model_calls"], provider.requests, strict=True):
        assert call["details"]["reservation"]["input_bytes"] == len(
            request.model_dump_json().encode("utf-8")
        )
        assert call["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits,actual",
    [
        (ModelBudgetLimits(max_calls=1), 1),
        (ModelBudgetLimits(max_input_bytes=1), 0),
        (ModelBudgetLimits(max_output_tokens=767), 0),
        (ModelBudgetLimits(max_output_tokens=3839), 1),
    ],
)
async def test_budget_exhaustion_fails_closed_without_dispatch(tmp_path, storage, limits, actual):
    provider = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider, model_budget=limits)
    error = caught.value
    assert error.code == "model_budget_exceeded"
    assert len(provider.requests) == actual
    store, journal, _ = storage
    snapshot = journal.snapshot(error.run_id)
    assert snapshot["summary"]["stop_reason"] == "model_budget_exceeded"
    assert snapshot["model_budget"]["reserved"]["calls"] == actual
    assert len(snapshot["model_calls"]) == actual
    assert store.get_run(error.run_id)["status"] == "failed"
    assert store.get_report(error.run_id) is None


@pytest.mark.asyncio
async def test_aggregate_bytes_exact_boundary_and_one_byte_less(tmp_path, storage):
    first = RecordingProvider()
    result = await execute(tmp_path, storage, first)
    exact = result["model_budget"]["reserved"]["input_bytes"]
    # Retrieval timestamps affect input lengths, not their fixed ISO width.
    second = RecordingProvider()
    await execute(tmp_path, storage, second, model_budget=ModelBudgetLimits(max_input_bytes=exact))
    third = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path, storage, third, model_budget=ModelBudgetLimits(max_input_bytes=exact - 1)
        )
    assert caught.value.code == "model_budget_exceeded"
    assert len(third.requests) == 1


@pytest.mark.asyncio
async def test_schema_bytes_checked_before_dispatch(tmp_path, storage):
    baseline = RecordingProvider()
    await execute(tmp_path, storage, baseline)
    final_bytes = len(baseline.requests[0].model_dump_json().encode("utf-8"))
    provider = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            provider,
            model_budget=ModelBudgetLimits(max_input_bytes=final_bytes - 1),
        )
    assert caught.value.code == "model_budget_exceeded"
    assert provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_at", [1, 2])
async def test_structured_parse_failure_keeps_known_usage(tmp_path, storage, bad_at):
    provider = RecordingProvider(bad_at=bad_at)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider)
    snapshot = storage[1].snapshot(caught.value.run_id)
    budget = snapshot["model_budget"]
    assert budget["reserved"]["calls"] == bad_at
    assert budget["usage_known_calls"] == bad_at
    assert budget["usage_unknown_calls"] == 0
    assert snapshot["model_calls"][-1]["status"] == "failed"
    assert snapshot["model_calls"][-1]["details"]["usage"]["total_tokens"] == 13


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [True, False])
async def test_failed_adapter_usage_is_known_only_when_provided(tmp_path, storage, known):
    error = ModelError(
        "model_truncated",
        "Incomplete",
        response_usage=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        if known
        else None,
    )
    provider = RecordingProvider(error=error)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider)
    budget = storage[1].snapshot(caught.value.run_id)["model_budget"]
    assert budget["usage_known_calls"] == int(known)
    assert budget["usage_unknown_calls"] == int(not known)
    assert budget["reserved"]["output_tokens"] == 768
    assert budget["known_usage"]["output_tokens"] == (50 if known else 0)


@pytest.mark.asyncio
async def test_unknown_success_usage_is_not_zero_usage(tmp_path, storage):
    result = await execute(tmp_path, storage, RecordingProvider(usage=False))
    assert result["model_budget"]["usage_unknown_calls"] == 2
    assert result["model_budget"]["known_usage"]["total_tokens"] == 0
    assert result["model_budget"]["usage_complete"] is False


@pytest.mark.asyncio
async def test_cancelled_call_remains_reserved_after_reopen(tmp_path, storage):
    provider = RecordingProvider(stall_at=1)
    run_ids = []
    task = asyncio.create_task(execute(tmp_path, storage, provider, on_run_created=run_ids.append))
    await asyncio.wait_for(provider.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.drained.is_set()
    with DagJournal(tmp_path / "research.sqlite3") as reopened:
        value = reopened.snapshot(run_ids[0])
        assert value["model_calls"][0]["status"] == "cancelled"
        assert value["model_budget"]["reserved"]["calls"] == 1
        assert value["model_budget"]["usage_unknown_calls"] == 1
        assert value["summary"]["stop_reason"] == "cancelled"


@pytest.mark.asyncio
async def test_reservation_saved_before_dispatch(tmp_path, storage):
    provider = RecordingProvider(stall_at=1)
    run_ids = []
    task = asyncio.create_task(execute(tmp_path, storage, provider, on_run_created=run_ids.append))
    await asyncio.wait_for(provider.entered.wait(), 2)
    snapshot = storage[1].snapshot(run_ids[0])
    assert snapshot["summary"] is None
    assert snapshot["model_calls"][0]["status"] == "in_flight"
    assert snapshot["model_budget"]["reserved"]["calls"] == 1
    assert snapshot["model_budget"]["usage_unknown_calls"] == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_during_reservation_never_dispatches(tmp_path, storage, monkeypatch):
    original = storage[1].start_model_call

    def cancel(*args):
        result = original(*args)
        asyncio.current_task().cancel()
        return result

    monkeypatch.setattr(storage[1], "start_model_call", cancel)
    provider = RecordingProvider()
    ids = []
    with pytest.raises(asyncio.CancelledError):
        await execute(tmp_path, storage, provider, on_run_created=ids.append)
    assert not provider.requests
    snapshot = storage[1].snapshot(ids[0])
    assert snapshot["model_calls"][0]["status"] == "cancelled"
    assert snapshot["model_budget"]["reserved"]["calls"] == 1


@pytest.mark.asyncio
async def test_failed_reservation_audit_prevents_external_request(tmp_path, storage, monkeypatch):
    def broken(*_):
        raise DagJournalError("dag_audit_io", "Disk failed")

    monkeypatch.setattr(storage[1], "start_model_call", broken)
    provider = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider)
    assert provider.requests == []
    snapshot = storage[1].snapshot(caught.value.run_id)
    assert snapshot["model_calls"] == []
    assert snapshot["model_budget"]["reserved"]["calls"] == 0


@pytest.mark.asyncio
async def test_failed_final_usage_audit_does_not_continue(tmp_path, storage, monkeypatch):
    def broken(*_):
        raise DagJournalError("dag_audit_io", "Disk failed")

    monkeypatch.setattr(storage[1], "finish_model_call", broken)
    provider = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, storage, provider)
    assert len(provider.requests) == 1
    snapshot = storage[1].snapshot(caught.value.run_id)
    assert snapshot["model_calls"][0]["status"] == "in_flight"
    assert snapshot["model_budget"]["usage_unknown_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["passed", "repair", "oscillation", "stalled", "max-rounds", "format"]
)
async def test_review_all_calls_including_format_repair_share_one_budget(
    tmp_path, storage, scenario
):
    provider = RecordingProvider(inner=ReviewDemoProvider(scenario))
    result = await execute(
        tmp_path, storage, provider, review_options=ReviewOptions(), review_journal=storage[2]
    )
    budget = result["model_budget"]
    assert budget["reserved"]["calls"] == len(provider.requests)
    assert 3 <= len(provider.requests) <= 9
    assert budget["usage_known_calls"] == len(provider.requests)
    assert budget["reserved"]["input_bytes"] == sum(
        len(item.model_dump_json().encode("utf-8")) for item in provider.requests
    )
    calls = storage[1].snapshot(result["run_id"])["model_calls"]
    if scenario == "format":
        assert any(call["status"] == "failed" for call in calls)
        assert any("format" in call["stage"] for call in calls)


@pytest.mark.asyncio
async def test_review_budget_exhaustion_is_not_a_format_retry(tmp_path, storage):
    provider = RecordingProvider(inner=ReviewDemoProvider("format"))
    with pytest.raises(ResearchRunError) as caught:
        await execute(
            tmp_path,
            storage,
            provider,
            review_options=ReviewOptions(),
            review_journal=storage[2],
            model_budget=ModelBudgetLimits(max_calls=3),
        )
    assert caught.value.code == "model_budget_exceeded"
    assert len(provider.requests) == 3
    snapshot = storage[1].snapshot(caught.value.run_id)
    assert snapshot["model_budget"]["usage_known_calls"] == 3
    assert snapshot["summary"]["stop_reason"] == "model_budget_exceeded"


@pytest.mark.asyncio
async def test_deadline_summary_does_not_refund_in_flight_report(tmp_path, storage):
    provider = RecordingProvider(stall_at=2)
    result = await execute(
        tmp_path, storage, provider, options=DagOptions(total_timeout_seconds=0.2)
    )
    assert result["report_kind"] == "evidence_digest"
    assert result["model_budget"]["reserved"]["calls"] == 2
    assert result["model_budget"]["usage_known_calls"] == 1
    assert result["model_budget"]["usage_unknown_calls"] == 1


def test_journal_reservation_cannot_be_changed_or_bypassed(storage):
    store, journal, _ = storage
    run = store.create_run("Budget test")
    journal.create_run(run, {"model_budget": ModelBudgetLimits(max_calls=1).model_dump()})
    with pytest.raises(DagJournalError):
        journal.start_model_call(run, "plan")
    call = journal.start_model_call(
        run, "plan", {"reservation": {"input_bytes": 100, "output_tokens": 50}}
    )
    with pytest.raises(DagJournalError):
        journal.finish_model_call(
            run, call, "completed", {"reservation": {"input_bytes": 1, "output_tokens": 1}}
        )
    journal.finish_model_call(run, call, "failed", {"error_code": "network"})
    with pytest.raises(ResearchError) as caught:
        journal.start_model_call(
            run, "retry", {"reservation": {"input_bytes": 1, "output_tokens": 1}}
        )
    assert caught.value.code == "model_budget_exceeded"
    assert journal.snapshot(run)["model_budget"]["reserved"]["input_bytes"] == 100


@pytest.mark.parametrize(
    "tampered",
    [
        {"api_key": "never-return-this-secret"},
        {"reservation": {"input_bytes": -1, "output_tokens": 5}},
        {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 1}},
        {"provider": "x\nsecret"},
        {"request_id": "x" * 257},
    ],
)
def test_journal_revalidates_read_allowlist_for_tampered_database(tmp_path, storage, tampered):
    store, journal, _ = storage
    run = store.create_run("Audit safety")
    journal.create_run(run, {})
    journal.start_model_call(run, "plan")
    with sqlite3.connect(tmp_path / "research.sqlite3") as connection:
        connection.execute(
            "UPDATE dag_model_calls SET details_json = ? WHERE run_id = ?",
            (json.dumps(tampered), run),
        )
    with pytest.raises(DagJournalError) as caught:
        journal.snapshot(run)
    assert caught.value.code == "dag_audit_validation"
    assert "secret" not in str(caught.value)


def test_partial_default_budget_config_normalizes_before_recording(storage):
    store, journal, _ = storage
    run = store.create_run("Default limits")
    journal.create_run(run, {"model_budget": {"max_calls": 1}})
    journal.start_model_call(
        run, "plan", {"reservation": {"input_bytes": 100, "output_tokens": 50}}
    )
    snapshot = journal.snapshot(run)
    assert snapshot["model_budget"]["limits"] == ModelBudgetLimits(max_calls=1).model_dump()
