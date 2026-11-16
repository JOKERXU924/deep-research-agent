"""A judge is separately audited; all model responses here are scripted fixtures."""

import asyncio
import copy
import hashlib
import json
import sqlite3

import pytest

from deep_research import benchmark_judge as module
from deep_research.benchmark_judge import (
    DIMENSIONS,
    RUBRIC_VERSION,
    FakeJudgeProvider,
    judge_report,
    report_hash,
)
from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.errors import ModelError, ResearchError
from deep_research.gateway import ModelGateway
from deep_research.model_budget import ModelBudgetLimits
from deep_research.research import ResearchRunError
from deep_research.research_store import ResearchStore
from deep_research.schemas import ModelResponse, TokenUsage


def candidate():
    text = "甲城市的河流长度为十千米。"
    return {
        "question": "甲城市的河流有多长？",
        "run_id": "source-research-id",
        "model_profile": {"model": "PRIVATE_GENERATOR_MODEL"},
        "usage": [{"provider": "PRIVATE_GENERATOR_PROVIDER"}],
        "report": {
            "title": "河流长度",
            "sections": [
                {
                    "heading": "长度",
                    "claims": [{"text": "河流长十千米。", "citations": ["E1"]}],
                }
            ],
            "limitations": ["仅有单一来源，尚未独立核查。"],
        },
        "sources": [
            {
                "source_id": "S1",
                "title": "河流资料",
                "url": "https://example.org/river",
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "truncated": False,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "source_id": "S1",
                "start": 0,
                "end": len(text),
                "quote": text,
            }
        ],
    }


def profile(*, offline=False):
    return {
        "id": "judge-fixture",
        "label": "评审测试",
        "provider": "offline_demo" if offline else "vllm",
        "model": "scripted-judge",
        "fingerprint": "b" * 64,
    }


class RecordingProvider:
    def __init__(self, change=None, *, failure=None, known=True, stall=False):
        self.change = change
        self.failure = failure
        self.known = known
        self.stall = stall
        self.requests = []
        self.entered = asyncio.Event()
        self.drained = asyncio.Event()

    async def complete(self, request):
        self.requests.append(request)
        self.entered.set()
        if self.stall:
            try:
                await asyncio.Event().wait()
            finally:
                self.drained.set()
        if self.failure is not None:
            raise self.failure
        response = await FakeJudgeProvider().complete(request)
        content = json.loads(response.content)
        if self.change:
            replacement = self.change(content)
            if replacement is not None:
                content = replacement
        data = response.model_dump()
        data["content"] = content if isinstance(content, str) else json.dumps(content)
        data["usage"] = (
            TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15) if self.known else None
        )
        return ModelResponse.model_validate(data)


async def execute(tmp_path, provider=None, *, bundle=None, **kwargs):
    bundle = bundle or candidate()
    return await judge_report(
        question=kwargs.pop("question", bundle["question"]),
        bundle=bundle,
        gold=kwargs.pop("gold", {"answer": "十千米"}),
        gateway=kwargs.pop("gateway", ModelGateway(provider or RecordingProvider())),
        db=tmp_path / "judge.sqlite3",
        profile=kwargs.pop("profile", profile()),
        **kwargs,
    )


def saved(tmp_path, run_id):
    db = tmp_path / "judge.sqlite3"
    with ResearchStore(db) as store, DagJournal(db) as journal:
        return store.get_run(run_id), store.get_report(run_id), journal.snapshot(run_id)


@pytest.mark.asyncio
async def test_one_call_separate_budget_durable_result_no_candidate_mutation(tmp_path):
    bundle = candidate()
    original = copy.deepcopy(bundle)
    provider = RecordingProvider()
    result = await execute(tmp_path, provider, bundle=bundle)
    assert bundle == original
    assert result["judge_run_id"] != bundle["run_id"]
    assert result["report_hash"] == report_hash(bundle, bundle["question"])
    assert result["rubric_version"] == RUBRIC_VERSION
    assert result["method"] == "llm_judge"
    assert result["calibrated"] is False
    assert set(result["scores"]) == set(DIMENSIONS)
    assert len(provider.requests) == 1
    budget = result["model_budget"]
    assert budget["effective_max_calls"] == 1
    assert budget["reserved"] == {
        "calls": 1,
        "input_bytes": len(provider.requests[0].model_dump_json().encode()),
        "output_tokens": 4096,
    }
    assert budget["known_usage"]["total_tokens"] == 15
    run, report, audit = saved(tmp_path, result["judge_run_id"])
    assert run["status"] == "completed"
    assert run["output_dir"] == "benchmark_judge:audit_only"
    assert report == result == audit["summary"]["result"]
    assert audit["config"]["kind"] == "benchmark_judge"
    assert audit["model_calls"][0]["status"] == "completed"
    assert audit["model_budget"] == budget
    assert not (tmp_path / "outputs").exists()


@pytest.mark.asyncio
async def test_prompt_blinds_backend_metadata_and_keeps_rubric_gold_and_schema(tmp_path):
    provider = RecordingProvider()
    await execute(tmp_path, provider)
    request = provider.requests[0]
    text = request.model_dump_json()
    assert "PRIVATE_GENERATOR" not in text
    assert "source-research-id" not in text
    assert "JSON Schema" in text
    assert "内部思考过程" in text
    payload = json.loads(request.messages[-1].content)
    assert payload["evaluator_reference"] == {"answer": "十千米"}
    assert payload["claim_inventory"][0]["claim_id"] == "C1"
    for dimension in DIMENSIONS:
        assert len(module.RUBRICS[dimension]) == 5
        assert dimension in request.messages[1].content


def test_hash_ignores_backend_metadata_but_binds_content_and_evidence():
    bundle = candidate()
    original = report_hash(bundle, bundle["question"])
    bundle["model_profile"] = {"model": "other"}
    bundle["usage"] = []
    bundle["run_id"] = "other"
    assert report_hash(bundle, bundle["question"]) == original
    bundle["report"]["limitations"] = ["changed"]
    assert report_hash(bundle, bundle["question"]) != original


@pytest.mark.asyncio
async def test_offline_fixture_is_explicit_and_does_not_claim_quality(tmp_path):
    result = await execute(
        tmp_path,
        gateway=ModelGateway(FakeJudgeProvider()),
        profile=profile(offline=True),
        offline=True,
    )
    assert result["method"] == "offline_fixture"
    assert all(item["score"] == 2 for item in result["scores"].values())
    assert all(item["label"] == "insufficient" for item in result["claims"])
    assert result["model_budget"]["usage_unknown_calls"] == 1
    assert result["model_budget"]["usage_complete"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "use_fake,offline_profile,offline",
    [(False, True, True), (True, False, True), (True, True, False), (False, False, True)],
)
async def test_offline_flag_cannot_disguise_live_provider_or_profile(
    tmp_path, use_fake, offline_profile, offline
):
    provider = FakeJudgeProvider() if use_fake else RecordingProvider()
    with pytest.raises(ResearchError, match="校验") as caught:
        await execute(
            tmp_path,
            gateway=ModelGateway(provider),
            profile=profile(offline=offline_profile),
            offline=offline,
        )
    assert caught.value.code == "judge_configuration"
    assert not (tmp_path / "judge.sqlite3").exists()


def _duplicate_claim(value):
    value["claims"].append(value["claims"][0])


def _unknown_claim(value):
    value["claims"][0]["claim_id"] = "C2"


def _unknown_evidence(value):
    value["claims"][0]["evidence_ids"] = ["E999"]


def _unknown_dimension_evidence(value):
    value["scores"]["clarity"]["evidence_ids"] = ["E999"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change,code",
    [
        (lambda value: "broken JSON", "invalid_json"),
        (lambda value: value.update(report_hash="a" * 64), "judge_binding"),
        (lambda value: value.update(rubric_version="old-version"), "schema_mismatch"),
        (_duplicate_claim, "judge_binding"),
        (_unknown_claim, "judge_binding"),
        (_unknown_evidence, "judge_binding"),
        (_unknown_dimension_evidence, "judge_binding"),
        (lambda value: value.update(claims=[]), "schema_mismatch"),
        (lambda value: value["scores"]["clarity"].update(score=True), "schema_mismatch"),
        (lambda value: value["scores"]["clarity"].update(score="4"), "schema_mismatch"),
        (lambda value: value["scores"]["clarity"].update(score=4.0), "schema_mismatch"),
        (lambda value: value["scores"]["clarity"].update(score=5), "schema_mismatch"),
        (lambda value: value["scores"]["clarity"].update(score=-1), "schema_mismatch"),
        (lambda value: value["scores"]["clarity"].update(rationale=""), "schema_mismatch"),
        (
            lambda value: value["scores"]["clarity"].update(evidence_ids=["E1", "E1"]),
            "schema_mismatch",
        ),
        (lambda value: value["claims"][0].update(label="supported"), "schema_mismatch"),
        (lambda value: value["claims"][0].update(label="contradicted"), "schema_mismatch"),
        (lambda value: value["claims"][0].update(label="true"), "schema_mismatch"),
        (lambda value: value.update(extra="secret"), "schema_mismatch"),
    ],
)
async def test_invalid_verdict_never_defaults_zero_or_retries_and_keeps_known_usage(
    tmp_path, change, code
):
    provider = RecordingProvider(change)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider)
    assert caught.value.code == code
    assert len(provider.requests) == 1
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert report is None
    assert audit["model_calls"][0]["status"] == "failed"
    assert audit["model_budget"]["known_usage"]["total_tokens"] == 15
    assert audit["model_budget"]["reserved"]["calls"] == 1


@pytest.mark.asyncio
async def test_missing_one_of_multiple_claims_rejected(tmp_path):
    bundle = candidate()
    bundle["report"]["sections"][0]["claims"].append(
        {"text": "该资料称长度为十千米。", "citations": ["E1"]}
    )
    provider = RecordingProvider(lambda value: value.update(claims=value["claims"][:1]))
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider, bundle=bundle)
    assert caught.value.code == "judge_binding"


@pytest.mark.asyncio
@pytest.mark.parametrize("label", ["supported", "contradicted", "insufficient"])
async def test_known_reference_does_not_establish_truth(tmp_path, label):
    def change(value):
        value["claims"][0].update(label=label, evidence_ids=["E1"])

    result = await execute(tmp_path, RecordingProvider(change))
    assert result["claims"][0]["label"] == label
    assert result["calibrated"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        lambda value: value.update(question="different"),
        lambda value: value["report"]["sections"][0]["claims"][0].update(citations=["E999"]),
        lambda value: value["evidence"][0].update(quote="tampered"),
        lambda value: value["evidence"][0].update(start=True),
        lambda value: value["evidence"].append(value["evidence"][0]),
        lambda value: value["sources"][0].update(sha256="a" * 64),
        lambda value: value["sources"][0].update(source_id="INVALID"),
        lambda value: value["sources"][0].update(title="\ud800"),
        lambda value: value.update(evidence=[]),
        lambda value: value.update(sources=[]),
    ],
)
async def test_invalid_input_has_no_call_or_run(tmp_path, change):
    bundle = candidate()
    question = bundle["question"]
    change(bundle)
    provider = RecordingProvider()
    with pytest.raises(ResearchError) as caught:
        await execute(tmp_path, provider, bundle=bundle, question=question)
    assert caught.value.code == "judge_input"
    assert provider.requests == []
    assert not (tmp_path / "judge.sqlite3").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("gold", [[], {"answer": float("nan")}, {"answer": "\ud800"}])
async def test_invalid_gold_fails_before_call(tmp_path, gold):
    provider = RecordingProvider()
    with pytest.raises(ResearchError):
        await execute(tmp_path, provider, gold=gold)
    assert provider.requests == []
    assert not (tmp_path / "judge.sqlite3").exists()


@pytest.mark.asyncio
async def test_schema_inclusive_input_limit_no_silent_truncation(tmp_path):
    provider = RecordingProvider()
    with pytest.raises(ResearchError) as caught:
        await execute(tmp_path, provider, gold={"answer": "界" * 22000})
    assert caught.value.code == "judge_input_too_large"
    assert provider.requests == []
    assert not (tmp_path / "judge.sqlite3").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits", [ModelBudgetLimits(max_input_bytes=1), ModelBudgetLimits(max_output_tokens=4095)]
)
async def test_budget_exhaustion_prevents_dispatch_but_keeps_failed_run(tmp_path, limits):
    provider = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider, budget=limits)
    assert caught.value.code == "model_budget_exceeded"
    assert provider.requests == []
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert report is None
    assert audit["model_calls"] == []
    assert audit["model_budget"]["reserved"]["calls"] == 0


@pytest.mark.asyncio
async def test_exact_budget_boundary(tmp_path):
    provider = RecordingProvider()
    first = await execute(tmp_path, provider)
    size = first["model_budget"]["reserved"]["input_bytes"]
    result = await execute(
        tmp_path,
        RecordingProvider(),
        budget=ModelBudgetLimits(max_input_bytes=size, max_output_tokens=4096),
    )
    assert result["model_budget"]["reserved"]["input_bytes"] == size


@pytest.mark.asyncio
async def test_durable_reservation_exists_before_provider_io(tmp_path):
    class Observer(RecordingProvider):
        async def complete(self, request):
            with sqlite3.connect(tmp_path / "judge.sqlite3") as connection:
                row = connection.execute(
                    "SELECT status,details_json FROM dag_model_calls"
                ).fetchone()
            assert row[0] == "in_flight"
            assert json.loads(row[1])["reservation"]["input_bytes"] == len(
                request.model_dump_json().encode()
            )
            return await super().complete(request)

    await execute(tmp_path, Observer())


@pytest.mark.asyncio
async def test_start_audit_failure_prevents_provider_io(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise DagJournalError("dag_audit_io", "safe")

    monkeypatch.setattr(DagJournal, "start_model_call", broken)
    provider = RecordingProvider()
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider)
    assert caught.value.code == "dag_audit_io"
    assert provider.requests == []
    assert saved(tmp_path, caught.value.run_id)[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_final_audit_failure_does_not_publish_result_or_invent_usage(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise DagJournalError("dag_audit_io", "safe")

    monkeypatch.setattr(DagJournal, "finish_model_call", broken)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path)
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert report is None
    assert caught.value.audit_complete is False
    assert audit["model_calls"][0]["status"] == "in_flight"
    assert audit["model_budget"]["usage_unknown_calls"] == 1
    assert audit["model_budget"]["known_usage"]["total_tokens"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("known", [True, False])
async def test_provider_failure_usage_known_independent_of_output_validity(tmp_path, known):
    usage = TokenUsage(input_tokens=7, output_tokens=2, total_tokens=9) if known else None
    provider = RecordingProvider(failure=ModelError("rate_limited", "safe", response_usage=usage))
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider)
    assert caught.value.code == "rate_limited"
    budget = saved(tmp_path, caught.value.run_id)[2]["model_budget"]
    assert budget["known_usage"]["total_tokens"] == (9 if known else 0)
    assert budget["usage_unknown_calls"] == (0 if known else 1)


@pytest.mark.asyncio
async def test_unknown_usage_is_not_zero_cost(tmp_path):
    result = await execute(tmp_path, RecordingProvider(known=False))
    assert result["model_budget"]["usage_unknown_calls"] == 1
    assert result["model_budget"]["usage_complete"] is False


@pytest.mark.asyncio
async def test_timeout_cancels_request_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "JUDGE_TIMEOUT_SECONDS", 0.01)
    provider = RecordingProvider(stall=True)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider)
    assert caught.value.code == "judge_timeout"
    assert provider.drained.is_set()
    assert len(provider.requests) == 1
    run, _, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert audit["model_calls"][0]["status"] == "cancelled"
    assert audit["model_budget"]["usage_unknown_calls"] == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_persists(tmp_path):
    provider = RecordingProvider(stall=True)
    task = asyncio.create_task(execute(tmp_path, provider))
    await provider.entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert provider.drained.is_set()
    run, report, audit = saved(tmp_path, caught.value.judge_run_id)
    assert run["status"] == "cancelled"
    assert report is None
    assert audit["model_calls"][0]["status"] == "cancelled"
    assert audit["model_budget"]["reserved"]["calls"] == 1


@pytest.mark.asyncio
async def test_cancel_cleanup_failure_preserves_original_cancel(tmp_path, monkeypatch):
    provider = RecordingProvider(stall=True)
    task = asyncio.create_task(execute(tmp_path, provider))
    await provider.entered.wait()

    def broken(*_args, **_kwargs):
        raise RuntimeError("DO_NOT_LEAK")

    monkeypatch.setattr(DagJournal, "finish_model_call", broken)
    monkeypatch.setattr(ResearchStore, "fail_run", broken)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert caught.value.status_saved is False
    assert caught.value.audit_complete is False
    assert "DO_NOT_LEAK" not in str(caught.value)


@pytest.mark.asyncio
async def test_cancellation_requested_in_reservation_hook_never_dispatches(tmp_path, monkeypatch):
    original = DagJournal.start_model_call

    def cancelling(*args, **kwargs):
        result = original(*args, **kwargs)
        asyncio.current_task().cancel()
        return result

    monkeypatch.setattr(DagJournal, "start_model_call", cancelling)
    provider = RecordingProvider()
    task = asyncio.create_task(execute(tmp_path, provider))
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert provider.requests == []
    audit = saved(tmp_path, caught.value.run_id)[2]
    assert audit["model_budget"]["reserved"]["calls"] == 1
    assert audit["model_budget"]["usage_unknown_calls"] == 1


@pytest.mark.asyncio
async def test_failure_cleanup_preserves_original_public_error(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("DO_NOT_LEAK")

    monkeypatch.setattr(ResearchStore, "fail_run", broken)
    provider = RecordingProvider(failure=ModelError("rate_limited", "safe"))
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, provider)
    assert caught.value.code == "rate_limited"
    assert caught.value.status_saved is False
    assert "DO_NOT_LEAK" not in str(caught.value)


@pytest.mark.asyncio
async def test_missing_dimension_fails_instead_of_zero_default(tmp_path):
    def change(value):
        del value["scores"]["completeness"]

    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, RecordingProvider(change))
    assert caught.value.code == "schema_mismatch"
    assert saved(tmp_path, caught.value.run_id)[1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ['{"report_hash":"a","report_hash":"b"}', '{"value":NaN}', '{"value":1e999}', "[]"],
)
async def test_malformed_json_objects_fail_closed(tmp_path, content):
    with pytest.raises(ResearchRunError):
        await execute(tmp_path, RecordingProvider(lambda _value: content))


@pytest.mark.asyncio
async def test_reference_hash_audits_evaluator_gold_separately(tmp_path):
    first = await execute(tmp_path, gold={"answer": "ten"})
    second = await execute(tmp_path, gold={"answer": "twenty"})
    assert first["report_hash"] == second["report_hash"]
    assert first["reference_hash"] != second["reference_hash"]
    assert (
        saved(tmp_path, first["judge_run_id"])[2]["config"]["reference_hash"]
        == first["reference_hash"]
    )


@pytest.mark.asyncio
async def test_valid_memory_extended_sources_supported(tmp_path):
    bundle = candidate()
    for index in range(2, 9):
        bundle["sources"].append({**bundle["sources"][0], "source_id": f"S{index}"})
    result = await execute(tmp_path, bundle=bundle)
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_ninth_source_rejected(tmp_path):
    bundle = candidate()
    for index in range(2, 10):
        bundle["sources"].append({**bundle["sources"][0], "source_id": f"S{index}"})
    with pytest.raises(ResearchError) as caught:
        await execute(tmp_path, bundle=bundle)
    assert caught.value.code == "judge_input"


@pytest.mark.asyncio
async def test_mutation_during_call_cannot_change_binding_or_profile(tmp_path):
    bundle = candidate()
    selected = profile()
    original_hash = report_hash(bundle, bundle["question"])

    class Mutating(RecordingProvider):
        async def complete(self, request):
            bundle["report"]["title"] = "changed during I/O"
            selected["model"] = "changed"
            return await super().complete(request)

    result = await execute(tmp_path, Mutating(), bundle=bundle, profile=selected)
    assert result["report_hash"] == original_hash
    assert result["model_profile"]["model"] == "scripted-judge"


@pytest.mark.asyncio
async def test_tampered_budget_model_is_revalidated(tmp_path):
    with pytest.raises(ResearchError) as caught:
        await execute(tmp_path, budget=ModelBudgetLimits.model_construct(max_calls=0))
    assert caught.value.code == "model_budget_invalid"
    assert not (tmp_path / "judge.sqlite3").exists()


@pytest.mark.asyncio
async def test_blocking_provider_cannot_accept_late_result(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "JUDGE_TIMEOUT_SECONDS", 0.001)
    loop = asyncio.get_running_loop()
    clock = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: clock[0])

    class Blocking(RecordingProvider):
        async def complete(self, request):
            # Advance time without yielding, like a blocking synchronous callback.
            # Windows' monotonic clock can tick at 15.625 ms: a 10 ms real sleep
            # does not reliably advance it. Test the deadline guard deterministically.
            clock[0] += module.JUDGE_TIMEOUT_SECONDS + 1.0
            return await super().complete(request)

    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path, Blocking())
    assert caught.value.code == "judge_timeout"
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert report is None
    assert audit["model_budget"]["known_usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_summary_save_failure_does_not_publish_result(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise DagJournalError("dag_audit_io", "safe")

    monkeypatch.setattr(DagJournal, "save_summary", broken)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path)
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert report is None
    assert audit["model_calls"][0]["status"] == "completed"
    assert audit["model_budget"]["known_usage"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_final_store_failure_is_not_success(tmp_path, monkeypatch):
    def broken(*_args, **_kwargs):
        raise ResearchError("store_io", "safe")

    monkeypatch.setattr(ResearchStore, "complete_run", broken)
    with pytest.raises(ResearchRunError) as caught:
        await execute(tmp_path)
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "failed"
    assert report is None
    # Separate SQLite connections are not a single cross-table transaction.
    # A saved judgment is diagnostic; research_runs.status is the commit marker.
    assert audit["summary"]["result"]["status"] == "completed"
    assert audit["model_calls"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_cancellation_as_provider_returns_retains_known_usage(tmp_path):
    class Cancelling(RecordingProvider):
        async def complete(self, request):
            response = await super().complete(request)
            asyncio.current_task().cancel()
            return response

    task = asyncio.create_task(execute(tmp_path, Cancelling()))
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    run, report, audit = saved(tmp_path, caught.value.run_id)
    assert run["status"] == "cancelled"
    assert report is None
    assert audit["model_calls"][0]["status"] == "cancelled"
    assert audit["model_budget"]["known_usage"]["total_tokens"] == 15
