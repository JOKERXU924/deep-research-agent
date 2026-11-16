"""Deterministic Red--Blue state transitions without a live model."""

import asyncio
import copy
import hashlib
import json

import pytest

from deep_research.errors import ResearchError, StructuredOutputError
from deep_research.gateway import StructuredResult, structured_request
from deep_research.research_models import CitedClaim, ReportSection, ResearchReport
from deep_research.review import review_report
from deep_research.review_models import (
    BlueRepairs,
    RedReview,
    RepairOperation,
    ReviewFinding,
    ReviewOptions,
    ReviewScores,
    report_fingerprint,
)
from deep_research.schemas import MAX_INPUT_BYTES, ModelResponse


def report(text="Original claim"):
    return ResearchReport(
        title="Offline report",
        sections=[
            ReportSection(heading="Evidence", claims=[CitedClaim(text=text, citations=["E1"])])
        ],
        limitations=["Only the supplied sources were examined."],
    )


def inputs():
    text = "Original evidence supports only limited observations."
    sources = [
        {
            "source_id": "S1",
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "title": "Saved source",
            "url": "https://example.org/source",
            "retrieved_at": "2025-01-01T00:00:00Z",
            "truncated": False,
        }
    ]
    evidence = [
        {"evidence_id": "E1", "source_id": "S1", "start": 0, "end": len(text), "quote": text}
    ]
    return sources, evidence


def finding(*, claim_id="C1", evidence_id="E1"):
    return ReviewFinding(
        id="F1",
        category="evidence",
        severity="medium",
        claim_ids=[claim_id],
        evidence_ids=[evidence_id],
        description="The claim needs a narrower qualification.",
    )


def red(score=60, *, findings=None, scores=None, hash_override=None):
    def action(schema, payload):
        assert schema is RedReview
        return RedReview(
            report_hash=hash_override or payload["report_hash"],
            scores=scores
            or ReviewScores(
                evidence_support=score, logical_consistency=score, citation_quality=score
            ),
            findings=[] if findings is None else findings,
        )

    return action


def blue(text="Revised claim", *, target="C1", citations=None, hash_override=None):
    def action(schema, payload):
        assert schema is BlueRepairs
        return BlueRepairs(
            report_hash=hash_override or payload["report_hash"],
            operations=[
                RepairOperation(
                    op="MODIFY",
                    target_id=target,
                    claim=CitedClaim(text=text, citations=citations or ["E1"]),
                    rationale="Narrow the claim to the available evidence.",
                )
            ],
        )

    return action


def structured(data, call_number):
    return StructuredResult(
        data=data,
        response=ModelResponse(
            provider="fake",
            model="offline",
            request_id=f"call-{call_number}",
            content=data.model_dump_json(),
            latency_ms=0.0,
        ),
    )


class ScriptedModel:
    def __init__(self, actions, before_call=None):
        self.actions = list(actions)
        self.calls = []
        self.before_call = before_call

    async def __call__(self, stage, request, schema):
        if self.before_call:
            self.before_call(stage)
        outgoing = structured_request(request, schema)
        assert len(outgoing.model_dump_json().encode("utf-8")) <= MAX_INPUT_BYTES
        payload = json.loads(request.messages[-1].content)
        self.calls.append(
            {"stage": stage, "request": request, "schema": schema, "payload": payload}
        )
        assert self.actions, "Unexpected additional model call"
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return structured(action(schema, payload), len(self.calls))


async def run(
    model, *, events=None, initial=None, options=None, on_event=None, memory_context=None
):
    sources, evidence = inputs()
    if events is None:
        events = []
    return await review_report(
        "What does the evidence actually support?",
        initial or report(),
        evidence=evidence,
        sources=sources,
        model_call=model,
        on_event=on_event or events.append,
        options=options,
        memory_context=memory_context,
    )


async def test_initial_pass_stops_after_one_red_and_persists_before_return():
    events = []
    model = ScriptedModel([red(80)])
    result = await run(model, events=events)
    assert result.report == report()
    assert result.audit["status"] == result.audit["stop_reason"] == "passed"
    assert result.audit["calls"] == 1
    assert result.audit["rounds"] == result.audit["format_repairs"] == 0
    assert [event["kind"] for event in events] == ["initial", "red", "stop"]
    assert events[-1]["report"] == result.report.model_dump()
    assert events[-1]["audit"] == result.audit
    assert len(result.audit["history"]) == 1


async def test_validated_durable_events_precede_every_following_phase():
    events = []
    previous = {"review.red.0": "initial", "review.blue.1": "red", "review.red.1": "blue"}

    def before(stage):
        assert events[-1]["kind"] == previous[stage]

    model = ScriptedModel([red(50, findings=[finding()]), blue(), red(90)], before)
    result = await run(model, events=events)
    assert [event["kind"] for event in events] == ["initial", "red", "blue", "red", "stop"]
    assert result.report == report("Revised claim")
    assert result.audit["calls"] == 3
    assert result.audit["best_round"] == 1
    for event in events:
        assert len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()) <= 65536
    assert "report" not in events[2]  # Blue stores validated patches, not two full report copies.


@pytest.mark.parametrize("rounds", [0, 1, 2])
@pytest.mark.parametrize("format_repair", [False, True])
async def test_round_and_global_repair_budgets_bound_calls_to_one_three_five_or_two_four_six(
    rounds, format_repair
):
    actions = [red(60)]
    if format_repair:
        actions.insert(0, StructuredOutputError("invalid_json", "safe invalid JSON category"))
    for index in range(rounds):
        actions += [blue(f"Revision {index + 1}"), red(60)]
    model = ScriptedModel(actions)
    result = await run(model, options=ReviewOptions(max_rounds=rounds))
    assert result.audit["calls"] == 1 + 2 * rounds + int(format_repair)
    assert len(model.calls) == result.audit["calls"]
    assert result.audit["rounds"] == rounds
    assert result.audit["format_repairs"] == int(format_repair)
    assert result.audit["stop_reason"] == "max_rounds"
    assert result.audit["status"] == "unresolved"


async def test_one_low_dimension_or_remaining_finding_prevents_passing():
    model = ScriptedModel(
        [
            red(
                scores=ReviewScores(
                    evidence_support=100, logical_consistency=79, citation_quality=100
                )
            )
        ]
    )
    result = await run(model, options=ReviewOptions(max_rounds=0))
    assert result.audit["status"] == "unresolved"
    second = await run(
        ScriptedModel([red(100, findings=[finding()])]), options=ReviewOptions(max_rounds=0)
    )
    assert second.audit["status"] == "unresolved"


async def test_noop_patch_stalls_without_reviewer_call_for_unchanged_report():
    model = ScriptedModel([red(70), blue("Original claim")])
    result = await run(model)
    assert result.audit["stop_reason"] == "stalled"
    assert result.audit["calls"] == 2
    assert len(result.audit["history"]) == 1
    assert result.report == report()


async def test_supported_verify_records_judgment_but_does_not_invent_report_changes():
    def verify(schema, payload):
        return BlueRepairs(
            report_hash=payload["report_hash"],
            operations=[
                RepairOperation(
                    op="VERIFY",
                    target_id="C1",
                    verdict="supported",
                    evidence_ids=["E1"],
                    rationale="The saved quotation is relevant; this remains a model judgment.",
                )
            ],
        )

    events = []
    result = await run(ScriptedModel([red(70), verify]), events=events)
    assert result.audit["stop_reason"] == "stalled"
    assert result.report == report()
    assert events[2]["verifications"][0]["verdict"] == "supported"


async def test_oscillation_returns_best_previously_reviewed_version_not_last_patch():
    model = ScriptedModel([red(40), blue("Better claim"), red(70), blue("Original claim")])
    result = await run(model)
    assert result.audit["stop_reason"] == "oscillation"
    assert result.audit["calls"] == 4
    assert result.audit["rounds"] == 2
    assert result.audit["best_round"] == 1
    assert result.report == report("Better claim")
    assert len(result.audit["history"]) == 2


async def test_max_rounds_preserves_best_reviewed_intermediate_not_last_worse_candidate():
    model = ScriptedModel([red(20), blue("Best claim"), red(75), blue("Worse claim"), red(30)])
    result = await run(model)
    assert result.report == report("Best claim")
    assert result.audit["best_hash"] == report_fingerprint(report("Best claim"))
    assert result.audit["final_scores"]["evidence_support"] == 75
    assert result.audit["best_round"] == 1
    assert result.audit["stop_reason"] == "max_rounds"


async def test_equal_quality_preserves_earliest_reviewed_report_deterministically():
    model = ScriptedModel([red(60), blue("New but not better"), red(60)])
    result = await run(model, options=ReviewOptions(max_rounds=1))
    assert result.report == report()
    assert result.audit["best_round"] == 0


async def test_quality_prefers_weakest_dimension_before_total_score():
    model = ScriptedModel(
        [
            red(
                scores=ReviewScores(
                    evidence_support=40, logical_consistency=100, citation_quality=100
                )
            ),
            blue("Balanced claim"),
            red(50),
        ]
    )
    result = await run(model, options=ReviewOptions(max_rounds=1))
    assert result.report == report("Balanced claim")


async def test_format_retry_regenerates_same_context_without_raw_invalid_payload():
    secret = "PRIVATE raw response must never enter the next prompt"
    model = ScriptedModel([StructuredOutputError("schema_mismatch", secret), red(90)])
    result = await run(model)
    assert [call["stage"] for call in model.calls] == ["review.red.0", "review.red.0.format"]
    assert model.calls[0]["payload"] == model.calls[1]["payload"]
    assert secret not in model.calls[1]["request"].model_dump_json()
    assert "schema_mismatch" in model.calls[1]["request"].messages[0].content
    assert result.audit["format_repairs"] == 1


async def test_format_repair_allowance_is_global_across_red_and_blue():
    failure = StructuredOutputError("invalid_json", "format error")
    model = ScriptedModel([failure, red(60), failure])
    with pytest.raises(StructuredOutputError) as captured:
        await run(model)
    assert captured.value is failure
    assert [call["stage"] for call in model.calls] == [
        "review.red.0",
        "review.red.0.format",
        "review.blue.1",
    ]


async def test_blue_can_use_the_single_format_repair_if_red_did_not():
    model = ScriptedModel([red(60), StructuredOutputError("invalid_json", "safe"), blue(), red(90)])
    result = await run(model)
    assert result.audit["calls"] == 4
    assert model.calls[2]["stage"] == "review.blue.1.format"
    assert model.calls[1]["payload"] == model.calls[2]["payload"]


@pytest.mark.parametrize("allowance", [0, 1])
async def test_invalid_repair_is_not_repaired_recursively(allowance):
    error = StructuredOutputError("invalid_json", "invalid")
    model = ScriptedModel([error, error])
    with pytest.raises(StructuredOutputError):
        await run(model, options=ReviewOptions(max_format_repairs=allowance))
    assert len(model.calls) == 1 + allowance


@pytest.mark.parametrize(
    "code",
    ["authentication", "network", "rate_limit", "store_io", "dag_budget", "output_too_large"],
)
async def test_non_format_failures_are_never_retried(code):
    error = ResearchError(code, "safe failure")
    model = ScriptedModel([error])
    with pytest.raises(ResearchError) as captured:
        await run(model)
    assert captured.value is error
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "action",
    [
        red(60, hash_override="0" * 64),
        red(60, findings=[finding(claim_id="C2")]),
        red(60, findings=[finding(evidence_id="E999")]),
    ],
)
async def test_invalid_red_hash_or_reference_is_fatal_not_format_repaired(action):
    events = []
    model = ScriptedModel([action])
    with pytest.raises(ResearchError) as captured:
        await run(model, events=events)
    assert captured.value.code == "review_validation"
    assert len(model.calls) == 1
    assert [event["kind"] for event in events] == ["initial"]


@pytest.mark.parametrize(
    "action", [blue(target="C2"), blue(citations=["E999"]), blue(hash_override="0" * 64)]
)
async def test_invalid_blue_target_citation_or_hash_is_fatal_and_not_audited_as_applied(action):
    events = []
    initial = report()
    model = ScriptedModel([red(50), action])
    with pytest.raises(ResearchError) as captured:
        await run(model, events=events, initial=initial)
    assert captured.value.code == "review_patch"
    assert len(model.calls) == 2
    assert [event["kind"] for event in events] == ["initial", "red"]
    assert initial == report()


@pytest.mark.parametrize(
    ("kind", "call_count"), [("initial", 0), ("red", 1), ("blue", 2), ("stop", 3)]
)
async def test_durable_sink_failure_stops_before_any_subsequent_phase(kind, call_count):
    model = ScriptedModel([red(60), blue(), red(90)])

    def fail(event):
        if event["kind"] == kind:
            raise OSError("private local path")

    with pytest.raises(ResearchError) as captured:
        await run(model, on_event=fail)
    assert captured.value.code == "review_audit_io"
    assert "private local path" not in str(captured.value)
    assert len(model.calls) == call_count


@pytest.mark.parametrize("code", ["store_io", "review_audit_storage"])
async def test_existing_storage_error_is_preserved_by_identity(code):
    error = ResearchError(code, "safe storage category")
    model = ScriptedModel([])

    def fail(event):
        raise error

    with pytest.raises(ResearchError) as captured:
        await run(model, on_event=fail)
    assert captured.value is error
    assert not model.calls


async def test_model_cancellation_propagates_and_no_stop_or_retry_is_written():
    events = []
    model = ScriptedModel([asyncio.CancelledError("cancelled by caller")])
    with pytest.raises(asyncio.CancelledError):
        await run(model, events=events)
    assert len(model.calls) == 1
    assert [event["kind"] for event in events] == ["initial"]


async def test_external_cancellation_drains_active_model_and_does_not_launch_blue():
    entered, cleaned = asyncio.Event(), asyncio.Event()
    calls, events = [], []

    async def model(stage, request, schema):
        calls.append(stage)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    task = asyncio.create_task(run(model, events=events))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert calls == ["review.red.0"]
    assert [event["kind"] for event in events] == ["initial"]


@pytest.mark.parametrize(("kind", "count"), [("initial", 0), ("red", 1), ("blue", 2), ("stop", 3)])
async def test_cancellation_requested_by_sink_is_observed_before_any_new_call(kind, count):
    model = ScriptedModel([red(60), blue(), red(90)])

    def cancel(event):
        if event["kind"] == kind:
            asyncio.current_task().cancel()

    task = asyncio.create_task(run(model, on_event=cancel))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(model.calls) == count


async def test_mutating_callback_event_copy_cannot_change_engine_report_or_audit():
    def mutate(event):
        if "report" in event:
            event["report"]["sections"][0]["claims"][0]["text"] = "corrupted by sink"
        event.clear()

    result = await run(ScriptedModel([red(90)]), on_event=mutate)
    assert result.report == report()
    assert result.audit["status"] == "passed"


async def test_prompts_contain_only_actual_evidence_and_referenced_source_metadata():
    sources, evidence = inputs()
    sources[0]["text"] += " HIDDEN FULL SOURCE TAIL"
    sources[0]["sha256"] = hashlib.sha256(sources[0]["text"].encode()).hexdigest()
    sources.append({**sources[0], "source_id": "S2", "title": "UNSENT SOURCE"})
    model = ScriptedModel([red(90)])
    await review_report(
        "Question",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=model,
        on_event=lambda event: None,
        memory_context={"recalled": [{"evidence_id": "E1", "retrieved_at": "2020-01-01"}]},
    )
    payload = model.calls[0]["payload"]
    serialized = json.dumps(payload)
    assert "HIDDEN FULL SOURCE TAIL" not in serialized
    assert "UNSENT SOURCE" not in serialized
    assert payload["evidence"] == evidence
    assert payload["claim_inventory"][0]["claim_id"] == "C1"
    assert payload["report_hash"] == report_fingerprint(report())
    assert payload["memory_context"]["recalled"][0]["retrieved_at"] == "2020-01-01"
    assert "untrusted" in model.calls[0]["request"].messages[0].content


async def test_prompt_plus_schema_budget_is_checked_before_callback_without_truncation():
    events = []
    model = ScriptedModel([])
    with pytest.raises(ResearchError) as captured:
        await run(model, events=events, memory_context={"note": "中" * 23000})
    assert captured.value.code == "review_context_budget"
    assert model.calls == []
    assert [event["kind"] for event in events] == ["initial"]


async def test_schema_overhead_is_counted_not_just_base_messages():
    # Find a base request just below 64 KiB whose schema cannot fit. The engine
    # must refuse before the injected callback (which may charge real tokens).
    model = ScriptedModel([])
    with pytest.raises(ResearchError) as captured:
        await run(model, memory_context={"note": "x" * 62000})
    assert captured.value.code == "review_context_budget"
    assert not model.calls


@pytest.mark.parametrize(
    "mutation", ["bad_quote", "bad_hash", "unknown_id", "bool_offset", "duplicate", "bad_unicode"]
)
async def test_invalid_input_provenance_fails_before_initial_event_or_call(mutation):
    sources, evidence = inputs()
    if mutation == "bad_quote":
        evidence[0]["quote"] = "changed quote"
    elif mutation == "bad_hash":
        sources[0]["sha256"] = "0" * 64
    elif mutation == "unknown_id":
        evidence[0]["evidence_id"] = "E2"
    elif mutation == "bool_offset":
        evidence[0]["start"] = False
    elif mutation == "duplicate":
        evidence.append(copy.deepcopy(evidence[0]))
    else:
        evidence[0]["quote"] = "\ud800"
    events = []
    model = ScriptedModel([])
    with pytest.raises(ResearchError):
        await review_report(
            "Question",
            report(),
            evidence=evidence,
            sources=sources,
            model_call=model,
            on_event=events.append,
        )
    assert not model.calls
    assert not events


@pytest.mark.parametrize(
    "memory", [{"value": float("nan")}, {"value": object()}, {"value": "\ud800"}, []]
)
async def test_invalid_optional_context_is_rejected_without_model_call(memory):
    model = ScriptedModel([])
    with pytest.raises(ResearchError) as captured:
        await run(model, memory_context=memory)
    assert captured.value.code == "review_input"
    assert not model.calls


async def test_async_sink_and_unvalidated_options_are_rejected():
    async def async_sink(event):
        pass

    for kwargs in ({"on_event": async_sink}, {"options": {"max_rounds": 1}}):
        model = ScriptedModel([])
        with pytest.raises(ResearchError) as captured:
            await run(model, **kwargs)
        assert captured.value.code == "review_configuration"
        assert not model.calls


async def test_invalid_callback_contract_is_fatal_not_a_format_retry():
    calls = []

    async def malformed(stage, request, schema):
        calls.append(stage)
        return {"data": "not a StructuredResult"}

    with pytest.raises(ResearchError) as captured:
        await run(malformed)
    assert captured.value.code == "review_model_contract"
    assert len(calls) == 1


async def test_pending_cancellation_is_not_replaced_by_simultaneous_audit_failure():
    model = ScriptedModel([])

    def failing_cancel(event):
        asyncio.current_task().cancel()
        raise OSError("audit failed during cancellation")

    task = asyncio.create_task(run(model, on_event=failing_cancel))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not model.calls


async def test_envelope_parser_mode_is_recorded_without_spending_format_repair():
    async def envelope(stage, request, schema):
        payload = json.loads(request.messages[-1].content)
        value = structured(red(90)(schema, payload), 1)
        return StructuredResult(data=value.data, response=value.response, parse_mode="envelope")

    events = []
    result = await run(envelope, events=events)
    assert result.audit["calls"] == 1
    assert result.audit["format_repairs"] == 0
    assert result.audit["history"][0]["parse_mode"] == "envelope"
    assert events[1]["parse_mode"] == "envelope"
