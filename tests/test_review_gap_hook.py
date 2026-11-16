"""One bounded gap callback; all evidence and model responses are offline fixtures."""

import asyncio
import copy
import hashlib

import pytest
from pydantic import ValidationError
from test_review import ScriptedModel, blue, finding, inputs, red, report

from deep_research.errors import ResearchError
from deep_research.research_store import ResearchStore
from deep_research.review import review_report
from deep_research.review_journal import ReviewJournal
from deep_research.review_models import ReviewFinding, ReviewOptions
from deep_research.structured_diagnostics import validation_diagnostic


def expanded():
    sources, evidence = inputs()
    text = "Fresh saved evidence adds a qualification missing from the initial source."
    sources.append(
        {
            "source_id": "S2",
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "title": "Additional source",
            "url": "https://example.org/additional",
        }
    )
    evidence.append(
        {"evidence_id": "E2", "source_id": "S2", "start": 0, "end": len(text), "quote": text}
    )
    return sources, evidence


async def run(model, callback, *, events=None, options=None, on_event=None):
    sources, evidence = inputs()
    return await review_report(
        "What does the evidence support?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=model,
        on_event=on_event or (events if events is not None else []).append,
        options=options,
        on_evidence_gap=callback,
    )


async def test_supplement_runs_once_between_initial_red_and_blue_and_preserves_old_ids():
    events, calls = [], []

    async def supplement(review):
        calls.append(review.model_dump())
        assert [event["kind"] for event in events] == ["initial", "red"]
        return expanded()

    model = ScriptedModel([red(40, findings=[finding()]), blue(citations=["E2"]), red(90)])
    result = await run(model, supplement, events=events)
    assert len(calls) == 1
    assert [event["kind"] for event in events] == [
        "initial",
        "red",
        "evidence_gap",
        "blue",
        "red",
        "stop",
    ]
    assert events[2]["added_source_ids"] == ["S2"]
    assert events[2]["added_evidence_ids"] == ["E2"]
    assert events[1]["evidence_ids"] == ["E1"]
    assert events[4]["evidence_ids"] == ["E1", "E2"]
    assert result.audit["best_evidence_ids"] == ["E1", "E2"]
    assert model.calls[1]["payload"]["evidence"] == expanded()[1]
    assert model.calls[1]["payload"]["report"] == report().model_dump()
    assert result.audit["calls"] == 3
    assert result.audit["format_repairs"] == 0


async def test_new_evidence_allows_one_reassessment_of_unchanged_report():
    async def supplement(review):
        return expanded()

    model = ScriptedModel([red(50, findings=[finding()]), blue("Original claim"), red(90)])
    result = await run(model, supplement)
    assert result.report == report()
    assert result.audit["best_round"] == 1
    assert result.audit["stop_reason"] == "passed"
    assert len(model.calls) == 3


async def test_reassessment_exemption_does_not_disable_stalling_or_repeat_retrieval():
    calls = []

    async def supplement(review):
        calls.append(1)
        return expanded()

    model = ScriptedModel(
        [red(50, findings=[finding()]), blue("Original claim"), red(60), blue("Original claim")]
    )
    result = await run(model, supplement)
    assert result.audit["stop_reason"] == "stalled"
    assert len(model.calls) == 4
    assert calls == [1]


async def test_best_scores_and_findings_retain_their_actual_evidence_snapshot():
    async def supplement(review):
        return expanded()

    model = ScriptedModel([red(70, findings=[finding()]), blue(), red(20)])
    result = await run(model, supplement, options=ReviewOptions(max_rounds=1))
    assert result.audit["best_round"] == 0
    assert result.audit["best_evidence_ids"] == ["E1"]
    assert result.audit["final_scores"]["evidence_support"] == 70
    assert result.audit["history"][1]["evidence_ids"] == ["E1", "E2"]


@pytest.mark.parametrize("case", ["passed", "no_findings", "logic", "citation", "zero_rounds"])
async def test_no_gap_for_passing_no_gap_findings_or_no_repair_rounds(case):
    item = finding().model_dump()
    if case in ("logic", "citation"):
        item["category"] = case
    findings = [] if case in ("passed", "no_findings") else [ReviewFinding(**item)]
    actions = [red(90 if case == "passed" else 60, findings=findings)]
    rounds = 0 if case == "zero_rounds" else 1
    if case != "passed" and rounds:
        actions += [blue("Original claim")]

    async def unexpected(review):
        pytest.fail("This review must not perform gap retrieval")

    result = await run(ScriptedModel(actions), unexpected, options=ReviewOptions(max_rounds=rounds))
    assert result.audit["calls"] == len(actions)


async def test_coverage_gap_with_no_claim_target_can_request_a_supplement():
    item = finding().model_dump()
    item.update(category="coverage", claim_ids=[], search_queries=["Calvin cycle"])
    received = []

    async def supplement(review):
        received.extend(review.findings[0].search_queries)
        return expanded()

    await run(
        ScriptedModel([red(40, findings=[ReviewFinding(**item)]), blue(), red(90)]), supplement
    )
    assert received == ["Calvin cycle"]


@pytest.mark.parametrize("unchanged", [None, "same", "sources_only"])
async def test_no_useful_supplement_keeps_normal_stall_and_no_gap_event(unchanged):
    events = []

    async def supplement(review):
        if unchanged is None:
            return None
        if unchanged == "same":
            return inputs()
        return expanded()[0], inputs()[1]

    result = await run(
        ScriptedModel([red(60, findings=[finding()]), blue("Original claim")]),
        supplement,
        events=events,
    )
    assert result.audit["stop_reason"] == "stalled"
    assert [event["kind"] for event in events] == ["initial", "red", "blue", "stop"]
    assert result.audit["best_evidence_ids"] == ["E1"]


@pytest.mark.parametrize(
    "mutation", ["old_quote", "old_title", "old_source_id", "remove_old", "reorder", "duplicate"]
)
async def test_changed_removed_reordered_or_duplicate_old_snapshots_fail_before_blue(mutation):
    sources, evidence = expanded()
    if mutation == "old_quote":
        # Internally valid but no longer the source/quote originally reviewed.
        sources[0]["text"] = "Altered original."
        sources[0]["sha256"] = hashlib.sha256(sources[0]["text"].encode()).hexdigest()
        evidence[0].update(quote=sources[0]["text"], end=len(sources[0]["text"]))
    elif mutation == "old_title":
        sources[0]["title"] = "Changed old provenance"
    elif mutation == "old_source_id":
        sources[0]["source_id"] = evidence[0]["source_id"] = "changed"
    elif mutation == "remove_old":
        sources.pop(0)
        evidence.pop(0)
    elif mutation == "reorder":
        sources.reverse()
        evidence.reverse()
    else:
        evidence.append(copy.deepcopy(evidence[0]))

    async def supplement(review):
        return sources, evidence

    events = []
    model = ScriptedModel([red(60, findings=[finding()])])
    with pytest.raises(ResearchError):
        await run(model, supplement, events=events)
    assert len(model.calls) == 1
    assert [event["kind"] for event in events] == ["initial", "red"]


@pytest.mark.parametrize("invalid", ["bad_hash", "bad_quote", "too_many_sources", "too_many_items"])
async def test_added_evidence_must_validate_provenance_and_size_limits(invalid):
    sources, evidence = expanded()
    if invalid == "bad_hash":
        sources[-1]["sha256"] = "0" * 64
    elif invalid == "bad_quote":
        evidence[-1]["quote"] = "Not in the saved source"
    elif invalid == "too_many_sources":
        sources.extend(dict(sources[-1], source_id=f"S{i}") for i in range(3, 14))
    else:
        evidence.extend(dict(evidence[-1], evidence_id=f"E{i}") for i in range(3, 34))

    async def supplement(review):
        return sources, evidence

    model = ScriptedModel([red(60, findings=[finding()])])
    with pytest.raises(ResearchError):
        await run(model, supplement)
    assert len(model.calls) == 1


async def test_callback_cannot_mutate_red_used_by_following_blue():
    async def supplement(review):
        review.findings.clear()
        return expanded()

    model = ScriptedModel([red(60, findings=[finding()]), blue(), red(90)])
    await run(model, supplement)
    assert len(model.calls[1]["payload"]["red_review"]["findings"]) == 1


async def test_audit_failure_prevents_using_supplement_or_launching_blue():
    async def supplement(review):
        return expanded()

    def sink(event):
        if event["kind"] == "evidence_gap":
            raise OSError("private path must never leak")

    model = ScriptedModel([red(60, findings=[finding()])])
    with pytest.raises(ResearchError) as caught:
        await run(model, supplement, on_event=sink)
    assert caught.value.code == "review_audit_io"
    assert "private" not in str(caught.value)
    assert len(model.calls) == 1


async def test_external_cancellation_drains_gap_without_blue_or_stop_event():
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def supplement(review):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    events = []
    model = ScriptedModel([red(60, findings=[finding()])])
    task = asyncio.create_task(run(model, supplement, events=events))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert len(model.calls) == 1
    assert [event["kind"] for event in events] == ["initial", "red"]


@pytest.mark.parametrize("pending_cancel", [False, True])
async def test_callback_failure_is_safe_and_preserves_pending_cancellation(pending_cancel):
    async def supplement(review):
        if pending_cancel:
            asyncio.current_task().cancel()
        raise OSError("private remote details")

    model = ScriptedModel([red(60, findings=[finding()])])
    task = asyncio.create_task(run(model, supplement))
    with pytest.raises(asyncio.CancelledError if pending_cancel else ResearchError) as caught:
        await task
    if not pending_cancel:
        assert caught.value.code == "review_gap_io"
        assert "private" not in str(caught.value)
    assert len(model.calls) == 1


async def test_research_error_from_budget_or_audit_callback_keeps_original_identity():
    failure = ResearchError("store_io", "Safe failure category")

    async def supplement(review):
        raise failure

    with pytest.raises(ResearchError) as caught:
        await run(ScriptedModel([red(60, findings=[finding()])]), supplement)
    assert caught.value is failure


async def test_global_deadline_from_gap_callback_keeps_timeout_identity_and_stops_review():
    failure = TimeoutError("global deadline")

    async def supplement(review):
        raise failure

    events = []
    model = ScriptedModel([red(60, findings=[finding()])])
    with pytest.raises(TimeoutError) as caught:
        await run(model, supplement, events=events)
    assert caught.value is failure
    assert len(model.calls) == 1
    assert [event["kind"] for event in events] == ["initial", "red"]


async def test_pending_external_cancel_is_not_changed_to_callback_timeout():
    async def supplement(review):
        asyncio.current_task().cancel()
        raise TimeoutError("deadline and cancellation")

    model = ScriptedModel([red(60, findings=[finding()])])
    task = asyncio.create_task(run(model, supplement))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(model.calls) == 1


@pytest.mark.parametrize("result", [[], {}, ([{}],), [[], []], "invalid"])
async def test_invalid_callback_result_is_not_retried(result):
    async def supplement(review):
        return result

    model = ScriptedModel([red(60, findings=[finding()])])
    with pytest.raises(ResearchError) as caught:
        await run(model, supplement)
    assert caught.value.code == "review_gap_contract"
    assert len(model.calls) == 1


async def test_noncallable_and_synchronous_callback_contracts_are_rejected():
    model = ScriptedModel([])
    with pytest.raises(ResearchError, match="callable"):
        await run(model, 7)
    assert not model.calls
    model = ScriptedModel([red(60, findings=[finding()])])
    with pytest.raises(ResearchError) as caught:
        await run(model, lambda review: expanded())
    assert caught.value.code == "review_gap_contract"
    assert len(model.calls) == 1


def test_old_findings_remain_valid_and_queries_are_optional_plain_bounded_text():
    assert finding().search_queries == []
    value = finding().model_dump()
    value["search_queries"] = ["  Calvin cycle  ", "光反应"]
    assert ReviewFinding(**value).search_queries == ["Calvin cycle", "光反应"]


@pytest.mark.parametrize(
    "queries",
    [
        [""],
        [" "],
        ["x" * 121],
        ["a", "b", "c"],
        ["a\n"],
        ["a\tb"],
        ["a\x00b"],
        ["a\u202eb"],
        ["\ud800"],
        ["https://example.org"],
        ["WWW.example.org"],
    ],
)
def test_invalid_suggested_queries_fail_strict_validation_without_echoing_content(queries):
    value = finding().model_dump()
    value["search_queries"] = queries
    with pytest.raises(ValidationError) as caught:
        ReviewFinding(**value)
    diagnostic = validation_diagnostic(
        "RedReview",
        [
            dict(item, loc=("findings", 0, *item["loc"]))
            for item in caught.value.errors(include_input=False)
        ],
    )
    assert all(
        item["path"].startswith("findings.*.search_queries") for item in diagnostic["issues"]
    )


async def test_gap_event_survives_durable_journal_reopen(tmp_path):
    path = tmp_path / "review.sqlite"

    async def supplement(review):
        return expanded()

    with ResearchStore(path) as store, ReviewJournal(path) as journal:
        run_id = store.create_run("Offline gap test")
        await run(
            ScriptedModel([red(60, findings=[finding()]), blue(), red(90)]),
            supplement,
            on_event=lambda event: journal.append(run_id, event),
        )
    with ReviewJournal(path) as reopened:
        gap = [
            row["event"]
            for row in reopened.snapshot(run_id)
            if row["event"]["kind"] == "evidence_gap"
        ]
    assert len(gap) == 1
    assert gap[0]["added_evidence_ids"] == ["E2"]
