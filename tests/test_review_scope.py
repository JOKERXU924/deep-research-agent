import pytest
from test_review import inputs, report

from deep_research.errors import ResearchError
from deep_research.gateway import StructuredResult
from deep_research.review import review_report
from deep_research.review_models import ReviewOptions
from deep_research.review_scope import ScopedRed, project
from deep_research.schemas import ModelResponse


def value(**kwargs):
    return ScopedRed(
        scores=dict(evidence_support=90, logical_consistency=90, citation_quality=90), **kwargs
    )


def claim_issue(**kwargs):
    return dict(
        kind="claim_error",
        category="evidence",
        severity="low",
        claim_id="C1",
        claim_excerpt="Original claim",
        evidence_ids=["E1"],
        evidence_excerpt="Original evidence",
        description="The claim is stronger than its evidence",
        **kwargs,
    )


def test_advice_is_saved_without_becoming_a_blocking_finding():
    sources, evidence = inputs()
    red, audit = project(
        value(advice=["Consider additional background"]), "What is known?", report(), evidence
    )
    assert red.findings == [] and audit["advice"] == ["Consider additional background"]
    assert audit["report_hash"] and audit["binding_hash"]


def test_low_severity_real_claim_error_still_blocks():
    _, evidence = inputs()
    red, audit = project(value(findings=[claim_issue()]), "What is known?", report(), evidence)
    assert len(red.findings) == 1 and red.findings[0].severity == "low"
    assert len(audit["issues"]) == 1


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("claim_excerpt", "zero overhead"),
        ("evidence_excerpt", "invented quotation"),
        ("claim_id", "C2"),
        ("evidence_ids", ["E2"]),
        ("point_id", "P8"),
    ],
)
def test_invalid_witness_does_not_become_a_pass(field, replacement):
    _, evidence = inputs()
    f = claim_issue()
    f[field] = replacement
    with pytest.raises(ResearchError, match="审查问题"):
        project(value(findings=[f]), "What is known?", report(), evidence)


def test_missing_answer_must_name_an_existing_full_requirement():
    from deep_research.robust_contracts import research_requirements

    _, evidence = inputs()
    q = "What is known?"
    f = dict(
        kind="missing_answer",
        severity="low",
        point_id="P1",
        requirement=research_requirements(q)[0]["question"],
        description="Required answer absent",
    )
    red, _ = project(value(findings=[f]), q, report(), evidence)
    assert red.findings[0].category == "coverage"
    f["requirement"] = "Explain unrelated memory and version history"
    with pytest.raises(ResearchError):
        project(value(findings=[f]), q, report(), evidence)


async def test_scoped_review_stops_without_blue_for_advice_and_saves_journal_fields():
    sources, evidence = inputs()
    calls = []
    events = []

    async def call(stage, request, schema):
        calls.append(stage)
        assert schema is ScopedRed and "requirements" in request.messages[-1].content
        response = ModelResponse(
            provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
        )
        return StructuredResult(data=value(advice=["Optional topic"]), response=response)

    result = await review_report(
        "What is known?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=call,
        on_event=events.append,
        scope_bound=True,
        options=ReviewOptions(),
    )
    assert result.audit["status"] == "passed" and calls == ["review.red.0"]
    assert result.audit["scope_review"]["snapshots"][0]["advice"] == ["Optional topic"]
    assert events[1]["kind"] == "red" and events[1]["scope_review"]["raw"]["advice"]


async def test_audit_failure_and_cancel_are_not_hidden():
    import asyncio

    sources, evidence = inputs()

    async def call(*args):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await review_report(
            "Question?",
            report(),
            evidence=evidence,
            sources=sources,
            model_call=call,
            on_event=lambda event: None,
            scope_bound=True,
        )

    def fail(event):
        raise OSError("audit unavailable")

    with pytest.raises(ResearchError):
        await review_report(
            "Question?",
            report(),
            evidence=evidence,
            sources=sources,
            model_call=call,
            on_event=fail,
            scope_bound=True,
        )


async def test_empty_blue_is_stalled_and_never_clears_an_existing_finding():
    sources, evidence = inputs()
    events = []

    async def call(stage, request, schema):
        if "blue" in stage:
            raise ResearchError("review_no_change", "No edits")
        response = ModelResponse(
            provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
        )
        return StructuredResult(data=value(findings=[claim_issue()]), response=response)

    result = await review_report(
        "Question?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=call,
        on_event=events.append,
        scope_bound=True,
        options=ReviewOptions(max_rounds=1, max_format_repairs=0),
    )
    assert result.report == report() and result.audit["status"] == "unresolved"
    assert result.audit["stop_reason"] == "stalled" and result.audit["calls"] == 2
    assert next(e for e in events if e["kind"] == "blue")["status"] == "no_change"


async def test_empty_blue_with_new_evidence_still_gets_a_fresh_red_review():
    import json

    sources, evidence = inputs()
    events = []
    calls = []

    async def supplement(red):
        return sources, [*evidence, {**evidence[0], "evidence_id": "E2"}]

    async def call(stage, request, schema):
        calls.append(stage)
        if "blue" in stage:
            raise ResearchError("review_no_change", "No edits")
        payload = json.loads(request.messages[-1].content)
        fresh = len(payload["evidence"]) == 2
        response = ModelResponse(
            provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
        )
        return StructuredResult(
            data=value(findings=[] if fresh else [claim_issue()]), response=response
        )

    result = await review_report(
        "Question?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=call,
        on_event=events.append,
        on_evidence_gap=supplement,
        scope_bound=True,
        options=ReviewOptions(max_rounds=1, max_format_repairs=0),
    )
    assert calls == ["review.red.0", "review.blue.1", "review.red.1"]
    assert result.audit["status"] == "passed" and result.report == report()
    assert result.audit["scope_review"]["selected_round"] == 1
    assert not result.audit["scope_review"]["issues"][0]["active_in_selected_report"]
    assert len(result.audit["scope_review"]["issues"][0]["observations"]) == 1


async def test_advice_without_findings_does_not_bypass_the_score_threshold():
    sources, evidence = inputs()

    async def call(stage, request, schema):
        response = ModelResponse(
            provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
        )
        return StructuredResult(
            data=ScopedRed(
                scores=dict(evidence_support=79, logical_consistency=90, citation_quality=90),
                advice=["Optional background"],
            ),
            response=response,
        )

    result = await review_report(
        "Question?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=call,
        on_event=lambda e: None,
        scope_bound=True,
        options=ReviewOptions(max_rounds=0),
    )
    assert result.audit["status"] == "unresolved" and result.audit["finding_count"] == 0
