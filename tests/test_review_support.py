import pytest
from test_review import inputs, report

from deep_research.errors import ResearchError
from deep_research.gateway import StructuredResult
from deep_research.review import review_report
from deep_research.review_models import ReviewOptions
from deep_research.review_support import SupportRed, project_support
from deep_research.schemas import ModelResponse


def value(verdict="supported", status="answered"):
    return SupportRed(
        scores=dict(evidence_support=95, logical_consistency=95, citation_quality=95),
        checks=[
            dict(
                claim_id="C1",
                verdict=verdict,
                claim_excerpt="Original claim",
                evidence_ids=["E1"],
                evidence_excerpt="Original evidence",
                explanation="Compare the actual claim and evidence",
            )
        ],
        coverage=[
            dict(
                point_id="P1",
                status=status,
                claim_ids=["C1"] if status == "answered" else [],
                explanation="Requested fact",
            )
        ],
        advice=["Optional background"],
    )


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient"])
def test_failed_claim_check_cannot_be_hidden_in_advice_or_high_scores(verdict):
    _, evidence = inputs()
    red, audit = project_support(value(verdict), "What is known?", report(), evidence)
    assert len(red.findings) == 1 and red.findings[0].claim_ids == ["C1"]
    assert audit["version"] == 2 and audit["raw"]["checks"][0]["verdict"] == verdict


def test_missing_answer_is_bound_by_program_to_the_actual_requirement():
    _, evidence = inputs()
    red, _ = project_support(value(status="missing"), "What is known?", report(), evidence)
    assert red.findings[0].category == "coverage" and red.findings[0].claim_ids == []


@pytest.mark.parametrize(
    "kind", ["missing_claim", "duplicate_claim", "unknown_point", "fake_quote", "fake_claim"]
)
def test_incomplete_or_invented_checks_fail_closed(kind):
    _, evidence = inputs()
    v = value()
    if kind == "missing_claim":
        v = v.model_copy(update={"checks": []})
    elif kind == "duplicate_claim":
        v.checks.append(v.checks[0])
    elif kind == "unknown_point":
        v = v.model_copy(update={"coverage": [v.coverage[0].model_copy(update={"point_id": "P8"})]})
    elif kind == "fake_quote":
        v = v.model_copy(
            update={"checks": [v.checks[0].model_copy(update={"evidence_excerpt": "invented"})]}
        )
    else:
        v = v.model_copy(
            update={"checks": [v.checks[0].model_copy(update={"claim_excerpt": "never asserted"})]}
        )
    with pytest.raises(ResearchError):
        project_support(v, "What is known?", report(), evidence)


async def test_complete_ledger_is_requested_and_stored_without_an_extra_model_call():
    sources, evidence = inputs()
    events = []
    calls = []

    async def call(stage, request, schema):
        assert schema is SupportRed
        calls.append(stage)
        return StructuredResult(
            data=value(),
            response=ModelResponse(
                provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
            ),
        )

    result = await review_report(
        "What is known?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=call,
        on_event=events.append,
        scope_bound=True,
        support_check=True,
    )
    assert result.audit["status"] == "passed" and len(calls) == 1
    assert result.audit["scope_review"]["version"] == 2
    assert next(e for e in events if e["kind"] == "red")["scope_review"]["raw"]["checks"]


@pytest.mark.parametrize("allow_repair", [0, 1])
async def test_invalid_anchor_has_a_strict_shared_repair_limit(allow_repair):
    sources, evidence = inputs()
    calls = []

    async def call(stage, request, schema):
        calls.append(stage)
        v = value()
        v = v.model_copy(
            update={
                "checks": [
                    v.checks[0].model_copy(update={"evidence_excerpt": "not ... contiguous"})
                ]
            }
        )
        return StructuredResult(
            data=v,
            response=ModelResponse(
                provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
            ),
        )

    with pytest.raises(ResearchError) as error:
        await review_report(
            "What is known?",
            report(),
            evidence=evidence,
            sources=sources,
            model_call=call,
            on_event=lambda e: None,
            scope_bound=True,
            support_check=True,
            options=ReviewOptions(max_rounds=0, max_format_repairs=allow_repair),
        )
    assert error.value.code == "schema_mismatch" and len(calls) == 1 + allow_repair


async def test_contract_repair_revalidates_witness_and_records_reason():
    sources, evidence = inputs()
    calls = []

    async def call(stage, request, schema):
        calls.append(stage)
        v = value()
        if len(calls) == 1:
            v = v.model_copy(
                update={"checks": [v.checks[0].model_copy(update={"claim_excerpt": "wrong side"})]}
            )
        else:
            assert "no ellipses" in request.messages[0].content
        return StructuredResult(
            data=v,
            response=ModelResponse(
                provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
            ),
        )

    result = await review_report(
        "What is known?",
        report(),
        evidence=evidence,
        sources=sources,
        model_call=call,
        on_event=lambda e: None,
        scope_bound=True,
        support_check=True,
        options=ReviewOptions(max_rounds=0, max_format_repairs=1),
    )
    assert result.audit["status"] == "passed" and result.audit["calls"] == 2
    assert result.audit["format_repairs"] == 1
    assert result.audit["format_repair_errors"] == [
        dict(stage="review.red.0", code="schema_mismatch")
    ]
