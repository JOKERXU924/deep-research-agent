import hashlib
import json

import pytest
from pydantic import ValidationError
from test_review import inputs, report

from deep_research.errors import ResearchError, StructuredOutputError
from deep_research.gateway import StructuredResult
from deep_research.review import review_report
from deep_research.review_models import ReviewOptions
from deep_research.review_references import project_references, reference_schema
from deep_research.schemas import ModelResponse
from deep_research.structured import parse_structured


def raw(verdict="supported", reason="entailed"):
    return dict(
        scores=dict(evidence_support=95, logical_consistency=95, citation_quality=95),
        checks=dict(C1=dict(evidence_ids=["E1"], reason=reason, verdict=verdict)),
        coverage=dict(P1=dict(claim_ids=["C1"], status="answered")),
    )


@pytest.mark.parametrize(
    "verdict,reason,findings",
    [
        ("supported", "entailed", 0),
        ("contradicted", "opposite", 1),
        ("insufficient", "not_established", 1),
    ],
)
def test_id_only_decisions_bind_full_original_text_without_copying(verdict, reason, findings):
    _, evidence = inputs()
    evidence[0]["quote"] = "证据原文。" * 100
    evidence[0]["end"] = len(evidence[0]["quote"])
    schema = reference_schema("What is known?", report())
    value = schema.model_validate(raw(verdict, reason))
    red, audit = project_references(value, "What is known?", report(), evidence)
    assert len(red.findings) == findings and audit["version"] == 3
    assert (
        audit["reference_catalogue"][0]["quote_sha256"]
        == hashlib.sha256(evidence[0]["quote"].encode()).hexdigest()
    )
    assert "quote" not in json.dumps(value.model_dump())


@pytest.mark.parametrize("field", ["missing", "extra", "copy"])
def test_fixed_object_keys_close_missing_and_extra_fields(field):
    r = raw()
    if field == "missing":
        r["checks"] = {}
    elif field == "extra":
        r["checks"]["C2"] = r["checks"]["C1"]
    else:
        r["checks"]["C1"]["evidence_excerpt"] = "not allowed"
    with pytest.raises(ValidationError):
        reference_schema("What is known?", report()).model_validate(r)


@pytest.mark.parametrize("kind", ["unknown", "duplicate", "inconsistent", "fake_conflict"])
def test_invalid_reference_or_reason_never_becomes_a_pass(kind):
    _, evidence = inputs()
    r = raw()
    if kind == "unknown":
        r["checks"]["C1"]["evidence_ids"] = ["E9"]
    elif kind == "duplicate":
        r["checks"]["C1"]["evidence_ids"] = ["E1", "E1"]
    elif kind == "fake_conflict":
        r["checks"]["C1"].update(verdict="insufficient", reason="source_conflict")
    else:
        r["checks"]["C1"]["reason"] = "opposite"
    value = reference_schema("What is known?", report()).model_validate(r)
    with pytest.raises(ResearchError):
        project_references(value, "What is known?", report(), evidence)


def test_duplicate_json_keys_are_rejected_before_keyed_model_parsing():
    data = json.dumps(raw())
    data = data.replace('"checks": {', '"checks": {"C1": {},', 1)
    with pytest.raises(StructuredOutputError):
        parse_structured(data, reference_schema("What is known?", report()))


async def test_reference_review_requests_no_excerpts_and_saves_catalogue():
    sources, evidence = inputs()
    events = []

    async def call(stage, request, schema):
        assert "Do NOT copy" in request.messages[0].content
        return StructuredResult(
            data=schema.model_validate(raw()),
            response=ModelResponse(
                provider="fixture", model="fixture", request_id="1", content="{}", latency_ms=0
            ),
        )

    result = await review_report(
        "What is known?",
        report(),
        sources=sources,
        evidence=evidence,
        model_call=call,
        on_event=events.append,
        scope_bound=True,
        support_check=True,
        reference_check=True,
        options=ReviewOptions(max_rounds=0),
    )
    assert result.audit["status"] == "passed" and result.audit["scope_review"]["version"] == 3
    assert events[1]["scope_review"]["witnesses"][0]["claim_id"] == "C1"
