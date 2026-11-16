"""Compact contracts retain valid rows and fail closed on references and audit errors."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_dag_research import FakeRetriever
from test_review import ScriptedModel, finding, red, report, run
from test_review_research import storage as review_storage

from deep_research.errors import ResearchError, StructuredOutputError
from deep_research.review_models import ReviewOptions
from deep_research.robust_contracts import (
    Answer,
    LeanBlue,
    LeanRed,
    PointAnswers,
    ShortQueries,
    assemble_report,
    bounded_query,
    canonical_review,
    generate_queries,
    generate_report,
    partial_answers,
    partial_queries,
    repair_capacity,
    research_requirements,
)
from deep_research.schemas import ChatMessage, ModelRequest

storage = review_storage


def test_real_technical_phrase_is_not_rejected_by_arbitrary_twenty_char_limit():
    result = ShortQueries(queries=[["Serializable", "serialization failure", "事务重试"]])
    assert bounded_query(result.queries[0]) == "Serializable serialization failure 事务重试"
    assert bounded_query(["x" * 64, "y" * 64, "WAL"]) == "x" * 64 + " WAL"


def test_web_requires_explicit_live_mode_for_robust_contracts():
    from deep_research.web_models import WebRequest

    with pytest.raises(ValueError):
        WebRequest(robust_contracts=True)
    request = WebRequest(mode="live", question="问题？", confirm_live=True, robust_contracts=True)
    assert request.robust_contracts is True


@pytest.mark.parametrize("term", ["https://example.org", "www.example.org", "bad\nquery"])
def test_unsafe_query_is_rejected_even_when_it_would_not_fit(term):
    with pytest.raises(ValueError):
        bounded_query(["x" * 64, "y" * 50, term])
    with pytest.raises(ValueError):
        ShortQueries(queries=[[term]])


def audit():
    return dict(repairs=0, repair_limit=1, report_attempts=[], query_fallback_point_ids=[])


def test_program_owns_structure_and_counts():
    points = [dict(id=f"P{i + 1}", question=f"Requirement {i}?") for i in range(8)]
    answers = {i: Answer(text=f"Answer {i}", citations=["E1"]) for i in range(8)}
    result = assemble_report("Question", points, answers)
    assert len(result.sections) == 4
    assert [len(s.claims) for s in result.sections] == [2, 2, 2, 2]
    assert repair_capacity(result)["remaining_total"] == 4
    assert assemble_report("Question", points, {}) is None


def test_only_valid_rows_survive_and_format_instructions_are_not_questions():
    raw = json.dumps(
        dict(
            answers=[
                dict(text="Valid", citations=["E1"]),
                dict(text="Bad", citations=["E1"], limitations_note="extra"),
            ]
        )
    )
    count, rows = partial_answers(raw)
    assert count == 2 and list(rows) == [0]
    assert partial_answers('{"answers": [], "extra": 1}') is None
    assert len(research_requirements("甲是什么？乙是什么？每点分别回答并引用证据。")) == 2


@pytest.mark.parametrize("invalid", ["E999", "duplicate", "schema"])
async def test_repairs_only_invalid_point_and_keeps_valid_answers(invalid):
    calls = []
    state = audit()

    async def call(stage, request, schema):
        payload = json.loads(request.messages[-1].content)
        calls.append(payload)
        assert schema is PointAnswers
        if len(calls) == 1:
            rows = [
                dict(text="Keep me", citations=["E1"]),
                dict(text="Invalid", citations=["E999"] if invalid == "E999" else ["E1", "E1"]),
            ]
            if invalid == "schema":
                rows[1]["extra"] = "invalid"
                error = StructuredOutputError("schema_mismatch", "Invalid row")
                error.partial_answers = partial_answers(json.dumps(dict(answers=rows)))
                raise error
            return SimpleNamespace(data=PointAnswers(answers=rows))
        return SimpleNamespace(data=PointAnswers(answers=[dict(text="Repaired", citations=["E1"])]))

    result = await generate_report(
        call,
        "甲是什么？乙是什么？",
        dict(evidence=[dict(evidence_id="E1")]),
        "report",
        state,
        lambda: state["repairs"] < 1,
    )
    assert [c.text for s in result.sections for c in s.claims] == ["Keep me", "Repaired"]
    assert len(calls[1]["requirements"]) == 1
    assert calls[1]["requirements"][0]["id"] == "P2"
    assert state["repairs"] == 1 and state["report_attempts"][0]["repaired"]


async def test_unrepaired_rewrite_retains_prior_valid_draft():
    state = audit()

    async def call(*args):
        raise StructuredOutputError("invalid_json", "Invalid")

    original = report()
    result = await generate_report(
        call,
        "问题？",
        dict(evidence=[]),
        "directed.report",
        state,
        lambda: state["repairs"] < 1,
        fallback=original,
    )
    assert result is original and state["repairs"] == 1
    assert state["report_attempts"][0]["retained_draft"] is True
    assert state["report_attempts"][0]["unresolved_point_ids"] == ["P1"]


@pytest.mark.parametrize("error", [asyncio.CancelledError(), ResearchError("cost_audit", "Audit")])
async def test_report_never_swallows_cancel_or_audit_failure(error):
    async def call(*args):
        raise error

    with pytest.raises(type(error)):
        await generate_report(call, "Question?", dict(evidence=[]), "report", audit(), lambda: True)


async def test_query_repair_is_local_and_program_binds_point_ids():
    state, calls = audit(), []
    targets = [dict(point_id="P1", query="first"), dict(point_id="P2", query="second")]
    request = ModelRequest(messages=(ChatMessage(role="user", content='{"targets": []}'),))

    async def call(stage, request, schema):
        calls.append(json.loads(request.messages[-1].content))
        assert schema is ShortQueries
        if len(calls) == 1:
            error = StructuredOutputError("schema_mismatch", "Too long")
            error.partial_queries = partial_queries(
                json.dumps(dict(queries=[["WAL"], ["x" * 121]]))
            )
            raise error
        return SimpleNamespace(data=ShortQueries(queries=[["checkpoint", "reader"]]))

    result = await generate_queries(call, request, targets, state, lambda: state["repairs"] < 1)
    assert result == [
        dict(point_id="P1", query="WAL"),
        dict(point_id="P2", query="checkpoint reader"),
    ]
    assert calls[1]["targets"] == [targets[1]]


async def test_query_failure_uses_audited_original_hint_without_more_calls():
    async def call(*args):
        raise StructuredOutputError("invalid_json", "Invalid")

    state = audit()
    targets = [dict(point_id="P1", query="original keywords")]
    request = ModelRequest(messages=(ChatMessage(role="user", content="{}"),))
    result = await generate_queries(call, request, targets, state, lambda: False)
    assert result == targets and state["query_fallback_point_ids"] == ["P1"]


def test_canonical_review_owns_hash_ids_and_preserves_strict_canonical_schema():
    wire = LeanRed(
        scores=dict(evidence_support=50, logical_consistency=50, citation_quality=50),
        findings=[
            dict(
                category="coverage",
                severity="medium",
                description="Missing",
                query_keywords=["WAL", "checkpoint"],
            )
        ],
    )
    canonical = canonical_review(wire, "a" * 64)
    assert canonical.report_hash == "a" * 64
    assert canonical.findings[0].id == "F1"
    assert canonical.findings[0].search_queries == ["WAL checkpoint"]
    blue = canonical_review(
        LeanBlue(edits=[dict(target_id="C1", text="Fixed", citations=["E1"])]), "b" * 64
    )
    assert blue.report_hash == "b" * 64 and blue.operations[0].op == "MODIFY"


async def test_failed_blue_retains_reviewed_draft_but_never_passes():
    model = ScriptedModel(
        [
            red(findings=[finding()]),
            StructuredOutputError("invalid_json", "Invalid"),
            StructuredOutputError("invalid_json", "Invalid"),
        ]
    )
    result = await run(model, options=ReviewOptions(robust_repairs=True, max_rounds=1))
    assert result.report == report()
    assert result.audit["status"] == "unresolved"
    assert result.audit["stop_reason"] == "repair_failed"
    assert result.audit["draft_retained"] is True
    assert model.calls[1]["payload"]["repair_capacity"]["remaining_total"] == 11


@pytest.mark.parametrize("error", [asyncio.CancelledError(), ResearchError("storage", "Audit")])
async def test_review_does_not_hide_cancel_or_audit_error(error):
    model = ScriptedModel([red(findings=[finding()]), error])
    with pytest.raises(type(error)):
        await run(model, options=ReviewOptions(robust_repairs=True, max_rounds=1))


@pytest.mark.parametrize("broken_red", [False, True])
async def test_pipeline_roundtrip_and_first_review_failure_retains_draft(
    tmp_path, storage, broken_red
):
    from deep_research.dag_research import DagOptions, run_dag_pipeline
    from deep_research.gateway import ModelGateway
    from deep_research.retrieval import SourceDocument
    from deep_research.retrieval_quality import RetrievalQualityOptions
    from deep_research.schemas import ModelResponse

    class Provider:
        async def complete(self, request):
            payload = json.loads(request.messages[-1].content)
            if "language" in payload:
                value = dict(queries=["initial"])
            elif "requirements" in payload:
                value = dict(answers=[dict(text="光合作用提供能量。", citations=["E1"])])
            elif broken_red:
                value = dict(invalid=True)
            else:
                value = dict(
                    scores=dict(evidence_support=90, logical_consistency=90, citation_quality=90),
                    findings=[],
                )
            return ModelResponse(
                provider="fixture",
                model="fixture",
                request_id="fixture",
                content=json.dumps(value, ensure_ascii=False),
                latency_ms=0,
            )

    result = await run_dag_pipeline(
        "光合作用提供什么？",
        gateway=ModelGateway(Provider()),
        retriever=FakeRetriever(
            {
                "initial": [
                    SourceDocument("https://example.org/light", "光合作用", "光合作用提供能量。")
                ]
            }
        ),
        store=storage[0],
        journal=storage[1],
        review_journal=storage[2],
        output_dir=tmp_path / "reports",
        options=DagOptions(max_replans=0),
        quality_options=RetrievalQualityOptions(robust_contracts=True),
        review_options=ReviewOptions(max_rounds=1),
        source_mode="local_corpus",
    )
    saved = storage[0].get_report(result["run_id"])
    assert saved["report"]["sections"][0]["claims"][0]["text"] == "光合作用提供能量。"
    assert saved["review"]["status"] == ("unresolved" if broken_red else "passed")
    assert saved["contract_recovery"]["repairs"] == 0
    assert saved["research_map"]["report_hash"]
