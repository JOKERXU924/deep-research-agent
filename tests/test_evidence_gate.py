"""No paid calls: ranking, abstention, immutable projection and DAG ablations."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from test_dag_research import FakeRetriever
from test_review import report
from test_review_research import storage as review_storage

from deep_research.errors import ResearchError, StructuredOutputError
from deep_research.evidence_gate import (
    Ranking,
    Supports,
    assess_support,
    rerank,
    validate_publication,
)

storage = review_storage


@pytest.mark.parametrize("rank,support", [(True, False), (False, True), (True, True)])
def test_new_options_require_live_mode(rank, support):
    from deep_research.web_models import WebRequest

    with pytest.raises(ValueError):
        WebRequest(rerank_evidence=rank, check_support=support)
    value = WebRequest(
        mode="live",
        question="问题？",
        confirm_live=True,
        rerank_evidence=rank,
        check_support=support,
    )
    assert value.rerank_evidence == rank and value.check_support == support


def evidence():
    return [dict(evidence_id="E1", source_id="S1", start=0, end=5, quote="proof")]


async def test_ranking_uses_original_spans_and_bounded_round_robin():
    pool = evidence() + [dict(evidence_id="E2", source_id="S1", start=6, end=12, quote="better")]

    async def call(stage, request, schema):
        assert stage == "evidence.rerank" and schema is Ranking
        return SimpleNamespace(
            data=Ranking(rankings=[dict(point_id="P1", candidates=["R2", "R1"])])
        )

    chosen, audit = await rerank(call, "Question?", pool, pool[:1], max_bytes=1000, max_items=1)
    assert chosen[0]["quote"] == "better" and chosen[0]["start"] == 6
    assert audit["status"] == "completed" and len(audit["candidate_spans"]) == 2


@pytest.mark.parametrize(
    "rows",
    [
        [dict(point_id="P1", candidates=["R999"])],
        [dict(point_id="P1", candidates=["R1", "R1"])],
        [dict(point_id="P2", candidates=["R1"])],
    ],
)
async def test_malformed_ranking_falls_back_without_inventing_evidence(rows):
    async def call(*args):
        return SimpleNamespace(data=Ranking(rankings=rows))

    original = evidence()
    chosen, audit = await rerank(call, "Question?", original, original, max_bytes=1000)
    assert chosen == original and audit["status"] == "fallback"


@pytest.mark.parametrize("verdict", ["supported", "partial", "contradicted", "insufficient"])
async def test_support_gate_projects_only_supported_claims_and_binds_context(verdict):
    async def call(stage, request, schema):
        assert stage == "evidence.support" and schema is Supports
        return SimpleNamespace(
            data=Supports(
                decisions=[dict(claim_id="C1", verdict=verdict, reason="Check original quote")]
            )
        )

    original = report()
    audit, view = await assess_support(call, "Question?", original, evidence())
    bundle = dict(
        question="Question?",
        report=original.model_dump(),
        evidence=evidence(),
        support_assessment=audit,
        publication=view,
    )
    validate_publication(bundle)
    assert view["abstained"] == (verdict != "supported")
    assert len(view["claim_inventory"]) == (1 if verdict == "supported" else 0)
    mutated = copy.deepcopy(bundle)
    mutated["evidence"][0]["quote"] = "other"
    with pytest.raises(ValueError, match="Stale support evidence"):
        validate_publication(mutated)
    mutated = copy.deepcopy(bundle)
    mutated["publication"]["abstained"] = not view["abstained"]
    with pytest.raises(ValueError):
        validate_publication(mutated)


async def test_invalid_support_closes_publication_instead_of_approving_everything():
    async def call(*args):
        raise StructuredOutputError("invalid_json", "Invalid")

    audit, view = await assess_support(call, "Question?", report(), evidence())
    assert audit["status"] == "unavailable" and view["abstained"]
    assert view["report"]["sections"] == []


@pytest.mark.parametrize("error", [asyncio.CancelledError(), ResearchError("cost_audit", "Audit")])
async def test_audit_and_cancel_are_never_swallowed(error):
    async def call(*args):
        raise error

    with pytest.raises(type(error)):
        await assess_support(call, "Question?", report(), evidence())
    with pytest.raises(type(error)):
        await rerank(call, "Question?", evidence(), evidence(), max_bytes=1000)


@pytest.mark.parametrize(
    "rank,support", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("selection", ["coverage", "contextual"])
async def test_four_pipeline_groups_and_export_projection(
    tmp_path, storage, rank, support, supported, selection
):
    from deep_research.dag_research import DagOptions, run_dag_pipeline
    from deep_research.gateway import ModelGateway
    from deep_research.retrieval import SourceDocument
    from deep_research.retrieval_quality import RetrievalQualityOptions
    from deep_research.review_models import ReviewOptions
    from deep_research.schemas import ModelResponse

    class Provider:
        async def complete(self, request):
            payload = json.loads(request.messages[-1].content)
            if "language" in payload:
                value = dict(queries=["initial"])
            elif "candidates" in payload:
                value = dict(
                    rankings=[dict(point_id=p["id"], candidates=["R1"]) for p in payload["points"]]
                )
            elif "requirements" in payload:
                value = dict(
                    answers=[
                        dict(
                            text="光合作用提供能量。",
                            citations=[payload["context"]["evidence"][0]["evidence_id"]],
                        )
                    ]
                )
            elif "claims" in payload:
                value = dict(
                    decisions=[
                        dict(
                            claim_id=c["claim_id"],
                            verdict="supported" if supported else "insufficient",
                            reason="原文支持范围判定",
                        )
                        for c in payload["claims"]
                    ]
                )
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
        quality_options=RetrievalQualityOptions(
            robust_contracts=not (rank or support),
            evidence_selection=selection,
            rerank_evidence=rank,
            check_support=support,
        ),
        review_options=ReviewOptions(max_rounds=1),
        source_mode="local_corpus",
    )
    bundle = storage[0].get_report(result["run_id"])
    assert ("evidence_rerank" in bundle) == rank
    assert ("support_assessment" in bundle) == support
    assert result["model_call_count"] == 3 + rank + support
    text = __import__("pathlib").Path(result["report_path"]).read_text(encoding="utf-8")
    if support:
        assert bundle["publication"]["abstained"] == (not supported)
        assert ("本次弃答" in text) == (not supported)
        assert ("- 光合作用提供能量。" in text) == supported
    assert bundle["report"]["sections"]  # Original candidate remains auditable.
