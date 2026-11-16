import asyncio
import json
from types import SimpleNamespace

import pytest
from test_dag_research import FakeRetriever
from test_review import report
from test_review_research import storage as review_storage

from deep_research.errors import ResearchError
from deep_research.research_loop import (
    LocalRepair,
    ResearchAssessment,
    assess,
    repair,
    rewrite_plan,
    validate_bundle,
)
from deep_research.research_models import CitedClaim, ReportSection, ResearchReport

storage = review_storage
QUESTION = "写者并发限制是什么？读者会阻塞写者吗？"


def original():
    return ResearchReport(
        title="Research",
        sections=[
            ReportSection(
                heading="Answers",
                claims=[
                    CitedClaim(text="Only one writer can write at a time.", citations=["E1"]),
                    CitedClaim(text="现有证据不足，无法回答。", citations=["E1"]),
                ],
            )
        ],
        limitations=["Limited evidence"],
    )


def assessment():
    return ResearchAssessment(
        points=[
            dict(point_id="P1", status="answered", claim_ids=["C1"]),
            dict(
                point_id="P2",
                status="abstained",
                claim_ids=["C2"],
                gap_terms=["reader", "concurrency"],
            ),
        ],
        claims=[
            dict(claim_id="C1", kind="fact", support="supported", reason="Direct quote"),
            dict(claim_id="C2", kind="abstention", support="not_applicable", reason="No answer"),
        ],
    )


async def test_refusal_is_neither_supported_fact_nor_answered_point():
    async def call(*args):
        return SimpleNamespace(data=assessment())

    evidence = [dict(evidence_id="E1", quote="Only one writer", source_id="S1", start=0, end=15)]
    audit, view = await assess(call, QUESTION, original(), evidence)
    assert len(view["claim_inventory"]) == 1 and len(view["nonfactual"]) == 1
    assert [p["status"] for p in view["points"]] == ["answered", "abstained"]
    bundle = dict(
        question=QUESTION,
        report=original().model_dump(),
        evidence=evidence,
        support_assessment=audit,
        publication=view,
    )
    validate_bundle(bundle)
    bundle["evidence"][0]["quote"] = "changed"
    with pytest.raises(ValueError):
        validate_bundle(bundle)


@pytest.mark.parametrize("target", ["C1", "C2"])
async def test_local_repair_preserves_correct_claim_and_rejects_wrong_target(target):
    async def call(*args):
        return SimpleNamespace(
            data=LocalRepair(
                edits=[
                    dict(target_id=target, text="Readers do not block writers.", citations=["E2"])
                ]
            )
        )

    audit = dict(
        points=[p.model_dump() for p in assessment().points],
        claims=[c.model_dump() for c in assessment().claims],
    )
    revised, receipt = await repair(
        call, QUESTION, original(), [dict(evidence_id="E1"), dict(evidence_id="E2")], audit
    )
    assert revised.sections[0].claims[0] == original().sections[0].claims[0]
    assert receipt["status"] == ("candidate" if target == "C2" else "retained")


async def test_real_empty_patch_replay_retains_original():
    # Real pilot returned all three arrays empty. It is a valid no-op, not a crash.
    async def call(*args):
        return SimpleNamespace(data=LocalRepair(edits=[], additions=[], removals=[]))

    audit = dict(
        points=[p.model_dump() for p in assessment().points],
        claims=[c.model_dump() for c in assessment().claims],
    )
    revised, receipt = await repair(call, QUESTION, original(), [], audit)
    assert revised == original()
    assert receipt["status"] == "unchanged" and receipt["operations"] == []


@pytest.mark.parametrize("error", [asyncio.CancelledError(), ResearchError("cost_audit", "Audit")])
async def test_loop_never_hides_cancel_or_cost_audit_error(error):
    async def call(*args):
        raise error

    with pytest.raises(type(error)):
        await assess(call, "Question?", report(), [])
    with pytest.raises(type(error)):
        await rewrite_plan(call, "Question?")


@pytest.mark.parametrize("rerank", [False, True])
@pytest.mark.parametrize("failure", ["none", "deadline", "recheck"])
@pytest.mark.parametrize("strict", [False, True])
async def test_complete_loop_gap_repair_reassessment_and_storage(
    tmp_path, storage, rerank, failure, strict
):
    from deep_research.dag_research import DagOptions, run_dag_pipeline
    from deep_research.gateway import ModelGateway
    from deep_research.retrieval import SourceDocument
    from deep_research.retrieval_quality import RetrievalQualityOptions
    from deep_research.review_models import ReviewOptions
    from deep_research.schemas import ModelResponse

    first = SourceDocument(
        "https://example.org/writer", "Writer", "Only one writer can write at a time."
    )
    extra = SourceDocument(
        "https://example.org/reader",
        "reader concurrency",
        "Readers do not block writers. Reader concurrency is supported.",
    )
    calls = []

    class Provider:
        async def complete(self, request):
            p = json.loads(request.messages[-1].content)
            if "allowed_targets" in p:
                key = "repair"
                eid = next(
                    e["evidence_id"] for e in p["evidence"] if "Readers do not" in e["quote"]
                )
                value = dict(edits=[dict(target_id="C2", text=extra.text, citations=[eid])])
            elif "points" in p and "claims" in p:
                key = "assess"
                done = p["claims"][1]["text"] == extra.text
                if failure == "deadline":
                    await asyncio.sleep(1)
                value = assessment().model_dump()
                if done:
                    value["points"][1] = dict(
                        point_id="P2", status="answered", claim_ids=["C2"], gap_terms=[]
                    )
                    value["claims"][1] = dict(
                        claim_id="C2", kind="fact", support="supported", reason="New quote"
                    )
                    if failure == "recheck":
                        value = dict(invalid=True)
            elif "points" in p:
                key = "plan"
                value = dict(
                    points=[
                        dict(point_id="P1", terms=["writer"]),
                        dict(point_id="P2", terms=["reader"]),
                    ]
                )
            elif "candidates" in p:
                key = "rerank"
                value = dict(
                    rankings=[dict(point_id=t["id"], candidates=["R1"]) for t in p["points"]]
                )
            elif "requirements" in p and "context" in p:
                key = "report"
                eid = p["context"]["evidence"][0]["evidence_id"]
                value = dict(
                    answers=[
                        dict(text=first.text, citations=[eid]),
                        dict(text="现有证据不足，无法回答。", citations=[eid]),
                    ]
                )
            elif "red_review" in p:
                key = "blue"
                value = dict(edits=[], additions=[], removals=[])
            else:
                key = "red"
                value = dict(
                    scores=dict(evidence_support=60, logical_consistency=60, citation_quality=60),
                    findings=[
                        dict(
                            kind="missing_answer",
                            category="coverage",
                            severity="medium",
                            point_id="P2",
                            requirement=p["requirements"][1]["question"],
                            claim_id=None,
                            description="Need readers",
                            query_keywords=[],
                        )
                    ],
                )
                if '"title": "SupportRed"' in request.messages[0].content:
                    value.pop("findings")
                    value["checks"] = [
                        dict(
                            claim_id=c["claim_id"],
                            verdict="supported",
                            claim_excerpt=c["text"],
                            evidence_ids=[p["evidence"][0]["evidence_id"]],
                            evidence_excerpt=p["evidence"][0]["quote"][:180],
                            explanation="Fixture source check",
                        )
                        for c in p["claim_inventory"]
                    ]
                    value["coverage"] = [
                        dict(
                            point_id=r["id"],
                            status="answered" if i == 0 else "missing",
                            claim_ids=[p["claim_inventory"][0]["claim_id"]] if i == 0 else [],
                            explanation="Need readers" if i else "Writer answered",
                        )
                        for i, r in enumerate(p["requirements"])
                    ]
            calls.append(key)
            return ModelResponse(
                provider="fixture",
                model="fixture",
                request_id=str(len(calls)),
                content=json.dumps(value, ensure_ascii=False),
                latency_ms=0,
            )

    retriever = FakeRetriever(
        {
            QUESTION.split("？")[0]: [first],
            "writer": [first],
            "reader": [first],
            "reader concurrency": [extra],
        }
    )
    result = await run_dag_pipeline(
        QUESTION,
        gateway=ModelGateway(Provider()),
        retriever=retriever,
        store=storage[0],
        journal=storage[1],
        review_journal=storage[2],
        output_dir=tmp_path / "reports",
        options=DagOptions(
            max_replans=0, total_timeout_seconds=0.5 if failure == "deadline" else 240
        ),
        quality_options=RetrievalQualityOptions(
            optimize_research=True, rerank_evidence=rerank, route_gaps=True
        ),
        review_options=ReviewOptions(max_rounds=1),
        review_support=strict,
        source_mode="local_corpus",
    )
    saved = storage[0].get_report(result["run_id"])
    if failure == "deadline":
        assert saved["publication"]["abstained"] is True
        assert saved["support_assessment"]["status"] == "unavailable"
        assert "repair" not in calls
        return
    if failure == "recheck":
        assert saved["research_loop"]["repair"]["retained_draft"] is True
        assert len(saved["publication"]["claim_inventory"]) == 1
        assert saved["report"]["sections"][0]["claims"][1]["text"] != extra.text
        return
    assert [p["status"] for p in saved["publication"]["points"]] == ["answered", "answered"]
    assert len(saved["publication"]["claim_inventory"]) == 2
    assert saved["research_loop"]["repair"]["accepted"] is True
    assert saved["gap_retrieval"]["routing"]["targets"][0]["action"] == "search_new"
    assert saved["gap_retrieval"]["routing"]["targets"][0]["closure"] == "assessed_answered"
    assert (
        result["model_call_count"] <= 9
        and calls.count("repair") == 1
        and calls.count("assess") == 2
    )
    assert result["gap_retrieval"]["search_calls"] <= 2
    assert calls.count("blue") == 1
    assert saved["research_loop"]["previous_review"]["stop_reason"] == "stalled"
    snapshot = storage[1].snapshot(result["run_id"])
    blue_calls = [c for c in snapshot["model_calls"] if "blue" in c["stage"]]
    assert len(blue_calls) == 1
    assert blue_calls[0]["status"] == "completed"
    assert blue_calls[0]["details"]["no_change"] is True
