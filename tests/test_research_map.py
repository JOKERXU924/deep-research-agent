"""Snapshot integrity, conservative states and production persistence without paid calls."""

import copy
import hashlib

import pytest
from test_retrieval_quality_pipeline import GapProvider, execute
from test_review_research import storage as review_storage

from deep_research.research_map import (
    ResearchMap,
    build_research_map,
    question_points,
    validate_research_map,
)
from deep_research.research_models import ResearchReport
from deep_research.review_models import report_fingerprint

storage = review_storage


@pytest.fixture
def bundle():
    text = "并发写入需要等待。"
    return dict(
        question="并发写入如何处理？部署成本怎样？",
        report=dict(
            title="比较",
            sections=[dict(heading="写入", claims=[dict(text=text, citations=["E1"])])],
            limitations=["部署成本资料不足。"],
        ),
        sources=[
            dict(
                source_id="S1",
                title="说明",
                text=text,
                url="https://example.org",
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        ],
        evidence=[dict(evidence_id="E1", source_id="S1", start=0, end=len(text), quote=text)],
    )


def test_points_exist_before_report_and_are_not_inferred_from_section_headings(bundle):
    points = question_points(bundle["question"])
    result = build_research_map(bundle, points=points)
    assert [p["question"] for p in result["points"]] == ["并发写入如何处理", "部署成本怎样"]
    assert result["points"][0]["claim_ids"] == ["C1"]
    assert result["points"][1]["status"] == "no_conclusion"
    assert any(g["kind"] == "no_conclusion" and g["point_ids"] == ["P2"] for g in result["gaps"])


def test_traceable_citation_is_not_automatically_verified(bundle):
    result = build_research_map(bundle)
    assert result["points"][0]["status"] == "needs_review"
    assert result["evidence"][0]["quote"] == bundle["sources"][0]["text"]
    assert all(g["status"] == "open" for g in result["gaps"])
    assert any(g["kind"] == "support_unverified" for g in result["gaps"])


def test_explicit_comparison_dimensions_and_conditions_are_retained():
    points = question_points("请比较并发写入、部署成本、事务能力和迁移风险，并说明成立条件。")
    assert len(points) == 5
    assert [p["question"] for p in points[:4]] == [
        "比较：并发写入",
        "比较：部署成本",
        "比较：事务能力",
        "比较：迁移风险",
    ]
    assert "成立条件" in points[-1]["question"]


def test_two_questions_split_but_leading_condition_stays_attached():
    assert len(question_points("为什么异步适合等待，并发需要什么限制？")) == 2
    assert len(question_points("在高并发条件下，是否需要迁移？")) == 1


def test_point_limit_keeps_remaining_requirements_in_last_slot():
    points = question_points("；".join(f"问题{i}" for i in range(12)))
    assert len(points) == 8
    assert "问题11" in points[-1]["question"]


@pytest.mark.parametrize("corruption", ["missing", "offset", "hash"])
def test_missing_or_corrupt_evidence_becomes_explicit_gap(bundle, corruption):
    if corruption == "missing":
        bundle["evidence"] = []
    elif corruption == "offset":
        bundle["evidence"][0]["end"] -= 1
    else:
        bundle["sources"][0]["sha256"] = "0" * 64
    result = build_research_map(bundle)
    assert result["conclusions"][0]["evidence_ids"] == []
    assert any(g["kind"] == "missing_evidence" for g in result["gaps"])


def test_unassigned_claims_and_global_coverage_findings_are_not_discarded(bundle):
    bundle["question"] = "迁移价格是多少？"
    result = build_research_map(bundle)
    assert result["conclusions"][0]["point_ids"] == []
    assert result["conclusions"][0]["link_method"] == "unassigned"
    assert any(g["claim_ids"] == ["C1"] and not g["point_ids"] for g in result["gaps"])


def attach_review(bundle):
    h = report_fingerprint(ResearchReport.model_validate(bundle["report"]))
    finding = dict(
        id="F1",
        category="evidence",
        severity="high",
        claim_ids=["C1"],
        evidence_ids=["E1"],
        description="需要说明写入条件。",
        search_queries=["写入条件"],
    )
    red = dict(
        report_hash=h,
        findings=[finding],
        scores=dict(evidence_support=50, logical_consistency=70, citation_quality=70),
    )
    bundle["review"] = dict(best_hash=h, best_round=0, finding_count=1)
    bundle["review_events"] = [dict(event=dict(kind="red", round=0, report_hash=h, review=red))]


def test_review_binds_to_selected_snapshot_not_latest_round(bundle):
    attach_review(bundle)
    stale = copy.deepcopy(bundle["review_events"][0])
    stale["event"]["round"] = 1
    stale["event"]["review"]["findings"][0]["description"] = "另一版本问题"
    bundle["review_events"].append(stale)
    result = build_research_map(bundle)
    assert result["review_binding"] == "matched"
    assert result["points"][0]["status"] == "review_flagged"
    gap = next(g for g in result["gaps"] if g["kind"] == "review_finding")
    assert gap["description"] == "需要说明写入条件。"
    assert gap["suggested_queries"] == ["写入条件"]
    assert gap["finding_id"] == "F1"


def test_missing_review_identity_is_unknown_not_no_issues(bundle):
    attach_review(bundle)
    bundle["review"]["best_hash"] = "a" * 64
    result = build_research_map(bundle)
    assert result["review_binding"] == "unavailable"
    assert result["points"][0]["status"] == "needs_review"


@pytest.mark.parametrize("field", ["report", "question", "map", "source"])
def test_saved_map_rejects_snapshot_tampering(bundle, field):
    bundle["research_map"] = build_research_map(bundle)
    if field == "report":
        bundle["report"]["sections"][0]["claims"][0]["text"] = "不同结论。"
    elif field == "question":
        bundle["question"] = "新问题"
    elif field == "source":
        bundle["sources"][0]["text"] = "被改变的原文"
    else:
        bundle["research_map"]["points"][0]["claim_ids"] = ["C12"]
    with pytest.raises(ValueError):
        validate_research_map(bundle)


def test_old_reports_remain_readable_without_backfilling_assumptions(bundle):
    validate_research_map(bundle)
    assert "research_map" not in bundle


def test_dangling_ids_and_duplicate_points_are_rejected(bundle):
    result = build_research_map(bundle)
    result["points"].append(copy.deepcopy(result["points"][0]))
    with pytest.raises(ValueError):
        ResearchMap.model_validate(result)


def test_database_reopen_preserves_projection_and_rejects_tampering(tmp_path, bundle):
    import json
    import sqlite3

    from deep_research.research_store import ResearchStore, ResearchStoreError

    path = tmp_path / "map.sqlite3"
    bundle["research_map"] = build_research_map(bundle)
    with ResearchStore(path) as store:
        run_id = store.create_run(bundle["question"])
        store.complete_run(run_id, bundle, [], str(tmp_path))
    with ResearchStore(path) as store:
        assert store.get_report(run_id)["research_map"] == bundle["research_map"]
    bundle["research_map"]["points"][0]["status"] = "no_conclusion"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE research_runs SET report_json=? WHERE id=?", (json.dumps(bundle), run_id)
        )
    with ResearchStore(path) as store, pytest.raises(ResearchStoreError):
        store.get_report(run_id)


async def test_dag_saves_same_map_to_database_and_download_without_extra_model_calls(
    tmp_path, storage
):
    import json

    provider = GapProvider(scenario="passed")
    result = await execute(tmp_path, storage, provider=provider)
    saved = storage[0].get_report(result["run_id"])
    assert saved["research_map"]["review_binding"] == "matched"
    assert [s for s, _ in provider.calls] == ["plan", "report", "red"]
    files = list((tmp_path / "reports").rglob("evidence.json"))
    assert len(files) == 1
    exported = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["research_map"] == exported["research_map"]
    validate_research_map(saved)
