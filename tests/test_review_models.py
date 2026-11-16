"""Offline strict review contracts and atomic snapshot-based patch semantics."""

import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from deep_research.research_models import ResearchReport
from deep_research.review_models import (
    BlueRepairs,
    RedReview,
    RepairOperation,
    ReviewError,
    ReviewFinding,
    ReviewOptions,
    ReviewScores,
    apply_repairs,
    claim_inventory,
    report_fingerprint,
    validate_red,
)


def report(counts=(2, 1), *, limitations=None) -> ResearchReport:
    sections = []
    count = 0
    for index, length in enumerate(counts):
        claims = []
        for _ in range(length):
            count += 1
            claims.append({"text": f"原陈述 {count}。", "citations": [f"E{(count - 1) % 4 + 1}"]})
        sections.append({"heading": f"章节 {index}", "claims": claims})
    return ResearchReport.model_validate(
        {
            "title": "研究标题",
            "sections": sections,
            "limitations": limitations or ["引用存在不代表事实已经核实。"],
        }
    )


def evidence() -> list[dict]:
    return [
        {"evidence_id": f"E{index}", "source_id": "S1", "start": 0, "end": 2, "quote": "原文"}
        for index in range(1, 5)
    ]


def finding(**updates) -> dict:
    return {
        "id": "F1",
        "category": "evidence",
        "severity": "high",
        "claim_ids": ["C1"],
        "evidence_ids": ["E1"],
        "description": "需要检查原文支持范围。",
        **updates,
    }


def red(snapshot, **updates) -> RedReview:
    return RedReview.model_validate(
        {
            "report_hash": report_fingerprint(snapshot),
            "scores": {"evidence_support": 80, "logical_consistency": 81, "citation_quality": 82},
            "findings": [finding()],
            **updates,
        }
    )


def operation(op: str, **updates) -> dict:
    return {"op": op, "rationale": "按已保存证据作结构性调整。", **updates}


def blue(snapshot, *operations, **updates) -> BlueRepairs:
    return BlueRepairs.model_validate(
        {
            "report_hash": report_fingerprint(snapshot),
            "operations": list(operations),
            **updates,
        }
    )


def claim(text="修订陈述。", citations=None) -> dict:
    return {"text": text, "citations": citations or ["E1"]}


def test_review_options_are_strict_frozen_and_bounded() -> None:
    options = ReviewOptions()
    assert (options.max_rounds, options.score_threshold, options.max_format_repairs) == (2, 80, 1)
    assert options.max_calls == 6
    assert ReviewOptions(max_rounds=0, max_format_repairs=0).max_calls == 1
    assert ReviewOptions(max_rounds=1, max_format_repairs=0).max_calls == 3
    assert ReviewOptions(max_rounds=2, max_format_repairs=0).max_calls == 5
    with pytest.raises(ValidationError):
        options.max_rounds = 1


@pytest.mark.parametrize(
    "values",
    [
        {"max_rounds": -1},
        {"max_rounds": 3},
        {"max_rounds": True},
        {"max_rounds": "1"},
        {"max_rounds": 1.0},
        {"score_threshold": -1},
        {"score_threshold": 101},
        {"score_threshold": True},
        {"max_format_repairs": -1},
        {"max_format_repairs": 2},
        {"max_format_repairs": True},
        {"unexpected": "private"},
    ],
)
def test_invalid_options_are_rejected(values) -> None:
    with pytest.raises(ValidationError):
        ReviewOptions.model_validate(values)


def test_fingerprint_is_canonical_and_changes_with_content() -> None:
    snapshot = report()
    data = snapshot.model_dump(mode="json")
    expected = hashlib.sha256(
        json.dumps(
            data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    assert report_fingerprint(snapshot) == expected
    reordered = {
        "limitations": data["limitations"],
        "sections": data["sections"],
        "title": data["title"],
    }
    assert report_fingerprint(ResearchReport.model_validate(reordered)) == expected
    data["sections"][0]["claims"][0]["citations"] = ["E4"]
    assert report_fingerprint(ResearchReport.model_validate(data)) != expected


def test_inventory_is_sequential_across_sections_and_detached() -> None:
    snapshot = report((2, 2))
    inventory = claim_inventory(snapshot)
    assert inventory[0] == {
        "claim_id": "C1",
        "section_index": 0,
        "claim_index": 0,
        "text": "原陈述 1。",
        "citations": ["E1"],
    }
    assert inventory[2]["claim_id"] == "C3"
    assert inventory[2]["section_index"] == 1
    assert inventory[2]["claim_index"] == 0
    inventory[0]["citations"].append("E4")
    assert snapshot.sections[0].claims[0].citations == ["E1"]


@pytest.mark.parametrize("bad", [-1, 101, True, "90", 90.0])
def test_scores_require_bounded_real_integers(bad) -> None:
    with pytest.raises(ValidationError):
        ReviewScores(evidence_support=bad, logical_consistency=80, citation_quality=80)


@pytest.mark.parametrize(
    "updates",
    [
        {"id": "F0"},
        {"id": "F100"},
        {"category": "unknown"},
        {"severity": "critical"},
        {"claim_ids": []},
        {"claim_ids": ["C13"]},
        {"claim_ids": ["C1", "C1"]},
        {"evidence_ids": ["E1", "E1"]},
        {"evidence_ids": ["E0"]},
        {"description": ""},
        {"description": "x" * 501},
        {"description": "\ud800"},
        {"unexpected": "private"},
    ],
)
def test_finding_schema_is_strict(updates) -> None:
    with pytest.raises(ValidationError):
        ReviewFinding.model_validate(finding(**updates))


def test_coverage_can_reference_no_existing_claims() -> None:
    snapshot = report()
    review = red(snapshot, findings=[finding(category="coverage", claim_ids=[], evidence_ids=[])])
    validate_red(review, snapshot, evidence())


def test_valid_red_accepts_empty_findings_and_preserves_inputs() -> None:
    snapshot = report()
    before = snapshot.model_dump()
    validate_red(red(snapshot), snapshot, evidence())
    validate_red(red(snapshot, findings=[]), snapshot, evidence())
    assert snapshot.model_dump() == before


@pytest.mark.parametrize(
    "updates",
    [
        {"report_hash": "0" * 64},
        {"findings": [finding(), finding()]},
        {"findings": [finding(claim_ids=["C4"])]},
        {"findings": [finding(evidence_ids=["E99"])]},
    ],
)
def test_red_rejects_stale_duplicate_or_invented_references(updates) -> None:
    snapshot = report()
    with pytest.raises(ReviewError) as error:
        validate_red(red(snapshot, **updates), snapshot, evidence())
    assert error.value.code == "review_validation"
    assert "原陈述" not in str(error.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"report_hash": "A" * 64},
        {"report_hash": "1" * 63},
        {"unexpected": "private"},
        {"findings": [finding(id=f"F{i}") for i in range(1, 14)]},
    ],
)
def test_red_shape_limits(updates) -> None:
    with pytest.raises(ValidationError):
        red(report(), **updates)


@pytest.mark.parametrize(
    "bad",
    [
        operation("ADD", claim=claim()),
        operation("ADD", section_index=0),
        operation("ADD", section_index=0, claim=claim(), target_id="C1"),
        operation("ADD", section_index=0, claim=claim(), verdict="supported"),
        operation("ADD", section_index=0, claim=claim(), evidence_ids=["E1"]),
        operation("MODIFY", claim=claim()),
        operation("MODIFY", target_id="C1"),
        operation("MODIFY", target_id="C1", claim=claim(), section_index=0),
        operation("MODIFY", target_id="C1", claim=claim(), evidence_ids=["E1"]),
        operation("DELETE"),
        operation("DELETE", target_id="C1", claim=claim()),
        operation("DELETE", target_id="C1", section_index=0),
        operation("DELETE", target_id="C1", verdict="unsupported"),
        operation("VERIFY", target_id="C1"),
        operation("VERIFY", target_id="C1", verdict="supported"),
        operation("VERIFY", target_id="C1", verdict="uncertain", claim=claim()),
        operation("VERIFY", target_id="C1", verdict="supported", evidence_ids=["E1", "E1"]),
        operation("EXECUTE", target_id="C1"),
        operation("DELETE", target_id="C0"),
        operation("DELETE", target_id="C1", rationale=""),
        operation("DELETE", target_id="C1", rationale="x" * 501),
        operation("DELETE", target_id="C1", unexpected="private"),
    ],
)
def test_operation_field_combinations_are_strict(bad) -> None:
    with pytest.raises(ValidationError):
        RepairOperation.model_validate(bad)


def test_delete_and_modify_use_original_snapshot_ids() -> None:
    snapshot = report((2, 1))
    before = snapshot.model_dump()
    repairs = blue(
        snapshot,
        operation("DELETE", target_id="C1"),
        operation("MODIFY", target_id="C2", claim=claim("原C2被修订。", ["E4"])),
    )
    revised, verifications = apply_repairs(snapshot, repairs, evidence=evidence())
    assert revised.sections[0].claims[0].text == "原C2被修订。"
    assert revised.sections[0].claims[0].citations == ["E4"]
    assert revised.sections[1].claims[0].text == "原陈述 3。"
    assert revised.title == snapshot.title
    assert revised.limitations == snapshot.limitations
    assert verifications == []
    assert snapshot.model_dump() == before


def test_delete_empty_section_and_additions_use_original_section_indexes() -> None:
    snapshot = report((1, 1, 1))
    revised, _ = apply_repairs(
        snapshot,
        blue(
            snapshot,
            operation("DELETE", target_id="C1"),
            operation("ADD", section_index=2, claim=claim("追加在原第三节。")),
            operation("ADD", section_index=2, claim=claim("第二个追加。")),
        ),
        evidence=evidence(),
    )
    assert [section.heading for section in revised.sections] == ["章节 1", "章节 2"]
    assert [item.text for item in revised.sections[1].claims] == [
        "原陈述 3。",
        "追加在原第三节。",
        "第二个追加。",
    ]


def test_add_and_delete_can_replace_the_only_claim_without_deleting_section() -> None:
    snapshot = report((1,))
    revised, _ = apply_repairs(
        snapshot,
        blue(
            snapshot,
            operation("DELETE", target_id="C1"),
            operation("ADD", section_index=0, claim=claim("替代陈述。")),
        ),
        evidence=evidence(),
    )
    assert len(revised.sections) == 1
    assert revised.sections[0].claims[0].text == "替代陈述。"


def test_verify_supported_records_a_judgement_without_creating_facts() -> None:
    snapshot = report()
    revised, checks = apply_repairs(
        snapshot,
        blue(
            snapshot,
            operation("VERIFY", target_id="C1", verdict="supported", evidence_ids=["E1"]),
        ),
        evidence=evidence(),
    )
    assert revised == snapshot
    assert revised is not snapshot
    assert checks == [
        {
            "report_hash": report_fingerprint(snapshot),
            "target_id": "C1",
            "verdict": "supported",
            "evidence_ids": ["E1"],
            "rationale": "按已保存证据作结构性调整。",
        }
    ]


@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_negative_or_uncertain_verify_only_adds_caveat(verdict) -> None:
    snapshot = report()
    revised, checks = apply_repairs(
        snapshot,
        blue(
            snapshot,
            operation("VERIFY", target_id="C1", verdict=verdict),
        ),
        evidence=evidence(),
    )
    assert revised.sections == snapshot.sections
    assert revised.limitations[:1] == snapshot.limitations
    assert len(revised.limitations) == 2
    assert "C1" in revised.limitations[-1]
    assert report_fingerprint(snapshot)[:12] in revised.limitations[-1]
    assert "尚未独立核验" in revised.limitations[-1]
    assert checks[0]["verdict"] == verdict


@pytest.mark.parametrize(
    "operations",
    [
        [operation("DELETE", target_id="C4")],
        [operation("ADD", section_index=3, claim=claim())],
        [operation("MODIFY", target_id="C1", claim=claim(citations=["E99"]))],
        [operation("ADD", section_index=0, claim=claim(citations=["E99"]))],
        [operation("VERIFY", target_id="C1", verdict="supported", evidence_ids=["E99"])],
        [operation("DELETE", target_id="C1"), operation("MODIFY", target_id="C1", claim=claim())],
        [
            operation("MODIFY", target_id="C1", claim=claim()),
            operation("VERIFY", target_id="C1", verdict="uncertain"),
        ],
    ],
)
def test_patch_rejects_unknown_or_duplicate_targets_atomically(operations) -> None:
    snapshot = report()
    before = snapshot.model_dump()
    with pytest.raises(ReviewError) as error:
        apply_repairs(snapshot, blue(snapshot, *operations), evidence=evidence())
    assert error.value.code == "review_patch"
    assert snapshot.model_dump() == before


def test_stale_patch_and_repeat_application_are_rejected() -> None:
    snapshot = report()
    repairs = blue(snapshot, operation("MODIFY", target_id="C1", claim=claim()))
    revised, _ = apply_repairs(snapshot, repairs, evidence=evidence())
    with pytest.raises(ReviewError) as error:
        apply_repairs(revised, repairs, evidence=evidence())
    assert error.value.code == "review_patch"
    with pytest.raises(ReviewError):
        apply_repairs(
            snapshot,
            blue(snapshot, operation("DELETE", target_id="C1"), report_hash="0" * 64),
            evidence=evidence(),
        )


def test_cannot_delete_every_claim() -> None:
    snapshot = report((1, 1))
    with pytest.raises(ReviewError) as error:
        apply_repairs(
            snapshot,
            blue(
                snapshot,
                operation("DELETE", target_id="C1"),
                operation("DELETE", target_id="C2"),
            ),
            evidence=evidence(),
        )
    assert error.value.code == "review_patch"
    assert len(snapshot.sections) == 2


@pytest.mark.parametrize("counts,addition_count", [((6,), 1), ((4, 4, 4), 1)])
def test_section_and_total_claim_limits_are_checked_after_entire_batch(
    counts, addition_count
) -> None:
    snapshot = report(counts)
    before = snapshot.model_dump()
    with pytest.raises(ReviewError):
        apply_repairs(
            snapshot,
            blue(
                snapshot,
                *[operation("ADD", section_index=0, claim=claim()) for _ in range(addition_count)],
            ),
            evidence=evidence(),
        )
    assert snapshot.model_dump() == before


def test_late_invalid_operation_rolls_back_prior_valid_changes_and_verification() -> None:
    snapshot = report()
    before = snapshot.model_dump()
    operations = [
        operation("MODIFY", target_id="C1", claim=claim("不会部分保存。")),
        operation("VERIFY", target_id="C2", verdict="uncertain"),
        operation("DELETE", target_id="C12"),
    ]
    with pytest.raises(ReviewError):
        apply_repairs(snapshot, blue(snapshot, *operations), evidence=evidence())
    assert snapshot.model_dump() == before


def test_full_limitations_fail_closed_for_new_uncertainty_but_allow_supported() -> None:
    snapshot = report(limitations=[f"原局限 {index}" for index in range(6)])
    with pytest.raises(ReviewError):
        apply_repairs(
            snapshot,
            blue(
                snapshot,
                operation("VERIFY", target_id="C1", verdict="unsupported"),
            ),
            evidence=evidence(),
        )
    assert len(snapshot.limitations) == 6
    revised, _ = apply_repairs(
        snapshot,
        blue(
            snapshot,
            operation("VERIFY", target_id="C1", verdict="supported", evidence_ids=["E1"]),
        ),
        evidence=evidence(),
    )
    assert revised.limitations == snapshot.limitations


@pytest.mark.parametrize(
    "items",
    [None, ["E1"], [{"evidence_id": []}], [{"evidence_id": "E0"}], [{"evidence_id": "E1"}] * 2, []],
)
def test_evidence_inventory_must_be_valid_unique_and_cover_snapshot(items) -> None:
    snapshot = report()
    with pytest.raises(ReviewError):
        validate_red(red(snapshot), snapshot, items)
    with pytest.raises(ReviewError):
        apply_repairs(snapshot, blue(snapshot, operation("DELETE", target_id="C1")), evidence=items)


def test_empty_and_oversized_blue_batches_are_schema_errors() -> None:
    snapshot = report()
    for operations in ([], [operation("ADD", section_index=0, claim=claim())] * 9):
        with pytest.raises(ValidationError):
            blue(snapshot, *operations)


def test_nested_lists_cannot_mutate_original_via_revised_result() -> None:
    snapshot = report()
    before = snapshot.model_dump()
    repairs = blue(snapshot, operation("MODIFY", target_id="C1", claim=claim()))
    revised, _ = apply_repairs(snapshot, repairs, evidence=evidence())
    revised.sections[0].claims[0].citations.append("E4")
    revised.limitations.append("调用者新增")
    assert snapshot.model_dump() == before
    assert repairs.operations[0].claim.citations == ["E1"]


def test_mutated_or_model_construct_bypassed_models_are_revalidated() -> None:
    snapshot = report()
    reviews = red(snapshot)
    reviews.findings[0].claim_ids.append("C99")
    with pytest.raises(ReviewError) as error:
        validate_red(reviews, snapshot, evidence())
    assert error.value.code == "review_validation"
    bad = RepairOperation.model_construct(op="EXECUTE", rationale="private", target_id="C1")
    patch = BlueRepairs.model_construct(report_hash=report_fingerprint(snapshot), operations=[bad])
    with pytest.raises(ReviewError) as error:
        apply_repairs(snapshot, patch, evidence=evidence())
    assert error.value.code == "review_patch"
    assert "private" not in str(error.value)


def test_invalid_mutated_report_is_rejected_without_exposing_content() -> None:
    snapshot = report()
    snapshot.sections[0].claims[0].citations.clear()
    with pytest.raises(ReviewError) as error:
        report_fingerprint(snapshot)
    assert error.value.code == "review_validation"
    assert "原陈述" not in str(error.value)


def test_inputs_and_results_are_deterministic() -> None:
    snapshot = report()
    patch = blue(
        snapshot,
        operation("ADD", section_index=1, claim=claim()),
        operation("VERIFY", target_id="C1", verdict="uncertain"),
    )
    inputs_before = copy.deepcopy((snapshot.model_dump(), patch.model_dump(), evidence()))
    first = apply_repairs(snapshot, patch, evidence=evidence())
    second = apply_repairs(snapshot, patch, evidence=evidence())
    assert first == second
    assert inputs_before == (snapshot.model_dump(), patch.model_dump(), evidence())
