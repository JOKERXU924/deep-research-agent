"""Actual Blue duplicate-target failure: reject it at wire parsing with safe feedback."""

import json

import pytest

from deep_research.errors import StructuredOutputError
from deep_research.review_json import parse_review_output
from deep_research.review_models import BlueRepairs


def test_modify_and_verify_same_target_get_safe_duplicate_feedback():
    value = dict(
        report_hash="a" * 64,
        operations=[
            dict(
                op="MODIFY",
                target_id="C1",
                claim=dict(text="Public claim.", citations=["E1"]),
                rationale="Public rationale.",
            ),
            dict(
                op="VERIFY",
                target_id="C1",
                verdict="supported",
                evidence_ids=["E1"],
                rationale="Public rationale.",
            ),
        ],
    )
    with pytest.raises(StructuredOutputError) as caught:
        parse_review_output(json.dumps(value), BlueRepairs)
    assert caught.value.code == "schema_mismatch"
    assert caught.value.validation["issues"] == [
        dict(path="operations.*.target_id", kind="duplicate_reference")
    ]
    assert "Public" not in json.dumps(caught.value.validation)


def test_unique_targets_remain_valid_and_no_automatic_patch_mutation():
    value = dict(
        report_hash="a" * 64,
        operations=[
            dict(
                op="VERIFY",
                target_id="C1",
                verdict="supported",
                evidence_ids=["E1"],
                rationale="OK",
            ),
            dict(
                op="VERIFY",
                target_id="C2",
                verdict="supported",
                evidence_ids=["E1"],
                rationale="OK",
            ),
        ],
    )
    parsed, mode = parse_review_output(json.dumps(value), BlueRepairs)
    assert mode == "strict" and parsed.report_hash == "a" * 64
    assert len(parsed.operations) == 2
