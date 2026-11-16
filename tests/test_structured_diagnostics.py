"""No live APIs: safe validation details survive retries and audit persistence."""

import copy
import json
import sqlite3

import pytest
from pydantic import model_validator
from pydantic_core import PydanticCustomError

from deep_research.dag_journal import DagJournal, DagJournalError
from deep_research.errors import StructuredOutputError
from deep_research.research_store import ResearchStore
from deep_research.review_models import BlueRepairs, RedReview
from deep_research.schemas import StrictModel
from deep_research.structured import parse_structured
from deep_research.structured_diagnostics import validate_diagnostic


def blue(operation):
    return {"report_hash": "a" * 64, "operations": [operation]}


def diagnostic(data, schema=BlueRepairs):
    with pytest.raises(StructuredOutputError) as caught:
        parse_structured(json.dumps(data), schema)
    assert caught.value.code == "schema_mismatch"
    return caught.value.validation


def test_hidden_blue_combination_has_specific_safe_category():
    value = diagnostic(
        blue(
            {
                "op": "MODIFY",
                "target_id": "C1",
                "rationale": "private rationale",
                "claim": {"text": "private claim", "citations": ["E1"]},
                "evidence_ids": ["E1"],
            }
        )
    )
    assert value == {
        "schema": "BlueRepairs",
        "truncated": False,
        "issues": [{"path": "operations.*", "kind": "operation_fields"}],
    }
    assert "private" not in json.dumps(value)


def test_nested_paths_are_static_and_unknown_keys_values_are_redacted():
    value = diagnostic(
        blue(
            {
                "op": "MODIFY",
                "target_id": "C1",
                "rationale": "private rationale",
                "claim": {"text": "private claim", "citations": "private wrong type"},
                "SECRET-KEY-DO-NOT-LOG": "SECRET-VALUE-DO-NOT-LOG",
            }
        )
    )
    assert {"path": "operations.*.claim.citations", "kind": "list_type"} in value["issues"]
    assert {"path": "<unknown>", "kind": "extra_forbidden"} in value["issues"]
    assert "SECRET" not in json.dumps(value)
    assert "private" not in json.dumps(value)


def test_red_noncoverage_requires_claims_without_recording_description():
    value = diagnostic(
        {
            "report_hash": "b" * 64,
            "scores": {"evidence_support": 50, "logical_consistency": 50, "citation_quality": 50},
            "findings": [
                {
                    "id": "F1",
                    "category": "evidence",
                    "severity": "high",
                    "claim_ids": [],
                    "evidence_ids": [],
                    "description": "private",
                }
            ],
        },
        RedReview,
    )
    assert value["issues"] == [{"path": "findings.*", "kind": "finding_claim_required"}]


def test_diagnostic_is_bounded_and_top_level_nonobject_is_explained():
    value = diagnostic({f"secret-key-{index}": "private" for index in range(30)})
    assert len(value["issues"]) == 8 and value["truncated"] is True
    assert "secret" not in json.dumps(value)
    assert diagnostic([])["issues"] == [{"path": "$", "kind": "object_required"}]


def test_arbitrary_custom_validator_message_kind_context_and_schema_name_not_logged():
    class PrivateSchema(StrictModel):
        @model_validator(mode="after")
        def reject(self):
            raise PydanticCustomError("SECRET-KIND", "SECRET-MSG", {"secret": "SECRET-VALUE"})

    value = diagnostic({}, PrivateSchema)
    assert value == {
        "schema": "other",
        "truncated": False,
        "issues": [{"path": "$", "kind": "validation_error"}],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "SECRET-SCHEMA"},
        {"truncated": 1},
        {"raw": "secret"},
        {"issues": []},
        {"issues": [{"path": "operations.*.SECRET", "kind": "missing"}]},
        {"issues": [{"path": "$", "kind": "SECRET-KIND"}]},
        {"issues": [{"path": "$", "kind": "missing", "input": "SECRET"}]},
        {"issues": [{"path": "$", "kind": "missing"}] * 9},
    ],
)
def test_diagnostic_allowlist_rejects_payload_injection(change):
    value = diagnostic([]) | change
    with pytest.raises(ValueError, match="invalid structured diagnostic"):
        validate_diagnostic(value)


def test_diagnostic_is_detached_when_stored_in_exception():
    value = diagnostic([])
    error = StructuredOutputError("schema_mismatch", "safe", validation=value)
    value["issues"][0]["kind"] = "SECRET"
    assert error.validation["issues"][0]["kind"] == "object_required"


def test_unknown_dotted_keys_cannot_impersonate_known_paths():
    value = diagnostic({"operations.*.claim": "secret", "$": "secret"})
    extras = [item for item in value["issues"] if item["kind"] == "extra_forbidden"]
    assert len(extras) == 2 and all(item["path"] == "<unknown>" for item in extras)


def test_structured_error_preserves_status_code_keyword_compatibility():
    error = StructuredOutputError("schema_mismatch", "safe", status_code=422)
    assert error.status_code == 422 and error.validation is None


def test_failed_call_diagnostic_survives_reopen_and_tampering_is_rejected(tmp_path):
    path = tmp_path / "diagnostics.sqlite"
    value = diagnostic(
        blue(
            {
                "op": "VERIFY",
                "target_id": "C1",
                "rationale": "safe",
                "verdict": "supported",
                "evidence_ids": [],
            }
        )
    )
    details = {
        "error_code": "schema_mismatch",
        "validation": value,
        "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
    }
    with ResearchStore(path) as store, DagJournal(path) as journal:
        run_id = store.create_run("offline")
        journal.create_run(run_id, {"model_call_limit": 9})
        call_id = journal.start_model_call(run_id, "review.blue.1")
        bad = copy.deepcopy(details)
        bad["validation"]["issues"][0]["input"] = "SECRET"
        with pytest.raises(DagJournalError):
            journal.finish_model_call(run_id, call_id, "failed", bad)
        assert journal.snapshot(run_id)["model_calls"][0]["status"] == "in_flight"
        journal.finish_model_call(run_id, call_id, "failed", details)
    with DagJournal(path) as journal:
        call = journal.snapshot(run_id)["model_calls"][0]
        assert call["details"] == details
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE dag_model_calls SET details_json=? WHERE run_id=?",
                (json.dumps(bad), run_id),
            )
        with pytest.raises(DagJournalError):
            journal.snapshot(run_id)
