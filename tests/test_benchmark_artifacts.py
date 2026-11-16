"""Reject edited/corrupted result bookkeeping before statistical comparison."""

import copy
from pathlib import Path

import pytest

from deep_research.benchmark_artifacts import validate_results
from deep_research.benchmark_metrics import score_report
from deep_research.benchmark_runner import _offline_profile, read_json, run_benchmark, summarize
from deep_research.errors import ResearchError
from deep_research.model_budget import ModelBudget, ModelBudgetLimits

RUN_ID = "7a9df439-0067-4a8e-9f3f-77799d1ddbc9"


def artifact_fixture():
    limits = ModelBudgetLimits(max_calls=3)
    manifest = {
        "schema_version": 1,
        "experiment_id": "283f39a7-f71e-49aa-a306-d150e4bddbb3",
        "status": "completed",
        "created_at": "2026-10-16T12:00:00+00:00",
        "python": "3.12.11",
        "engine_fingerprint": "a" * 64,
        "dataset_id": "researchbench-v1",
        "dataset_fingerprint": "b" * 64,
        "provenance": "Test fixture only",
        "case_count": 1,
        "case_ids": ["case-one"],
        "presets": {
            "baseline": {"compression": False, "review": False, "concurrency": 3, "max_calls": 3}
        },
        "repeats": 1,
        "seed": 42,
        "planned_runs": 1,
        "planned_call_ceiling": 3,
        "max_total_model_calls": 90,
        "model_profile": _offline_profile(),
        "model_limits": limits.model_dump(),
        "memory": "disabled",
        "retrieval": "frozen_local_lexical",
        "compression_backend": "lexical",
        "mode": "offline_fixture",
        "quality_claim": False,
        "metric_version": "m8-rules-v1",
        "task_order": [{"case_id": "case-one", "preset": "baseline", "repeat": 0}],
    }
    metrics = score_report(
        {
            "sources": [{"source_id": "S1", "text": "A fact"}],
            "evidence": [
                {"evidence_id": "E1", "source_id": "S1", "start": 0, "end": 6, "quote": "A fact"}
            ],
            "report": {"sections": [{"claims": [{"text": "A fact", "citations": ["E1"]}]}]},
        },
        {"facts": ["A fact"]},
    )
    rows = [
        {
            "case_id": "case-one",
            "domain": "test",
            "preset": "baseline",
            "repeat": 0,
            "status": "completed",
            "run_id": RUN_ID,
            "error_code": None,
            "degraded": False,
            "bundle_path": f"reports/{RUN_ID}/evidence.json",
            "bundle_sha256": "c" * 64,
            "model_budget": ModelBudget(limits).summary(),
            "metrics": metrics,
            "elapsed_ms": 3.5,
        }
    ]
    return {
        "schema_version": 1,
        "manifest": manifest,
        "records": rows,
        "summary": summarize(rows, manifest),
    }


def test_accepts_valid_results_and_returns_detached_copy():
    artifact = artifact_fixture()
    validated = validate_results(artifact)
    assert validated == artifact
    validated["records"][0]["metrics"]["citation_coverage"] = 0
    assert artifact["records"][0]["metrics"]["citation_coverage"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("schema_version", 1.0),
        ("experiment_id", "not-a-uuid"),
        ("status", "successful"),
        ("status", []),
        ("created_at", "2026-10-16"),
        ("python", "secret\n3.12.11"),
        ("engine_fingerprint", "z" * 64),
        ("dataset_fingerprint", "A" * 64),
        ("dataset_id", "../example"),
        ("provenance", ""),
        ("case_count", True),
        ("case_count", 2),
        ("case_ids", ["case-one", "case-one"]),
        ("case_ids", []),
        ("case_ids", [None]),
        ("repeats", True),
        ("repeats", 0),
        ("repeats", 4),
        ("seed", -1),
        ("planned_runs", 2),
        ("planned_call_ceiling", 2),
        ("max_total_model_calls", 2),
        ("memory", "shared"),
        ("retrieval", "live_web"),
        ("compression_backend", "semantic"),
        ("mode", "live_model_frozen_corpus"),
        ("quality_claim", 0),
        ("quality_claim", True),
        ("metric_version", "m9"),
        ("task_order", []),
        ("presets", []),
    ],
)
def test_manifest_strict_types_shape_identity_and_denominators(field, value):
    artifact = artifact_fixture()
    artifact["manifest"][field] = value
    with pytest.raises(ResearchError) as error:
        validate_results(artifact)
    assert error.value.code == "benchmark_artifact"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda a: a.update(schema_version=True),
        lambda a: a.update(extra="unexpected"),
        lambda a: a["manifest"].update(extra="unexpected"),
        lambda a: a["manifest"]["model_limits"].pop("max_calls"),
        lambda a: a["manifest"]["model_limits"].update(max_calls=True),
        lambda a: a["manifest"]["model_profile"].update(api_key="must-not-leak"),
        lambda a: a["manifest"]["presets"]["baseline"].update(compression=0),
        lambda a: a["manifest"]["presets"]["baseline"].update(compression=True),
        lambda a: a["manifest"]["presets"]["baseline"].update(concurrency=True),
        lambda a: a["manifest"]["presets"]["baseline"].update(max_calls=2),
        lambda a: a["manifest"]["presets"].update(unknown={}),
        lambda a: a["manifest"]["task_order"][0].update(repeat=True),
        lambda a: a["manifest"]["task_order"][0].update(case_id="unknown"),
        lambda a: a["manifest"]["task_order"].append(copy.deepcopy(a["manifest"]["task_order"][0])),
        lambda a: a["records"].append(copy.deepcopy(a["records"][0])),
        lambda a: a["records"][0].update(extra="unexpected"),
        lambda a: a["records"][0].update(audit_error="benchmark_audit"),
        lambda a: a["records"][0].update(model_budget=None),
        lambda a: a["records"][0]["model_budget"]["limits"].update(max_calls=2),
        lambda a: a["records"][0]["model_budget"].update(effective_max_calls=2),
        lambda a: a["records"][0]["metrics"].update(failure_penalty=False),
        lambda a: a["records"][0]["metrics"].update(citation_coverage=True),
        lambda a: a["records"][0]["metrics"].update(citation_coverage=float("nan")),
        lambda a: a["records"][0]["metrics"].update(citation_coverage=0.5),
        lambda a: a["records"][0]["metrics"]["counts"].update(claims=True),
        lambda a: a["summary"]["baseline"].update(completed=True),
        lambda a: a["summary"]["baseline"].update(completed=0),
        lambda a: a["summary"]["baseline"]["metrics"]["citation_coverage"].update(mean=0),
        lambda a: a["summary"]["baseline"]["metrics"]["citation_coverage"].update(mean=True),
        lambda a: a["summary"].update(unlisted={}),
    ],
)
def test_corrupt_nested_values_cannot_sneak_through(mutation):
    artifact = artifact_fixture()
    mutation(artifact)
    with pytest.raises(ResearchError) as error:
        validate_results(artifact)
    assert error.value.code == "benchmark_artifact"
    assert "must-not-leak" not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("case_id", "unknown"),
        ("preset", "not-in-plan"),
        ("repeat", True),
        ("repeat", 1),
        ("status", "success"),
        ("run_id", None),
        ("run_id", "broken"),
        ("error_code", "failed"),
        ("domain", "../outside"),
        ("degraded", 0),
        ("elapsed_ms", True),
        ("elapsed_ms", -1),
        ("elapsed_ms", float("inf")),
        ("bundle_sha256", "bad"),
        ("bundle_path", "../outside.json"),
        ("bundle_path", "C:/reports/secret.json"),
    ],
)
def test_row_strict_types_and_bindings(field, value):
    artifact = artifact_fixture()
    artifact["records"][0][field] = value
    with pytest.raises(ResearchError):
        validate_results(artifact)


def failed_artifact(status="failed"):
    artifact = artifact_fixture()
    artifact["records"][0].update(
        status=status,
        bundle_path=None,
        bundle_sha256=None,
        error_code="cancelled" if status == "cancelled" else "model_budget_exceeded",
        metrics={
            "citation_coverage": 0.0,
            "citation_validity": 0.0,
            "gold_fact_coverage": 0.0,
            "nonempty_output": False,
            "failure_penalty": True,
        },
    )
    artifact["summary"] = summarize(artifact["records"], artifact["manifest"])
    return artifact


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_failure_rows_remain_in_completed_experiment_denominator(status):
    artifact = failed_artifact(status)
    assert validate_results(artifact)["summary"]["baseline"]["attempted"] == 1
    assert artifact["summary"]["baseline"]["metrics"]["citation_coverage"]["mean"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["metrics"].update(citation_coverage=1.0),
        lambda r: r["metrics"].update(citation_validity=None),
        lambda r: r["metrics"].update(gold_fact_coverage=True),
        lambda r: r["metrics"].update(nonempty_output=True),
        lambda r: r["metrics"].update(failure_penalty=False),
        lambda r: r.update(degraded=True),
        lambda r: r.update(error_code=None),
        lambda r: r.update(bundle_path=f"reports/{RUN_ID}/evidence.json"),
    ],
)
def test_edited_failure_cannot_become_positive_or_undefined_score(mutation):
    artifact = failed_artifact()
    mutation(artifact["records"][0])
    with pytest.raises(ResearchError):
        validate_results(artifact)


@pytest.mark.parametrize("status", ["running", "interrupted", "cancelled"])
def test_partial_experiments_retain_plan_and_allow_missing_records(status):
    artifact = artifact_fixture()
    artifact["manifest"]["status"] = status
    artifact["records"] = []
    artifact["summary"] = summarize([], artifact["manifest"])
    validated = validate_results(artifact)
    assert validated["manifest"]["planned_runs"] == 1
    assert validated["summary"]["baseline"]["attempted"] == 0


def test_completed_experiment_cannot_silently_drop_planned_rows():
    artifact = artifact_fixture()
    artifact["records"] = []
    artifact["summary"] = summarize([], artifact["manifest"])
    with pytest.raises(ResearchError):
        validate_results(artifact)


def test_interrupted_report_can_retain_missing_budget_audit_error():
    artifact = artifact_fixture()
    artifact["manifest"]["status"] = "interrupted"
    artifact["records"][0].update(model_budget=None, audit_error="benchmark_audit")
    artifact["summary"] = summarize(artifact["records"], artifact["manifest"])
    assert validate_results(artifact)["records"][0]["audit_error"] == "benchmark_audit"


def test_duplicate_run_identity_is_rejected_even_for_distinct_repeats():
    artifact = artifact_fixture()
    artifact["manifest"].update(repeats=2, planned_runs=2, planned_call_ceiling=6)
    artifact["manifest"]["task_order"].append(
        {"case_id": "case-one", "preset": "baseline", "repeat": 1}
    )
    row = copy.deepcopy(artifact["records"][0])
    row["repeat"] = 1
    artifact["records"].append(row)
    artifact["summary"] = summarize(artifact["records"], artifact["manifest"])
    with pytest.raises(ResearchError):
        validate_results(artifact)


def test_secret_extra_fields_never_appear_in_error():
    artifact = artifact_fixture()
    artifact["manifest"]["model_profile"]["secret"] = "very-secret-value"
    with pytest.raises(ResearchError) as caught:
        validate_results(artifact)
    assert "very-secret-value" not in str(caught.value)


async def test_real_offline_runner_artifact_passes_strict_validation(tmp_path):
    result = await run_benchmark(
        output_dir=tmp_path,
        presets=("baseline", "full", "tight-budget"),
        limit=1,
        max_total_model_calls=13,
    )
    value = validate_results(read_json(Path(result["results_path"])))
    assert len(value["records"]) == 3
    assert value["manifest"]["status"] == "completed"
    assert value["summary"]["tight-budget"]["failed"] == 1
