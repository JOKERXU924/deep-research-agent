"""Pure structural validation of untrusted M8 artifacts, not authentication.

The caller must independently bind dataset and report bytes and recompute scores.
Hashes stored in the same editable document cannot establish authenticity.
"""

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from uuid import UUID

from deep_research.errors import ResearchError
from deep_research.model_budget import (
    ModelBudgetLimits,
    validate_budget_summary,
    validate_profile_snapshot,
)

_METRICS = ("citation_coverage", "citation_validity", "gold_fact_coverage")
_PRESETS = {
    "baseline": (False, False, 3, 3),
    "compression": (True, False, 3, 3),
    "review": (False, True, 3, 9),
    "full": (True, True, 3, 9),
    "tight-budget": (False, False, 3, 1),
    "serial": (False, False, 1, 3),
}
_MANIFEST_KEYS = {
    "schema_version",
    "experiment_id",
    "status",
    "created_at",
    "python",
    "engine_fingerprint",
    "dataset_id",
    "dataset_fingerprint",
    "provenance",
    "case_count",
    "case_ids",
    "presets",
    "repeats",
    "seed",
    "planned_runs",
    "planned_call_ceiling",
    "max_total_model_calls",
    "model_profile",
    "model_limits",
    "memory",
    "retrieval",
    "compression_backend",
    "mode",
    "quality_claim",
    "metric_version",
    "task_order",
}
_ROW_KEYS = {
    "case_id",
    "domain",
    "preset",
    "repeat",
    "status",
    "run_id",
    "error_code",
    "degraded",
    "bundle_path",
    "bundle_sha256",
    "model_budget",
    "metrics",
    "elapsed_ms",
}
_COUNT_KEYS = {
    "claims",
    "cited_claims",
    "citation_refs",
    "valid_citation_refs",
    "gold_facts",
    "matched_gold_facts",
    "valid_evidence",
}


def _require(condition: bool) -> None:
    if not condition:
        raise ValueError("invalid artifact")


def _object(value, keys: set, optional: set | None = None) -> dict:
    _require(
        isinstance(value, dict)
        and keys <= value.keys()
        and value.keys() <= keys | (optional or set())
    )
    return value


def _integer(value, low=0, high=2**63 - 1) -> int:
    _require(type(value) is int and low <= value <= high)
    return value


def _string(value, limit=200, pattern: str | None = None) -> str:
    _require(
        isinstance(value, str)
        and 0 < len(value) <= limit
        and bool(value.strip())
        and not any(ord(c) < 32 and c not in "\r\n\t" for c in value)
    )
    value.encode("utf-8")
    if pattern is not None:
        _require(re.fullmatch(pattern, value) is not None)
    return value


def _identifier(value) -> str:
    return _string(value, 96, r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}")


def _digest(value) -> str:
    return _string(value, 64, r"[a-f0-9]{64}")


def _uuid(value) -> str:
    value = _string(value, 36)
    _require(str(UUID(value)) == value)
    return value


def _number(value, *, low=0.0, high=1.0, nullable=False):
    if nullable and value is None:
        return None
    _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high)
    return value


def _array(value, low=0, high=300) -> list:
    _require(isinstance(value, list) and low <= len(value) <= high)
    return value


def _task(value: dict, manifest: dict) -> tuple[str, str, int]:
    _object(value, {"case_id", "preset", "repeat"})
    _identifier(value["case_id"])
    _string(value["preset"], 20)
    _require(value["case_id"] in manifest["case_ids"] and value["preset"] in manifest["presets"])
    _integer(value["repeat"], 0, manifest["repeats"] - 1)
    return value["case_id"], value["preset"], value["repeat"]


def _manifest(value: dict) -> list[tuple[str, str, int]]:
    _object(value, _MANIFEST_KEYS)
    _integer(value["schema_version"], 1, 1)
    _uuid(value["experiment_id"])
    _require(value["status"] in {"running", "completed", "interrupted", "cancelled"})
    _string(value["created_at"], 100)
    _require(datetime.fromisoformat(value["created_at"]).utcoffset() is not None)
    _string(value["python"], 40, r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]+)?")
    _digest(value["engine_fingerprint"])
    _digest(value["dataset_fingerprint"])
    _identifier(value["dataset_id"])
    _string(value["provenance"], 2000)
    _integer(value["case_count"], 1, 50)
    cases = _array(value["case_ids"], 1, 50)
    for case in cases:
        _identifier(case)
    _require(len(cases) == len(set(cases)) == value["case_count"])
    _integer(value["repeats"], 1, 3)
    _integer(value["seed"], 0, 2**32 - 1)
    _integer(value["planned_runs"], 1, 300)
    _integer(value["planned_call_ceiling"], 1, 2700)
    _integer(value["max_total_model_calls"], 1, 2700)
    limits = ModelBudgetLimits.model_validate(value["model_limits"]).model_dump()
    _require(set(value["model_limits"]) == set(limits))
    profile = validate_profile_snapshot(value["model_profile"])
    _require(
        value["memory"] == "disabled"
        and value["retrieval"] == "frozen_local_lexical"
        and value["compression_backend"] == "lexical"
        and value["metric_version"] == "m8-rules-v1"
        and value["quality_claim"] is False
    )
    _require(value["mode"] in {"offline_fixture", "live_model_frozen_corpus"})
    _require((profile["provider"] == "offline_demo") == (value["mode"] == "offline_fixture"))
    presets = value["presets"]
    _require(isinstance(presets, dict) and 1 <= len(presets) <= 6)
    for name, config in presets.items():
        _require(name in _PRESETS)
        _object(config, {"compression", "review", "concurrency", "max_calls"})
        compression, review, concurrency, maximum = _PRESETS[name]
        _require(type(config["compression"]) is bool and config["compression"] == compression)
        _require(type(config["review"]) is bool and config["review"] == review)
        _integer(config["concurrency"], concurrency, concurrency)
        _integer(
            config["max_calls"],
            min(maximum, limits["max_calls"]),
            min(maximum, limits["max_calls"]),
        )
    expected = {(c, p, r) for c in cases for p in presets for r in range(value["repeats"])}
    order = [_task(task, value) for task in _array(value["task_order"], 1, 300)]
    _require(len(order) == len(set(order)) == len(expected) == value["planned_runs"])
    _require(set(order) == expected)
    ceiling = len(cases) * value["repeats"] * sum(c["max_calls"] for c in presets.values())
    _require(value["planned_call_ceiling"] == ceiling <= value["max_total_model_calls"])
    return order


def _metrics(value: dict, completed: bool) -> None:
    base = {*_METRICS, "nonempty_output"}
    if not completed:
        _object(value, base | {"failure_penalty"})
        _require(value["failure_penalty"] is True and value["nonempty_output"] is False)
        for key in _METRICS:
            _number(value[key], high=0.0, nullable=key == "gold_fact_coverage")
        return
    _object(value, base | {"counts", "metric_notes"})
    _require(type(value["nonempty_output"]) is bool)
    counts = _object(value["counts"], _COUNT_KEYS)
    for number in counts.values():
        _integer(number, 0, 100000)
    _require(
        counts["cited_claims"] <= counts["claims"]
        and counts["valid_citation_refs"] <= counts["citation_refs"]
        and counts["matched_gold_facts"] <= counts["gold_facts"]
    )
    ratios = {
        "citation_coverage": (counts["cited_claims"], counts["claims"]),
        "citation_validity": (counts["valid_citation_refs"], counts["citation_refs"]),
        "gold_fact_coverage": (counts["matched_gold_facts"], counts["gold_facts"]),
    }
    for key, (numerator, denominator) in ratios.items():
        _number(value[key], nullable=True)
        _require(value[key] == (numerator / denominator if denominator else None))
    for note in _array(value["metric_notes"], 1, 8):
        _string(note, 1000)


def _row(value: dict, manifest: dict) -> tuple[str, str, int]:
    _object(value, _ROW_KEYS, {"audit_error"})
    key = _task({k: value[k] for k in ("case_id", "preset", "repeat")}, manifest)
    _string(value["domain"], 48, r"[a-z][a-z0-9_-]{0,47}")
    _require(value["status"] in {"completed", "failed", "cancelled"})
    _require(type(value["degraded"]) is bool)
    _number(value["elapsed_ms"], high=1e12)
    if value["run_id"] is not None:
        _uuid(value["run_id"])
    if "audit_error" in value:
        _require(value["audit_error"] == "benchmark_audit")
    completed = value["status"] == "completed"
    _metrics(value["metrics"], completed)
    if completed:
        _require(value["run_id"] is not None and value["error_code"] is None)
        _digest(value["bundle_sha256"])
        _require(value["bundle_path"] == f"reports/{value['run_id']}/evidence.json")
    else:
        _require(
            value["bundle_path"] is None
            and value["bundle_sha256"] is None
            and value["degraded"] is False
        )
        _string(value["error_code"], 100, r"[a-z][a-z0-9_]{0,99}")
        if value["status"] == "cancelled":
            _require(value["error_code"] == "cancelled")
    if value["model_budget"] is not None:
        _require(value["run_id"] is not None)
        budget = validate_budget_summary(value["model_budget"])
        expected = {
            **manifest["model_limits"],
            "max_calls": manifest["presets"][value["preset"]]["max_calls"],
        }
        _require(
            budget["limits"] == expected and budget["effective_max_calls"] == expected["max_calls"]
        )
    elif completed:
        _require(value.get("audit_error") == "benchmark_audit")
    return key


def _validate_summary(summary: dict, records: list[dict], manifest: dict) -> None:
    _object(summary, set(manifest["presets"]))
    for preset, value in summary.items():
        _object(
            value,
            {
                "attempted",
                "planned",
                "completed",
                "failed",
                "cancelled",
                "degraded",
                "metrics",
                "known_tokens",
                "unknown_usage_calls",
                "unavailable_budget_rows",
            },
        )
        rows = [row for row in records if row["preset"] == preset]
        for key, item in value.items():
            if key != "metrics":
                _integer(item)
        expected = {
            "attempted": len(rows),
            "planned": manifest["case_count"] * manifest["repeats"],
            "completed": sum(row["status"] == "completed" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "cancelled": sum(row["status"] == "cancelled" for row in rows),
            "degraded": sum(row["degraded"] for row in rows),
            "known_tokens": sum(
                row["model_budget"]["known_usage"]["total_tokens"]
                for row in rows
                if row["model_budget"] is not None
            ),
            "unknown_usage_calls": sum(
                row["model_budget"]["usage_unknown_calls"]
                for row in rows
                if row["model_budget"] is not None
            ),
            "unavailable_budget_rows": sum(row["model_budget"] is None for row in rows),
        }
        _require(all(value[key] == count for key, count in expected.items()))
        _object(value["metrics"], set(_METRICS))
        for metric, scores in value["metrics"].items():
            _object(scores, {"mean", "scored_cases", "attempted_cases"})
            _number(scores["mean"], nullable=True)
            _integer(scores["scored_cases"], 0, manifest["case_count"])
            _integer(scores["attempted_cases"], 0, manifest["case_count"])
            grouped = defaultdict(list)
            for row in rows:
                if row["metrics"][metric] is not None:
                    grouped[row["case_id"]].append(row["metrics"][metric])
            means = [sum(values) / len(values) for values in grouped.values()]
            _require(
                scores["mean"] == (sum(means) / len(means) if means else None)
                and scores["scored_cases"] == len(means)
                and scores["attempted_cases"] == len({row["case_id"] for row in rows})
            )


def validate_results(artifact: object) -> dict:
    """Validate schema, planned-task coverage, penalties, budgets, and summaries.

    Running/interrupted/cancelled experiments may have an ordered prefix of the
    plan. A completed experiment must have all planned rows, including failures.
    Returned dictionaries share no mutable objects with the supplied artifact.
    """
    try:
        # One bounded JSON copy excludes cycles, non-finite values and custom objects.
        raw = json.dumps(artifact, ensure_ascii=False, allow_nan=False).encode("utf-8")
        _require(len(raw) <= 16 * 1024 * 1024)
        value = json.loads(raw)
        _object(value, {"schema_version", "manifest", "records", "summary"})
        _integer(value["schema_version"], 1, 1)
        manifest = value["manifest"]
        order = _manifest(manifest)
        records = _array(value["records"])
        keys = [_row(row, manifest) for row in records]
        _require(keys == order[: len(keys)])
        run_ids = [row["run_id"] for row in records if row["run_id"] is not None]
        _require(len(run_ids) == len(set(run_ids)))
        if manifest["status"] == "completed":
            _require(len(keys) == len(order) and all("audit_error" not in row for row in records))
        _validate_summary(value["summary"], records, manifest)
        return value
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        UnicodeError,
        RecursionError,
        ResearchError,
    ):
        raise ResearchError(
            "benchmark_artifact", "评测记录结构、任务分母或审计数据不一致。"
        ) from None
