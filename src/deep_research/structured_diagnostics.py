"""Bounded schema diagnostics: never persist model values, messages or unknown keys."""

MAX_ISSUES = 8
# Paths are code-owned, not copied from arbitrary response keys. Array indexes
# are collapsed to '*' so a diagnostic does not reproduce model content.
_PATHS = {
    "ResearchOutline": {"topic", "questions", "questions.*"},
    "ResearchPlan": {"queries", "queries.*"},
    "ResearchReport": {
        "title",
        "sections",
        "sections.*",
        "sections.*.heading",
        "sections.*.claims",
        "sections.*.claims.*",
        "sections.*.claims.*.text",
        "sections.*.claims.*.citations",
        "sections.*.claims.*.citations.*",
        "limitations",
        "limitations.*",
    },
    "RedReview": {
        "report_hash",
        "scores",
        "scores.evidence_support",
        "scores.logical_consistency",
        "scores.citation_quality",
        "findings",
        "findings.*",
        "findings.*.id",
        "findings.*.category",
        "findings.*.severity",
        "findings.*.claim_ids",
        "findings.*.claim_ids.*",
        "findings.*.evidence_ids",
        "findings.*.evidence_ids.*",
        "findings.*.description",
        "findings.*.search_queries",
        "findings.*.search_queries.*",
    },
    "BlueRepairs": {
        "report_hash",
        "operations",
        "operations.*",
        "operations.*.op",
        "operations.*.target_id",
        "operations.*.section_index",
        "operations.*.claim",
        "operations.*.claim.text",
        "operations.*.claim.citations",
        "operations.*.claim.citations.*",
        "operations.*.verdict",
        "operations.*.evidence_ids",
        "operations.*.evidence_ids.*",
        "operations.*.rationale",
    },
    "other": set(),
}
_PATHS.update(
    {
        "PointAnswers": {
            "answers",
            "answers.*",
            "answers.*.text",
            "answers.*.citations",
            "answers.*.citations.*",
        },
        "ShortQueries": {"queries", "queries.*", "queries.*.*"},
        "LeanRed": {
            "scores",
            "scores.evidence_support",
            "scores.logical_consistency",
            "scores.citation_quality",
            "findings",
            "findings.*",
            "findings.*.claim_id",
            "findings.*.category",
            "findings.*.severity",
            "findings.*.description",
            "findings.*.evidence_ids",
            "findings.*.evidence_ids.*",
            "findings.*.query_keywords",
            "findings.*.query_keywords.*",
        },
        "LeanBlue": {
            "edits",
            "edits.*",
            "edits.*.target_id",
            "edits.*.text",
            "edits.*.citations",
            "edits.*.citations.*",
            "additions",
            "additions.*",
            "additions.*.section_index",
            "additions.*.text",
            "additions.*.citations",
            "additions.*.citations.*",
            "removals",
            "removals.*",
        },
    }
)
_KINDS = frozenset(
    {
        "missing",
        "extra_forbidden",
        "int_type",
        "string_type",
        "list_type",
        "dict_type",
        "model_type",
        "bool_type",
        "float_type",
        "none_required",
        "literal_error",
        "string_pattern_mismatch",
        "string_too_long",
        "string_too_short",
        "greater_than_equal",
        "less_than_equal",
        "greater_than",
        "less_than",
        "too_long",
        "too_short",
        "value_error",
        "validation_error",
        "object_required",
        "operation_fields",
        "duplicate_reference",
        "finding_claim_required",
    }
)
_LOCATIONS = {name: {tuple(path.split(".")) for path in paths} for name, paths in _PATHS.items()}


def validation_diagnostic(schema_name: str, errors: list[dict]) -> dict:
    """Extract only static paths and categories from Pydantic errors."""
    name = schema_name if schema_name in _PATHS else "other"
    issues = []
    for error in errors[:MAX_ISSUES]:
        loc = error.get("loc", ())
        parts = tuple("*" if type(part) is int else part for part in loc)
        # Match components before rendering: an unknown key containing dots
        # must not masquerade as a known nested field (or the root '$').
        path = ".".join(parts) if parts in _LOCATIONS[name] else "<unknown>" if parts else "$"
        kind = error.get("type")
        issues.append({"path": path, "kind": kind if kind in _KINDS else "validation_error"})
    return validate_diagnostic(
        {
            "schema": name,
            "issues": issues,
            "truncated": len(errors) > MAX_ISSUES,
        }
    )


def validate_diagnostic(value: object) -> dict:
    """Revalidate at persistence/read/retry boundaries; return a detached copy."""
    if not isinstance(value, dict) or set(value) != {"schema", "issues", "truncated"}:
        raise ValueError("invalid structured diagnostic")
    name, issues = value["schema"], value["issues"]
    if (
        not isinstance(name, str)
        or name not in _PATHS
        or type(value["truncated"]) is not bool
        or not isinstance(issues, list)
        or not 1 <= len(issues) <= MAX_ISSUES
    ):
        raise ValueError("invalid structured diagnostic")
    for item in issues:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "kind"}
            or not isinstance(item["path"], str)
            or item["path"] not in _PATHS[name] | {"$", "<unknown>"}
            or not isinstance(item["kind"], str)
            or item["kind"] not in _KINDS
        ):
            raise ValueError("invalid structured diagnostic")
    return {
        "schema": name,
        "issues": [dict(item) for item in issues],
        "truncated": value["truncated"],
    }
