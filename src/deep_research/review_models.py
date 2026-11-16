"""M5 review contracts and atomic, snapshot-bound report patches.

These checks establish structural validity and known reference IDs. A Red score
or Blue VERIFY verdict is a model opinion, not factual or entailment verification.
All patch targets refer to the original snapshot, before any deletion or addition.
"""

import hashlib
import json
import re
import unicodedata
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)
from pydantic_core import PydanticCustomError

from deep_research.errors import ResearchError
from deep_research.research_models import CitedClaim, EvidenceID, QueryText, ResearchReport
from deep_research.schemas import StrictModel


def _utf8(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ValueError("review text must be valid UTF-8") from None
    return value


ReviewText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    AfterValidator(_utf8),
]


def _search_query(value: object) -> object:
    # Queries remain untrusted search data, never tool instructions or URLs.
    # Check before whitespace stripping so trailing controls cannot disappear.
    if isinstance(value, str):
        if (
            any(unicodedata.category(character).startswith("C") for character in value)
            or "://" in value
            or "www." in value.casefold()
        ):
            raise ValueError("search queries must be plain topic text")
        _utf8(value)
    return value


GapQueryText = Annotated[QueryText, BeforeValidator(_search_query)]
ReportHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ClaimID = Annotated[str, StringConstraints(pattern=r"^C(?:[1-9]|1[0-2])$")]
FindingID = Annotated[str, StringConstraints(pattern=r"^F[1-9][0-9]?$")]
_EVIDENCE_ID = re.compile(r"E[1-9][0-9]{0,2}\Z")


class ReviewError(ResearchError):
    """Stable public errors, never raw report text or validation payloads."""


def _error(code: str = "review_validation") -> ReviewError:
    message = (
        "报告修订批次校验失败；原报告未改变。"
        if code == "review_patch"
        else "审查结果或报告引用校验失败。"
    )
    return ReviewError(code, message)


class ReviewOptions(StrictModel):
    robust_repairs: bool = False
    max_rounds: int = Field(default=2, ge=0, le=2)
    score_threshold: int = Field(default=80, ge=0, le=100)
    max_format_repairs: int = Field(default=1, ge=0, le=1)

    @property
    def max_calls(self) -> int:
        return 1 + 2 * self.max_rounds + self.max_format_repairs


class ReviewScores(StrictModel):
    evidence_support: int = Field(ge=0, le=100)
    logical_consistency: int = Field(ge=0, le=100)
    citation_quality: int = Field(ge=0, le=100)


class ReviewFinding(StrictModel):
    id: FindingID
    category: Literal["evidence", "logic", "citation", "coverage"]
    severity: Literal["low", "medium", "high"]
    claim_ids: list[ClaimID] = Field(max_length=4)
    evidence_ids: list[EvidenceID] = Field(max_length=4)
    description: ReviewText
    search_queries: list[GapQueryText] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def valid_references(self) -> Self:
        if not self.claim_ids and self.category != "coverage":
            raise PydanticCustomError(
                "finding_claim_required", "only coverage findings may omit claim targets"
            )
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise PydanticCustomError("duplicate_reference", "duplicate finding claim reference")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise PydanticCustomError("duplicate_reference", "duplicate finding evidence reference")
        return self


class RedReview(StrictModel):
    report_hash: ReportHash
    scores: ReviewScores
    findings: list[ReviewFinding] = Field(max_length=12)


class RepairOperation(StrictModel):
    """ADD/MODIFY use claim.citations, never operation-level evidence_ids.

    ADD requires section_index and claim, with no target_id/verdict/evidence_ids.
    MODIFY requires target_id and claim, with no section_index/verdict/evidence_ids.
    DELETE requires target_id, with no section_index/claim/verdict/evidence_ids.
    VERIFY requires target_id and verdict, with no section_index/claim; supported
    requires nonempty evidence_ids. Each operation requires rationale.
    Inapplicable nullable fields may be omitted or null; unused arrays must be [].
    """

    op: Literal["ADD", "DELETE", "MODIFY", "VERIFY"]
    target_id: ClaimID | None = None
    section_index: int | None = Field(default=None, ge=0, le=3)
    claim: CitedClaim | None = None
    verdict: Literal["supported", "unsupported", "uncertain"] | None = None
    evidence_ids: list[EvidenceID] = Field(default_factory=list, max_length=4)
    rationale: ReviewText

    @model_validator(mode="after")
    def valid_combination(self) -> Self:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise PydanticCustomError(
                "duplicate_reference", "duplicate verification evidence reference"
            )
        if self.op == "ADD":
            valid = (
                self.target_id is None
                and self.section_index is not None
                and self.claim is not None
                and self.verdict is None
                and not self.evidence_ids
            )
        elif self.op == "MODIFY":
            valid = (
                self.target_id is not None
                and self.section_index is None
                and self.claim is not None
                and self.verdict is None
                and not self.evidence_ids
            )
        elif self.op == "DELETE":
            valid = (
                self.target_id is not None
                and self.section_index is None
                and self.claim is None
                and self.verdict is None
                and not self.evidence_ids
            )
        else:
            valid = (
                self.target_id is not None
                and self.section_index is None
                and self.claim is None
                and self.verdict is not None
                and (self.verdict != "supported" or bool(self.evidence_ids))
            )
        if not valid:
            raise PydanticCustomError(
                "operation_fields", "operation fields do not match the requested operation"
            )
        return self


class BlueRepairs(StrictModel):
    report_hash: ReportHash
    operations: list[RepairOperation] = Field(min_length=1, max_length=8)


def _report_copy(report: ResearchReport, *, code: str = "review_validation") -> ResearchReport:
    if not isinstance(report, ResearchReport):
        raise _error(code)
    try:
        # Frozen Pydantic models may still contain mutable lists. Revalidate a
        # detached snapshot instead of trusting model_construct or mutated data.
        result = ResearchReport.model_validate(report.model_dump(mode="json"), strict=True)
        result.model_dump_json().encode("utf-8")
        return result
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise _error(code) from None


def _fingerprint(report: ResearchReport) -> str:
    encoded = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def report_fingerprint(report: ResearchReport) -> str:
    """Canonical content hash, not a publisher signature or proof of correctness."""
    try:
        return _fingerprint(_report_copy(report))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise _error() from None


def _inventory(report: ResearchReport) -> list[dict]:
    items = []
    for section_index, section in enumerate(report.sections):
        for claim_index, claim in enumerate(section.claims):
            items.append(
                {
                    "claim_id": f"C{len(items) + 1}",
                    "section_index": section_index,
                    "claim_index": claim_index,
                    "text": claim.text,
                    "citations": list(claim.citations),
                }
            )
    return items


def claim_inventory(report: ResearchReport) -> list[dict]:
    """Return detached C1..C12 identifiers in original section/claim order."""
    return _inventory(_report_copy(report))


def _evidence_ids(evidence: list[dict], *, code: str) -> set[str]:
    if not isinstance(evidence, list):
        raise _error(code)
    ids: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict):
            raise _error(code)
        key = item.get("evidence_id")
        if not isinstance(key, str) or _EVIDENCE_ID.fullmatch(key) is None or key in ids:
            raise _error(code)
        ids.add(key)
    return ids


def _known_report_references(report: ResearchReport, evidence_ids: set[str], *, code: str) -> None:
    if any(
        citation not in evidence_ids
        for section in report.sections
        for claim in section.claims
        for citation in claim.citations
    ):
        raise _error(code)


def validate_red(review: RedReview, report: ResearchReport, evidence: list[dict]) -> None:
    """Validate snapshot and reference existence, not the review's factual assertions."""
    snapshot = _report_copy(report)
    if not isinstance(review, RedReview):
        raise _error()
    try:
        checked = RedReview.model_validate(review.model_dump(mode="json"), strict=True)
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise _error() from None
    if checked.report_hash != _fingerprint(snapshot):
        raise _error()
    evidence_ids = _evidence_ids(evidence, code="review_validation")
    _known_report_references(snapshot, evidence_ids, code="review_validation")
    claim_ids = {item["claim_id"] for item in _inventory(snapshot)}
    finding_ids: set[str] = set()
    for finding in checked.findings:
        if (
            finding.id in finding_ids
            or not set(finding.claim_ids) <= claim_ids
            or not set(finding.evidence_ids) <= evidence_ids
        ):
            raise _error()
        finding_ids.add(finding.id)


def apply_repairs(
    report: ResearchReport, repairs: BlueRepairs, *, evidence: list[dict]
) -> tuple[ResearchReport, list[dict]]:
    """Apply a full batch or reject it. No operation ever mutates the input report.

    VERIFY only records a model's evidence judgement. Unsupported/uncertain
    judgements add bounded caveats; they do not silently replace claim text.
    """
    code = "review_patch"
    snapshot = _report_copy(report, code=code)
    if not isinstance(repairs, BlueRepairs):
        raise _error(code)
    try:
        checked = BlueRepairs.model_validate(repairs.model_dump(mode="json"), strict=True)
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise _error(code) from None
    fingerprint = _fingerprint(snapshot)
    if checked.report_hash != fingerprint:
        raise _error(code)
    evidence_ids = _evidence_ids(evidence, code=code)
    _known_report_references(snapshot, evidence_ids, code=code)
    inventory = {item["claim_id"]: item for item in _inventory(snapshot)}
    targeted: set[str] = set()
    deleted: set[str] = set()
    modified: dict[str, dict] = {}
    additions: dict[int, list[dict]] = {}
    verifications: list[dict] = []
    limitations = list(snapshot.limitations)
    for operation in checked.operations:
        target = operation.target_id
        if target is not None:
            if target not in inventory or target in targeted:
                raise _error(code)
            targeted.add(target)
        if operation.claim is not None and not set(operation.claim.citations) <= evidence_ids:
            raise _error(code)
        if not set(operation.evidence_ids) <= evidence_ids:
            raise _error(code)
        if operation.op == "ADD":
            section = operation.section_index
            if section is None or section >= len(snapshot.sections):
                raise _error(code)
            additions.setdefault(section, []).append(operation.claim.model_dump(mode="json"))
        elif operation.op == "MODIFY":
            modified[target] = operation.claim.model_dump(mode="json")
        elif operation.op == "DELETE":
            deleted.add(target)
        else:
            verifications.append(
                {
                    "report_hash": fingerprint,
                    "target_id": target,
                    "verdict": operation.verdict,
                    "evidence_ids": list(operation.evidence_ids),
                    "rationale": operation.rationale,
                }
            )
            if operation.verdict != "supported":
                assessment = (
                    "证据支持不足" if operation.verdict == "unsupported" else "支持情况不确定"
                )
                caveat = (
                    f"蓝方将原报告快照 {fingerprint[:12]} 的 {target} 标记为{assessment}；"
                    "这是审查意见，尚未独立核验。"
                )
                if caveat not in limitations:
                    if len(limitations) >= 6:
                        raise _error(code)
                    limitations.append(caveat)
    sections: list[dict] = []
    claim_number = 0
    for section_index, section in enumerate(snapshot.sections):
        claims: list[dict] = []
        for claim in section.claims:
            claim_number += 1
            claim_id = f"C{claim_number}"
            if claim_id not in deleted:
                claims.append(modified.get(claim_id, claim.model_dump(mode="json")))
        claims.extend(additions.get(section_index, []))
        if claims:
            sections.append({"heading": section.heading, "claims": claims})
    try:
        revised = ResearchReport.model_validate(
            {"title": snapshot.title, "sections": sections, "limitations": limitations}, strict=True
        )
        _known_report_references(revised, evidence_ids, code=code)
        _fingerprint(revised)  # UTF-8 validity also matters before returning a usable revision.
    except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError):
        raise _error(code) from None
    return revised, verifications
