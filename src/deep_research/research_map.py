"""Versioned question/claim/evidence/gap snapshot; links are candidates, not entailment."""

import hashlib
import json
import re
from typing import Literal, Self

from pydantic import Field, model_validator

from deep_research.relevance import _terms
from deep_research.research_models import ResearchReport
from deep_research.review_models import claim_inventory
from deep_research.schemas import StrictModel


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Point(StrictModel):
    id: str = Field(pattern=r"^P[1-8]$")
    question: str = Field(min_length=1, max_length=1000)
    origin: Literal["question_rule"] = "question_rule"
    claim_ids: list[str] = Field(max_length=12)
    gap_ids: list[str] = Field(max_length=64)
    status: Literal["no_conclusion", "needs_review", "review_flagged"]


class Conclusion(StrictModel):
    id: str = Field(pattern=r"^C(?:[1-9]|1[0-2])$")
    text: str = Field(min_length=1, max_length=700)
    section_index: int = Field(ge=0, le=3)
    claim_index: int = Field(ge=0, le=5)
    evidence_ids: list[str] = Field(max_length=4)
    point_ids: list[str] = Field(max_length=8)
    link_method: Literal["lexical_candidate", "unassigned"]


class Evidence(StrictModel):
    id: str
    source_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=65536)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class Gap(StrictModel):
    id: str = Field(pattern=r"^G[1-9][0-9]?$")
    point_ids: list[str] = Field(max_length=8)
    claim_ids: list[str] = Field(max_length=12)
    kind: Literal["no_conclusion", "missing_evidence", "support_unverified", "review_finding"]
    description: str = Field(min_length=1, max_length=700)
    status: Literal["open"] = "open"
    priority: Literal["high", "medium", "low"]
    origin: Literal["structural", "model_review"]
    finding_id: str | None = None
    suggested_queries: list[str] = Field(default_factory=list, max_length=2)


class ResearchMap(StrictModel):
    version: Literal[1] = 1
    method: Literal["question-lexical-v1"] = "question-lexical-v1"
    report_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    question_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    points: list[Point] = Field(min_length=1, max_length=8)
    conclusions: list[Conclusion] = Field(max_length=12)
    evidence: list[Evidence] = Field(max_length=999)
    gaps: list[Gap] = Field(max_length=64)
    review_binding: Literal["matched", "unavailable", "not_enabled"]

    @model_validator(mode="after")
    def references(self) -> Self:
        collections = [self.points, self.conclusions, self.evidence, self.gaps]
        sets = [{v.id for v in values} for values in collections]
        if any(len(keys) != len(values) for keys, values in zip(sets, collections, strict=True)):
            raise ValueError("Duplicate research map IDs")
        p, c, e, g = sets
        for point in self.points:
            if not set(point.claim_ids) <= c or not set(point.gap_ids) <= g:
                raise ValueError("Dangling point reference")
            if set(point.claim_ids) != {v.id for v in self.conclusions if point.id in v.point_ids}:
                raise ValueError("Asymmetric conclusion link")
            if set(point.gap_ids) != {v.id for v in self.gaps if point.id in v.point_ids}:
                raise ValueError("Asymmetric gap link")
        if any(not set(v.point_ids) <= p or not set(v.evidence_ids) <= e for v in self.conclusions):
            raise ValueError("Dangling conclusion reference")
        if any(not set(v.point_ids) <= p or not set(v.claim_ids) <= c for v in self.gaps):
            raise ValueError("Dangling gap reference")
        return self


def question_points(question):
    """Before writing: preserve all question text, at most eight provisional requirements."""
    pieces = [v.strip() for v in re.split(r"[？?！!；;\n]+", question) if v.strip()]
    expanded = []
    interrogative = r"为什么|如何|什么|哪些|是否|怎样|多少|何时|能否"
    for piece in pieces:
        # Separate two explicit questions, but keep a leading condition with its question.
        halves = re.split(r"[，,]", piece, maxsplit=1)
        if len(halves) == 2 and all(re.search(interrogative, h) for h in halves):
            expanded.extend(halves)
            continue
        compare = re.fullmatch(r"(?:请)?(?:比较|对比|评估)([^，,]+)(?:[，,](.+))?", piece)
        if compare and "、" in compare[1]:
            expanded.extend("比较：" + item for item in re.split(r"、|以及|和", compare[1]) if item)
            if compare[2]:
                expanded.append(compare[2])
        else:
            expanded.append(piece)
    pieces = expanded
    if len(pieces) > 8:
        pieces = pieces[:7] + ["；".join(pieces[7:])]
    return [dict(id=f"P{i}", question=q) for i, q in enumerate(pieces or [question], 1)]


def _findings(bundle):
    review = bundle.get("review")
    if not review:
        return [], "not_enabled"
    # Use exactly the selected report's review, never the last chronological round.
    from deep_research.review_models import RedReview, report_fingerprint

    expected = report_fingerprint(ResearchReport.model_validate(bundle["report"]))
    if review.get("best_hash") != expected:
        return [], "unavailable"
    matches = [
        v["event"]
        for v in bundle.get("review_events", [])
        if v.get("event", {}).get("kind") == "red"
        and v["event"].get("round") == review.get("best_round")
        and v["event"].get("report_hash") == expected
        and v["event"].get("review", {}).get("report_hash") == expected
    ]
    if len(matches) != 1:
        return [], "unavailable"
    try:
        findings = [f.model_dump() for f in RedReview.model_validate(matches[0]["review"]).findings]
    except ValueError:
        return [], "unavailable"
    known = {
        c["claim_id"] for c in claim_inventory(ResearchReport.model_validate(bundle["report"]))
    }
    if any(not set(f["claim_ids"]) <= known for f in findings):
        return [], "unavailable"
    if len(findings) != review.get("finding_count"):
        return [], "unavailable"
    return findings, "matched"


def build_research_map(bundle, *, points=None):
    points = [
        dict(p, claim_ids=[], gap_ids=[], status="no_conclusion")
        for p in (points if points is not None else question_points(bundle["question"]))
    ]
    claims = claim_inventory(ResearchReport.model_validate(bundle["report"]))
    sources = {s["source_id"]: s for s in bundle["sources"]}
    archive = {e["evidence_id"]: e for e in bundle["evidence"]}
    evidence, conclusions, gaps = {}, [], []

    def gap(kind, description, point_ids, claim_ids, priority="medium", finding=None):
        item = dict(
            id=f"G{len(gaps) + 1}",
            kind=kind,
            description=description,
            point_ids=point_ids,
            claim_ids=claim_ids,
            priority=priority,
            origin="model_review" if finding else "structural",
            finding_id=finding["id"] if finding else None,
            suggested_queries=finding.get("search_queries", []) if finding else [],
        )
        gaps.append(item)
        for p in points:
            if p["id"] in point_ids:
                p["gap_ids"].append(item["id"])

    for claim in claims:
        wanted = _terms(claim["text"], topic=True)
        links = [p["id"] for p in points if wanted & _terms(p["question"], topic=True)]
        valid = []
        for eid in claim["citations"]:
            e = archive.get(eid)
            s = sources.get(e["source_id"]) if e else None
            if (
                s
                and type(e["start"]) is int
                and type(e["end"]) is int
                and 0 <= e["start"] < e["end"] <= len(s["text"])
                and s["text"][e["start"] : e["end"]] == e["quote"]
                and hashlib.sha256(s["text"].encode()).hexdigest() == s["sha256"]
            ):
                valid.append(eid)
                evidence[eid] = dict(
                    id=eid,
                    source_id=e["source_id"],
                    start=e["start"],
                    end=e["end"],
                    quote=e["quote"],
                    source_sha256=s["sha256"],
                )
        conclusions.append(
            dict(
                id=claim["claim_id"],
                text=claim["text"],
                section_index=claim["section_index"],
                claim_index=claim["claim_index"],
                evidence_ids=valid,
                point_ids=links,
                link_method="lexical_candidate" if links else "unassigned",
            )
        )
        for p in points:
            if p["id"] in links:
                p["claim_ids"].append(claim["claim_id"])
                p["status"] = "needs_review"
        if len(valid) != len(claim["citations"]):
            gap(
                "missing_evidence",
                "部分引用不存在或原文位置/哈希校验失败。",
                links,
                [claim["claim_id"]],
                "high",
            )
        gap(
            "support_unverified",
            "引用尚需核对是否支持该结论及其适用条件。",
            links,
            [claim["claim_id"]],
            "low",
        )
    for p in points:
        if not p["claim_ids"]:
            gap("no_conclusion", "尚未关联到回答此要点的候选结论。", [p["id"]], [], "high")
    findings, binding = _findings(bundle)
    for f in findings:
        ids = [c["id"] for c in conclusions if c["id"] in f["claim_ids"]]
        linked = list(
            dict.fromkeys(p for c in conclusions if c["id"] in ids for p in c["point_ids"])
        )
        gap("review_finding", f["description"], linked, ids, f["severity"], f)
        for p in points:
            if p["id"] in linked:
                p["status"] = "review_flagged"
    return ResearchMap.model_validate(
        dict(
            report_hash=fingerprint(bundle["report"]),
            question_hash=fingerprint(bundle["question"]),
            points=points,
            conclusions=conclusions,
            evidence=list(evidence.values()),
            gaps=gaps,
            review_binding=binding,
        )
    ).model_dump()


def validate_research_map(bundle):
    """Read/write guard: a saved projection must still match its report and source snapshot."""
    from deep_research.evidence_gate import validate_publication

    validate_publication(bundle)
    if "research_map" not in bundle:
        return
    ResearchMap.model_validate(bundle["research_map"])
    if bundle["research_map"] != build_research_map(bundle):
        raise ValueError("Research map does not match its saved snapshot")
