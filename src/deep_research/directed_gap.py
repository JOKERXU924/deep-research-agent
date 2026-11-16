"""Select bounded retrieval targets from provisional research-map gaps, not truth verdicts."""

import re

from pydantic import Field

from deep_research.relevance import _terms
from deep_research.research_map import build_research_map
from deep_research.review_models import GapQueryText
from deep_research.schemas import StrictModel


class DirectedQuery(StrictModel):
    point_id: str = Field(pattern=r"^P[1-8]$")
    query: GapQueryText


class DirectedQueries(StrictModel):
    targets: list[DirectedQuery] = Field(min_length=1, max_length=2)


def plan_directed_gaps(bundle, *, maximum=2):
    mapping = build_research_map(bundle)
    conclusions = {c["id"]: c for c in mapping["conclusions"]}
    targets = []
    for point in mapping["points"]:
        if re.match(r"^(?:每点|每个|请引用|引用|资料不足|约\d|用\d+字)", point["question"]):
            continue
        wanted = _terms(point["question"], topic=True)
        coverage = max(
            (
                len(wanted & _terms(conclusions[c]["text"], topic=True)) / max(1, len(wanted))
                for c in point["claim_ids"]
            ),
            default=0,
        )
        if point["status"] == "no_conclusion" or coverage < 0.6:
            # Queries are data only; do not turn arbitrary user URLs/controls into search requests.
            query = point["question"].strip()[:120]
            if "://" in query or "www." in query or any(ord(c) < 32 for c in query):
                continue
            targets.append(
                dict(
                    point_id=point["id"],
                    requirement=point["question"],
                    query=query,
                    reason="no_conclusion" if not point["claim_ids"] else "weak_lexical_coverage",
                    coverage=coverage,
                    gap_ids=point["gap_ids"],
                )
            )
    targets.sort(key=lambda t: (t["reason"] != "no_conclusion", t["coverage"], t["point_id"]))
    return dict(
        base_report_hash=mapping["report_hash"],
        method="map-gap-lexical-v1",
        targets=targets[:maximum],
        before_map=mapping,
        limitation="Provisional lexical gaps; not a semantic completeness or truth check.",
    )
