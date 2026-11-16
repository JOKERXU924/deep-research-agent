"""Route provisional evidence gaps. Local matches are hypotheses, never closure proofs."""

import re

from pydantic import TypeAdapter

from deep_research.relevance import _terms
from deep_research.review_models import GapQueryText


def reported_conflict_ids(review, report_hash):
    scope = (review or {}).get("scope_review", {})
    return {
        c["claim_id"]
        for s in scope.get("snapshots", [])
        if s.get("round") == scope.get("selected_round") and s.get("report_hash") == report_hash
        for c in s.get("source_conflicts", [])
    }


def plan_gap_actions(
    publication, support, sources, evidence, *, rewrites=(), maximum=2, conflict_claims=()
):
    claims = {c["claim_id"]: c for c in (support or {}).get("claims", [])}
    hints = {r["point_id"]: r["query"] for r in rewrites}
    targets = []
    point_claims = {p["point_id"]: p["claim_ids"] for p in (support or {}).get("points", [])}
    for point in publication.get("points", []):
        ids = set(point.get("claim_ids", [])) | set(point_claims.get(point["id"], []))
        linked = [claims[c] for c in ids if c in claims]
        conflict = bool(ids & set(conflict_claims))
        contradicted = any(c["support"] == "contradicted" for c in linked)
        if point["status"] == "answered" and not (conflict or contradicted):
            continue
        query = point.get("gap_query") or hints.get(point["id"]) or point.get("question", "")[:120]
        try:
            query = TypeAdapter(GapQueryText).validate_python(query)
        except ValueError:
            continue
        wanted = _terms(query, topic=True)
        local = []
        if wanted:
            for source in sources:
                for p in re.finditer(r"[^\n]+", source["text"]):
                    found = _terms(p.group(), topic=True)
                    if len(wanted & found) / len(wanted) < 0.6:
                        continue
                    if any(
                        e["source_id"] == source["source_id"]
                        and e["start"] <= p.start()
                        and e["end"] >= p.end()
                        for e in evidence
                    ):
                        continue
                    local.append(dict(source_id=source["source_id"], start=p.start(), end=p.end()))
                    if len(local) >= 3:
                        break
                if len(local) >= 3:
                    break
        action = (
            "search_alternative"
            if conflict
            else "revise_existing"
            if contradicted
            else "restore_local"
            if local
            else "search_new"
        )
        targets.append(
            dict(
                point_id=point["id"],
                query=query,
                action=action,
                reason="model_reported_source_conflict"
                if conflict
                else "claim_opposes_existing_quote"
                if contradicted
                else "possible_omitted_context"
                if local
                else "no_local_candidate",
                local_candidates=local,
                closure="unverified",
            )
        )
    targets.sort(
        key=lambda t: (
            t["action"] != "search_alternative",
            t["action"] != "restore_local",
            t["point_id"],
        )
    )
    return dict(
        version=1,
        targets=targets[:maximum],
        limitation=(
            "Lexical local candidates are not answers; "
            "closure requires post-repair support assessment"
        ),
    )
