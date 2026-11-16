"""Blind human-review packets. Pending labels never become claimed accuracy."""

import hashlib
import json
from pathlib import Path

from deep_research.research import validate_citations
from deep_research.research_models import ResearchReport
from deep_research.review_models import claim_inventory
from deep_research.robust_contracts import research_requirements


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def build_packet(items, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    packets = []
    ratings = []
    mapping = []
    markdown = [
        "# 人工复核包",
        "",
        "先独立阅读问题、报告与原文，再填写 ratings.json。不要用模型意见代替人工标签。",
        "",
    ]
    for index, item in enumerate(
        sorted(items, key=lambda r: fingerprint(r["bundle"]["report"])), 1
    ):
        bundle = item["bundle"]
        report = ResearchReport.model_validate(bundle["report"])
        validate_citations(report, bundle["sources"], bundle["evidence"])
        case_id = f"R{index}"
        claims = claim_inventory(report)
        points = research_requirements(item["question"])
        packet = dict(
            case_id=case_id,
            question=item["question"],
            report=report.model_dump(),
            claims=claims,
            requirements=points,
            evidence=bundle["evidence"],
            sources=bundle["sources"],
        )
        packet["snapshot_hash"] = fingerprint(packet)
        packets.append(packet)
        ratings.append(
            dict(
                case_id=case_id,
                snapshot_hash=packet["snapshot_hash"],
                claims=[dict(claim_id=c["claim_id"], label=None, rationale="") for c in claims],
                requirements=[dict(point_id=p["id"], label=None, rationale="") for p in points],
                question_issue="",
            )
        )
        mapping.append(
            dict(case_id=case_id, variant=item["variant"], original_id=item.get("case_id"))
        )
        markdown.extend([f"## {case_id}", item["question"], ""])
        for c in claims:
            markdown.append(f"- {c['claim_id']}: {c['text']} [{', '.join(c['citations'])}]")
        markdown.append("")
        for e in bundle["evidence"]:
            s = next(s for s in bundle["sources"] if s["source_id"] == e["source_id"])
            markdown.extend(
                [
                    f"### {case_id} / {e['evidence_id']}",
                    f"来源：{s.get('url', s['source_id'])}；字符区间 [{e['start']}, {e['end']})",
                    e["quote"],
                    "",
                ]
            )
    payload = dict(version=1, reviewer="", independent_human_review=False, cases=ratings)
    instructions = (
        "claims.label: supported / partial / contradicted / insufficient / non_factual；"
        "requirements.label: answered / partial / unanswered / unanswerable_with_sources。"
        "每项填写原文依据 rationale；完成后填写 reviewer。"
        "独立人工复核属实后才设 independent_human_review=true。"
        "空标签表示待评，禁止自动当作正确或用模型回填。"
    )
    for name, value in [
        ("packet.json", dict(version=1, cases=packets, instructions=instructions)),
        ("ratings.json", payload),
        ("unblind-after-review.json", mapping),
    ]:
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "READ-ME.md").write_text("\n".join([instructions, "", *markdown]), encoding="utf-8")
    return dict(
        cases=len(packets),
        claims=sum(len(p["claims"]) for p in packets),
        status="pending_independent_human_review",
    )


def score_packet(packet, ratings):
    canonical = {c["case_id"]: c for c in packet["cases"]}
    rows = ratings["cases"]
    if len(rows) != len(canonical) or {r["case_id"] for r in rows} != canonical.keys():
        raise ValueError("Missing or duplicate review cases")
    claim_labels = []
    point_labels = []
    pending = 0
    for row in rows:
        case = canonical[row["case_id"]]
        if (
            row["snapshot_hash"] != case["snapshot_hash"]
            or fingerprint({k: v for k, v in case.items() if k != "snapshot_hash"})
            != case["snapshot_hash"]
        ):
            raise ValueError("Review snapshot changed")
        for key, id_field, expected, allowed, dest in [
            (
                "claims",
                "claim_id",
                {c["claim_id"] for c in case["claims"]},
                {"supported", "partial", "contradicted", "insufficient", "non_factual"},
                claim_labels,
            ),
            (
                "requirements",
                "point_id",
                {p["id"] for p in case["requirements"]},
                {"answered", "partial", "unanswered", "unanswerable_with_sources"},
                point_labels,
            ),
        ]:
            values = row[key]
            if len(values) != len(expected) or {v[id_field] for v in values} != expected:
                raise ValueError("Incomplete or unknown annotation IDs")
            for value in values:
                label = value["label"]
                if label is None:
                    pending += 1
                    continue
                if (
                    label not in allowed
                    or not isinstance(value.get("rationale"), str)
                    or not value["rationale"].strip()
                ):
                    raise ValueError("Invalid label or missing source rationale")
                dest.append(label)
    complete = not pending
    confirmed = (
        complete
        and bool(ratings.get("reviewer", "").strip())
        and ratings.get("independent_human_review") is True
    )
    facts = [v for v in claim_labels if v != "non_factual"]
    return dict(
        status="completed_independent_review" if confirmed else "pending_independent_human_review",
        pending_labels=pending,
        labels_filled=len(claim_labels) + len(point_labels),
        independently_confirmed=confirmed,
        supported_fact_pct=100 * facts.count("supported") / len(facts)
        if confirmed and facts
        else None,
        unsupported_fact_pct=100
        * sum(v in {"contradicted", "insufficient"} for v in facts)
        / len(facts)
        if confirmed and facts
        else None,
        fully_answered_point_pct=100 * point_labels.count("answered") / len(point_labels)
        if confirmed and point_labels
        else None,
    )
