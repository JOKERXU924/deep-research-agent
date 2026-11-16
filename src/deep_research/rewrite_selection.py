"""Original-query span protection; translated hints may add, never evict anchors.

This is an evidence-selection invariant, not a factual-support guarantee. The
subsequent model reranker still makes its own bounded selection.
"""

from deep_research.compression_runtime import json_bytes
from deep_research.evidence_selection import Selection
from deep_research.evidence_selection_v3 import MAX_WINDOW_CHARS

VERSION = "original-span-guard-v1"


def contains(outer, inner):
    return (
        outer["source_id"] == inner["source_id"]
        and outer["start"] <= inner["start"]
        and outer["end"] >= inner["end"]
    )


def protect(
    original,
    supplemental,
    sources,
    *,
    max_bytes,
    max_items,
    existing=None,
    next_id=1,
    allow_split=False,
):
    """Keep original spans at this exact budget. Only union overlapping source text.

    Existing context is immutable. A union is allowed only within one source,
    <=600 characters, and the complete renumbered JSON must fit the budget.
    No labels, generated facts, or substring heuristics decide protected spans.
    """
    existing = list(existing or [])
    kept = [dict(e) for e in original.evidence]
    texts = {s["source_id"]: s["text"] for s in sources}
    receipts = []

    def numbered(rows):
        return [{**e, "evidence_id": f"E{next_id + i}"} for i, e in enumerate(rows)]

    for candidate in supplemental.evidence:
        span = {k: candidate[k] for k in ("source_id", "start", "end", "quote")}
        if any(contains(e, span) for e in [*existing, *kept]):
            receipts.append(
                dict(**{k: span[k] for k in ("source_id", "start", "end")}, status="covered")
            )
            continue
        overlaps = [
            i
            for i, e in enumerate(kept)
            if e["source_id"] == span["source_id"]
            and e["start"] < span["end"]
            and span["start"] < e["end"]
        ]
        proposal = list(kept)
        split_spans = []
        if overlaps:
            left = min([span["start"], *[kept[i]["start"] for i in overlaps]])
            right = max([span["end"], *[kept[i]["end"] for i in overlaps]])
            if right - left > MAX_WINDOW_CHARS:
                if not allow_split:
                    receipts.append(
                        dict(
                            source_id=span["source_id"],
                            start=span["start"],
                            end=span["end"],
                            status="window_limit",
                        )
                    )
                    continue
                cursor = span["start"]
                for item in sorted([*existing, *kept], key=lambda e: e["start"]):
                    if item["source_id"] != span["source_id"] or item["end"] <= cursor:
                        continue
                    if item["start"] >= span["end"]:
                        continue
                    if item["start"] > cursor:
                        split_spans.append((cursor, item["start"]))
                    cursor = max(cursor, item["end"])
                if cursor < span["end"]:
                    split_spans.append((cursor, span["end"]))
                proposal = [
                    *kept,
                    *[
                        dict(
                            source_id=span["source_id"],
                            start=a,
                            end=b,
                            quote=texts[span["source_id"]][a:b],
                        )
                        for a, b in split_spans
                    ],
                ]
            else:
                merged = dict(
                    source_id=span["source_id"],
                    start=left,
                    end=right,
                    quote=texts[span["source_id"]][left:right],
                )
                proposal = [
                    merged if i == overlaps[0] else e
                    for i, e in enumerate(kept)
                    if i == overlaps[0] or i not in overlaps
                ]
        else:
            proposal.append(span)
        proposal = numbered(proposal)
        status = "split_added" if split_spans else "merged" if overlaps else "added"
        if len(existing) + len(proposal) > max_items:
            status = "item_limit"
        elif json_bytes([*existing, *proposal]) > max_bytes:
            status = "byte_limit"
        else:
            kept = proposal
        receipts.append(
            dict(
                source_id=span["source_id"],
                start=span["start"],
                end=span["end"],
                status=status,
                **({"split_spans": split_spans} if split_spans else {}),
            )
        )
    kept = numbered(kept)
    # This invariant is checked over real offsets, not scores or assumed relevance.
    assert all(any(contains(e, p) for e in kept) for p in original.evidence)
    assert json_bytes([*existing, *kept]) <= max_bytes
    assert len(existing) + len(kept) <= max_items
    audit = dict(original.audit)
    audit.update(
        version=VERSION,
        max_bytes=max_bytes,
        context_bytes=json_bytes([*existing, *kept]),
        rewrite_targets=supplemental.audit.get("rewrite_targets", 0),
        selected=[dict(evidence_id=e["evidence_id"]) for e in kept],
        rewrite_guard=dict(
            version=1,
            method="protect-original-query-spans",
            protected_spans=[
                {k: e[k] for k in ("source_id", "start", "end")} for e in original.evidence
            ],
            original_context_bytes=json_bytes([*existing, *original.evidence]),
            unguarded_context_bytes=supplemental.audit["context_bytes"],
            unguarded_selection=supplemental.audit,
            original_selection=original.audit,
            additions=sum(r["status"] == "added" for r in receipts),
            merges=sum(r["status"] == "merged" for r in receipts),
            decisions=receipts,
        ),
    )
    return Selection(kept, audit)
