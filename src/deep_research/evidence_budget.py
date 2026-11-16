"""Lossless source-span compaction, then marginal retrieval-score per JSON-byte filling."""

import numpy as np

from deep_research.compression_runtime import json_bytes
from deep_research.evidence_selection import Selection
from deep_research.rewrite_selection import contains, protect


def allocate_marginal(
    base, chunks, scores, sources, *, max_bytes, max_items=16, existing=None, next_id=1
):
    existing = list(existing or [])
    kept = [dict(e) for e in base.evidence]
    limit = min(max_bytes, json_bytes([*existing, *kept]))
    texts = {s["source_id"]: s["text"] for s in sources}
    merges = 0

    def numbered(rows):
        return [{**e, "evidence_id": f"E{next_id + i}"} for i, e in enumerate(rows)]

    # Merge only touching/whitespace-adjacent spans, never erase source characters.
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(kept):
            for j in range(i + 1, len(kept)):
                b = kept[j]
                if a["source_id"] != b["source_id"]:
                    continue
                left, right = sorted((a, b), key=lambda e: e["start"])
                gap = texts[a["source_id"]][left["end"] : right["start"]]
                start, end = min(a["start"], b["start"]), max(a["end"], b["end"])
                if (
                    end - start > 600
                    or right["start"] > left["end"]
                    and (gap.strip() or len(gap) > 2)
                ):
                    continue
                merged = dict(
                    source_id=a["source_id"],
                    start=start,
                    end=end,
                    quote=texts[a["source_id"]][start:end],
                )
                proposal = numbered([merged if k == i else e for k, e in enumerate(kept) if k != j])
                if json_bytes([*existing, *proposal]) < json_bytes([*existing, *kept]):
                    kept = proposal
                    merges += 1
                    changed = True
                    break
            if changed:
                break
    compact_bytes = json_bytes([*existing, *kept])
    receipts = []
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(chunks) or not np.isfinite(values).all():
        raise ValueError("Invalid marginal candidate scores")
    normal = (
        values / np.maximum(values.max(axis=1, keepdims=True), 1e-12) if len(chunks) else values
    )
    assigned = np.zeros(len(values), dtype=float)
    for e in kept:
        indices = [i for i, c in enumerate(chunks) if contains(e, c)]
        if indices:
            assigned[int(np.argmax(normal[:, indices].max(axis=1)))] += 1
    pool = sorted({int(i) for row in values for i in np.argsort(-row, kind="stable")[:12]})
    for _ in range(max_items):
        best = None
        for index in pool:
            span = chunks[index]
            if any(contains(e, span) for e in [*existing, *kept]):
                continue
            covered = set()
            for e in [*existing, *kept]:
                if e["source_id"] == span["source_id"]:
                    covered.update(
                        range(max(e["start"], span["start"]), min(e["end"], span["end"]))
                    )
            novelty = 1 - len(covered) / max(1, span["end"] - span["start"])
            gain = float(np.sum(normal[:, index] / (1 + assigned))) * novelty
            if gain <= 0:
                continue
            option = dict(span)
            for e in kept:
                if e["source_id"] != option["source_id"]:
                    continue
                left, right = sorted((e, option), key=lambda p: p["start"])
                gap = texts[e["source_id"]][left["end"] : right["start"]]
                start, end = min(e["start"], option["start"]), max(e["end"], option["end"])
                if end - start <= 600 and (
                    right["start"] <= left["end"] or not gap.strip() and len(gap) <= 2
                ):
                    option = dict(
                        source_id=e["source_id"],
                        start=start,
                        end=end,
                        quote=texts[e["source_id"]][start:end],
                    )
            proposed = protect(
                Selection(kept, base.audit),
                Selection([option], {"context_bytes": json_bytes([*existing, option])}),
                sources,
                max_bytes=limit,
                max_items=max_items,
                existing=existing,
                next_id=next_id,
                allow_split=True,
            )
            if proposed.evidence == kept:
                continue
            cost = json_bytes([*existing, *proposed.evidence]) - json_bytes([*existing, *kept])
            utility = gain / max(16, cost)
            if best is None or utility > best[0]:
                best = (utility, index, gain, cost, proposed.evidence)
        if best is None:
            break
        utility, index, gain, cost, kept = best
        assigned[int(np.argmax(normal[:, index] / (1 + assigned)))] += 1
        receipts.append(
            dict(
                candidate=index,
                estimated_marginal_gain=gain,
                incremental_json_bytes=cost,
                utility_per_byte=utility,
            )
        )
    assert all(any(contains(e, p) for e in kept) for p in base.evidence)
    assert json_bytes([*existing, *kept]) <= limit and len(existing) + len(kept) <= max_items
    audit = dict(base.audit)
    audit.update(
        context_bytes=json_bytes([*existing, *kept]),
        selected=[dict(evidence_id=e["evidence_id"]) for e in kept],
        adaptive_budget=dict(
            version=1,
            policy="lossless-compact-then-marginal-per-byte",
            base_bytes=limit,
            compact_bytes=compact_bytes,
            adjacent_merges=merges,
            additions=receipts,
            limitation="Estimated retrieval benefit, not factual coverage or support",
        ),
    )
    return Selection(kept, audit)
