"""Validated exact-source coverage. Gaps, negations and source versions are not inferred."""

import hashlib


def coverage(evidence, gold, sources):
    index = {s["source_id"]: s for s in sources}
    if len(index) != len(sources):
        raise ValueError("Duplicate source identity")
    for source in sources:
        if hashlib.sha256(source["text"].encode()).hexdigest() != source["sha256"]:
            raise ValueError("Source hash mismatch")
    for span in [*evidence, *gold]:
        source = index.get(span["source_id"])
        if (
            source is None
            or type(span["start"]) is not int
            or type(span["end"]) is not int
            or not 0 <= span["start"] < span["end"] <= len(source["text"])
            or source["text"][span["start"] : span["end"]] != span["quote"]
        ):
            raise ValueError("Invalid evidence/gold interval")
    rows = []
    for label in gold:
        relevant = sorted(
            (
                e
                for e in evidence
                if e["source_id"] == label["source_id"]
                and e["start"] < label["end"]
                and e["end"] > label["start"]
            ),
            key=lambda e: (e["start"], e["end"]),
        )
        cursor, covered, gaps = label["start"], 0, []
        for span in relevant:
            left, right = max(label["start"], span["start"]), min(label["end"], span["end"])
            if left > cursor:
                gaps.append([cursor, left])
            covered += max(0, right - max(cursor, left))
            cursor = max(cursor, right)
        if cursor < label["end"]:
            gaps.append([cursor, label["end"]])
        rows.append(
            dict(
                single_span=any(
                    e["start"] <= label["start"] and e["end"] >= label["end"] for e in relevant
                ),
                union_full=not gaps,
                covered_chars=covered,
                gold_chars=label["end"] - label["start"],
                missing_intervals=gaps,
                source_id=label["source_id"],
                start=label["start"],
                end=label["end"],
            )
        )
    return dict(
        single_hits=sum(r["single_span"] for r in rows),
        union_hits=sum(r["union_full"] for r in rows),
        gold_count=len(gold),
        labels=rows,
    )
