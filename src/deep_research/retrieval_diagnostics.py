"""Post-hoc interval diagnostics. Gold labels never enter ranking or packing."""


def covers(spans, gold):
    cursor = gold["start"]
    for span in sorted(spans, key=lambda s: s["start"]):
        if span["source_id"] == gold["source_id"] and span["start"] <= cursor < span["end"]:
            cursor = max(cursor, span["end"])
    return cursor >= gold["end"]


def diagnose(
    gold, *, candidates, eligible, selected, wide_bytes, wide_items, orders, raw_candidates=None
):
    """Shadow probes diagnose constraints, not independent causal effects.

    wide_bytes uses the existing 12KB cap; wide_items uses 32 slots under that cap.
    Top-32 rank is descriptive only, not an invented cutoff in the baseline.
    """
    rows = []
    for i, label in enumerate(gold):
        hit = covers(selected, label)
        raw = covers(raw_candidates if raw_candidates is not None else candidates, label)
        after_cleaning = covers(candidates, label)
        one = any(covers([c], label) for c in candidates)
        admitted = covers(eligible, label)
        large = covers(wide_bytes, label)
        many = covers(wide_items, label)
        if hit:
            reason = "hit"
        elif not raw:
            reason = "candidate_gap"
        elif not after_cleaning:
            reason = "cleaning"
        elif not admitted:
            reason = "eligibility"
        elif large:
            reason = "byte_budget"
        elif many:
            reason = "item_limit_or_rank"
        else:
            reason = "ranking_or_context"
        ranks = []
        for order in orders:
            ranks.append(
                next(
                    (
                        rank
                        for rank, index in enumerate(order, 1)
                        if covers([candidates[index]], label)
                    ),
                    None,
                )
            )
        rows.append(
            dict(
                gold_index=i,
                source_id=label["source_id"],
                start=label["start"],
                end=label["end"],
                hit=hit,
                lost_at=reason,
                candidate_union_coverage=raw,
                after_cleaning_coverage=after_cleaning,
                single_candidate_coverage=one,
                fragmented=raw and not one,
                eligible_coverage=admitted,
                wide_byte_probe=large,
                wide_item_probe=many,
                first_full_candidate_rank_by_point=ranks,
            )
        )
    return rows
