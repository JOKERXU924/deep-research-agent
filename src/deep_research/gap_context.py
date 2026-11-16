"""Restore the immediately following paragraph when a gap hit is only a heading.

This is gap-only context expansion, not a change to the frozen initial ranker.
Never synthesize text, cross sources, or exceed the original context budget.
"""

import re

from deep_research.compression_runtime import json_bytes


def expand_gap_headings(selected, sources, *, existing, max_bytes):
    index = {s["source_id"]: s for s in sources}
    result = [dict(e) for e in selected]
    for offset, item in enumerate(result):
        quote = item["quote"].strip()
        if len(quote) > 80 or re.search(r"[。！？.!?]$", quote):
            continue
        text = index[item["source_id"]]["text"]
        following = re.search(r"[^\r\n]+", text[item["end"] :])
        if following is None:
            continue
        end = item["end"] + following.end()
        if end - item["start"] > 300:
            continue
        expanded = {**item, "end": end, "quote": text[item["start"] : end]}
        proposed = [*result[:offset], expanded, *result[offset + 1 :]]
        if json_bytes([*existing, *proposed]) <= max_bytes:
            result = proposed
    return result
