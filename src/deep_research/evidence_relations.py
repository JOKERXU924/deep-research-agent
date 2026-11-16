"""Bounded API attribution and referential/conditional source windows.

API names select bounded dependency windows; they never change evidence text.
No dataset labels, API-specific names, generated facts or model calls.
"""

import re
from dataclasses import dataclass

from deep_research.compression_runtime import json_bytes
from deep_research.evidence_selection import Selection
from deep_research.evidence_selection_v3 import MAX_WINDOW_CHARS, navigation

_API = re.compile(
    r"^(?:(?:async\s+)?(?:def|class)\s+|@)?[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*\([^\n]{0,220}\)\s*$"
)
_CLASS = re.compile(r"^class\s+[A-Za-z_]")
_REF = re.compile(r"^(?:the|this)\s+(?:function|method|return(?:ed)?\b|result\b)", re.I)
_CONDITION = re.compile(r"(?:^|(?<=[.!?])\s+)(?:if|unless|only if|provided that)\b", re.I)
_SCOPE_LIMIT = 1200


@dataclass
class Linked:
    windows: list[dict]
    audit: dict


def link(ranked):
    owners, windows = {}, []
    texts = {s["source_id"]: s["text"] for s in ranked.sources}
    for source in ranked.sources:
        chunks = [
            (i, c) for i, c in enumerate(ranked.chunks) if c["source_id"] == source["source_id"]
        ]
        owner = None
        for pos, (i, c) in enumerate(chunks):
            value = c["quote"].strip()
            if _API.fullmatch(value):
                owner = c
            elif _CLASS.match(value) or navigation(value):
                owner = None
            if owner is None or c["end"] - owner["start"] > _SCOPE_LIMIT:
                continue
            owners[i] = owner
            if _REF.match(value) and c["end"] - owner["start"] <= MAX_WINDOW_CHARS:
                windows.append((owner["start"], c["end"], owner, "api_reference"))
            for match in _CONDITION.finditer(c["quote"]):
                left = c["start"] + match.start()
                while left < c["end"] and source["text"][left].isspace():
                    left += 1
                right = c["end"]
                linked = False
                for _, following in chunks[pos + 1 : pos + 3]:
                    gap = source["text"][right : following["start"]]
                    if gap.strip() or _API.fullmatch(following["quote"].strip()):
                        break
                    if following["start"] != right and not _REF.match(following["quote"].strip()):
                        break
                    if following["end"] - left > MAX_WINDOW_CHARS:
                        break
                    right = following["end"]
                    linked = True
                if linked:
                    windows.append((left, right, owner, "conditional_continuation"))
    records = []
    seen = set()
    for left, right, owner, reason in windows:
        key = (owner["source_id"], left, right)
        if key in seen or len(records) >= 64:
            continue
        seen.add(key)
        records.append(
            dict(
                source_id=key[0],
                start=left,
                end=right,
                quote=texts[key[0]][left:right],
                api=owner["quote"],
                reason=reason,
            )
        )
    return Linked(
        records,
        dict(
            version=1,
            method="bounded-api-dependencies",
            scope_max_chars=_SCOPE_LIMIT,
            window_max_chars=MAX_WINDOW_CHARS,
            windows=records,
            source_candidate_count=len(ranked.chunks),
        ),
    )


_COMPLEXITY = re.compile(r"复杂度|\bcomplexity\b|\bO\s*\(", re.I)
_PARTITION = re.compile(r"分区|划分|\bpartitions?\b|\bslices?\b", re.I)


def obligations(linked, points, rewrites):
    hints = {r["point_id"]: r["query"] for r in rewrites}
    selected = []
    decisions = []
    for point in points:
        query = point["question"] + " " + hints.get(point["id"], "")
        intents = [pattern for pattern in (_COMPLEXITY, _PARTITION) if pattern.search(query)]
        if not intents:
            continue
        useful = []
        for window in linked.windows:
            name = re.search(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", window["api"])
            if not name:
                continue
            leaf = name.group(1).split(".")[-1]
            if not re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(leaf) + r"(?![A-Za-z0-9_])", query, re.I
            ):
                continue
            if not any(pattern.search(window["quote"]) for pattern in intents):
                continue
            useful.append(window)
        if useful:
            window = min(useful, key=lambda w: (w["end"] - w["start"], w["start"]))
            key = (window["source_id"], window["start"], window["end"])
            if not any((e["source_id"], e["start"], e["end"]) == key for e in selected):
                selected.append(
                    {
                        "evidence_id": f"E{len(selected) + 1}",
                        **{k: window[k] for k in ("source_id", "start", "end", "quote")},
                    }
                )
                decisions.append(
                    dict(
                        point_id=point["id"],
                        api=window["api"],
                        source_id=key[0],
                        start=key[1],
                        end=key[2],
                        reason=window["reason"],
                    )
                )
        if len(selected) >= 2:
            break
    return Selection(
        selected, dict(context_bytes=json_bytes(selected), selected_relations=decisions)
    )
