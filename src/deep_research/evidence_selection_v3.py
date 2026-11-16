"""Dependency-aware source windows over immutable v1 short-anchor ranking.

No labels, translation, generated facts, or paid calls. Original text offsets are
preserved. Context expansion may fall back to a short anchor under byte pressure.
"""

import asyncio
import re
from dataclasses import dataclass

import numpy as np

from deep_research.compression_runtime import json_bytes
from deep_research.errors import ResearchError
from deep_research.evidence_selection import Selection, _lexical, _ranks, candidates, pack, targets

VERSION = "dependency-window-v3"
MAX_WINDOW_CHARS = 600
_LINK = re.compile(
    r"^(?:however\b|but\b|unless\b|otherwise\b|provided\b|if\b|this\b|these\b|"
    r"therefore\b|in (?:this|that) case\b|但|然而|如果|否则|上述|这些|因此)",
    re.I,
)
_END = re.compile(r"[。！？.!?][\"”’')\]]?$|[。！？.!?]$")


@dataclass
class Ranked:
    chunks: list[dict]
    aspects: list[str]
    lexical: np.ndarray
    semantic: np.ndarray | None
    orders: list[list[int]]
    eligible: np.ndarray
    sources: list[dict]
    encoder_id: str | None


async def prepare(sources, question, *, queries=None, encoder=None):
    chunks, aspects = candidates(sources), targets(question, queries)
    lexical = _lexical(chunks, aspects)
    semantic = None
    if encoder is not None and chunks:
        q = await encoder.encode_queries(aspects)
        docs = np.concatenate(
            [
                await encoder.encode_documents([c["quote"] for c in chunks[i : i + 128]])
                for i in range(0, len(chunks), 128)
            ]
        )
        if (
            q.shape != (len(aspects), encoder.dimensions)
            or docs.shape != (len(chunks), encoder.dimensions)
            or not np.isfinite(q).all()
            or not np.isfinite(docs).all()
        ):
            raise ResearchError("retrieval_selection", "编码器返回无效向量。")
        qnorm, dnorm = (
            np.linalg.norm(q, axis=1, keepdims=True),
            np.linalg.norm(docs, axis=1, keepdims=True),
        )
        if (qnorm == 0).any() or (dnorm == 0).any():
            raise ResearchError("retrieval_selection", "编码器返回零向量。")
        semantic = (q / qnorm) @ (docs / dnorm).T
    rank = 1 / (20 + _ranks(lexical))
    eligible = lexical > 0
    if semantic is not None:
        rank = 0.5 * rank + 0.5 / (20 + _ranks(semantic))
        eligible |= semantic >= 0.78
    await asyncio.sleep(0)
    return Ranked(
        chunks,
        aspects,
        lexical,
        semantic,
        [np.argsort(-r, kind="stable").tolist() for r in rank],
        eligible,
        sources,
        encoder.encoder_id if encoder else None,
    )


def heading(text):
    return len(text.strip()) <= 80 and not _END.search(text.strip())


def window(ranked, index):
    anchor = ranked.chunks[index]
    source = next(s for s in ranked.sources if s["source_id"] == anchor["source_id"])
    text = source["text"]
    start, end = anchor["start"], anchor["end"]
    neighbors = [
        (i, c) for i, c in enumerate(ranked.chunks) if c["source_id"] == anchor["source_id"]
    ]
    position = next(i for i, (j, _) in enumerate(neighbors) if j == index)
    reasons = []
    if position:
        previous = neighbors[position - 1][1]
        gap = text[previous["end"] : start]
        if (
            not gap.strip()
            and not navigation(previous["quote"])
            and end - previous["start"] <= MAX_WINDOW_CHARS
            and (
                heading(previous["quote"])
                or _LINK.match(anchor["quote"].lstrip())
                or not _END.search(previous["quote"].rstrip())
                and not gap
            )
        ):
            start = previous["start"]
            reasons.append("preceding_dependency")
    # At most two following anchors; preserve lists/conditions without consuming a whole document.
    for _, following in neighbors[position + 1 : position + 3]:
        gap = text[end : following["start"]]
        current = text[start:end].rstrip()
        need = (
            heading(current) or not _END.search(current) or _LINK.match(following["quote"].lstrip())
        )
        if gap.strip() or not need or following["end"] - start > MAX_WINDOW_CHARS:
            break
        end = following["end"]
        reasons.append("following_dependency")
    return dict(source_id=anchor["source_id"], start=start, end=end, quote=text[start:end]), reasons


def baseline(ranked, *, max_bytes, max_items=16, existing=None, next_id=1):
    return pack(
        ranked.chunks,
        ranked.aspects,
        ranked.lexical,
        ranked.semantic,
        max_bytes=max_bytes,
        max_items=max_items,
        existing=existing or [],
        next_id=next_id,
    )


def navigation(text):
    value = text.strip()
    if not value or not any(c.isalnum() for c in value):
        return True
    if len(value) > 100 or _END.search(value):
        return False
    return bool(
        value.endswith(("»", "›"))
        or re.fullmatch(
            r"(?:previous|next)(?: topic)?|modules? \|?|index|this page|"
            r"table of contents|navigation|search|quick search",
            value,
            re.I,
        )
    )


def cleaned(ranked):
    indices = [i for i, c in enumerate(ranked.chunks) if not structural(c["quote"])]
    lexical = ranked.lexical[:, indices]
    semantic = ranked.semantic[:, indices] if ranked.semantic is not None else None
    # Transfer a section/API label's lexical signal to its adjacent body.
    # The label itself is not standalone evidence; navigation never transfers.
    for column, original in enumerate(indices):
        if not original:
            continue
        previous, body = ranked.chunks[original - 1], ranked.chunks[original]
        if (
            previous["source_id"] == body["source_id"]
            and structural(previous["quote"])
            and not navigation(previous["quote"])
            and body["end"] - previous["start"] <= MAX_WINDOW_CHARS
        ):
            text = next(s["text"] for s in ranked.sources if s["source_id"] == body["source_id"])
            if not text[previous["end"] : body["start"]].strip():
                lexical[:, column] = np.maximum(lexical[:, column], ranked.lexical[:, original - 1])
    scores = 1 / (20 + _ranks(lexical))
    eligible = lexical > 0
    if semantic is not None:
        scores = 0.5 * scores + 0.5 / (20 + _ranks(semantic))
        eligible |= semantic >= 0.78
    return Ranked(
        [ranked.chunks[i] for i in indices],
        ranked.aspects,
        lexical,
        semantic,
        [np.argsort(-r, kind="stable").tolist() for r in scores],
        eligible,
        ranked.sources,
        ranked.encoder_id,
    )


def structural(text):
    """Heuristic: bare labels/questions are not standalone declarative evidence.

    They may still accompany a selected body as context. Short unpunctuated
    factual prose can be misclassified; diagnostics report cleaning losses.
    """
    return (
        navigation(text)
        or heading(text)
        or (len(text.strip()) <= 100 and text.strip().endswith(("?", "？")))
    )


def contextual(ranked, *, max_bytes, max_items=16, existing=None, next_id=1):
    if (
        type(max_bytes) is not int
        or not 2 <= max_bytes <= 12000
        or type(max_items) is not int
        or not 1 <= max_items <= 32
    ):
        raise ResearchError("retrieval_selection", "检索上下文预算无效。")
    existing = list(existing or [])
    if json_bytes(existing) > max_bytes or len(existing) > max_items:
        raise ResearchError("retrieval_selection", "已有证据超过预算。")
    original = baseline(
        ranked, max_bytes=max_bytes, max_items=max_items, existing=existing, next_id=next_id
    )
    selected = [dict(e) for e in original.evidence if not structural(e["quote"])]
    filtered = cleaned(ranked)

    def renumber(rows):
        return [{**row, "evidence_id": f"E{next_id + i}"} for i, row in enumerate(rows)]

    selected = renumber(selected)
    first_fill = baseline(
        filtered,
        max_bytes=max_bytes,
        max_items=max_items,
        existing=[*existing, *selected],
        next_id=next_id + len(selected),
    )
    selected.extend(first_fill.evidence)
    events = []
    # Preserve every non-structural baseline span before spending spare bytes.
    for anchor in list(selected):
        index = next(
            (
                i
                for i, c in enumerate(ranked.chunks)
                if c["source_id"] == anchor["source_id"]
                and c["start"] == anchor["start"]
                and c["end"] == anchor["end"]
            ),
            None,
        )
        if index is None:
            continue
        option, reasons = window(ranked, index)
        if not reasons:
            continue
        overlap = [
            j
            for j, e in enumerate(selected)
            if e["source_id"] == option["source_id"]
            and e["start"] < option["end"]
            and option["start"] < e["end"]
        ]
        if not overlap:
            continue
        start = min([option["start"], *[selected[j]["start"] for j in overlap]])
        end = max([option["end"], *[selected[j]["end"] for j in overlap]])
        if end - start > MAX_WINDOW_CHARS:
            continue
        text = next(s["text"] for s in ranked.sources if s["source_id"] == option["source_id"])
        merged = dict(source_id=option["source_id"], start=start, end=end, quote=text[start:end])
        proposal = []
        for j, e in enumerate(selected):
            if j == overlap[0]:
                proposal.append(merged)
            elif j not in overlap:
                proposal.append(e)
        proposal = renumber(proposal)
        accepted = json_bytes([*existing, *proposal]) <= max_bytes
        if accepted:
            selected = proposal
        events.append(
            dict(
                anchor=index,
                expansion=reasons,
                expanded=accepted,
                fallback=not accepted,
                merged=accepted and len(overlap) > 1,
            )
        )
    fill = baseline(
        filtered,
        max_bytes=max_bytes,
        max_items=max_items,
        existing=[*existing, *selected],
        next_id=next_id + len(selected),
    )
    selected.extend(fill.evidence)
    return Selection(
        selected,
        dict(
            version=VERSION,
            targets=ranked.aspects,
            candidates=len(ranked.chunks),
            max_bytes=max_bytes,
            context_bytes=json_bytes([*existing, *selected]),
            encoder_kind="semantic" if ranked.semantic is not None else "lexical",
            encoder_id=ranked.encoder_id,
            selected=[dict(evidence_id=e["evidence_id"]) for e in selected],
            windows=events,
            window_max_chars=MAX_WINDOW_CHARS,
            removed_navigation=sum(navigation(c["quote"]) for c in ranked.chunks),
            removed_structural=len(ranked.chunks) - len(filtered.chunks),
            retained_seed_spans=[
                {k: e[k] for k in ("source_id", "start", "end")}
                for e in original.evidence
                if not structural(e["quote"])
            ],
        ),
    )


async def select_coverage(
    sources,
    question,
    *,
    queries=None,
    encoder=None,
    max_bytes=12000,
    max_items=16,
    existing=None,
    next_id=1,
):
    ranked = await prepare(sources, question, queries=queries, encoder=encoder)
    return contextual(
        ranked, max_bytes=max_bytes, max_items=max_items, existing=existing, next_id=next_id
    )
