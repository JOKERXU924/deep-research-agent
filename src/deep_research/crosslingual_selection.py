"""Per-requirement bilingual RRF with bounded extractive packing; no gold or translations here."""

import asyncio

import numpy as np

from deep_research.compression_runtime import json_bytes
from deep_research.errors import ResearchError
from deep_research.evidence_selection import Selection, _lexical, _ranks, candidates
from deep_research.evidence_selection_v3 import Ranked, structural, window
from deep_research.rewrite_selection import contains, protect
from deep_research.robust_contracts import research_requirements


async def select_bilingual(
    sources,
    question,
    *,
    rewrites,
    encoder,
    max_bytes=12000,
    max_items=16,
    existing=None,
    next_id=1,
    adaptive_budget=False,
):
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
    chunks = candidates(sources)
    points = research_requirements(question)
    original = [p["question"] for p in points]
    hints = {r["point_id"]: r["query"] for r in rewrites}
    translated = [hints.get(p["id"], p["question"]) for p in points]
    audit = dict(
        version="per-point-bilingual-v1",
        targets=original,
        candidates=len(chunks),
        max_bytes=max_bytes,
        encoder_kind="semantic",
        encoder_id=encoder.encoder_id,
        rewrite_targets=sum(p["id"] in hints for p in points),
        policy="paired-query-rrf",
    )
    if not chunks:
        return Selection([], dict(**audit, context_bytes=json_bytes(existing), selected=[]))
    queries = await encoder.encode_queries([*original, *translated])
    docs = np.concatenate(
        [
            await encoder.encode_documents([c["quote"] for c in chunks[i : i + 128]])
            for i in range(0, len(chunks), 128)
        ]
    )
    if (
        queries.shape != (len(points) * 2, encoder.dimensions)
        or docs.shape != (len(chunks), encoder.dimensions)
        or not np.isfinite(queries).all()
        or not np.isfinite(docs).all()
    ):
        raise ResearchError("retrieval_selection", "编码器返回无效向量。")
    qnorm = np.linalg.norm(queries, axis=1, keepdims=True)
    dnorm = np.linalg.norm(docs, axis=1, keepdims=True)
    if (qnorm == 0).any() or (dnorm == 0).any():
        raise ResearchError("retrieval_selection", "编码器返回零向量。")
    normalized = docs / dnorm
    semantic = (queries / qnorm) @ normalized.T
    semantic = 0.3 * semantic[: len(points)] + 0.7 * semantic[len(points) :]
    lexical = _lexical(chunks, translated)
    # Zero lexical matches have no rank signal. Their arbitrary source order must not win RRF.
    scores = 0.8 / (20 + _ranks(semantic)) + 0.2 * (lexical > 0) / (20 + _ranks(lexical))
    eligible = (lexical > 0) | (semantic >= 0.70)
    eligible[:, [i for i, c in enumerate(chunks) if structural(c["quote"])]] = False
    ranked = Ranked(chunks, original, lexical, semantic, [], eligible, sources, encoder.encoder_id)
    selected = []
    chosen = set()
    receipts = []

    def add(index, point, expand):
        nonlocal selected
        anchor = chunks[index]
        if any(contains(e, anchor) for e in [*existing, *selected]):
            chosen.add(index)
            return False
        span, _ = window(ranked, index) if expand else (anchor, [])
        before = json_bytes([*existing, *selected])
        result = protect(
            Selection(selected, audit),
            Selection([span], {"context_bytes": json_bytes([*existing, span])}),
            sources,
            max_bytes=max_bytes,
            max_items=max_items,
            existing=existing,
            next_id=next_id,
            allow_split=True,
        )
        if result.evidence == selected and expand and span != anchor:
            result = protect(
                Selection(selected, audit),
                Selection([anchor], {"context_bytes": json_bytes([*existing, anchor])}),
                sources,
                max_bytes=max_bytes,
                max_items=max_items,
                existing=existing,
                next_id=next_id,
                allow_split=True,
            )
        chosen.add(index)
        if result.evidence == selected:
            return False
        selected = result.evidence
        receipts.append(
            dict(
                point_id=points[point]["id"],
                candidate=index,
                expanded=expand,
                added_json_bytes=json_bytes([*existing, *selected]) - before,
            )
        )
        return True

    orders = [np.argsort(-row, kind="stable").tolist() for row in scores]
    # Establish one answer-bearing window per requirement before allowing repeated API matches.
    for point, order in enumerate(orders):
        for index in order:
            if eligible[point, index] and index not in chosen and add(index, point, True):
                break
    for _ in range(max_items):
        await asyncio.sleep(0)
        progress = False
        for point, order in enumerate(orders):
            for index in order:
                if index not in chosen and eligible[point, index] and add(index, point, False):
                    progress = True
                    break
        if not progress:
            break
    audit.update(
        context_bytes=json_bytes([*existing, *selected]),
        allocation_receipts=receipts,
        selected=[dict(evidence_id=e["evidence_id"]) for e in selected],
    )
    assert audit["context_bytes"] <= max_bytes and len(existing) + len(selected) <= max_items
    result = Selection(selected, audit)
    if adaptive_budget:
        from deep_research.evidence_budget import allocate_marginal

        result = allocate_marginal(
            result,
            chunks,
            np.where(eligible, scores, 0),
            sources,
            max_bytes=max_bytes,
            max_items=max_items,
            existing=existing,
            next_id=next_id,
        )
    return result
