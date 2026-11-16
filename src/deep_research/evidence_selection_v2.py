"""Source-grounded multilingual retrieval with tie-aware fusion and parent context.

v1 remains immutable in evidence_selection.py. No translations, aliases, gold
answers, or domain-specific facts are added to a user's query or the evidence.
"""

import asyncio
import hashlib
import re
from collections import Counter

import numpy as np

from deep_research.compression_runtime import json_bytes
from deep_research.errors import ResearchError
from deep_research.evidence_selection import Selection, _lexical, candidates
from deep_research.relevance import _terms

VERSION = "multilingual-parent-v2"
MAX_PARENT_CHARS = 600
MAX_CHILD_CHARS = 300


async def cross_language_rescue(question, text, title, encoder):
    """Bounded post-fetch gate rescue; cosine is a heuristic, not a relevance probability.

    Only invoked after lexical rejection. Does not translate search requests or
    fetch extra sources, and never silently falls back on encoder errors.
    """
    if abs(_han_ratio(question) - _han_ratio(text)) <= 0.35:
        return None
    result = await select_coverage(
        [
            dict(
                source_id="S1",
                title=title,
                text=text,
                sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        ],
        question,
        encoder=encoder,
        max_items=1,
    )
    scores = [r["semantic_score"] for r in result.audit["selected"]]
    score = max(scores, default=-1)
    return min(1.0, max(0.0, score)) if score >= 0.84 else None


def _heading(text):
    return len(text.strip()) <= 80 and not re.search(r"[。！？.!?]$", text.strip())


def passages(sources):
    candidates(sources)  # Retain the original source hash, size, and candidate count validation.
    parents = []
    for source in sources:
        text = source["text"]
        paragraphs = list(re.finditer(r"[^\r\n]+", text))
        skip = set()
        for i, paragraph in enumerate(paragraphs):
            if i in skip:
                continue
            start, end = paragraph.span()
            if _heading(paragraph.group()) and i + 1 < len(paragraphs):
                following = paragraphs[i + 1]
                if following.end() - start <= MAX_PARENT_CHARS:
                    end = following.end()
                    skip.add(i + 1)
            # Keep complete sentences together; oversized sentences have bounded fallback spans.
            cursor = start
            boundaries = [
                start + m.end() for m in re.finditer(r"[。！？]|[.!?](?=\s|$)", text[start:end])
            ]
            while end - cursor > MAX_PARENT_CHARS:
                possible = [b for b in boundaries if cursor < b <= cursor + MAX_PARENT_CHARS]
                cut = max(possible) if possible else cursor + MAX_PARENT_CHARS
                parents.append(
                    dict(
                        source_id=source["source_id"], start=cursor, end=cut, quote=text[cursor:cut]
                    )
                )
                cursor = cut
            if end > cursor:
                parents.append(
                    dict(
                        source_id=source["source_id"], start=cursor, end=end, quote=text[cursor:end]
                    )
                )
    # Bare navigation/section labels cannot establish a factual claim.
    return [p for p in parents if not _heading(p["quote"]) and p["quote"].strip()]


def rank_support(scores):
    """Ties share a rank; zero lexical evidence contributes zero."""
    output = np.zeros_like(scores)
    for row, values in enumerate(scores):
        for score in np.unique(values):
            if score > 0:
                output[row, values == score] = 1 / (20 + 1 + np.count_nonzero(values > score))
    return output


def _han_ratio(text):
    letters = re.findall(r"[A-Za-z\u3400-\u9fff]", text)
    return sum(ord(c) > 127 for c in letters) / max(1, len(letters))


def _bridge_candidates(parents, orders, semantic):
    """One bounded expansion using only terms found in first-hop original passages."""
    if semantic is None:
        return [], [], {}
    tokens = [
        set(re.findall(r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_+]{1,19}(?![A-Za-z0-9_])", p["quote"]))
        for p in parents
    ]
    support = Counter()
    for order in np.argsort(-semantic, axis=1, kind="stable").tolist():
        for rank, i in enumerate(order[:8]):
            for token in tokens[i]:
                support[token] += 1 / (rank + 1)
    # Favor concepts supported by first-hop hits, not rare incidental identifiers.
    terms = sorted(support, key=lambda t: (-support[t], t))[:12]
    selected, windows = [], {}
    # Look for a compact link between first-hop concepts. A long paragraph with
    # many incidental symbols must not win simply by accumulating more tokens.
    for term in terms:
        matches = []
        for i, row in enumerate(tokens):
            if (
                term not in row
                or len(row & set(terms)) < 2
                or i in selected
                or np.max(semantic[:, i]) < max(0.78, np.max(semantic) - 0.07)
            ):
                continue
            for sentence in re.finditer(r"[^。！？.!?]+(?:[。！？.!?]|$)", parents[i]["quote"]):
                found = set(
                    re.findall(
                        r"(?<![A-Za-z0-9_])[A-Z][A-Z0-9_+]{1,19}(?![A-Za-z0-9_])",
                        sentence.group(),
                    )
                )
                if term in found and len(found & set(terms)) >= 2:
                    matches.append((i, sentence.start(), sentence.end()))
        if matches:
            i, start, end = min(
                matches, key=lambda m: (m[2] - m[1], -float(np.max(semantic[:, m[0]])), m[0], m[1])
            )
            selected.append(i)
            windows[i] = dict(
                source_id=parents[i]["source_id"],
                start=parents[i]["start"] + start,
                end=parents[i]["start"] + end,
                quote=parents[i]["quote"][start:end],
            )
        if len(selected) == 2:
            break
    return selected, terms, windows


def _queries(question, queries):
    # Keep quoted assertions intact; a comma inside a quotation is not a new coverage goal.
    clauses, current, quoted = [], [], False
    for char in question[:2048]:
        if char in '“”"':
            quoted = not quoted
        if char in "\n。！？!?；;，," and not quoted:
            clauses.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    clauses.append("".join(current).strip())
    clauses = [
        p[:300]
        for p in clauses
        if _terms(p, topic=True)
        and not re.match(
            r"^(?:每个关键|每个结论|约\d|资料不足|请引用|cite |quote |about \d)", p, re.I
        )
    ]
    topic = ""
    if len(clauses) > 1 and re.match(
        r"^(?:in |for |regarding |about |请研究|关于|对于)", clauses[0], re.I
    ):
        topic = clauses.pop(0)
    aspects = list(dict.fromkeys(clauses))[:8] or [question[:300]]
    # Planner search keywords are not independent questions and must not steal evidence slots.
    return aspects, [(topic[:100] + "; " + a)[:300] if topic else a for a in aspects]


async def _vectors(encoder, values, *, query=False):
    chunks = []
    for offset in range(0, len(values), 128):
        batch = values[offset : offset + 128]
        raw = await (encoder.encode_queries(batch) if query else encoder.encode_documents(batch))
        if (
            not isinstance(raw, np.ndarray)
            or raw.shape != (len(batch), encoder.dimensions)
            or not np.isfinite(raw).all()
        ):
            raise ResearchError("retrieval_selection", "编码器返回无效向量。")
        norm = np.linalg.norm(raw, axis=1, keepdims=True)
        if (norm == 0).any() or not np.isfinite(norm).all():
            raise ResearchError("retrieval_selection", "编码器返回零向量或无效范数。")
        chunks.append(raw / norm)
    return np.concatenate(chunks)


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
    if (
        type(max_bytes) is not int
        or not 2 <= max_bytes <= 12000
        or type(max_items) is not int
        or not 1 <= max_items <= 32
        or type(next_id) is not int
        or next_id < 1
    ):
        raise ResearchError("retrieval_selection", "检索上下文预算或证据编号无效。")
    existing = list(existing or [])
    if len(existing) > max_items or json_bytes(existing) > max_bytes:
        raise ResearchError("retrieval_selection", "已有证据超过预算；未静默删除。")
    await asyncio.sleep(0)
    parents = passages(sources)
    aspects, contextual = _queries(question, queries)
    lexical = _lexical(parents, aspects)
    scores = rank_support(lexical)
    semantic = None
    raw_semantic = None
    weights = np.full(len(aspects), 1.0)
    if encoder is not None and parents:
        q = await _vectors(encoder, contextual, query=True)
        # The best bounded child retrieves its complete parent. Titles are ranking metadata only.
        children, owners = [], []
        titles = {s["source_id"]: s.get("title", "")[:80] for s in sources}
        for index, parent in enumerate(parents):
            quote = parent["quote"]
            for start in range(0, len(quote), 220):
                children.append(
                    titles[parent["source_id"]] + "\n" + quote[start : start + MAX_CHILD_CHARS]
                )
                owners.append(index)
        documents = await _vectors(encoder, children)
        child_scores = q @ documents.T
        semantic = np.zeros_like(lexical)
        for index in range(len(parents)):
            semantic[:, index] = np.max(child_scores[:, np.array(owners) == index], axis=1)
        raw_semantic = semantic.copy()
        # Match explicit query terms against source titles to keep an API question in its
        # named source domain. Metadata changes ranks, never the preserved quote.
        wanted = _terms(question, topic=True)
        title_terms = {s["source_id"]: _terms(s.get("title", ""), topic=True) for s in sources}
        anchors = wanted & set().union(*title_terms.values())
        if anchors:
            for index, parent in enumerate(parents):
                semantic[:, index] += (
                    0.03 * len(anchors & title_terms[parent["source_id"]]) / len(anchors)
                )
        # Script mismatch is detected from inputs, not labels. Weak lexical matches on API names
        # must not dominate the multilingual encoder for an otherwise untranslated query.
        corpus_ratio = _han_ratio(" ".join(p["quote"] for p in parents))
        weights = np.array(
            [0.05 if abs(_han_ratio(a) - corpus_ratio) > 0.35 else 0.25 for a in aspects]
        )
        scores = weights[:, None] * scores + (1 - weights[:, None]) * rank_support(semantic)
    eligible = lexical > 0
    if semantic is not None:
        eligible |= raw_semantic >= 0.78
    orders = [np.argsort(-row, kind="stable").tolist() for row in scores]
    bridges, bridge_terms, bridge_windows = _bridge_candidates(parents, orders, raw_semantic)
    # Give at most two bridge passages a slot after the best hit for each coverage target.
    for row, order in enumerate(orders):
        assigned = [i for i in bridges if int(np.argmax(semantic[:, i])) == row]
        orders[row] = order[:1] + assigned + [i for i in order[1:] if i not in assigned]
    selected, receipts, seen = [], [], set()
    normalized = {" ".join(e["quote"].split()).casefold() for e in existing}
    while len(existing) + len(selected) < max_items:
        progress = False
        for row, order in enumerate(orders):
            for index in order:
                if index in seen or not eligible[row, index]:
                    continue
                parent = bridge_windows.get(index, parents[index])
                key = " ".join(parent["quote"].split()).casefold()
                if key in normalized or any(
                    e["source_id"] == parent["source_id"]
                    and e["start"] <= parent["start"]
                    and e["end"] >= parent["end"]
                    for e in [*existing, *selected]
                ):
                    seen.add(index)
                    continue
                item = dict(evidence_id=f"E{next_id + len(selected)}", **parent)
                if json_bytes([*existing, *selected, item]) > max_bytes:
                    continue
                selected.append(item)
                normalized.add(key)
                seen.add(index)
                receipts.append(
                    dict(
                        evidence_id=item["evidence_id"],
                        target=row,
                        bridge=index in bridge_windows,
                        lexical_score=float(lexical[row, index]),
                        semantic_score=float(raw_semantic[row, index])
                        if semantic is not None
                        else None,
                    )
                )
                progress = True
                break
            if len(existing) + len(selected) >= max_items:
                break
        if not progress:
            break
    await asyncio.sleep(0)
    return Selection(
        selected,
        dict(
            version=VERSION,
            bridge_terms=bridge_terms,
            bridge_candidates=bridges,
            bridge_spans=list(bridge_windows.values()),
            targets=aspects,
            encoding_queries=contextual,
            candidates=len(parents),
            max_bytes=max_bytes,
            context_bytes=json_bytes([*existing, *selected]),
            selected=receipts,
            lexical_weights=weights.tolist(),
            encoder_kind="semantic" if encoder is not None else "lexical",
            encoder_id=encoder.encoder_id if encoder is not None else None,
        ),
    )
