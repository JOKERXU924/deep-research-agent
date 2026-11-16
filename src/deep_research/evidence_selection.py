"""Bounded, extractive retrieval: clause coverage + BM25 + optional local E5 RRF.

No model-generated text is treated as evidence. The encoder never sees gold labels.
Candidates, offsets and UTF-8 context sizes are independently auditable.
"""

import asyncio
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from deep_research.compression_runtime import json_bytes
from deep_research.embeddings import EmbeddingEncoder
from deep_research.errors import ResearchError
from deep_research.relevance import _terms

VERSION = "coverage-bm25-rrf-v1"
MAX_CANDIDATES = 2048
MAX_SPAN_CHARS = 300


@dataclass
class Selection:
    evidence: list[dict]
    audit: dict


def targets(question: str, queries: list[str] | None = None) -> list[str]:
    """Deterministic clause splitting, not an LLM subquestion planner."""
    result = []
    for part in [*re.split(r"[\n。！？!?；;，,]+", question[:2048]), *(queries or [])]:
        part = part.strip()[:300]
        if re.match(
            r"^(?:每个关键|每个结论|约\d|资料不足|请引用|cite |quote |about \d)", part, re.I
        ):
            continue
        if _terms(part, topic=True) and part.casefold() not in {p.casefold() for p in result}:
            result.append(part)
    return result[:8] or [question[:300]]


def candidates(sources: list[dict]) -> list[dict]:
    if len(sources) > 12:
        raise ResearchError("retrieval_selection", "来源数量超过检索上限。")
    result = []
    for source in sources:
        text = source["text"]
        if (
            len(text.encode("utf-8")) > 65536
            or hashlib.sha256(text.encode()).hexdigest() != source["sha256"]
        ):
            raise ResearchError("retrieval_selection", "来源大小或哈希校验失败。")
        for paragraph in re.finditer(r"[^\r\n]+", text):
            # Sentence boundaries, with bounded fallback for pathological long sentences.
            units = list(
                re.finditer(r".+?(?:[。！？](?:[”’])?|[.!?](?=\s|$)|$)", paragraph.group())
            )
            start = end = paragraph.start()
            for unit in units:
                left, right = paragraph.start() + unit.start(), paragraph.start() + unit.end()
                if right - start > MAX_SPAN_CHARS and end > start:
                    result.append(
                        dict(
                            source_id=source["source_id"],
                            start=start,
                            end=end,
                            quote=text[start:end],
                        )
                    )
                    start = left
                while right - start > MAX_SPAN_CHARS:
                    boundary = text.rfind(" ", start + MAX_SPAN_CHARS // 2, start + MAX_SPAN_CHARS)
                    cut = boundary if boundary > start else start + MAX_SPAN_CHARS
                    result.append(
                        dict(
                            source_id=source["source_id"],
                            start=start,
                            end=cut,
                            quote=text[start:cut],
                        )
                    )
                    start = cut
                end = right
            if end > start and text[start:end].strip():
                result.append(
                    dict(source_id=source["source_id"], start=start, end=end, quote=text[start:end])
                )
    if len(result) > MAX_CANDIDATES:
        raise ResearchError("retrieval_selection", "候选片段超过上限；未静默丢弃尾部资料。")
    return result


def _lexical(chunks: list[dict], aspects: list[str]) -> np.ndarray:
    # Binary term frequency; document-length normalization prevents long keyword lists winning.
    terms = [_terms(item["quote"], topic=True) for item in chunks]
    df = Counter(term for row in terms for term in row)
    average = sum(map(len, terms)) / max(1, len(terms)) or 1
    scores = np.zeros((len(aspects), len(chunks)))
    for row, query in enumerate(aspects):
        wanted = _terms(query, topic=True)
        for col, found in enumerate(terms):
            norm = 1 + 1.2 * (0.25 + 0.75 * len(found) / average)
            scores[row, col] = sum(
                math.log(1 + (len(chunks) - df[t] + 0.5) / (df[t] + 0.5)) * 2.2 / norm
                for t in wanted & found
            )
    return scores


def _ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(scores)
    for row, values in enumerate(scores):
        order = np.argsort(-values, kind="stable")
        ranks[row, order] = np.arange(1, len(order) + 1)
    return ranks


def pack(chunks, aspects, lexical, semantic, *, max_bytes, max_items, existing, next_id):
    ranked = 1 / (20 + _ranks(lexical))
    eligible = lexical > 0
    if semantic is not None:
        ranked = 0.5 * ranked + 0.5 / (20 + _ranks(semantic))
        eligible |= semantic >= 0.78
    orders = [np.argsort(-row, kind="stable").tolist() for row in ranked]
    selected, chosen, receipts = [], set(), []
    normalized = {" ".join(item["quote"].split()).casefold() for item in existing}
    while len(selected) + len(existing) < max_items:
        progress = False
        for aspect, order in enumerate(orders):
            for index in order:
                if index in chosen or not eligible[aspect, index]:
                    continue
                span = chunks[index]
                quote_key = " ".join(span["quote"].split()).casefold()
                if quote_key in normalized:
                    chosen.add(index)
                    continue
                # Do not append a second quote already covered by the report context.
                if any(
                    item["source_id"] == span["source_id"]
                    and item["start"] <= span["start"]
                    and item["end"] >= span["end"]
                    for item in [*existing, *selected]
                ):
                    chosen.add(index)
                    continue
                item = {"evidence_id": f"E{next_id + len(selected)}", **span}
                if json_bytes([*existing, *selected, item]) > max_bytes:
                    continue
                selected.append(item)
                chosen.add(index)
                normalized.add(quote_key)
                receipts.append(
                    dict(
                        evidence_id=item["evidence_id"],
                        target=aspect,
                        lexical_score=float(lexical[aspect, index]),
                        semantic_score=float(semantic[aspect, index])
                        if semantic is not None
                        else None,
                    )
                )
                progress = True
                break
            if len(selected) + len(existing) >= max_items:
                break
        if not progress:
            break
    return Selection(
        selected,
        dict(
            version=VERSION,
            targets=aspects,
            candidates=len(chunks),
            max_bytes=max_bytes,
            context_bytes=json_bytes([*existing, *selected]),
            selected=receipts,
            encoder_kind="semantic" if semantic is not None else "lexical",
        ),
    )


async def select_coverage(
    sources: list[dict],
    question: str,
    *,
    queries=None,
    encoder: EmbeddingEncoder | None = None,
    max_bytes=12000,
    max_items=16,
    existing=None,
    next_id=1,
) -> Selection:
    if (
        type(max_bytes) is not int
        or not 2 <= max_bytes <= 12000
        or type(max_items) is not int
        or not 1 <= max_items <= 32
    ):
        raise ResearchError("retrieval_selection", "检索上下文预算无效。")
    existing = list(existing or [])
    if json_bytes(existing) > max_bytes or len(existing) > max_items:
        raise ResearchError("retrieval_selection", "已有证据超过预算；未静默删除。")
    await asyncio.sleep(0)
    chunks, aspects = candidates(sources), targets(question, queries)
    lexical = _lexical(chunks, aspects)
    semantic = None
    if encoder is not None and chunks:
        q = await encoder.encode_queries(aspects)
        batches = []
        for offset in range(0, len(chunks), 128):
            batches.append(
                await encoder.encode_documents([c["quote"] for c in chunks[offset : offset + 128]])
            )
        docs = np.concatenate(batches)
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
    await asyncio.sleep(0)
    result = pack(
        chunks,
        aspects,
        lexical,
        semantic,
        max_bytes=max_bytes,
        max_items=max_items,
        existing=existing,
        next_id=next_id,
    )
    result.audit["encoder_id"] = encoder.encoder_id if encoder is not None else None
    return result
