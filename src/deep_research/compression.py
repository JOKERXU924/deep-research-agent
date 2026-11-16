"""Bounded extractive L1 query ranking, L2 TextRank, and L3 exact quotations.

The configured encoder determines whether L1 is semantic or merely lexical.
TextRank centrality and similarity are selection signals, not fact checking.
Quotes always refer to contiguous offsets in a validated full source snapshot.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from pydantic import Field

from deep_research.errors import ResearchError
from deep_research.schemas import StrictModel

MAX_SOURCES = 8
MAX_SOURCE_BYTES = 64 * 1024
MAX_CHUNKS = 4096
MAX_SENTENCES = 512
ENCODER_BATCH_SIZE = 64
_SENTENCE_END = re.compile(r"[。！？.!?;；\r\n]+")
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ebef]+")
_DAMPING = 0.85
_TOLERANCE = 1e-8
_MAX_ITERATIONS = 100


class CompressionOptions(StrictModel):
    max_evidence: int = Field(default=8, ge=1, le=16)
    max_evidence_bytes: int = Field(default=8192, ge=512, le=24576)
    coarse_top_k: int = Field(default=24, ge=1, le=64)
    chunk_characters: int = Field(default=240, ge=32, le=300)


class CompressionEncoder(Protocol):
    encoder_id: str
    dimensions: int
    kind: str

    async def encode_queries(self, texts: list[str]) -> np.ndarray: ...

    async def encode_documents(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class CompressionResult:
    evidence: list[dict[str, Any]]
    audit: dict[str, Any]


@dataclass(frozen=True)
class _Source:
    id: str
    text: str


@dataclass(frozen=True)
class _Span:
    source_id: str
    start: int
    end: int
    quote: str
    coarse_score: float = 0.0

    def evidence(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "start": self.start,
            "end": self.end,
            "quote": self.quote,
        }


def _error(code: str, message: str) -> ResearchError:
    return ResearchError(code, message)


def _text(value: object, maximum: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise _error("compression_input", "Compression input text is invalid or exceeds its limit.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _error("compression_input", "Compression input must contain valid UTF-8.") from None
    return value


def _validate_sources(sources: list[dict]) -> tuple[list[_Source], int]:
    if not isinstance(sources, list) or len(sources) > MAX_SOURCES:
        raise _error("compression_input", "Compression accepts at most eight source snapshots.")
    validated: list[_Source] = []
    seen: set[str] = set()
    total_bytes = 0
    for source in sources:
        if not isinstance(source, dict) or not {"source_id", "text", "sha256"} <= source.keys():
            raise _error("compression_input", "Source snapshot fields are missing or invalid.")
        source_id = _text(source["source_id"], 64)
        text = _text(source["text"], MAX_SOURCE_BYTES)
        raw = text.encode("utf-8")
        if source_id in seen:
            raise _error("compression_input", "Source snapshot IDs must be unique.")
        if len(raw) > MAX_SOURCE_BYTES:
            raise _error("compression_capacity", "A source snapshot exceeds the 64 KiB limit.")
        if (
            not isinstance(source["sha256"], str)
            or hashlib.sha256(raw).hexdigest() != source["sha256"]
        ):
            raise _error("compression_integrity", "Source snapshot hash validation failed.")
        seen.add(source_id)
        total_bytes += len(raw)
        validated.append(_Source(source_id, text))
    # Canonical ordering makes ties independent of source arrival/list order.
    return sorted(validated, key=lambda source: source.id), total_bytes


def _trim(source_id: str, text: str, start: int, end: int, score: float = 0.0) -> _Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return _Span(source_id, start, end, text[start:end], score) if end > start else None


def _terms(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result = set(re.findall(r"[a-z0-9]+", normalized))
    for sequence in _HAN.findall(normalized):
        if len(sequence) == 1:
            result.add(sequence)
        else:
            result.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return result


def _unit_vectors(value: object, rows: int, dimensions: int) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 2
        or value.shape != (rows, dimensions)
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
    ):
        raise _error("embedding_vectors", "Encoder vectors have invalid shape, type, or values.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = value.astype(np.float64, copy=True)
        scale = np.max(np.abs(result), axis=1, keepdims=True)
        np.divide(result, scale, out=result, where=scale != 0)
        lengths = np.linalg.norm(result, axis=1, keepdims=True)
        np.divide(result, lengths, out=result, where=lengths != 0)
    if not np.isfinite(result).all():
        raise _error("embedding_vectors", "Encoder vectors cannot be normalized safely.")
    return result


def _evidence_bytes(evidence: list[dict]) -> int:
    reserved = [{**item, "evidence_id": "E999"} for item in evidence]
    return len(json.dumps(reserved, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _is_fragment(text: str, start: int, end: int) -> bool:
    """Disclose fixed chunk boundaries that appear to split a sentence."""
    previous = start
    while previous > 0 and text[previous - 1] in " \t":
        previous -= 1
    following = end
    while following < len(text) and text[following] in " \t":
        following += 1
    boundary = "。！？.!?;；\r\n"
    split_start = previous > 0 and text[previous - 1] not in boundary
    split_end = (
        following < len(text) and text[end - 1] not in boundary and text[following] not in "\r\n"
    )
    return split_start or split_end


class HierarchicalCompressor:
    def __init__(
        self, encoder: CompressionEncoder, options: CompressionOptions | None = None
    ) -> None:
        if options is not None and not isinstance(options, CompressionOptions):
            raise _error("compression_configuration", "Invalid compression options.")
        self.options = options or CompressionOptions()
        dimensions = getattr(encoder, "dimensions", None)
        encoder_id = getattr(encoder, "encoder_id", None)
        kind = getattr(encoder, "kind", None)
        if (
            type(dimensions) is not int
            or not 1 <= dimensions <= 4096
            or not isinstance(encoder_id, str)
            or not encoder_id.strip()
            or len(encoder_id) > 200
            or kind not in ("semantic", "lexical", "fixture")
            or not callable(getattr(encoder, "encode_queries", None))
            or not callable(getattr(encoder, "encode_documents", None))
        ):
            raise _error("embedding_configuration", "Encoder configuration is invalid.")
        try:
            encoder_id.encode("utf-8")
        except UnicodeEncodeError:
            raise _error("embedding_configuration", "Encoder identity is invalid.") from None
        self.encoder = encoder
        self._dimensions = dimensions
        self._encoder_metadata = {"encoder_id": encoder_id, "kind": kind, "dimensions": dimensions}

    async def _encode(self, texts: list[str], *, queries: bool) -> np.ndarray:
        await asyncio.sleep(0)
        try:
            if queries:
                value = await self.encoder.encode_queries(texts)
            else:
                value = await self.encoder.encode_documents(texts)
        except ResearchError as error:
            if error.code.startswith("embedding_"):
                raise
            raise _error(
                "embedding_failure", "The configured encoder failed; no fallback was used."
            ) from None
        except Exception:
            raise _error(
                "embedding_failure", "The configured encoder failed; no fallback was used."
            ) from None
        await asyncio.sleep(0)
        return _unit_vectors(value, len(texts), self._dimensions)

    async def _chunks(self, sources: list[_Source]) -> list[_Span]:
        chunks = []
        for source in sources:
            for start in range(0, len(source.text), self.options.chunk_characters):
                end = min(start + self.options.chunk_characters, len(source.text))
                span = _trim(source.id, source.text, start, end)
                if span is not None:
                    chunks.append(span)
                    if len(chunks) > MAX_CHUNKS:
                        raise _error("compression_capacity", "Source chunk count exceeds 4096.")
                if len(chunks) % 64 == 0:
                    await asyncio.sleep(0)
        return chunks

    async def _coarse(self, question: str, chunks: list[_Span]) -> tuple[list[_Span], dict]:
        query = (await self._encode([question], queries=True))[0]
        unique = list(dict.fromkeys(span.quote for span in chunks))
        scores: dict[str, float] = {}
        for start in range(0, len(unique), ENCODER_BATCH_SIZE):
            texts = unique[start : start + ENCODER_BATCH_SIZE]
            matrix = await self._encode(texts, queries=False)
            for text, score in zip(texts, np.clip(matrix @ query, -1.0, 1.0), strict=True):
                scores[text] = float(score)
        ranked = sorted(
            (
                _Span(span.source_id, span.start, span.end, span.quote, scores[span.quote])
                for span in chunks
            ),
            key=lambda span: (-span.coarse_score, span.source_id, span.start, span.end),
        )
        selected: list[_Span] = []
        represented: set[str] = set()
        selected_keys: set[tuple[str, int]] = set()
        # Give each source one chance before a long/repetitive source takes every slot.
        for span in ranked:
            if len(selected) >= self.options.coarse_top_k:
                break
            if span.source_id not in represented:
                selected.append(span)
                represented.add(span.source_id)
                selected_keys.add((span.source_id, span.start))
        for span in ranked:
            if len(selected) >= self.options.coarse_top_k:
                break
            if (span.source_id, span.start) not in selected_keys:
                selected.append(span)
                selected_keys.add((span.source_id, span.start))
        return selected, {
            "encoded_document_count": len(unique),
            "document_batches": (len(unique) + ENCODER_BATCH_SIZE - 1) // ENCODER_BATCH_SIZE,
            "query_vector_zero": not bool(np.any(query)),
            "positive_relevance_chunks": sum(span.coarse_score > 0 for span in ranked),
        }

    async def _sentences(self, chunks: list[_Span], sources: list[_Source]) -> list[_Span]:
        originals = {source.id: source.text for source in sources}
        sentences = []
        for chunk in chunks:
            text = originals[chunk.source_id]
            boundaries = [
                chunk.start + match.end() for match in _SENTENCE_END.finditer(chunk.quote)
            ]
            boundaries.append(chunk.end)
            start = chunk.start
            for end in boundaries:
                # Tiny neighbouring sentences are grouped as one contiguous extract.
                # This keeps punctuation-heavy inputs bounded without losing offsets.
                if end != chunk.end and end - start < 48:
                    continue
                span = _trim(chunk.source_id, text, start, end, chunk.coarse_score)
                if span is not None:
                    sentences.append(span)
                    if len(sentences) > MAX_SENTENCES:
                        raise _error("compression_capacity", "TextRank sentence count exceeds 512.")
                start = end
            await asyncio.sleep(0)
        return sorted(sentences, key=lambda span: (span.source_id, span.start, span.end))

    async def _textrank(self, sentences: list[_Span]) -> tuple[np.ndarray, dict]:
        count = len(sentences)
        terms = [_terms(span.quote) for span in sentences]
        weights = np.zeros((count, count), dtype=np.float64)
        for left in range(count):
            for right in range(left + 1, count):
                union = terms[left] | terms[right]
                weight = len(terms[left] & terms[right]) / len(union) if union else 0.0
                weights[left, right] = weights[right, left] = weight
            if left % 16 == 0:
                await asyncio.sleep(0)
        totals = weights.sum(axis=1)
        dangling = totals == 0
        transitions = np.divide(
            weights, totals[:, None], out=np.zeros_like(weights), where=totals[:, None] != 0
        )
        rank = np.full(count, 1.0 / count)
        converged = False
        delta = 0.0
        for iteration in range(1, _MAX_ITERATIONS + 1):
            updated = (1 - _DAMPING) / count + _DAMPING * (
                transitions.T @ rank + rank[dangling].sum() / count
            )
            delta = float(np.abs(updated - rank).sum())
            rank = updated
            if delta <= _TOLERANCE:
                converged = True
                break
            if iteration % 5 == 0:
                await asyncio.sleep(0)
        return rank, {
            "iterations": iteration,
            "converged": converged,
            "delta": delta,
            "edges": int(np.count_nonzero(weights) // 2),
            "dangling_nodes": int(dangling.sum()),
            "damping": _DAMPING,
            "tolerance": _TOLERANCE,
            "max_iterations": _MAX_ITERATIONS,
            "edge_metric": "lexical_jaccard",
        }

    async def _select(
        self, question: str, sentences: list[_Span], ranks: np.ndarray
    ) -> tuple[list, list, int]:
        wanted = _terms(question)
        max_rank = float(ranks.max())
        ranked = []
        for span, rank in zip(sentences, ranks, strict=True):
            overlap = len(wanted & _terms(span.quote)) / len(wanted) if wanted else 0.0
            query_score = 0.75 * max(0.0, span.coarse_score) + 0.25 * overlap
            centrality = float(rank / max_rank) if max_rank else 0.0
            # Centrality is a tie-breaking signal, not a licence for repeated
            # off-topic boilerplate to dominate a weaker, unique query match.
            score = 0.85 * query_score + 0.15 * centrality
            ranked.append((score, query_score, float(rank), span))
        ranked.sort(key=lambda entry: (-entry[0], entry[3].source_id, entry[3].start, entry[3].end))
        coverage = []
        seen_sources: set[str] = set()
        # Central, repeated boilerplate is not necessarily relevant. Coverage
        # reserves each source's strongest query match before graph ranking.
        for entry in sorted(
            ranked,
            key=lambda entry: (
                -entry[1],
                -entry[0],
                entry[3].source_id,
                entry[3].start,
                entry[3].end,
            ),
        ):
            if entry[3].source_id not in seen_sources:
                coverage.append(entry)
                seen_sources.add(entry[3].source_id)
        coverage_keys = {(entry[3].source_id, entry[3].start) for entry in coverage}
        remaining = [
            entry for entry in ranked if (entry[3].source_id, entry[3].start) not in coverage_keys
        ]
        evidence: list[dict] = []
        selections: list[dict] = []
        quotes: set[str] = set()
        selected_terms: list[set[str]] = []
        duplicates = 0
        term_cache = {span.quote: _terms(span.quote) for span in sentences}

        def redundancy(entry: tuple) -> float:
            terms = term_cache[entry[3].quote]
            return max(
                (
                    len(terms & previous) / len(terms | previous)
                    for previous in selected_terms
                    if terms | previous
                ),
                default=0.0,
            )

        while (coverage or remaining) and len(evidence) < self.options.max_evidence:
            await asyncio.sleep(0)
            if coverage:
                entry = coverage.pop(0)
                reason = "source_coverage"
            else:
                # Repetitive boilerplate should not use every remaining slot.
                # This changes order only: similar text is never treated as a
                # verified duplicate or silently merged into another source.
                remaining.sort(
                    key=lambda entry: (
                        -(entry[0] - 0.2 * redundancy(entry)),
                        entry[3].source_id,
                        entry[3].start,
                        entry[3].end,
                    )
                )
                entry = remaining.pop(0)
                reason = "ranked"
            score, query_score, centrality, span = entry
            overlap_penalty = redundancy(entry)
            if span.quote in quotes:
                duplicates += 1
                continue
            item = span.evidence()
            if _evidence_bytes([*evidence, item]) > self.options.max_evidence_bytes:
                continue
            evidence.append(item)
            quotes.add(span.quote)
            selected_terms.append(term_cache[span.quote])
            selections.append(
                {
                    "source_id": span.source_id,
                    "start": span.start,
                    "end": span.end,
                    "coarse_score": span.coarse_score,
                    "query_score": query_score,
                    "textrank_score": centrality,
                    "combined_score": score,
                    "redundancy_score": overlap_penalty,
                    "selection_score": score - 0.2 * overlap_penalty,
                    "reason": reason,
                    "trimmed_for_budget": False,
                }
            )
        if not evidence:
            # Never remove a qualifier or negation merely to fill a byte budget.
            raise _error("compression_budget", "No whole candidate fits the evidence budget.")
        return evidence, selections, duplicates

    async def compress(self, question: str, sources: list[dict]) -> CompressionResult:
        question = _text(question, 1000)
        validated, input_bytes = _validate_sources(sources)
        await asyncio.sleep(0)
        audit = {
            "schema_version": 1,
            "method": "hierarchical_extractive",
            "encoder": dict(self._encoder_metadata),
            "encoder_id": self._encoder_metadata["encoder_id"],
            "encoder_kind": self._encoder_metadata["kind"],
            "options": self.options.model_dump(),
            "input_source_count": len(validated),
            "input_source_bytes": input_bytes,
            "chunk_count": 0,
            "coarse_selected_count": 0,
            "sentence_count": 0,
            "selected_count": 0,
            "selected_quote_bytes": 0,
            "evidence_json_bytes": 2,
            "selected_source_ids": [],
            "uncovered_source_ids": [source.id for source in validated],
            "duplicate_quotes_skipped": 0,
            "coarse": {
                "encoded_document_count": 0,
                "document_batches": 0,
                "query_vector_zero": False,
                "positive_relevance_chunks": 0,
            },
            "byte_counting": "utf8_compact_json_with_reserved_E999_ids",
            "selection_policy": "source_query_coverage_then_redundancy_penalized_blend",
            "ranking_weights": {
                "query": 0.85,
                "centrality": 0.15,
                "redundancy": 0.2,
                "query_chunk_relevance": 0.75,
                "query_lexical_overlap": 0.25,
            },
            "textrank": {
                "iterations": 0,
                "converged": True,
                "edges": 0,
                "dangling_nodes": 0,
                "damping": _DAMPING,
                "tolerance": _TOLERANCE,
                "max_iterations": _MAX_ITERATIONS,
                "edge_metric": "lexical_jaccard",
            },
            "selected_spans": [],
            "limitations": [
                "Similarity and graph centrality do not verify truth or entailment.",
                "Fixed chunk boundaries can omit important context; fragment flags are heuristic.",
                "Evidence byte limits do not replace the complete model-request budget.",
            ],
        }
        if not validated:
            return CompressionResult([], audit)
        chunks = await self._chunks(validated)
        coarse, coarse_audit = await self._coarse(question, chunks)
        sentences = await self._sentences(coarse, validated)
        if not sentences:
            raise _error("compression_empty", "No extractable source spans remain.")
        ranks, textrank_audit = await self._textrank(sentences)
        await asyncio.sleep(0)
        evidence, selections, duplicates = await self._select(question, sentences, ranks)
        originals = {source.id: source.text for source in validated}
        for selected in selections:
            selected["fragment"] = _is_fragment(
                originals[selected["source_id"]], selected["start"], selected["end"]
            )
        selected_ids = sorted({item["source_id"] for item in evidence})
        audit.update(
            {
                "chunk_count": len(chunks),
                "coarse_selected_count": len(coarse),
                "sentence_count": len(sentences),
                "selected_count": len(evidence),
                "selected_quote_bytes": sum(
                    len(item["quote"].encode("utf-8")) for item in evidence
                ),
                "evidence_json_bytes": _evidence_bytes(evidence),
                "selected_source_ids": selected_ids,
                "uncovered_source_ids": [
                    source.id for source in validated if source.id not in selected_ids
                ],
                "duplicate_quotes_skipped": duplicates,
                "textrank": textrank_audit,
                "selected_spans": selections,
                "coarse": coarse_audit,
            }
        )
        return CompressionResult(evidence, audit)
