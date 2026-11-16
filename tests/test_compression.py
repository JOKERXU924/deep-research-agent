"""Offline extraction contracts; no semantic model or network is needed."""

import asyncio
import hashlib
import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from pydantic import ValidationError

from deep_research.compression import (
    CompressionOptions,
    HierarchicalCompressor,
)
from deep_research.errors import ResearchError
from deep_research.memory_vector import LexicalEncoder


def source(text, source_id="S1"):
    return {
        "source_id": source_id,
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def reserved_bytes(items):
    numbered = [{**item, "evidence_id": "E999"} for item in items]
    return len(json.dumps(numbered, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


class RecordingEncoder:
    encoder_id = "offline-lexical-fixture"
    kind = "fixture"
    dimensions = 1024

    def __init__(self):
        self.calls = []

    async def encode_queries(self, texts):
        self.calls.append(("query", list(texts)))
        return LexicalEncoder().encode(texts)

    async def encode_documents(self, texts):
        self.calls.append(("document", list(texts)))
        return LexicalEncoder().encode(texts)


class ConstantEncoder(RecordingEncoder):
    dimensions = 2

    def __init__(self, query=(1.0, 0.0), document=(1.0, 0.0)):
        super().__init__()
        self.query = query
        self.document = document

    async def encode_queries(self, texts):
        self.calls.append(("query", list(texts)))
        return np.tile(self.query, (len(texts), 1))

    async def encode_documents(self, texts):
        self.calls.append(("document", list(texts)))
        return np.tile(self.document, (len(texts), 1))


async def test_full_snapshot_includes_relevant_material_after_old_prefix():
    background = "An unrelated description of garden plants and watering. " * 35
    important = "The queryneedle research record preserves source addresses and exact quotations."
    document = source(background + important)
    result = await HierarchicalCompressor(
        RecordingEncoder(), CompressionOptions(coarse_top_k=1, max_evidence=2)
    ).compress("queryneedle research quotations", [document])
    assert any("queryneedle" in item["quote"] for item in result.evidence)
    assert result.audit["input_source_bytes"] == len(document["text"].encode("utf-8"))
    assert result.audit["chunk_count"] > 2
    assert result.audit["coarse_selected_count"] == 1


async def test_exact_unicode_offsets_and_no_rewriting_or_number_assignment():
    text = "  ＡＢＣ Cafe\u0301。\n证据出处🙂保留原始字符，不能改写。\n原文中的否定条件必须保留！ "
    document = source(text)
    result = await HierarchicalCompressor(RecordingEncoder()).compress("证据出处", [document])
    assert result.evidence
    for item in result.evidence:
        assert set(item) == {"source_id", "start", "end", "quote"}
        assert item["quote"] == text[item["start"] : item["end"]]
        assert 0 <= item["start"] < item["end"] <= len(text)
    assert document == source(text)
    assert result.audit["selected_quote_bytes"] == sum(
        len(item["quote"].encode("utf-8")) for item in result.evidence
    )
    assert result.audit["evidence_json_bytes"] == reserved_bytes(result.evidence)
    assert result.audit["encoder_id"] == result.audit["encoder"]["encoder_id"]
    assert result.audit["encoder_kind"] == "fixture"
    json.dumps(result.audit, ensure_ascii=False, allow_nan=False)


async def test_source_order_does_not_change_result_or_audit():
    documents = [source("alpha evidence one.", "S2"), source("beta evidence two.", "S1")]
    compressor = HierarchicalCompressor(RecordingEncoder())
    first = await compressor.compress("evidence", documents)
    second = await compressor.compress("evidence", list(reversed(documents)))
    assert first == second
    with pytest.raises(FrozenInstanceError):
        first.evidence = []


async def test_source_diversity_covers_eight_distinct_sources():
    documents = [source(f"Unique research fact {index}.", f"S{index}") for index in range(8)]
    result = await HierarchicalCompressor(ConstantEncoder()).compress("research", documents)
    assert {item["source_id"] for item in result.evidence} == {f"S{index}" for index in range(8)}
    assert result.audit["uncovered_source_ids"] == []
    assert all(item["reason"] == "source_coverage" for item in result.audit["selected_spans"])


async def test_exact_duplicate_quotes_are_not_published_twice_and_audit_uncovered_sources():
    result = await HierarchicalCompressor(ConstantEncoder()).compress(
        "evidence", [source("Identical evidence.", "S1"), source("Identical evidence.", "S2")]
    )
    assert result.evidence == [
        {"source_id": "S1", "start": 0, "end": 19, "quote": "Identical evidence."}
    ]
    assert result.audit["uncovered_source_ids"] == ["S2"]
    assert result.audit["duplicate_quotes_skipped"] == 1
    assert result.audit["coarse"]["encoded_document_count"] == 1


async def test_document_encoding_deduplicates_chunks_and_uses_bounded_batches():
    documents = [source("".join(f"record{index:05d}".ljust(32, "x") for index in range(150)))]
    encoder = RecordingEncoder()
    result = await HierarchicalCompressor(
        encoder, CompressionOptions(chunk_characters=32, coarse_top_k=4)
    ).compress("record", documents)
    batches = [texts for kind, texts in encoder.calls if kind == "document"]
    assert [len(batch) for batch in batches] == [64, 64, 22]
    assert result.audit["coarse"]["document_batches"] == 3
    assert all(len(text) <= 32 for batch in batches for text in batch)
    assert len([call for call in encoder.calls if call[0] == "query"]) == 1


async def test_zero_vectors_and_no_graph_edges_use_dangling_redistribution():
    result = await HierarchicalCompressor(ConstantEncoder((0, 0), (0, 0))).compress(
        "unmatched", [source("alpha", "S1"), source("beta", "S2"), source("gamma", "S3")]
    )
    assert result.audit["coarse"]["query_vector_zero"] is True
    assert result.audit["textrank"]["edges"] == 0
    assert result.audit["textrank"]["dangling_nodes"] == 3
    assert result.audit["textrank"]["converged"] is True
    assert result.audit["textrank"]["iterations"] == 1
    for span in result.audit["selected_spans"]:
        assert span["textrank_score"] == pytest.approx(1 / 3)
        assert span["query_score"] == 0
    json.dumps(result.audit, allow_nan=False)


@pytest.mark.parametrize("magnitude", [1.0, 1e-300, 1e300])
async def test_negative_and_non_unit_finite_vectors_are_safely_normalized(magnitude):
    result = await HierarchicalCompressor(
        ConstantEncoder((magnitude, 0.0), (-magnitude, 0.0))
    ).compress("unmatched", [source("alpha")])
    assert result.audit["selected_spans"][0]["coarse_score"] == pytest.approx(-1)
    assert result.audit["selected_spans"][0]["query_score"] == 0
    assert result.audit["coarse"]["positive_relevance_chunks"] == 0


async def test_textrank_nontrivial_graph_is_bounded_and_audited():
    documents = [
        source("research alpha beta", "S1"),
        source("research alpha gamma", "S2"),
        source("research beta gamma", "S3"),
        source("isolated", "S4"),
    ]
    result = await HierarchicalCompressor(RecordingEncoder()).compress("research", documents)
    graph = result.audit["textrank"]
    assert graph["edges"] == 3
    assert graph["dangling_nodes"] == 1
    assert 1 <= graph["iterations"] <= graph["max_iterations"] == 100
    assert sum(span["textrank_score"] for span in result.audit["selected_spans"]) == pytest.approx(
        1
    )
    assert graph["converged"]


async def test_small_budget_does_not_cut_long_chinese_candidate():
    document = source("条件" * 140 + "，但此结论并不适用于所有情况。")
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(
            ConstantEncoder(), CompressionOptions(chunk_characters=300, max_evidence_bytes=512)
        ).compress("条件", [document])
    assert captured.value.code == "compression_budget"


async def test_exact_utf8_json_budget_boundary_and_reserved_ids():
    text = "甲" * 160 + "。"
    span = {"source_id": "S1", "start": 0, "end": len(text), "quote": text}
    boundary = reserved_bytes([span])
    assert 512 < boundary < 8192
    result = await HierarchicalCompressor(
        ConstantEncoder(), CompressionOptions(max_evidence_bytes=boundary)
    ).compress("甲", [source(text)])
    assert result.evidence == [span]
    assert result.audit["evidence_json_bytes"] == boundary
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(
            ConstantEncoder(), CompressionOptions(max_evidence_bytes=boundary - 1)
        ).compress("甲", [source(text)])
    assert captured.value.code == "compression_budget"


async def test_budget_can_skip_large_candidate_without_truncating_small_one():
    result = await HierarchicalCompressor(
        ConstantEncoder(), CompressionOptions(max_evidence_bytes=512, chunk_characters=300)
    ).compress("证据", [source("证据" * 140 + "。", "S1"), source("短证据。", "S2")])
    assert [item["quote"] for item in result.evidence] == ["短证据。"]
    assert result.audit["uncovered_source_ids"] == ["S1"]
    assert not result.audit["selected_spans"][0]["trimmed_for_budget"]


async def test_fixed_chunk_fragment_is_disclosed_and_never_exceeds_300_characters():
    text = "条件" * 400
    result = await HierarchicalCompressor(
        ConstantEncoder(), CompressionOptions(chunk_characters=300)
    ).compress("条件", [source(text)])
    assert all(len(item["quote"]) <= 300 for item in result.evidence)
    assert all(item["fragment"] for item in result.audit["selected_spans"])
    assert all(not item["trimmed_for_budget"] for item in result.audit["selected_spans"])
    for item in result.evidence:
        assert item["quote"] == text[item["start"] : item["end"]]


async def test_punctuation_heavy_input_has_bounded_graph_without_silent_source_truncation():
    text = "a。" * 6000
    result = await HierarchicalCompressor(
        ConstantEncoder(), CompressionOptions(chunk_characters=300, coarse_top_k=64)
    ).compress("a", [source(text)])
    assert result.audit["input_source_bytes"] == len(text.encode("utf-8"))
    assert result.audit["chunk_count"] == 40
    assert result.audit["sentence_count"] <= 512
    assert result.audit["coarse"]["encoded_document_count"] == 1


async def test_max_evidence_count_is_enforced():
    documents = [source(f"A research fact {index}.", f"S{index}") for index in range(8)]
    result = await HierarchicalCompressor(
        RecordingEncoder(), CompressionOptions(max_evidence=2)
    ).compress("research", documents)
    assert len(result.evidence) == result.audit["selected_count"] == 2
    assert len(result.audit["uncovered_source_ids"]) == 6


async def test_empty_sources_do_not_call_encoder():
    encoder = RecordingEncoder()
    result = await HierarchicalCompressor(encoder).compress("question", [])
    assert result.evidence == []
    assert result.audit["input_source_bytes"] == 0
    assert result.audit["evidence_json_bytes"] == 2
    assert result.audit["coarse"]["encoded_document_count"] == 0
    assert encoder.calls == []


@pytest.mark.parametrize("question", [None, 1, True, "", " \t", "a" * 1001, "\ud800"])
async def test_invalid_question_is_rejected_before_encoding(question):
    encoder = RecordingEncoder()
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(encoder).compress(question, [source("text")])
    assert captured.value.code == "compression_input"
    assert encoder.calls == []


@pytest.mark.parametrize(
    "documents",
    [None, {}, (), [None], [{}], [source("valid")] * 2, [source("x", str(i)) for i in range(9)]],
)
async def test_invalid_source_container_and_duplicate_ids_are_rejected(documents):
    encoder = RecordingEncoder()
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(encoder).compress("question", documents)
    assert captured.value.code == "compression_input"
    assert encoder.calls == []


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("text", "", "compression_input"),
        ("text", " \n", "compression_input"),
        ("text", "\ud800", "compression_input"),
        ("text", 123, "compression_input"),
        ("source_id", "", "compression_input"),
        ("source_id", "a" * 65, "compression_input"),
        ("source_id", "\ud800", "compression_input"),
        ("source_id", True, "compression_input"),
        ("sha256", "f" * 64, "compression_integrity"),
        ("sha256", None, "compression_integrity"),
    ],
)
async def test_snapshot_text_id_and_hash_validation(key, value, code):
    encoder = RecordingEncoder()
    document = {**source("valid"), key: value}
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(encoder).compress("question", [document])
    assert captured.value.code == code
    assert encoder.calls == []


async def test_source_limit_is_utf8_bytes_not_characters():
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(RecordingEncoder()).compress("q", [source("中" * 21846)])
    assert captured.value.code == "compression_capacity"
    document = source("x" * (64 * 1024))
    result = await HierarchicalCompressor(ConstantEncoder()).compress("q", [document])
    assert result.audit["input_source_bytes"] == 64 * 1024


async def test_capacity_overflow_is_explicit_and_happens_before_encoder_calls():
    encoder = RecordingEncoder()
    documents = [source("x" * 65536, f"S{index}") for index in range(3)]
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(encoder, CompressionOptions(chunk_characters=32)).compress(
            "question", documents
        )
    assert captured.value.code == "compression_capacity"
    assert encoder.calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"max_evidence": 0},
        {"max_evidence": 17},
        {"max_evidence": True},
        {"max_evidence": "3"},
        {"max_evidence_bytes": 511},
        {"max_evidence_bytes": 24577},
        {"coarse_top_k": 0},
        {"coarse_top_k": 65},
        {"chunk_characters": 31},
        {"chunk_characters": 301},
        {"extra": 1},
    ],
)
def test_options_are_strict_and_bounded(options):
    with pytest.raises(ValidationError):
        CompressionOptions(**options)


def test_options_are_frozen_and_untyped_options_are_rejected():
    options = CompressionOptions()
    with pytest.raises(ValidationError):
        options.max_evidence = 1
    with pytest.raises(ResearchError) as captured:
        HierarchicalCompressor(RecordingEncoder(), {"max_evidence": 1})
    assert captured.value.code == "compression_configuration"


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("dimensions", True),
        ("dimensions", 0),
        ("dimensions", 4097),
        ("dimensions", 2.0),
        ("encoder_id", ""),
        ("encoder_id", "\ud800"),
        ("encoder_id", "a" * 201),
        ("kind", "unknown"),
        ("encode_queries", None),
        ("encode_documents", None),
    ],
)
def test_encoder_configuration_is_validated(attribute, value):
    encoder = RecordingEncoder()
    setattr(encoder, attribute, value)
    with pytest.raises(ResearchError) as captured:
        HierarchicalCompressor(encoder)
    assert captured.value.code == "embedding_configuration"


@pytest.mark.parametrize("stage", ["query", "document"])
@pytest.mark.parametrize(
    "bad_vectors",
    [
        [[1, 0]],
        np.ones(2),
        np.ones((1, 3)),
        np.ones((2, 2)),
        np.ones((1, 2), dtype=bool),
        np.ones((1, 2), dtype=complex),
        np.array([["secret", "value"]], dtype=object),
        np.array([[np.nan, 0]]),
        np.array([[np.inf, 0]]),
    ],
)
async def test_malformed_encoder_vectors_fail_without_fallback(stage, bad_vectors):
    encoder = ConstantEncoder()

    async def malformed(texts):
        return bad_vectors

    setattr(encoder, "encode_queries" if stage == "query" else "encode_documents", malformed)
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(encoder).compress("question", [source("a")])
    assert captured.value.code == "embedding_vectors"
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (ValueError("sensitive upstream payload"), "embedding_failure"),
        (ResearchError("retrieval_network", "sensitive upstream payload"), "embedding_failure"),
        (ResearchError("embedding_timeout", "safe error"), "embedding_timeout"),
    ],
)
async def test_backend_failure_keeps_stable_category_and_does_not_silently_fallback(
    failure, expected_code
):
    encoder = RecordingEncoder()

    async def fail(texts):
        raise failure

    encoder.encode_queries = fail
    with pytest.raises(ResearchError) as captured:
        await HierarchicalCompressor(encoder).compress("question", [source("text")])
    assert captured.value.code == expected_code
    assert "sensitive upstream" not in str(captured.value)
    assert encoder.calls == []


async def test_cancellation_propagates_and_awaits_encoder_cleanup():
    entered, cleaned = asyncio.Event(), asyncio.Event()
    encoder = RecordingEncoder()

    async def wait_for_cancel(texts):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    encoder.encode_queries = wait_for_cancel
    task = asyncio.create_task(
        HierarchicalCompressor(encoder).compress("question", [source("text")])
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert encoder.calls == []


async def test_cancellation_at_preprocessing_checkpoint_never_calls_encoder():
    encoder = RecordingEncoder()
    task = asyncio.create_task(
        HierarchicalCompressor(encoder).compress("question", [source("text" * 1000)])
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert encoder.calls == []


async def test_query_coverage_prevents_repeated_background_from_hiding_relevant_tail():
    # The real demonstration is deliberately difficult: repeated background has
    # stronger graph centrality than the one-off answer near each document's end.
    from deep_research.compression_demo import DOCUMENTS, QUESTION

    result = await HierarchicalCompressor(RecordingEncoder()).compress(
        QUESTION, [source(document.text, f"S{index}") for index, document in enumerate(DOCUMENTS)]
    )
    combined = "\n".join(item["quote"] for item in result.evidence)
    assert "证据出处" in combined
    assert "并发调度" in combined or "全局超时" in combined
    coverage = [
        item for item in result.audit["selected_spans"] if item["reason"] == "source_coverage"
    ]
    assert len(coverage) == 2
    assert all(item["query_score"] > 0 for item in coverage)
