"""Offline checks for lexical, not neural-semantic, memory vectors."""

import hashlib

import numpy as np
import pytest

from deep_research import memory_vector
from deep_research.errors import ResearchError
from deep_research.memory_vector import (
    MAX_ENCODE_BATCH,
    LexicalEncoder,
    conflict_reason,
    lexical_overlap,
    normalize_text,
    vector_scores,
)


def expected_counts(tokens):
    row = np.zeros(1024, dtype=np.float64)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        row[int.from_bytes(digest[:8], "big") % 1024] += 1
    return (row / np.linalg.norm(row)).astype(np.float32)


def test_encoder_contract_shape_dtype_identity_and_unit_vectors():
    encoder = LexicalEncoder()
    matrix = encoder.encode(["研究证据", "alpha beta", "", "🙂"])
    assert encoder.encoder_id == "lexical-hash-v1:1024"
    assert encoder.dimensions == 1024
    assert matrix.shape == (4, 1024)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1, 1, 0, 0], atol=1e-6)


def test_hash_buckets_are_stable_sha256_and_word_counts_are_preserved():
    actual = LexicalEncoder().encode(["ALPHA alpha Beta 2026"])[0]
    expected = expected_counts(["en:alpha", "en:alpha", "en:beta", "en:2026"])
    np.testing.assert_array_equal(actual, expected)


def test_chinese_uses_bigrams_and_single_character_fallback():
    actual = LexicalEncoder().encode(["研究证据", "水"])
    np.testing.assert_array_equal(actual[0], expected_counts(["zh:研究", "zh:究证", "zh:证据"]))
    np.testing.assert_array_equal(actual[1], expected_counts(["zh:水"]))


def test_encoder_does_not_use_randomized_python_hash(monkeypatch):
    expected = LexicalEncoder().encode(["模型 model 1024"])

    def forbidden_hash(value):
        raise AssertionError("Python hash must not determine stored vectors")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.hash", forbidden_hash)
        actual = LexicalEncoder().encode(["模型 model 1024"])
    np.testing.assert_array_equal(actual, expected)


def test_normalization_only_creates_key_and_keeps_source_string_unchanged():
    source = "  ＡＢＣ\t研究\n StraßE\u00a0１２３  "
    assert normalize_text(source) == "abc 研究 strasse 123"
    assert source == "  ＡＢＣ\t研究\n StraßE\u00a0１２３  "
    assert normalize_text("说明： 不同。") == "说明: 不同。"
    assert normalize_text("Cafe\u0301") == normalize_text("Café")


def test_equivalent_normalization_produces_identical_vectors():
    vectors = LexicalEncoder().encode(["ＡＢＣ 研究 １２３", "abc\t研究\n123"])
    np.testing.assert_array_equal(vectors[0], vectors[1])


def test_empty_batch_and_no_word_inputs_are_zero():
    encoder = LexicalEncoder()
    assert encoder.encode([]).shape == (0, 1024)
    assert encoder.encode([]).dtype == np.float32
    np.testing.assert_array_equal(encoder.encode(["", " \n\t", "!!!", "🙂"]), 0)


@pytest.mark.parametrize("text", [None, 1, True, b"text", ["text"], "\ud800", "a" * 4001])
def test_invalid_text_is_consistently_rejected_without_echoing_input(text):
    operations = [
        lambda: normalize_text(text),
        lambda: LexicalEncoder().encode([text]),
        lambda: lexical_overlap(text, "valid text"),
        lambda: conflict_reason("valid text", text),
    ]
    for operation in operations:
        with pytest.raises(ResearchError) as captured:
            operation()
        assert captured.value.code == "memory_vector"
        assert len(str(captured.value)) < 120


def test_exact_text_limit_is_allowed_and_normalization_expansion_stays_deterministic():
    assert len(normalize_text("a" * 4000)) == 4000
    assert LexicalEncoder().encode(["a" * 4000]).shape == (1, 1024)
    # The limit applies to original input characters, not the normalized expansion.
    assert normalize_text("ﬃ" * 4000) == "ffi" * 4000
    assert LexicalEncoder().encode(["ﬃ" * 4000]).shape == (1, 1024)


@pytest.mark.parametrize("texts", [None, "text", ("text",), {"text"}, [""] * 4097])
def test_invalid_encoding_batches_are_rejected(texts):
    with pytest.raises(ResearchError, match="list") as captured:
        LexicalEncoder().encode(texts)
    assert captured.value.code == "memory_vector"


def test_exact_batch_size_boundary_is_allowed():
    result = LexicalEncoder().encode([""] * MAX_ENCODE_BATCH)
    assert result.shape == (MAX_ENCODE_BATCH, 1024)
    assert not result.any()


def test_scores_normalize_vectors_without_mutating_inputs():
    query = np.zeros(1024, dtype=np.float32)
    query[0] = 7
    matrix = np.zeros((4, 1024), dtype=np.float32)
    matrix[0, 0] = 2
    matrix[1, 0] = -3
    matrix[2, 1] = 9
    original_query = query.copy()
    original_matrix = matrix.copy()
    scores = vector_scores(query, matrix)
    assert scores == [1.0, -1.0, 0.0, 0.0]
    assert all(isinstance(score, float) for score in scores)
    np.testing.assert_array_equal(query, original_query)
    np.testing.assert_array_equal(matrix, original_matrix)


@pytest.mark.parametrize("magnitude", [1e308, 1e-310])
def test_finite_extreme_values_do_not_overflow_or_underflow(magnitude):
    query = np.zeros(1024)
    query[:2] = magnitude
    matrix = np.stack([query, -query, np.zeros(1024)])
    assert vector_scores(query, matrix) == pytest.approx([1, -1, 0])


def test_zero_query_empty_matrix_and_integer_vectors():
    assert vector_scores(np.zeros(1024), np.ones((2, 1024))) == [0, 0]
    assert vector_scores(np.ones(1024), np.empty((0, 1024))) == []
    assert vector_scores(np.ones(1024, dtype=np.int64), np.ones((1, 1024), dtype=np.int64)) == [1]


def test_noncontiguous_and_readonly_arrays_are_supported():
    query = np.ones(2048)[::2]
    matrix = np.ones((2, 2048))[:, ::2]
    query.flags.writeable = False
    matrix.flags.writeable = False
    assert vector_scores(query, matrix) == [1, 1]


@pytest.mark.parametrize(
    "query,matrix",
    [
        ([1] * 1024, np.ones((1, 1024))),
        (np.ones(1024), [[1] * 1024]),
        (np.array(1), np.ones((1, 1024))),
        (np.ones((1, 1024)), np.ones((1, 1024))),
        (np.ones(1023), np.ones((1, 1024))),
        (np.ones(1024), np.ones(1024)),
        (np.ones(1024), np.ones((1, 1023))),
        (np.ones(1024), np.ones((1, 1, 1024))),
        (np.ones(1024, dtype=bool), np.ones((1, 1024))),
        (np.ones(1024), np.ones((1, 1024), dtype=bool)),
        (np.ones(1024, dtype=complex), np.ones((1, 1024))),
        (np.ones(1024), np.ones((1, 1024), dtype=object)),
        (np.full(1024, "1"), np.ones((1, 1024))),
        (np.full(1024, float("nan")), np.ones((1, 1024))),
        (np.ones(1024), np.full((1, 1024), float("inf"))),
        (np.full(1024, float("-inf")), np.empty((0, 1024))),
    ],
)
def test_invalid_vectors_raise_public_memory_error(query, matrix):
    with pytest.raises(ResearchError) as captured:
        vector_scores(query, matrix)
    assert captured.value.code == "memory_vector"


def test_lexical_overlap_uses_real_tokens_and_not_hash_buckets(monkeypatch):
    monkeypatch.setattr(memory_vector, "_bucket", lambda token: 0)
    vectors = LexicalEncoder().encode(["apple", "orange"])
    assert vector_scores(vectors[0], vectors[1:]) == [1]
    assert lexical_overlap("apple", "orange") == 0
    assert conflict_reason("apple", "orange") is None


def test_lexical_overlap_is_jaccard_not_frequency_or_semantic_similarity():
    assert lexical_overlap("alpha alpha beta", "alpha gamma") == pytest.approx(1 / 3)
    assert lexical_overlap("研究证据", "研究方法") == pytest.approx(1 / 5)
    assert lexical_overlap("ALPHA", "ａｌｐｈａ") == 1
    assert lexical_overlap("", "") == 0
    assert lexical_overlap("🙂", "!!!") == 0
    assert lexical_overlap("cars", "automobiles") == 0


@pytest.mark.parametrize(
    "left,right",
    [
        ("系统支持异步任务。", "系统不支持异步任务。"),
        ("地球是圆的。", "地球不是圆的。"),
        ("这个结论成立", "这个结论并不成立"),
        ("研究发现相关证据", "研究没有发现相关证据"),
        ("the service is available.", "the service is not available."),
        ("the service can cache results", "the service cannot cache results"),
        ("the service can cache results", "the service can't cache results"),
        ("the service is available", "the service isn't available"),
        ("the service does support caching", "the service doesn't support caching"),
    ],
)
def test_matching_stem_with_one_explicit_negation_is_only_a_possible_conflict(left, right):
    assert conflict_reason(left, right) == "possible_negation"
    assert conflict_reason(right, left) == "possible_negation"


@pytest.mark.parametrize(
    "left,right",
    [
        ("系统并发上限为3个任务。", "系统并发上限为8个任务。"),
        ("the timeout is 10 seconds", "the timeout is 20 seconds"),
        ("样本平均温度为-5度", "样本平均温度为5度"),
        ("the measured ratio is 0.3", "the measured ratio is 0.4"),
        ("the collection contains 1,000 items", "the collection contains 2,000 items"),
        ("the expected value is 1e3", "the expected value is 1e4"),
    ],
)
def test_same_stem_with_changed_numeric_values_is_only_a_candidate(left, right):
    assert conflict_reason(left, right) == "possible_value_change"
    assert conflict_reason(right, left) == "possible_value_change"


@pytest.mark.parametrize(
    "left,right",
    [
        ("系统支持本次配置变更", "系统反对本次配置变更"),
        ("系统允许外部访问", "系统禁止外部访问"),
        ("实验结果显示温度上升", "实验结果显示温度下降"),
        ("the service allows external access", "the service forbids external access"),
        ("the caching feature is enabled", "the caching feature is disabled"),
    ],
)
def test_explicit_opposition_with_matching_context_is_only_a_candidate(left, right):
    assert conflict_reason(left, right) == "possible_opposition"
    assert conflict_reason(right, left) == "possible_opposition"


@pytest.mark.parametrize(
    "left,right",
    [
        ("系统支持异步任务", "系统支持异步任务。"),
        ("系统支持异步任务", "系统支持异步操作"),
        ("研究报告使用证据", "研究报告保留证据"),
        ("系统不支持异步任务", "系统并不支持异步任务"),
        ("系统不得不支持异步任务", "系统支持异步任务"),
        ("系统不能不支持异步任务", "系统支持异步任务"),
        ("系统不仅支持异步任务", "系统支持异步任务"),
        ("the module not only supports caching", "the module supports caching"),
        ("the module does not not support caching", "the module does support caching"),
        ("系统不支持本次配置变更", "系统反对本次配置变更"),
        ("甲系统允许外部访问", "乙系统禁止外部访问"),
        ("the timeout is 10 seconds", "the timeout is 20 minutes"),
        ("the timeout is 10 seconds", "the timeout is 10.0 seconds"),
        ("the count is 1,000 items", "the count is 1000 items"),
        ("the count is 1e3 items", "the count is 1000 items"),
        ("model a1 supports caching", "model a2 supports caching"),
        ("123", "456"),
        ("", "the service is not available"),
        ("支持", "不支持"),
        ("🙂", "!!!"),
    ],
)
def test_unsupported_or_ambiguous_claims_are_not_declared_conflicts(left, right):
    assert conflict_reason(left, right) is None
    assert conflict_reason(right, left) is None
