"""Deterministic lexical vectors and conservative conflict candidate rules.

This is a word-count hash encoder, NOT a neural semantic embedding model. Hash
collisions and lexical similarity never prove that two passages mean the same
thing. Conflict rules only flag candidates for later inspection; they neither
verify facts nor choose which source should be believed.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from decimal import Decimal, InvalidOperation

import numpy as np

from deep_research.errors import ResearchError

MAX_TEXT_CHARS = 4000
MAX_ENCODE_BATCH = 4096
VECTOR_DIMENSIONS = 1024
_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ebef]+")
_WORDS = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(
    r"(?<![a-z0-9_])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[+-]?\d+)?"
    r"(?![a-z0-9_])"
)
_NEGATIVE_EN = re.compile(r"\b(?:not|never|no)\b")
_NEGATIVE_ZH = re.compile(
    r"(?:并没有|并不|并非|并未|未曾|从未|没有|不|未|非)"
    r"(?=支持|允许|需要|包含|存在|可以|具有|能够|符合|参与|增加|减少|发生|提供|适用|使用|"
    r"推荐|应当|发现|证实|是|能|会|正确|可靠|安全|成功|有效|成立|充足|稳定|相同)"
)
_COMPLEX_NEGATION = re.compile(
    r"不仅|不但|不论|不管|不得不|不能不|未必|不一定|\bnot\s+(?:only|just)\b"
)
_CONTRACTIONS = {
    "cannot": "can",
    "can't": "can",
    "isn't": "is",
    "aren't": "are",
    "wasn't": "was",
    "weren't": "were",
    "doesn't": "does",
    "don't": "do",
    "didn't": "did",
    "won't": "will",
    "wouldn't": "would",
    "shouldn't": "should",
    "couldn't": "could",
    "hasn't": "has",
    "haven't": "have",
    "hadn't": "had",
    "mustn't": "must",
}
_CONTRACTION = re.compile(r"\b(?:" + "|".join(map(re.escape, _CONTRACTIONS)) + r")\b")
_OPPOSITIONS = (
    ("支持", "反对"),
    ("允许", "禁止"),
    ("增加", "减少"),
    ("上升", "下降"),
    ("启用", "禁用"),
    ("成功", "失败"),
    ("有效", "无效"),
    ("support", "oppose"),
    ("supports", "opposes"),
    ("allow", "forbid"),
    ("allows", "forbids"),
    ("allowed", "forbidden"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("true", "false"),
)


def _invalid(message: str) -> ResearchError:
    return ResearchError("memory_vector", message)


def normalize_text(text: str) -> str:
    """Return a comparison key, never a replacement for stored source text."""
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
        raise _invalid("Memory text must be a string of at most 4000 characters.")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise _invalid("Memory text must contain valid UTF-8 characters.") from None
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(normalized: str) -> list[str]:
    tokens = ["en:" + word for word in _WORDS.findall(normalized)]
    for match in _HAN.finditer(normalized):
        word = match.group()
        # A lone Chinese character remains searchable; longer spans use bigrams.
        if len(word) == 1:
            tokens.append("zh:" + word)
        else:
            tokens.extend("zh:" + word[index : index + 2] for index in range(len(word) - 1))
    return tokens


def _bucket(token: str) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % VECTOR_DIMENSIONS


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    """Scale before squaring to avoid overflow/underflow on finite extremes."""
    values = rows.astype(np.float64, copy=True)
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    np.divide(values, scales, out=values, where=scales != 0)
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    np.divide(values, lengths, out=values, where=lengths != 0)
    return values


class LexicalEncoder:
    """English/digit words + Chinese bigram counts in fixed SHA-256 buckets."""

    encoder_id = "lexical-hash-v1:1024"
    dimensions = VECTOR_DIMENSIONS

    def encode(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list) or len(texts) > MAX_ENCODE_BATCH:
            raise _invalid("Encoding requires a list of at most 4096 text entries.")
        # Validate the entire batch before allocating the vector matrix.
        normalized = [normalize_text(text) for text in texts]
        rows = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for index, text in enumerate(normalized):
            for token, count in Counter(_tokens(text)).items():
                rows[index, _bucket(token)] += count
        if not texts:
            return rows
        return _unit_rows(rows).astype(np.float32)


def vector_scores(query_vector: np.ndarray, matrix: np.ndarray) -> list[float]:
    """Cosine scores after validation and safe normalization; zero rows score 0."""
    if not isinstance(query_vector, np.ndarray) or not isinstance(matrix, np.ndarray):
        raise _invalid("Memory vectors must be NumPy arrays.")
    if (
        query_vector.ndim != 1
        or query_vector.shape[0] != VECTOR_DIMENSIONS
        or matrix.ndim != 2
        or matrix.shape[1] != VECTOR_DIMENSIONS
    ):
        raise _invalid("Memory vectors must use the configured 1024 dimensions.")
    if query_vector.dtype.kind not in "fiu" or matrix.dtype.kind not in "fiu":
        raise _invalid("Memory vectors must contain real numeric values.")
    if not np.isfinite(query_vector).all() or not np.isfinite(matrix).all():
        raise _invalid("Memory vectors must contain finite values.")
    if matrix.shape[0] == 0:
        return []
    with np.errstate(over="ignore", invalid="ignore"):
        query = _unit_rows(query_vector.reshape(1, -1))[0]
        rows = _unit_rows(matrix)
    if not np.isfinite(query).all() or not np.isfinite(rows).all():
        raise _invalid("Memory vectors cannot be represented as finite float64 values.")
    scores = np.clip(rows @ query, -1.0, 1.0)
    return scores.tolist()


def lexical_overlap(left: str, right: str) -> float:
    """Jaccard overlap of actual lexical tokens, independent of hash collisions."""
    left_tokens = set(_tokens(normalize_text(left)))
    right_tokens = set(_tokens(normalize_text(right)))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _statement_key(text: str) -> str:
    return " ".join(text.split()).strip(" .。!?！？;；")


def _has_anchor(text: str) -> bool:
    return len(text.replace(" ", "")) >= 4 and len(set(_tokens(text))) >= 2


def _without_negation(text: str) -> tuple[str, int]:
    stem, contractions = _CONTRACTION.subn(lambda match: _CONTRACTIONS[match.group()], text)
    stem, english = _NEGATIVE_EN.subn("", stem)
    stem, chinese = _NEGATIVE_ZH.subn("", stem)
    return _statement_key(stem), contractions + english + chinese


def _numeric_shape(text: str) -> tuple[str, list[Decimal]] | None:
    matches = list(_NUMBER.finditer(text))
    if not matches:
        return None
    try:
        values = [Decimal(match.group().replace(",", "")) for match in matches]
    except InvalidOperation:
        return None
    return _NUMBER.sub(" <number> ", text), values


def _opposition_shape(text: str, word: str) -> tuple[str, int]:
    pattern = re.escape(word)
    if word.isascii():
        pattern = r"\b" + pattern + r"\b"
    return re.subn(pattern, "<opposition>", text)


def conflict_reason(left: str, right: str) -> str | None:
    """Return a possible conflict category, not a truth or contradiction verdict.

    Negation and opposition rules require matching residual sentence structure.
    Double negation, unfamiliar wording, and low-overlap statements deliberately
    produce false negatives rather than a fabricated semantic judgment.
    """
    left_key = _statement_key(normalize_text(left))
    right_key = _statement_key(normalize_text(right))
    if not left_key or not right_key or left_key == right_key:
        return None
    if _COMPLEX_NEGATION.search(left_key) or _COMPLEX_NEGATION.search(right_key):
        return None
    left_stem, left_negations = _without_negation(left_key)
    right_stem, right_negations = _without_negation(right_key)
    if (
        {left_negations, right_negations} == {0, 1}
        and left_stem == right_stem
        and _has_anchor(left_stem)
    ):
        return "possible_negation"

    left_numeric = _numeric_shape(left_key)
    right_numeric = _numeric_shape(right_key)
    if left_numeric is not None and right_numeric is not None:
        left_shape, left_values = left_numeric
        right_shape, right_values = right_numeric
        if (
            left_shape == right_shape
            and len(left_values) == len(right_values)
            and left_values != right_values
            and _has_anchor(left_shape.replace("<number>", ""))
        ):
            return "possible_value_change"

    # "Not supporting" and "opposing" must not be called opposites by stripping
    # their verbs alone. More complex negative statements are outside this rule.
    if left_negations or right_negations:
        return None
    for positive, negative in _OPPOSITIONS:
        for first, second in ((positive, negative), (negative, positive)):
            left_shape, left_count = _opposition_shape(left_key, first)
            right_shape, right_count = _opposition_shape(right_key, second)
            if (
                left_count == right_count == 1
                and left_shape == right_shape
                and _has_anchor(left_shape.replace("<opposition>", ""))
            ):
                return "possible_opposition"
    return None
