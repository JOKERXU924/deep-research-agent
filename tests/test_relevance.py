"""Synthetic relevance checks are not measured real-world retrieval accuracy."""

from dataclasses import asdict

import pytest

from deep_research.relevance import assess_relevance, title_relevance_score
from deep_research.retrieval import SourceDocument

QUESTION = (
    "请研究光合作用与碳固定的关系，说明二者的区别和联系。"
    "每个关键结论都要引用检索到的资料，证据不足的地方明确说明，不要编造来源。"
)


def source(title: str, text: str) -> SourceDocument:
    return SourceDocument("https://example.org/source", title, text)


@pytest.mark.parametrize(
    "document",
    [
        source("FAIRY TAIL角色列表", "虚构角色使用植物魔法。此条目列出作品角色与关系。"),
        source("放射性碳定年法", "根据碳同位素衰变测定年代，植物通过光合作用吸收碳。"),
        source(
            "放射性碳定年法",
            "根据碳同位素衰变测定年代，植物通过光合作用吸收碳。样本应采取固定的方法处理。",
        ),
        source("引用与研究报告", "每个关键结论都应引用资料，证据不足时明确说明。"),
        source("References and sources", "Research questions require evidence and cited claims."),
    ],
)
def test_filters_obvious_mismatches_without_page_specific_blacklist(document):
    decision = assess_relevance(QUESTION, document)
    assert not decision.accepted
    assert decision.reason == "insufficient_overlap"
    assert 0 <= decision.score <= 1


@pytest.mark.parametrize(
    "document,reason",
    [
        (source("光合作用", "植物转化光能。"), "title_match"),
        (source("碳固定", "将无机碳转变成有机物。"), "title_match"),
        (source("卡尔文循环", "卡尔文循环参与光合作用过程中的碳固定。"), "body_match"),
        (source("光呼吸", "光呼吸与光合作用相关，也影响碳固定的效率。"), "body_match"),
        (
            source("生物学词汇表", "遗传是亲代传递特征。光合作用利用光能。碳固定转化无机碳。"),
            "body_match",
        ),
    ],
)
def test_related_source_can_match_title_or_differently_named_body(document, reason):
    decision = assess_relevance(QUESTION, document)
    assert decision.accepted
    assert decision.reason == reason


def test_output_requirements_after_topic_cannot_supply_relevance():
    question = (
        "Explain photosynthesis and carbon fixation. "
        "Cite all sources and evidence; produce a detailed report with factual conclusions."
    )
    assert not assess_relevance(
        question,
        source(
            "How to cite sources", "A detailed report includes evidence and factual conclusions."
        ),
    ).accepted
    assert assess_relevance(
        question, source("Calvin cycle", "Photosynthesis uses carbon fixation.")
    ).accepted
    assert not assess_relevance(
        question,
        source("Radiocarbon dating", "Carbon enters plants by photosynthesis."),
    ).accepted


def test_english_title_matching_is_case_insensitive():
    decision = assess_relevance(
        "What is PHOTOSYNTHESIS and how is it related to carbon fixation?",
        source("Photosynthesis", "Energy conversion."),
    )
    assert decision.accepted
    assert decision.reason == "title_match"


@pytest.mark.parametrize("question", ["请研究以下问题：\n光合作用与碳固定", "光合作用与碳固定"])
def test_leading_scaffolding_does_not_replace_topic(question):
    assert assess_relevance(question, source("碳固定", "示例正文")).accepted
    assert not assess_relevance(question, source("虚构人物", "示例正文")).accepted


@pytest.mark.parametrize("question", ["请说明。", "?", "", "¿qué?", "请研究以下问题"])
def test_low_information_topic_is_explicitly_unassessed(question):
    # A lexical gate must not falsely claim an unsupported query is irrelevant.
    decision = assess_relevance(question, source("任意资料", "正文"))
    if question == "¿qué?":
        # Latin words can still be extracted; this is not a language detector.
        assert decision.reason == "insufficient_overlap"
    else:
        assert decision.accepted
        assert decision.reason == "no_topic_terms"
        assert decision.score == 0


def test_small_two_character_topic_and_mixed_language_titles():
    assert assess_relevance("癌症是什么？", source("癌症", "恶性肿瘤。")).accepted
    assert assess_relevance("Compare SQLite 与 PostgreSQL。", source("SQLite", "数据库")).accepted


def test_audit_decision_is_bounded_and_has_no_raw_question_or_document_values():
    decision = assess_relevance("研究光合作用与碳固定。PRIVATE", source("PRIVATE", "UNTRUSTED"))
    values = asdict(decision)
    assert set(values) == {
        "accepted",
        "reason",
        "topic_terms",
        "title_matches",
        "body_matches",
        "score",
    }
    assert "PRIVATE" not in str(values)
    assert "UNTRUSTED" not in str(values)
    assert 0 <= decision.topic_terms <= 64
    assert 0 <= decision.title_matches <= decision.topic_terms
    assert 0 <= decision.body_matches <= decision.topic_terms
    assert 0 <= decision.score <= 1


def test_large_input_and_normalized_characters_are_bounded():
    decision = assess_relevance(
        "photosynthesis " + "carbon " * 100_000, source("ＰＨＯＴＯＳＹＮＴＨＥＳＩＳ", "")
    )
    assert decision.accepted
    assert decision.topic_terms <= 64
    assert title_relevance_score("ＰＨＯＴＯＳＹＮＴＨＥＳＩＳ", "photosynthesis") == 1


def test_keyword_stuffing_is_a_documented_lexical_limit_not_semantic_truth():
    # Passing means lexically related, never verified, reliable or factually true.
    decision = assess_relevance(QUESTION, source("编造的条目", "光合作用与碳固定都是魔法。"))
    assert decision.accepted
    assert not hasattr(decision, "truth")


def test_short_topic_body_requires_every_core_term_not_scattered_partial_match():
    # General rule, not a photosynthesis/page-name blacklist. For a short query
    # three of four topic words still omit one core concept entirely.
    decision = assess_relevance(
        "Explain thermal energy storage efficiency.",
        source("Power supply", "Thermal energy storage is mentioned, but not its performance."),
    )
    assert decision.body_matches == 3
    assert decision.topic_terms == 4
    assert decision.score == 0.75
    assert not decision.accepted
    assert assess_relevance(
        "Explain thermal energy storage efficiency.",
        source("Power supply", "Thermal energy storage efficiency is measured."),
    ).accepted


def test_title_rank_does_not_invent_tokens_from_output_instructions():
    assert title_relevance_score(QUESTION, "光合作用") > title_relevance_score(
        QUESTION, "FAIRY TAIL"
    )
    assert title_relevance_score(QUESTION, "研究报告与引用") == 0
