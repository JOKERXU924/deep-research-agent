"""A bounded lexical topic gate, not a semantic relevance or truth classifier.

The original user's first substantive sentence supplies the topic. Later output
instructions ("cite every claim", for example) must not make unrelated pages
look relevant. Title matches are deliberately preferred; a differently named
    page may still pass when its body covers the topic. Synonyms, translation
and keyword stuffing are limitations of this inexpensive deterministic gate.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from deep_research.retrieval import SourceDocument

MAX_TOPIC_CHARACTERS = 2048
MAX_TOPIC_TERMS = 64
MAX_BODY_CHARACTERS = 64 * 1024

_ENGLISH_STOP = frozenset(
    "a an the of and or to from for with in on at by as is are was were be been "
    "being it its this that these those their them they we our you your i me my "
    "do does did can could would should will shall may might have has had "
    "what which who when where why how whether please research explain describe "
    "compare comparison relationship relationships difference differences between "
    "about into under over through using use used also following question questions".split()
)
_CHINESE_STOP = re.compile(
    "|".join(
        re.escape(word)
        for word in sorted(
            {
                "研究一下",
                "分析一下",
                "比较一下",
                "介绍一下",
                "说明一下",
                "请帮我",
                "是什么",
                "有什么",
                "有哪些",
                "为什么",
                "是否",
                "分别",
                "之间",
                "如何",
                "什么",
                "以下",
                "上述",
                "关于",
                "有关",
                "相关",
                "以及",
                "及其",
                "二者",
                "两者",
                "关系",
                "区别",
                "联系",
                "详细",
                "简要",
                "说明",
                "介绍",
                "总结",
                "分析",
                "比较",
                "研究",
                "问题",
                "帮我",
                "并且",
                "请问",
                "請問",
                "關於",
                "關係",
                "區別",
                "聯繫",
                "說明",
                "介紹",
                "問題",
                "请",
                "請",
                "与",
                "與",
                "和",
                "及",
                "的",
                "是",
                "呢",
                "吗",
                "嗎",
            },
            key=lambda word: (-len(word), word),
        )
    )
)
_HAN = re.compile(r"[\u3400-\u9fff]+")
_WORD = re.compile(r"[a-z][a-z0-9_-]*")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _terms(text: str, *, topic: bool = False) -> set[str]:
    normalized = _normalize(text)
    english = {word for word in _WORD.findall(normalized) if len(word) > 1}
    if topic:
        english.difference_update(_ENGLISH_STOP)
        normalized = _CHINESE_STOP.sub(" ", normalized)
    result = english
    for segment in _HAN.findall(normalized):
        result.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return result


def _topic_terms(question: str) -> set[str]:
    # Treat a leading "please research the following question:" as scaffolding,
    # then take the first actual topic. Never fill a missing topic with evidence
    # or report requirements from a later sentence.
    for sentence in re.split(r"[\n\r。！？.!?:：]+", question[:MAX_TOPIC_CHARACTERS]):
        terms = _terms(sentence, topic=True)
        if terms:
            return set(sorted(terms)[:MAX_TOPIC_TERMS])
    return set()


@dataclass(frozen=True)
class RelevanceDecision:
    """Safe audit metadata: counts and stable codes, never source/request text."""

    accepted: bool
    reason: Literal["title_match", "body_match", "no_topic_terms", "insufficient_overlap"]
    topic_terms: int
    title_matches: int
    body_matches: int
    score: float


def title_relevance_score(query: str, title: str) -> float:
    """Stable local ranking for already-returned search hits; no new requests."""
    wanted = _topic_terms(query)
    return len(wanted & _terms(title[:512])) / len(wanted) if wanted else 0.0


def assess_relevance(question: str, document: "SourceDocument") -> RelevanceDecision:
    """Conservatively screen obvious topical mismatches before evidence selection.

    A specific English title token, two Chinese title bigrams (or the sole
    two-character topic) suffice. Without a title match a short topic (up to
    eight core terms) must be completely covered; a longer topic needs at least
    75% coverage. A partial match on a short question often identifies only one
    of its main concepts, especially with overlapping Chinese bigrams.
    Low-information/unsupported-language questions are
    allowed explicitly instead of pretending the heuristic can assess them.
    """
    wanted = _topic_terms(question)
    title = wanted & _terms(document.title[:512])
    body = wanted & _terms(document.text[:MAX_BODY_CHARACTERS])
    if not wanted:
        return RelevanceDecision(True, "no_topic_terms", 0, 0, 0, 0.0)
    score = round(len(title | body) / len(wanted), 4)
    title_is_specific = (
        any(term[0].isascii() for term in title)
        or len(title) >= 2
        or (len(wanted) == 1 and bool(title))
    )
    if title_is_specific:
        reason = "title_match"
    elif len(body) / len(wanted) >= (1.0 if len(wanted) <= 8 else 0.75):
        reason = "body_match"
    else:
        reason = "insufficient_overlap"
    return RelevanceDecision(
        reason != "insufficient_overlap", reason, len(wanted), len(title), len(body), score
    )
