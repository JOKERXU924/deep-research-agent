"""Offline, bounded evaluation; provenance and lexical overlap are not truth.

HotpotQA formulas independently implemented to match its official evaluator:
https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
Dataset and task: Yang et al., EMNLP 2018, https://hotpotqa.github.io/.
No dataset is included or downloaded here. HotpotQA data is CC BY-SA 4.0.
"""

import hashlib
import re
import string
import unicodedata
from collections import Counter

MAX_CASES = 1000
MAX_TEXT = 262144
MAX_ITEMS = 1000


def _list(value, name: str, *, limit: int = MAX_ITEMS) -> list:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} must be a bounded list")
    return value


def _text(value, name: str, *, empty: bool = False, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{name} must be bounded text")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _lexical(text: str) -> str:
    # Unlike Hotpot's English normalization, preserve articles and Unicode letters.
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in text if char.isalnum())


def _unique_index(items: list, key: str) -> dict[str, dict]:
    """Duplicate IDs make every occurrence ambiguous; never last-write-wins."""
    index: dict[str, dict] = {}
    duplicates: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        identifier = item.get(key)
        if not isinstance(identifier, str) or not identifier or len(identifier) > 200:
            continue
        if identifier in index:
            duplicates.add(identifier)
        index[identifier] = item
    return {key: item for key, item in index.items() if key not in duplicates}


def _valid_evidence(sources: list, evidence: list) -> set[str]:
    source_index = _unique_index(sources, "source_id")
    valid_sources = {}
    for key, source in source_index.items():
        text = source.get("text")
        if not isinstance(text, str) or len(text) > MAX_TEXT:
            continue
        if "sha256" in source:
            try:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            except UnicodeError:
                continue
            if source["sha256"] != digest:
                continue
        valid_sources[key] = text
    valid = set()
    for key, item in _unique_index(evidence, "evidence_id").items():
        source_id = item.get("source_id")
        text = valid_sources.get(source_id) if isinstance(source_id, str) else None
        start, end, quote = item.get("start"), item.get("end"), item.get("quote")
        if (
            text is not None
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
            and isinstance(quote, str)
            and quote.strip()
            and quote == text[start:end]
        ):
            valid.add(key)
    return valid


def score_report(bundle: dict, gold: dict) -> dict:
    """Score all supplied claims, not only claims with valid citations.

    Gold facts are strings or {statement, aliases} dictionaries. An alias or the
    statement must be a normalized substring of ONE claim to count as covered.
    Even negated/repeated facts can match: this metric is explicitly a proxy.
    """
    if not isinstance(bundle, dict) or not isinstance(gold, dict):
        raise ValueError("bundle and gold must be objects")
    report = bundle.get("report", {})
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    sources = _list(bundle.get("sources", []), "sources", limit=100)
    evidence = _list(bundle.get("evidence", []), "evidence")
    valid = _valid_evidence(sources, evidence)
    claims: list[dict] = []
    for section in _list(report.get("sections", []), "sections", limit=100):
        if not isinstance(section, dict):
            raise ValueError("section must be an object")
        claims.extend(_list(section.get("claims", []), "claims"))
        if len(claims) > MAX_ITEMS:
            raise ValueError("too many claims")
    cited_claims = refs = valid_refs = 0
    normalized_claims = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("claim must be an object")
        text = _text(claim.get("text", ""), "claim text", empty=True)
        normalized_claims.append(_lexical(text))
        citations = _list(claim.get("citations", []), "citations", limit=100)
        refs += len(citations)
        count = sum(isinstance(cite, str) and cite in valid for cite in citations)
        valid_refs += count
        cited_claims += int(count > 0)
    facts = _list(gold.get("facts", []), "facts")
    matched = 0
    for fact in facts:
        if isinstance(fact, str):
            alternatives = [_text(fact, "fact")]
        elif isinstance(fact, dict):
            alternatives = [_text(fact.get("statement"), "fact statement")]
            alternatives.extend(
                _text(alias, "fact alias")
                for alias in _list(fact.get("aliases", []), "aliases", limit=20)
            )
        else:
            raise ValueError("fact must be text or an object")
        normalized = [_lexical(text) for text in alternatives]
        if any(not text for text in normalized):
            raise ValueError("facts and aliases must contain letters or numbers")
        matched += int(any(part in claim for part in normalized for claim in normalized_claims))
    return {
        "citation_coverage": _ratio(cited_claims, len(claims)),
        "citation_validity": _ratio(valid_refs, refs),
        "gold_fact_coverage": _ratio(matched, len(facts)),
        "nonempty_output": any(normalized_claims),
        "counts": {
            "claims": len(claims),
            "cited_claims": cited_claims,
            "citation_refs": refs,
            "valid_citation_refs": valid_refs,
            "gold_facts": len(facts),
            "matched_gold_facts": matched,
            "valid_evidence": len(valid),
        },
        "metric_notes": [
            "Citation validity checks snapshot provenance, not entailment or factual truth.",
            "Gold fact coverage is a lexical substring proxy, not accuracy or hallucination rate.",
            "Zero denominators are null; duplicate citations count per reference.",
        ],
    }


def score_claim_labels(claim_ids: list[str], labels: dict[str, str]) -> dict:
    """Summarize explicit manual/judge labels; missing labels never imply support."""
    _list(claim_ids, "claim ids")
    if not isinstance(labels, dict) or len(labels) > MAX_ITEMS:
        raise ValueError("labels must be a bounded object")
    for key in claim_ids:
        _text(key, "claim id", limit=200)
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("duplicate claim ids")
    if set(labels) - set(claim_ids):
        raise ValueError("labels contain unknown claim ids")
    allowed = {"supported", "contradicted", "insufficient"}
    if any(not isinstance(value, str) or value not in allowed for value in labels.values()):
        raise ValueError("invalid claim label")
    counts = Counter(labels.values())
    missing = [key for key in claim_ids if key not in labels]
    return {
        "fact_support_rate": _ratio(counts["supported"], len(labels)),
        "unsupported_rate": _ratio(counts["contradicted"] + counts["insufficient"], len(labels)),
        "label_coverage": _ratio(len(labels), len(claim_ids)),
        "labeled_claims": len(labels),
        "total_claims": len(claim_ids),
        "counts": {label: counts[label] for label in sorted(allowed)},
        "missing_claim_ids": missing,
        "note": "Label-based support is relative to supplied evidence, not objective truth.",
    }


def normalize_answer(answer: str) -> str:
    """Official HotpotQA normalization: lower, ASCII punctuation, articles, spaces."""
    _text(answer, "answer", empty=True)
    no_punctuation = answer.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(?:a|an|the)\b", " ", no_punctuation).split())


def _harmonic(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def answer_scores(prediction: str, gold: str) -> dict[str, float]:
    """Official EM and bag-of-token F1, including yes/no/noanswer special cases."""
    actual, expected = normalize_answer(prediction), normalize_answer(gold)
    result = {"em": float(actual == expected), "f1": 0.0, "prec": 0.0, "recall": 0.0}
    if actual != expected and ({actual, expected} & {"yes", "no", "noanswer"}):
        return result
    actual_tokens, expected_tokens = actual.split(), expected.split()
    overlap = sum((Counter(actual_tokens) & Counter(expected_tokens)).values())
    if overlap:
        result["prec"] = overlap / len(actual_tokens)
        result["recall"] = overlap / len(expected_tokens)
        result["f1"] = _harmonic(result["prec"], result["recall"])
    return result


def _support_set(value: list) -> set[tuple[str, int]]:
    result = set()
    for item in _list(value, "supporting facts", limit=1000):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("supporting fact must be [title, sentence index]")
        title, index = item
        _text(title, "supporting title", limit=1000)
        if type(index) is not int or not 0 <= index <= 100000:
            raise ValueError("supporting sentence index must be a nonnegative integer")
        result.add((title, index))
    return result


def supporting_fact_scores(prediction: list, gold: list) -> dict[str, float]:
    """HotpotQA treats supporting facts as exact, case-sensitive (title,index) sets."""
    actual, expected = _support_set(prediction), _support_set(gold)
    overlap = len(actual & expected)
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    return {
        "sp_em": float(actual == expected),
        "sp_f1": _harmonic(precision, recall),
        "sp_prec": precision,
        "sp_recall": recall,
    }


def evaluate_hotpot(gold: list[dict], predictions: dict) -> dict:
    """Score every gold ID; missing predictions score zero without reducing N.

    Deliberately stricter ingestion than the original CLI: reject duplicate gold
    IDs, extra prediction IDs, malformed inputs, and unbounded collections.
    Duplicate supporting facts are allowed, matching the official set semantics.
    """
    _list(gold, "gold cases", limit=MAX_CASES)
    if not gold:
        raise ValueError("gold cases must not be empty")
    if not isinstance(predictions, dict) or set(predictions) != {"answer", "sp"}:
        raise ValueError("predictions must contain answer and sp objects")
    answers, supports = predictions["answer"], predictions["sp"]
    if not isinstance(answers, dict) or not isinstance(supports, dict):
        raise ValueError("predictions must contain answer and sp objects")
    if len(answers) > MAX_CASES or len(supports) > MAX_CASES:
        raise ValueError("too many predictions")
    case_index = {}
    for case in gold:
        if not isinstance(case, dict):
            raise ValueError("gold case must be an object")
        key = _text(case.get("_id"), "case id", limit=200)
        if key in case_index:
            raise ValueError("duplicate gold case id")
        _text(case.get("answer"), "gold answer", empty=True)
        _support_set(case.get("supporting_facts"))
        case_index[key] = case
    if set(answers) - case_index.keys() or set(supports) - case_index.keys():
        raise ValueError("unknown prediction case ids")
    fields = (
        "em",
        "f1",
        "prec",
        "recall",
        "sp_em",
        "sp_f1",
        "sp_prec",
        "sp_recall",
        "joint_em",
        "joint_f1",
        "joint_prec",
        "joint_recall",
    )
    totals = dict.fromkeys(fields, 0.0)
    per_case = {}
    for key, case in case_index.items():
        scores = dict.fromkeys(fields, 0.0)
        if key in answers:
            scores.update(answer_scores(answers[key], case["answer"]))
        if key in supports:
            scores.update(supporting_fact_scores(supports[key], case["supporting_facts"]))
        if key in answers and key in supports:
            scores["joint_prec"] = scores["prec"] * scores["sp_prec"]
            scores["joint_recall"] = scores["recall"] * scores["sp_recall"]
            scores["joint_f1"] = _harmonic(scores["joint_prec"], scores["joint_recall"])
            scores["joint_em"] = scores["em"] * scores["sp_em"]
        for field in fields:
            totals[field] += scores[field]
        per_case[key] = scores
    size = len(gold)
    return {
        "metrics": {field: value / size for field, value in totals.items()},
        "count": size,
        "answer_coverage": len(answers) / size,
        "supporting_fact_coverage": len(supports) / size,
        "joint_coverage": len(answers.keys() & supports.keys()) / size,
        "missing_answers": sorted(case_index.keys() - answers.keys()),
        "missing_supporting_facts": sorted(case_index.keys() - supports.keys()),
        "per_case": per_case,
        "scoring_reference": "https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py",
        "note": "Local selected-set evaluation; not an official leaderboard submission.",
    }
