"""Hand-calculated offline tests; fixture scores are not empirical quality claims."""

import copy
import hashlib

import pytest

from deep_research.benchmark_metrics import (
    answer_scores,
    evaluate_hotpot,
    normalize_answer,
    score_claim_labels,
    score_report,
    supporting_fact_scores,
)


def sample_bundle():
    text = "北京是中国的首都。Earth has one natural moon."
    return {
        "sources": [
            {"source_id": "S1", "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}
        ],
        "evidence": [
            {"evidence_id": "E1", "source_id": "S1", "start": 0, "end": 9, "quote": text[:9]}
        ],
        "report": {
            "title": "示例",
            "sections": [
                {
                    "claims": [
                        {"text": "北京是中国的首都。", "citations": ["E1"]},
                        {"text": "Earth has one natural moon.", "citations": ["E999"]},
                        {"text": "A claim with no citation.", "citations": []},
                    ]
                }
            ],
        },
    }


def test_report_denominators_and_lexical_aliases():
    gold = {
        "facts": [
            {"statement": "Beijing is the capital of China", "aliases": ["北京是中国的首都"]},
            "Earth has one natural moon",
            "Unmentioned fact",
        ]
    }
    result = score_report(sample_bundle(), gold)
    assert result["citation_coverage"] == pytest.approx(1 / 3)
    assert result["citation_validity"] == 0.5
    assert result["gold_fact_coverage"] == pytest.approx(2 / 3)
    assert result["nonempty_output"] is True
    assert result["counts"] == {
        "claims": 3,
        "cited_claims": 1,
        "citation_refs": 2,
        "valid_citation_refs": 1,
        "gold_facts": 3,
        "matched_gold_facts": 2,
        "valid_evidence": 1,
    }
    assert "not accuracy" in result["metric_notes"][1]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b["evidence"][0].update(source_id="unknown"),
        lambda b: b["evidence"][0].update(source_id=[]),
        lambda b: b["evidence"][0].update(start=True),
        lambda b: b["evidence"][0].update(end=9.0),
        lambda b: b["evidence"][0].update(start=-1),
        lambda b: b["evidence"][0].update(end=999),
        lambda b: b["evidence"][0].update(end=0),
        lambda b: b["evidence"][0].update(quote="fabricated"),
        lambda b: b["sources"][0].update(sha256="0" * 64),
        lambda b: b["sources"][0].update(sha256=None),
        lambda b: b["sources"].append(copy.deepcopy(b["sources"][0])),
        lambda b: b["evidence"].append(copy.deepcopy(b["evidence"][0])),
    ],
)
def test_invalid_provenance_counts_zero_not_exception(mutation):
    bundle = sample_bundle()
    mutation(bundle)
    score = score_report(bundle, {"facts": []})
    assert score["citation_coverage"] == 0.0
    assert score["citation_validity"] == 0.0
    assert score["counts"]["valid_evidence"] == 0


def test_hash_optional_and_offsets_unicode_not_utf8_bytes():
    bundle = sample_bundle()
    del bundle["sources"][0]["sha256"]
    assert score_report(bundle, {})["counts"]["valid_evidence"] == 1
    bundle["evidence"][0]["end"] = len(bundle["evidence"][0]["quote"].encode())
    assert score_report(bundle, {})["counts"]["valid_evidence"] == 0


def test_valid_provenance_does_not_prove_entailment():
    bundle = sample_bundle()
    bundle["report"]["sections"][0]["claims"] = [
        {"text": "The moon is made of cheese.", "citations": ["E1"]}
    ]
    result = score_report(bundle, {"facts": ["北京是中国的首都"]})
    assert result["citation_validity"] == 1
    assert result["gold_fact_coverage"] == 0
    assert "not entailment" in result["metric_notes"][0]


def test_lexical_proxy_can_match_a_negation_but_not_cross_claim_boundaries():
    bundle = sample_bundle()
    bundle["report"]["sections"][0]["claims"] = [
        {"text": "It is false that Beijing is the capital.", "citations": []},
        {"text": "First", "citations": []},
        {"text": "second", "citations": []},
    ]
    result = score_report(bundle, {"facts": ["Beijing is the capital", "First second"]})
    assert result["gold_fact_coverage"] == 0.5


def test_empty_denominators_null_and_not_nonempty_output():
    result = score_report({}, {})
    assert result["citation_coverage"] is None
    assert result["citation_validity"] is None
    assert result["gold_fact_coverage"] is None
    assert result["nonempty_output"] is False


def test_malformed_and_duplicate_reference_entries_are_not_removed():
    bundle = sample_bundle()
    bundle["report"]["sections"][0]["claims"] = [
        {"text": "Claim", "citations": ["E1", "E1", None, [], "E999"]}
    ]
    result = score_report(bundle, {})
    assert result["citation_validity"] == 2 / 5
    assert result["counts"]["citation_refs"] == 5


@pytest.mark.parametrize(
    "bundle,gold",
    [
        ([], {}),
        ({}, []),
        ({"report": []}, {}),
        ({"report": {"sections": [None]}}, {}),
        ({"report": {"sections": [{"claims": [None]}]}}, {}),
        ({"report": {"sections": [{"claims": [{"citations": "E1"}]}]}}, {}),
        ({}, {"facts": [None]}),
        ({}, {"facts": [""]}),
        ({}, {"facts": [{"statement": "x", "aliases": [""]}]}),
        ({}, {"facts": ["!!!"]}),
        ({"sources": [None] * 101}, {}),
    ],
)
def test_report_rejects_malformed_collection_or_gold(bundle, gold):
    with pytest.raises(ValueError):
        score_report(bundle, gold)


def test_manual_labels_denominator_excludes_missing_but_reports_coverage():
    result = score_claim_labels(
        ["c1", "c2", "c3", "c4"], {"c1": "supported", "c2": "contradicted", "c3": "insufficient"}
    )
    assert result["fact_support_rate"] == 1 / 3
    assert result["unsupported_rate"] == 2 / 3
    assert result["label_coverage"] == 3 / 4
    assert result["missing_claim_ids"] == ["c4"]
    empty = score_claim_labels([], {})
    assert empty["fact_support_rate"] is None
    assert empty["label_coverage"] is None


@pytest.mark.parametrize(
    "ids,labels",
    [
        (["a", "a"], {}),
        (["a"], {"b": "supported"}),
        (["a"], {"a": "true"}),
        (["a"], {"a": True}),
        (["a"], {"a": []}),
        ([""], {}),
        (["a"], []),
    ],
)
def test_manual_labels_strict(ids, labels):
    with pytest.raises(ValueError):
        score_claim_labels(ids, labels)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" The, Quick Brown FOX! ", "quick brown fox"),
        ("An apple; a pear", "apple pear"),
        ("U.S.A.", "usa"),
        ("The 中文，标题", "中文，标题"),
        ("a an the", ""),
    ],
)
def test_hotpot_normalization(raw, expected):
    assert normalize_answer(raw) == expected


def test_hotpot_token_multiplicity_and_hand_calculated_f1():
    assert answer_scores("cat cat dog", "cat dog dog") == {
        "em": 0.0,
        "prec": 2 / 3,
        "recall": 2 / 3,
        "f1": 2 / 3,
    }
    assert answer_scores("red fox", "the red fox jumps") == {
        "em": 0.0,
        "prec": 1.0,
        "recall": 2 / 3,
        "f1": 0.8,
    }


@pytest.mark.parametrize("special", ["yes", "no", "noanswer"])
def test_hotpot_special_answers_do_not_receive_partial_overlap(special):
    assert answer_scores(special, special + " question")["f1"] == 0
    assert answer_scores(special + " question", special)["f1"] == 0
    assert answer_scores(special, special)["f1"] == 1


def test_hotpot_official_empty_token_corner_case():
    assert answer_scores("the", "a") == {"em": 1.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    assert supporting_fact_scores([], []) == {
        "sp_em": 1.0,
        "sp_f1": 0.0,
        "sp_prec": 0.0,
        "sp_recall": 0.0,
    }


def test_supporting_facts_exact_set_semantics():
    result = supporting_fact_scores([["A", 0], ["A", 0], ["B", 1]], [["A", 0]])
    assert result == {"sp_em": 0.0, "sp_prec": 0.5, "sp_recall": 1.0, "sp_f1": 2 / 3}
    assert supporting_fact_scores([["a", 0]], [["A", 0]])["sp_f1"] == 0


def hotpot_gold():
    return [
        {"_id": "q1", "answer": "red fox jumps", "supporting_facts": [["A", 0]]},
        {"_id": "q2", "answer": "yes", "supporting_facts": [["B", 1]]},
    ]


def test_hotpot_joint_and_missing_ids_do_not_shrink_denominator():
    result = evaluate_hotpot(
        hotpot_gold(), {"answer": {"q1": "red fox"}, "sp": {"q1": [["A", 0], ["C", 0]]}}
    )
    assert result["metrics"]["f1"] == 0.4
    assert result["metrics"]["sp_f1"] == 1 / 3
    assert result["per_case"]["q1"]["joint_prec"] == 0.5
    assert result["per_case"]["q1"]["joint_recall"] == 2 / 3
    assert result["metrics"]["joint_f1"] == pytest.approx(2 / 7)
    assert result["count"] == 2
    assert result["answer_coverage"] == 0.5
    assert result["joint_coverage"] == 0.5
    assert result["missing_answers"] == ["q2"]
    assert set(result["per_case"]["q2"].values()) == {0.0}


def test_hotpot_joint_requires_both_answer_and_support():
    result = evaluate_hotpot(
        hotpot_gold(), {"answer": {"q1": "red fox jumps"}, "sp": {"q2": [["B", 1]]}}
    )
    assert result["metrics"]["em"] == 0.5
    assert result["metrics"]["sp_em"] == 0.5
    assert result["metrics"]["joint_em"] == 0
    assert result["joint_coverage"] == 0


@pytest.mark.parametrize(
    "gold,predictions",
    [
        ([], {"answer": {}, "sp": {}}),
        (hotpot_gold() * 2, {"answer": {}, "sp": {}}),
        (hotpot_gold(), {"answer": {"extra": "x"}, "sp": {}}),
        (hotpot_gold(), {"answer": {}, "sp": {"extra": []}}),
        (hotpot_gold(), {"answer": {"q1": None}, "sp": {}}),
        (hotpot_gold(), {"answer": {}, "sp": {"q1": [["A", True]]}}),
        (hotpot_gold(), {"answer": {}, "sp": {"q1": [["A", -1]]}}),
        (hotpot_gold(), {"answer": {}, "sp": {"q1": [["A", 1.0]]}}),
        (hotpot_gold(), {"answer": {}, "sp": {"q1": [["A"]]}}),
        (hotpot_gold(), {"answer": {}, "sp": {"q1": "A"}}),
        (hotpot_gold(), {"answer": {}, "sp": {}, "extra": {}}),
        (hotpot_gold(), {"answer": [], "sp": {}}),
        ([{"_id": "q", "answer": "x", "supporting_facts": None}], {"answer": {}, "sp": {}}),
    ],
)
def test_hotpot_rejects_malformed_inputs(gold, predictions):
    with pytest.raises(ValueError):
        evaluate_hotpot(gold, predictions)


def test_hotpot_maximum_case_bound():
    gold = [{"_id": str(i), "answer": "a", "supporting_facts": []} for i in range(1001)]
    with pytest.raises(ValueError):
        evaluate_hotpot(gold, {"answer": {}, "sp": {}})
