"""Paired resampling unit tests with known arithmetic and explicit missingness."""

import math

import pytest

from deep_research.benchmark_statistics import compare_paired


def test_paired_difference_and_cohen_dz_hand_calculated():
    result = compare_paired({"a": 0, "b": 10, "c": 20}, {"a": 1, "b": 12, "c": 23})
    assert result["mean_delta"] == 2
    assert result["cohen_dz"] == 2  # Differences 1,2,3 have sample SD = 1.
    assert result["effect_reason"] is None
    assert result["pairs_used"] == result["pairs_total"] == 3
    assert result["pair_coverage"] == 1
    assert 1 <= result["ci95"][0] <= 2 <= result["ci95"][1] <= 3


def test_pairing_not_independent_group_subtraction():
    result = compare_paired({"a": 1, "b": 2, "c": 100}, {"a": 3, "b": 4})
    assert result["mean_delta"] == 2
    assert result["missing_right"] == ["c"]
    assert result["unscored_ids"] == ["c"]
    assert result["pair_coverage"] == 2 / 3


def test_missing_nulls_reported_without_becoming_zero():
    result = compare_paired({"a": None, "b": 1, "left": 9}, {"a": 5, "b": 2, "right": 7})
    assert result["mean_delta"] == 1
    assert result["pairs_used"] == 1
    assert result["pairs_total"] == 4
    assert result["missing_left"] == ["right"]
    assert result["missing_right"] == ["left"]
    assert result["unscored_ids"] == ["a", "left", "right"]


@pytest.mark.parametrize("left,right", [({}, {}), ({"a": None}, {"a": 1}), ({"a": 1}, {"b": 1})])
def test_no_pairs_no_invented_interval_or_effect(left, right):
    result = compare_paired(left, right)
    assert result["mean_delta"] is None
    assert result["ci95"] is None
    assert result["cohen_dz"] is None
    assert result["effect_reason"] == "fewer_than_two_pairs"


def test_single_pair_has_delta_but_no_bootstrap_inference():
    result = compare_paired({"a": 1}, {"a": 2})
    assert result["mean_delta"] == 1
    assert result["ci95"] is None
    assert result["cohen_dz"] is None


@pytest.mark.parametrize("difference", [0, 1, -2])
def test_zero_variance_effect_not_infinite_or_zero_claim(difference):
    result = compare_paired({"a": 0, "b": 0}, {"a": difference, "b": difference})
    assert result["mean_delta"] == difference
    assert result["ci95"] == [difference, difference]
    assert result["cohen_dz"] is None
    assert result["effect_reason"] == "zero_difference_variance"


def test_seed_and_order_reproducibility():
    left = {"d": 1, "a": 2, "b": 3, "c": 4}
    right = {"d": 3, "a": 8, "b": 4, "c": 10}
    result = compare_paired(left, right, seed=17, n_resamples=337)
    assert result == compare_paired(
        dict(reversed(left.items())), dict(reversed(right.items())), seed=17, n_resamples=337
    )
    assert result["seed"] == 17
    assert result["n_resamples"] == 337
    assert result["paired_ids"] == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "bad", [True, "1", float("nan"), float("inf"), -float("inf"), [], {}, 10**13]
)
def test_invalid_scores_are_rejected_on_either_side(bad):
    with pytest.raises(ValueError):
        compare_paired({"a": bad}, {"a": 1})
    with pytest.raises(ValueError):
        compare_paired({"a": 1}, {"a": bad})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 2**32},
        {"seed": 1.0},
        {"n_resamples": 0},
        {"n_resamples": True},
        {"n_resamples": 10001},
        {"n_resamples": 2.0},
    ],
)
def test_rng_arguments_bounded_strict(kwargs):
    with pytest.raises(ValueError):
        compare_paired({"a": 0}, {"a": 1}, **kwargs)


@pytest.mark.parametrize("left", [[], {"": 1}, {True: 1}, {"x" * 201: 1}])
def test_mapping_and_ids_strict(left):
    with pytest.raises(ValueError):
        compare_paired(left, {})


def test_maximum_union_not_only_each_side():
    with pytest.raises(ValueError):
        compare_paired({str(i): 1 for i in range(501)}, {str(i): 1 for i in range(501, 1002)})


def test_chunked_maximum_has_finite_results():
    left = {str(i): 0 for i in range(1000)}
    right = {str(i): i % 2 for i in range(1000)}
    result = compare_paired(left, right, n_resamples=10000)
    assert result["mean_delta"] == 0.5
    assert math.isfinite(result["cohen_dz"])
    assert all(math.isfinite(value) for value in result["ci95"])


def test_limitations_and_no_significance_or_p_values():
    result = compare_paired({"a": 0, "b": 0}, {"a": 1, "b": 2})
    assert "repetitions must first be averaged" in result["notes"][1]
    assert "not proof" in result["notes"][2]
    assert "not multiplicity-adjusted" in result["notes"][3]
    assert "p_value" not in result
    assert "significant" not in result
