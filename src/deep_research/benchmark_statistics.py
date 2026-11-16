"""Paired case-level descriptive comparison, not a population/significance claim."""

import math
from collections.abc import Mapping

import numpy as np

MAX_CASES = 1000
MAX_RESAMPLES = 10000


def _validated_scores(value: Mapping[str, float | None]) -> dict[str, float | None]:
    if not isinstance(value, Mapping) or len(value) > MAX_CASES:
        raise ValueError("scores must be a mapping of at most 1000 cases")
    result = {}
    for key, score in value.items():
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ValueError("case ids must be nonempty bounded strings")
        if score is None:
            result[key] = None
        elif type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("scores must be finite numbers or null, never booleans")
        elif abs(score) > 1e12:
            raise ValueError("score magnitude exceeds the numerical safety bound")
        else:
            result[key] = float(score)
    return result


def compare_paired(
    left: Mapping[str, float | None],
    right: Mapping[str, float | None],
    *,
    seed: int = 42,
    n_resamples: int = 2000,
) -> dict:
    """Bootstrap paired right-minus-left differences; pairs are sorted by ID.

    Caller must average repetitions within each case FIRST. We resample cases,
    never runs, so repeat-heavy cases do not become artificial independent data.
    Effects use paired Cohen's dz (sample SD of differences), not pooled d.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    if type(n_resamples) is not int or not 1 <= n_resamples <= MAX_RESAMPLES:
        raise ValueError("n_resamples must be an integer in [1, 10000]")
    left_scores, right_scores = _validated_scores(left), _validated_scores(right)
    all_ids = set(left_scores) | set(right_scores)
    if len(all_ids) > MAX_CASES:
        raise ValueError("comparison contains more than 1000 unique cases")
    paired_ids = sorted(
        key
        for key in all_ids
        if left_scores.get(key) is not None and right_scores.get(key) is not None
    )
    differences = np.array([right_scores[key] - left_scores[key] for key in paired_ids])
    size = len(paired_ids)
    mean_delta = float(np.mean(differences)) if size else None
    ci = None
    effect = None
    reason = "fewer_than_two_pairs" if size < 2 else None
    if size >= 2:
        rng = np.random.default_rng(seed)
        means = np.empty(n_resamples, dtype=float)
        # At most 128 x 1000 temporary indices; deterministic across dict ordering.
        for start in range(0, n_resamples, 128):
            stop = min(start + 128, n_resamples)
            indices = rng.integers(0, size, size=(stop - start, size))
            means[start:stop] = np.mean(differences[indices], axis=1)
        ci = np.quantile(means, [0.025, 0.975], method="linear").tolist()
        deviation = float(np.std(differences, ddof=1))
        if deviation == 0.0:
            reason = "zero_difference_variance"
        else:
            effect = mean_delta / deviation
            if not math.isfinite(effect):
                effect = None
                reason = "nonfinite_effect"
    return {
        "mean_delta": mean_delta,
        "ci95": ci,
        "cohen_dz": effect,
        "effect_reason": reason,
        "pairs_used": size,
        "pairs_total": len(all_ids),
        "pair_coverage": size / len(all_ids) if all_ids else None,
        "paired_ids": paired_ids,
        "missing_left": sorted(set(right_scores) - set(left_scores)),
        "missing_right": sorted(set(left_scores) - set(right_scores)),
        "unscored_ids": sorted(all_ids - set(paired_ids)),
        "seed": seed,
        "n_resamples": n_resamples,
        "notes": [
            "Delta is right minus left; positive does not always mean better (e.g. latency).",
            "Percentile 95% interval resamples paired cases; repetitions must first be averaged.",
            "A small curated benchmark interval is not proof about a wider population.",
            "Exploratory comparisons are not multiplicity-adjusted significance tests.",
        ],
    }
