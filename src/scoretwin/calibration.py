"""Analyze a blinded human identity study for IDSR metric calibration.

The evaluator does not invent ratings or treat synthetic tests as human calibration.
It computes ordinal inter-rater agreement, repeated-item consistency, metric/rating
rank correlations and page/source-cluster bootstrap intervals from supplied ratings.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[cursor]:
            stop += 1
        # Statistical ranks are 1-based; ties receive their average rank.
        average_rank = ((cursor + 1) + stop) / 2.0
        ranks[order[cursor:stop]] = average_rank
        cursor = stop
    return ranks


def _spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float | None:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if len(xx) != len(yy) or len(xx) < 3:
        return None
    rank_x, rank_y = _average_ranks(xx), _average_ranks(yy)
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _quadratic_weighted_kappa(
    ratings_a: list[int],
    ratings_b: list[int],
    rating_min: int,
    rating_max: int,
) -> float | None:
    if len(ratings_a) != len(ratings_b) or len(ratings_a) < 2:
        return None
    categories = np.arange(rating_min, rating_max + 1, dtype=np.float64)
    span = max(1.0, float(rating_max - rating_min))
    squared_distance = ((categories[:, None] - categories[None, :]) / span) ** 2
    a = np.asarray(ratings_a, dtype=np.int64) - rating_min
    b = np.asarray(ratings_b, dtype=np.int64) - rating_min
    if np.any(a < 0) or np.any(b < 0) or np.any(a >= len(categories)) or np.any(b >= len(categories)):
        raise ValueError("Ratings fall outside the configured scale.")
    observed = float(np.mean(squared_distance[a, b]))
    proportions_a = np.bincount(a, minlength=len(categories)).astype(np.float64) / len(a)
    proportions_b = np.bincount(b, minlength=len(categories)).astype(np.float64) / len(b)
    expected = float(np.sum(proportions_a[:, None] * proportions_b[None, :] * squared_distance))
    if expected <= 1e-12:
        return None
    return float(1.0 - observed / expected)


def _correlation_bootstrap(
    rows: list[dict],
    metric_name: str,
    direction: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    usable = []
    for row in rows:
        metric_value = row["metrics"].get(metric_name)
        rating = row["mean_rating"]
        if metric_value is None or rating is None:
            continue
        metric_value = float(metric_value)
        rating = float(rating)
        if not math.isfinite(metric_value) or not math.isfinite(rating):
            continue
        quality_value = metric_value if direction == "higher" else -metric_value
        usable.append({
            "pair_id": row["pair_id"],
            "independence_group": row["independence_group"],
            "metric": quality_value,
            "rating": rating,
        })

    observed = _spearman(
        [row["metric"] for row in usable],
        [row["rating"] for row in usable],
    )
    if len(usable) < 3:
        return {
            "n_pairs": len(usable),
            "spearman_rho": observed,
            "bootstrap_95_ci": None,
            "bootstrap_valid": 0,
            "reason": "At least three usable pairs are required.",
        }

    grouped: dict[str, list[dict]] = {}
    for row in usable:
        grouped.setdefault(row["independence_group"], []).append(row)
    group_keys = sorted(grouped)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        selected_groups = rng.integers(0, len(group_keys), size=len(group_keys))
        sampled = []
        for selected_index in selected_groups:
            sampled.extend(grouped[group_keys[int(selected_index)]])
        rho = _spearman(
            [row["metric"] for row in sampled],
            [row["rating"] for row in sampled],
        )
        if rho is not None and math.isfinite(rho):
            estimates.append(rho)

    interval = None
    if estimates:
        interval = {
            "lower_95": float(np.quantile(estimates, 0.025)),
            "upper_95": float(np.quantile(estimates, 0.975)),
            "bootstrap_samples_requested": bootstrap_samples,
            "bootstrap_samples_valid": len(estimates),
            "bootstrap_seed": seed,
            "resampling_unit": "independence_group",
            "independence_group_count": len(group_keys),
        }
    return {
        "n_pairs": len(usable),
        "independence_group_count": len(group_keys),
        "direction": direction,
        "spearman_rho": observed,
        "bootstrap_95_ci": interval,
        "bootstrap_valid": len(estimates),
        "interpretation": "Higher adjusted metric values should associate with higher human identity ratings.",
    }


def _repeat_consistency(pairs: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for pair in pairs:
        repeat_group = pair.get("repeat_group")
        if repeat_group is not None:
            grouped.setdefault(str(repeat_group), []).append(pair)

    differences = []
    exact_pairs = 0
    total_pairs = 0
    by_rater: dict[str, list[tuple[int, int]]] = {}
    for repeat_group, group in grouped.items():
        if len(group) < 2:
            continue
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left_ratings = group[left_index].get("ratings", {})
                right_ratings = group[right_index].get("ratings", {})
                for rater in set(left_ratings) & set(right_ratings):
                    left = int(left_ratings[rater])
                    right = int(right_ratings[rater])
                    differences.append(abs(left - right))
                    exact_pairs += int(left == right)
                    total_pairs += 1
                    by_rater.setdefault(str(rater), []).append((left, right))

    return {
        "repeat_group_count": len(grouped),
        "within_rater_repeat_comparisons": total_pairs,
        "exact_repeat_agreement_rate": float(exact_pairs / total_pairs) if total_pairs else None,
        "mean_absolute_repeat_rating_difference": float(np.mean(differences)) if differences else None,
        "repeats_by_rater": {
            rater: {
                "comparison_count": len(values),
                "mean_absolute_difference": float(np.mean([abs(a - b) for a, b in values])),
                "exact_agreement_rate": float(np.mean([a == b for a, b in values])),
            }
            for rater, values in sorted(by_rater.items())
        },
        "caveat": "Repeat consistency is only computed when the manifest includes repeat_group labels for the same blinded comparison.",
    }


def _rater_agreement(pairs: list[dict], rating_min: int, rating_max: int) -> dict:
    raters = sorted({
        str(rater)
        for pair in pairs
        for rater in pair.get("ratings", {})
    })
    results = []
    for left_index, left_rater in enumerate(raters):
        for right_rater in raters[left_index + 1:]:
            left_ratings, right_ratings = [], []
            for pair in pairs:
                ratings = pair.get("ratings", {})
                if left_rater in ratings and right_rater in ratings:
                    left_ratings.append(int(ratings[left_rater]))
                    right_ratings.append(int(ratings[right_rater]))
            kappa = _quadratic_weighted_kappa(left_ratings, right_ratings, rating_min, rating_max)
            results.append({
                "rater_a": left_rater,
                "rater_b": right_rater,
                "overlapping_pairs": len(left_ratings),
                "quadratic_weighted_kappa": kappa,
            })
    valid = [item["quadratic_weighted_kappa"] for item in results if item["quadratic_weighted_kappa"] is not None]
    return {
        "rater_count": len(raters),
        "rater_pair_count": len(results),
        "mean_pairwise_quadratic_weighted_kappa": float(np.mean(valid)) if valid else None,
        "pairwise": results,
        "interpretation": "Kappa is undefined when expected disagreement is zero; all pairwise values and overlap counts are reported.",
    }


def recommended_study_design(target_rho: float = 0.4, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Approximate independent-pair N via Fisher-z; a planning heuristic, not a guarantee."""
    if not 0 < target_rho < 1 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("target_rho, alpha and power must be within (0,1).")
    # Standard normal approximations for two-sided alpha and desired power.
    z_alpha = 1.959963984540054 if abs(alpha - 0.05) < 1e-10 else 1.959963984540054
    z_power = 0.8416212335729143 if abs(power - 0.8) < 1e-10 else 0.8416212335729143
    required = int(math.ceil(((z_alpha + z_power) / math.atanh(target_rho)) ** 2 + 3))
    return {
        "approximate_independent_pairs_for_target_spearman_rho": required,
        "target_abs_correlation": target_rho,
        "two_sided_alpha": alpha,
        "target_power": power,
        "planned_unique_pairs_with_20_percent_exclusion_margin": int(math.ceil(required / 0.8)),
        "recommended_raters": 5,
        "recommended_hidden_repeat_comparisons_per_rater": 10,
        "design": "balanced set of easy/ambiguous/adversarial pages; randomized blind order; independent source-piece groups",
        "status": "planning heuristic based on a Fisher-z approximation; run a pilot or simulation before formal preregistration",
    }


def evaluate_human_calibration(
    study: dict,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20261029,
) -> dict:
    """Evaluate metric/rating association and inter-rater agreement from a blinded study."""
    if study.get("schema") != "scoretwin-human-calibration-v1":
        raise ValueError("Study schema must be scoretwin-human-calibration-v1.")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    scale = study.get("rating_scale", {})
    rating_min = int(scale.get("min", 1))
    rating_max = int(scale.get("max", 7))
    if rating_max <= rating_min:
        raise ValueError("rating_scale.max must be greater than rating_scale.min.")
    directions = study.get("metric_directions", {})
    if not isinstance(directions, dict) or not directions:
        raise ValueError("metric_directions must map metric names to 'higher' or 'lower'.")
    if any(direction not in {"higher", "lower"} for direction in directions.values()):
        raise ValueError("Metric direction values must be 'higher' or 'lower'.")

    pairs = study.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Study must contain a non-empty pairs list.")
    seen_ids: set[str] = set()
    normalized = []
    all_raters = set()
    rating_count = 0
    for index, pair in enumerate(pairs):
        pair_id = str(pair.get("id", ""))
        if not pair_id or pair_id in seen_ids:
            raise ValueError(f"Pair index {index} has a missing or duplicate id.")
        seen_ids.add(pair_id)
        ratings = pair.get("ratings", {})
        metrics = pair.get("metrics", {})
        if not isinstance(ratings, dict) or not ratings:
            raise ValueError(f"Pair {pair_id} must have at least one rater score.")
        for rater, rating in ratings.items():
            if isinstance(rating, bool) or int(rating) != rating or not rating_min <= int(rating) <= rating_max:
                raise ValueError(f"Pair {pair_id} has a rating outside {rating_min}..{rating_max}.")
            all_raters.add(str(rater))
            rating_count += 1
        row_metrics = {}
        for metric_name in directions:
            value = metrics.get(metric_name)
            if value is None:
                row_metrics[metric_name] = None
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Pair {pair_id}, metric {metric_name} must be finite.")
            row_metrics[metric_name] = number
        normalized.append({
            "pair_id": pair_id,
            "independence_group": str(pair.get("independence_group", pair_id)),
            "repeat_group": pair.get("repeat_group"),
            "ratings": {str(key): int(value) for key, value in ratings.items()},
            "metrics": row_metrics,
            "mean_rating": float(np.mean(list(ratings.values()))),
            "rater_count": len(ratings),
        })

    correlation_results = {
        metric: _correlation_bootstrap(
            normalized, metric, direction, bootstrap_samples, bootstrap_seed + index
        )
        for index, (metric, direction) in enumerate(sorted(directions.items()))
    }
    kappa = _rater_agreement(normalized, rating_min, rating_max)
    design = recommended_study_design()
    independent_groups = len({row["independence_group"] for row in normalized})
    return {
        "schema": "scoretwin-human-calibration-report-0.1",
        "data_status": "human ratings supplied to this evaluator; verify blinding, randomization and source independence from study metadata",
        "pair_count": len(normalized),
        "independence_group_count": independent_groups,
        "rater_count": len(all_raters),
        "total_rating_count": rating_count,
        "rating_scale": {"min": rating_min, "max": rating_max, "anchors": scale.get("anchors", {})},
        "metric_directions": directions,
        "metric_human_associations": correlation_results,
        "inter_rater_agreement": kappa,
        "within_rater_repeat_consistency": _repeat_consistency(normalized),
        "study_design_metadata": study.get("study_design", {}),
        "recommended_design": design,
        "scalar_did_weights": None,
        "scalar_did_status": "not_fit: correlation is evidence of association, not an approved DID aggregation; preregister model form and validate out of source group",
        "limitations": [
            "No human ratings are fabricated; results only apply to pairs present in the supplied study.",
            "Bootstrap resamples independence_group to reduce leakage when multiple views come from one source piece.",
            "The Fisher-z planning N is approximate for Spearman correlation and should be checked by simulation before preregistration.",
            "A high correlation on one dataset does not establish calibration under distribution shift.",
        ],
    }
