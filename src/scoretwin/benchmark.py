"""Reproducible, page-level synthetic notehead benchmark with uncertainty."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import statistics
import sys
import time
from typing import Sequence

import cv2
import numpy as np
import PIL

from . import __version__
from .imaging import detect_staff_systems
from .pipeline import analyze_array
from .proposal import get_default_model, get_default_model_metrics
from .synthetic import generate_synthetic_page


def _major_minor(version: str) -> tuple[int, int]:
    numbers = re.findall(r"\d+", str(version))
    return (int(numbers[0]), int(numbers[1]) if len(numbers) > 1 else 0) if numbers else (0, 0)


def _match_notes(truth: list[dict], predicted: list[dict], tolerance_px: float) -> list[tuple[int, int, float]]:
    candidates = []
    for truth_index, gt in enumerate(truth):
        for predicted_index, pred in enumerate(predicted):
            if int(gt["staff_index"]) != int(pred["staff_index"]):
                continue
            distance = float(np.hypot(
                float(gt["center"][0]) - float(pred["center"][0]),
                float(gt["center"][1]) - float(pred["center"][1]),
            ))
            if distance <= tolerance_px:
                candidates.append((distance, truth_index, predicted_index))
    candidates.sort()
    used_truth: set[int] = set()
    used_predicted: set[int] = set()
    matches = []
    for distance, truth_index, predicted_index in candidates:
        if truth_index in used_truth or predicted_index in used_predicted:
            continue
        used_truth.add(truth_index)
        used_predicted.add(predicted_index)
        matches.append((truth_index, predicted_index, distance))
    return matches


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(quantile * len(ordered)) - 1)))
    return ordered[index]


def _bootstrap_mean_ci(values: Sequence[float], seed: int, rounds: int = 2000, confidence_level: float = 0.95) -> list[float] | None:
    clean = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if clean.size == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(rounds, clean.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return [float(low), float(high)]


def _wilson_ci(successes: int, total: int, confidence_level: float = 0.95) -> list[float] | None:
    if total <= 0:
        return None
    z = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def evaluate_synthetic(
    pages: int = 10,
    seed: int = 20261029,
    staff_count: int = 2,
    notes_per_staff: int = 9,
    noise_sigma: float = 0.0,
    blur_sigma: float = 0.0,
    contract_staff_spaces: float = 0.3,
    bootstrap_rounds: int = 2000,
) -> dict:
    """Evaluate notehead detection on seeded, registered synthetic pages.

    Matching uses the requested fraction of detected staff spacing. Geometry
    error is normalized by staff spacing; the page p95 is conditional on matched
    noteheads, while precision/recall account for misses and extra proposals.
    The generated pages receive only blur and additive noise, so this benchmark
    does not measure perspective, color, ISP, page curvature or real manuscripts.
    """
    if pages < 1:
        raise ValueError("pages must be at least 1.")
    if staff_count < 1 or notes_per_staff < 1:
        raise ValueError("staff_count and notes_per_staff must be positive.")
    if contract_staff_spaces <= 0:
        raise ValueError("contract_staff_spaces must be positive.")
    if bootstrap_rounds < 100:
        raise ValueError("bootstrap_rounds must be at least 100.")

    setup_started = time.perf_counter()
    model = get_default_model()
    model_setup_seconds = time.perf_counter() - setup_started
    model_metrics = get_default_model_metrics()
    model_hash = hashlib.sha256(np.asarray(model["weights"], dtype=np.float32).tobytes()).hexdigest()
    total_truth = total_predicted = total_true_positive = total_pitch_correct = 0
    total_geometry_error_px = total_geometry_error_sp = total_runtime = 0.0
    per_page = []

    for page_index in range(pages):
        page_seed = seed + page_index
        gray, truth = generate_synthetic_page(
            seed=page_seed,
            staff_count=staff_count,
            notes_per_staff=notes_per_staff,
            noise_sigma=noise_sigma,
            blur_sigma=blur_sigma,
        )
        detected_staffs, _binary, _staff_mask = detect_staff_systems(gray)
        spacings = [float(staff.spacing) for staff in detected_staffs]
        mean_spacing = float(np.median(spacings)) if spacings else 16.0
        scene = analyze_array(gray, model=model)
        predicted = scene["notes"]
        tolerance_px = contract_staff_spaces * mean_spacing
        matches = _match_notes(truth, predicted, tolerance_px=tolerance_px)
        pitch_correct = 0
        geometry_errors_px = []
        geometry_errors_sp = []
        for truth_index, predicted_index, distance_px in matches:
            gt, pred = truth[truth_index], predicted[predicted_index]
            geometry_errors_px.append(distance_px)
            staff_index = int(gt["staff_index"])
            staff_space = spacings[staff_index] if staff_index < len(spacings) else mean_spacing
            geometry_errors_sp.append(distance_px / max(staff_space, 1e-6))
            if gt["pitch"] == pred["pitch"]:
                pitch_correct += 1

        true_positive = len(matches)
        false_positive = len(predicted) - true_positive
        false_negative = len(truth) - true_positive
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(truth) if truth else 0.0
        page_f1 = _f1(precision, recall)
        mean_error_px = statistics.mean(geometry_errors_px) if geometry_errors_px else None
        mean_error_sp = statistics.mean(geometry_errors_sp) if geometry_errors_sp else None
        p95_error_sp = _percentile(geometry_errors_sp, 0.95)
        localization_pass = p95_error_sp is not None and p95_error_sp <= contract_staff_spaces
        combined_pass = localization_pass and page_f1 >= 0.95

        total_truth += len(truth)
        total_predicted += len(predicted)
        total_true_positive += true_positive
        total_pitch_correct += pitch_correct
        total_geometry_error_px += sum(geometry_errors_px)
        total_geometry_error_sp += sum(geometry_errors_sp)
        total_runtime += float(scene["runtime_seconds"])
        per_page.append({
            "seed": page_seed,
            "ground_truth_notes": len(truth),
            "proposed_notes": len(predicted),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": page_f1,
            "pitch_correct_on_matches": pitch_correct,
            "pitch_accuracy_on_matches": pitch_correct / true_positive if true_positive else None,
            "mean_center_error_px": mean_error_px,
            "mean_center_error_staff_spaces": mean_error_sp,
            "p95_center_error_staff_spaces": p95_error_sp,
            "localization_contract_pass": localization_pass,
            "combined_f1_and_localization_pass": combined_pass,
            "staffs_detected": len(scene["staffs"]),
            "staff_spacing_px": mean_spacing,
            "runtime_seconds": scene["runtime_seconds"],
        })

    precision = total_true_positive / total_predicted if total_predicted else 0.0
    recall = total_true_positive / total_truth if total_truth else 0.0
    macro_f1 = statistics.mean(page["f1"] for page in per_page)
    page_mean_errors = [page["mean_center_error_staff_spaces"] for page in per_page if page["mean_center_error_staff_spaces"] is not None]
    page_p95_errors = [page["p95_center_error_staff_spaces"] for page in per_page if page["p95_center_error_staff_spaces"] is not None]
    page_pitch_accuracy = [page["pitch_accuracy_on_matches"] for page in per_page if page["pitch_accuracy_on_matches"] is not None]
    localization_passes = sum(bool(page["localization_contract_pass"]) for page in per_page)
    combined_passes = sum(bool(page["combined_f1_and_localization_pass"]) for page in per_page)

    numpy_version = _major_minor(np.__version__)
    opencv_version = _major_minor(cv2.__version__)
    pillow_version = _major_minor(PIL.__version__)
    python_supported = sys.version_info >= (3, 11)
    dependencies_supported = (
        (1, 26) <= numpy_version < (3, 0)
        and (4, 9) <= opencv_version < (5, 0)
        and pillow_version >= (10, 0)
    )

    return {
        "benchmark": "synthetic-notehead-proposal-v2",
        "epistemic_status": "MĂSURAT — procedurally generated, geometry-registered pages only",
        "data_status": "synthetic_only_not_a_real_photo_benchmark",
        "protocol": {
            "seed_start": seed,
            "seed_end": seed + pages - 1,
            "pages": pages,
            "staff_count_per_page": staff_count,
            "notes_per_staff": notes_per_staff,
            "capture_noise_sigma_grayscale": float(noise_sigma),
            "capture_blur_sigma_pixels": float(blur_sigma),
            "registered_geometry": True,
            "matching_tolerance_staff_spaces": float(contract_staff_spaces),
            "p95_error_is_conditional_on_matched_noteheads": True,
            "bootstrap_unit": "page",
            "bootstrap_rounds": bootstrap_rounds,
            "confidence_level": 0.95,
            "bootstrap_seed": seed + 91_019,
        },
        "proposal_model": {
            "sha256_weights": model_hash,
            "training": model_metrics,
            "training_pages": "not applicable; patch model uses procedural crops",
        },
        "measurements": {
            "n_ground_truth_noteheads": total_truth,
            "n_proposed_noteheads": total_predicted,
            "true_positive": total_true_positive,
            "false_positive": total_predicted - total_true_positive,
            "false_negative": total_truth - total_true_positive,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": _f1(precision, recall),
            "macro_page_f1": macro_f1,
            "macro_page_f1_ci95_bootstrap": _bootstrap_mean_ci([page["f1"] for page in per_page], seed + 91_019, bootstrap_rounds, 0.95),
            "pitch_accuracy_on_matched_notes": total_pitch_correct / total_true_positive if total_true_positive else None,
            "mean_center_error_px_on_matches": total_geometry_error_px / total_true_positive if total_true_positive else None,
            "mean_center_error_staff_spaces_on_matches": total_geometry_error_sp / total_true_positive if total_true_positive else None,
            "mean_page_center_error_staff_spaces_ci95_bootstrap": _bootstrap_mean_ci(page_mean_errors, seed + 91_020, bootstrap_rounds, 0.95),
            "mean_page_p95_center_error_staff_spaces": statistics.mean(page_p95_errors) if page_p95_errors else None,
            "mean_page_p95_center_error_staff_spaces_ci95_bootstrap": _bootstrap_mean_ci(page_p95_errors, seed + 91_021, bootstrap_rounds, 0.95),
            "pages_p95_at_or_below_contract": localization_passes,
            "page_localization_pass_rate_ci95_wilson": _wilson_ci(localization_passes, pages, 0.95),
            "pages_f1_ge_0_95_and_p95_at_or_below_contract": combined_passes,
            "combined_pass_rate_ci95_wilson": _wilson_ci(combined_passes, pages, 0.95),
            "pitch_accuracy_ci95_bootstrap": _bootstrap_mean_ci(page_pitch_accuracy, seed + 91_022, bootstrap_rounds, 0.95),
            "mean_runtime_seconds_per_page": total_runtime / pages,
            "model_setup_seconds_outside_page_runtime": model_setup_seconds,
        },
        "runtime_environment": {
            "python": sys.version,
            "project_requires_python": ">=3.11",
            "runtime_meets_project_python_requirement": python_supported,
            "runtime_meets_project_dependency_requirements": dependencies_supported,
            "runtime_meets_all_project_requirements": python_supported and dependencies_supported,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
        },
        "software": {
            "scoretwin": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "proposal_model_training_metrics": model_metrics,
        "per_page": per_page,
    }
