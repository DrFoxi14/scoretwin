"""Reproducible held-out synthetic regression benchmark."""

from __future__ import annotations

from itertools import product

import numpy as np

from .imaging import detect_staff_systems
from .pipeline import analyze_array
from .proposal import get_default_model
from .synthetic import generate_synthetic_page


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


def evaluate_synthetic(
    pages: int = 10,
    seed: int = 20261029,
    staff_count: int = 2,
    notes_per_staff: int = 9,
) -> dict:
    """Evaluate notehead proposals on generated pages with independent page seeds."""
    if pages < 1:
        raise ValueError("pages must be at least 1.")
    model = get_default_model()
    total_truth = 0
    total_predicted = 0
    total_true_positive = 0
    total_pitch_correct = 0
    total_geometry_error = 0.0
    total_runtime = 0.0
    per_page = []

    for page_index in range(pages):
        page_seed = seed + page_index
        gray, truth = generate_synthetic_page(
            seed=page_seed,
            staff_count=staff_count,
            notes_per_staff=notes_per_staff,
        )
        detected_staffs, _binary, _staff_mask = detect_staff_systems(gray)
        mean_spacing = float(np.median([staff.spacing for staff in detected_staffs])) if detected_staffs else 16.0
        scene = analyze_array(gray, model=model)
        predicted = scene["notes"]
        matches = _match_notes(truth, predicted, tolerance_px=0.55 * mean_spacing)
        pitch_correct = 0
        geometry_errors = []
        for truth_index, predicted_index, distance in matches:
            gt, pred = truth[truth_index], predicted[predicted_index]
            geometry_errors.append(distance)
            if gt["pitch"] == pred["pitch"]:
                pitch_correct += 1

        tp = len(matches)
        total_truth += len(truth)
        total_predicted += len(predicted)
        total_true_positive += tp
        total_pitch_correct += pitch_correct
        total_geometry_error += sum(geometry_errors)
        total_runtime += float(scene["runtime_seconds"])
        per_page.append({
            "seed": page_seed,
            "ground_truth_notes": len(truth),
            "proposed_notes": len(predicted),
            "true_positive": tp,
            "false_positive": len(predicted) - tp,
            "false_negative": len(truth) - tp,
            "pitch_correct_on_matches": pitch_correct,
            "mean_center_error_px": float(np.mean(geometry_errors)) if geometry_errors else None,
            "staffs_detected": len(scene["staffs"]),
            "runtime_seconds": scene["runtime_seconds"],
        })

    precision = total_true_positive / total_predicted if total_predicted else 0.0
    recall = total_true_positive / total_truth if total_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "benchmark": "synthetic-notehead-proposal-v1",
        "data_status": "synthetic_only_not_a_real_photo_benchmark",
        "seed_start": seed,
        "pages": pages,
        "staff_count_per_page": staff_count,
        "notes_per_staff": notes_per_staff,
        "n_ground_truth": total_truth,
        "n_proposed": total_predicted,
        "true_positive": total_true_positive,
        "false_positive": total_predicted - total_true_positive,
        "false_negative": total_truth - total_true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pitch_accuracy_on_matched_notes": total_pitch_correct / total_true_positive if total_true_positive else 0.0,
        "mean_center_error_px": total_geometry_error / total_true_positive if total_true_positive else None,
        "mean_runtime_seconds": total_runtime / pages,
        "per_page": per_page,
    }
