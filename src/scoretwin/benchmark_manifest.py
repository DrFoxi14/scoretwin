"""Evaluate externally annotated score pages with page-level bootstrap intervals."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .benchmark import _match_notes
from .imaging import load_grayscale
from .pipeline import analyze_array
from .proposal import load_model


def _scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _bootstrap_intervals(
    pages: list[dict],
    samples: int = 1000,
    seed: int = 20261029,
) -> dict:
    if samples < 100:
        raise ValueError("bootstrap_samples must be at least 100.")
    rng = np.random.default_rng(seed)
    names = ("precision", "recall", "f1", "mean_center_error_sp", "p95_center_error_sp")
    values = {name: [] for name in names}
    for _ in range(samples):
        selected = [pages[int(i)] for i in rng.integers(0, len(pages), size=len(pages))]
        tp = sum(int(page["true_positive"]) for page in selected)
        fp = sum(int(page["false_positive"]) for page in selected)
        fn = sum(int(page["false_negative"]) for page in selected)
        precision, recall, f1 = _scores(tp, fp, fn)
        values["precision"].append(precision)
        values["recall"].append(recall)
        values["f1"].append(f1)
        error_values = [
            float(value)
            for page in selected
            for value in page.get("center_error_sp_values", [])
        ]
        values["mean_center_error_sp"].append(
            float(np.mean(error_values)) if error_values else None
        )
        values["p95_center_error_sp"].append(
            float(np.percentile(error_values, 95)) if error_values else None
        )

    intervals = {}
    for name, metric_values in values.items():
        valid = np.asarray(
            [value for value in metric_values if value is not None and np.isfinite(value)],
            dtype=float,
        )
        intervals[name] = {
            "lower_95": float(np.quantile(valid, 0.025)) if len(valid) else None,
            "upper_95": float(np.quantile(valid, 0.975)) if len(valid) else None,
            "bootstrap_samples_requested": samples,
            "bootstrap_samples_valid": int(len(valid)),
            "bootstrap_seed": seed,
            "resampling_unit": "page",
        }
    return intervals


def evaluate_manifest(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
    tolerance_staff_spaces: float = 0.35,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20261029,
) -> dict:
    """Evaluate notehead proposals on a manifest of annotated pages.

    Manifest schema:
      {
        "schema": "scoretwin-annotated-pages-v1",
        "pages": [{
          "id": "page-001",
          "image": "images/page-001.jpg",
          "sha256": "optional SHA-256 of the original file",
          "page_index": 0,
          "staff_spacing_px": 16.0,
          "notes": [{"center": [x, y], "staff_index": 0,
                     "pitch": {"step": "E", "octave": 4}}]
        }]
      }

    All annotation coordinates are in the grayscale image coordinate frame produced
    by load_grayscale for the selected page. This evaluator can score real data only
    when the manifest contains independently prepared real annotations.
    """
    if tolerance_staff_spaces <= 0:
        raise ValueError("tolerance_staff_spaces must be positive.")
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "scoretwin-annotated-pages-v1":
        raise ValueError("Manifest schema must be scoretwin-annotated-pages-v1.")
    records = manifest.get("pages")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain a non-empty pages list.")

    root = manifest_file.parent.resolve()
    model = load_model(model_path) if model_path is not None else None
    per_page: list[dict] = []
    all_center_errors: list[float] = []
    all_center_errors_sp: list[float] = []
    pitch_correct = 0
    pitch_total = 0
    total_truth = 0
    total_proposed = 0

    for page_number, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("image"), str):
            raise ValueError(f"Page entry {page_number} must specify an image path.")
        image_path = (root / record["image"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"Page image path escapes the manifest directory: {record['image']}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        source_bytes = image_path.read_bytes()
        image_sha256 = hashlib.sha256(source_bytes).hexdigest()
        expected_sha = record.get("sha256")
        if expected_sha is not None and expected_sha != image_sha256:
            raise ValueError(f"SHA-256 mismatch for {image_path.name}.")

        page_index = int(record.get("page_index", 0))
        gray = load_grayscale(image_path, page=page_index)
        truth = record.get("notes")
        if not isinstance(truth, list):
            raise ValueError(f"Page entry {page_number} must contain an annotated notes list.")
        if any("center" not in note or "staff_index" not in note for note in truth):
            raise ValueError(f"Each note annotation on {image_path.name} needs center and staff_index.")

        scene = analyze_array(gray, model=model)
        predicted = scene["notes"]
        spacing_value = record.get("staff_spacing_px")
        if spacing_value is None:
            found_spacings = [float(staff["spacing"]) for staff in scene["staffs"]]
            spacing_value = float(np.median(found_spacings)) if found_spacings else 12.0
        spacing_px = float(spacing_value)
        if not np.isfinite(spacing_px) or spacing_px <= 0:
            raise ValueError(f"Invalid staff_spacing_px for {image_path.name}.")
        tolerance_px = tolerance_staff_spaces * spacing_px
        matches = _match_notes(truth, predicted, tolerance_px=tolerance_px)

        page_pitch_correct = 0
        page_pitch_total = 0
        center_errors = []
        center_errors_sp = []
        for truth_index, predicted_index, distance in matches:
            center_errors.append(float(distance))
            center_errors_sp.append(float(distance) / spacing_px)
            truth_pitch = truth[truth_index].get("pitch")
            if truth_pitch is not None:
                page_pitch_total += 1
                if predicted[predicted_index]["pitch"] == truth_pitch:
                    page_pitch_correct += 1

        tp = len(matches)
        fp = len(predicted) - tp
        fn = len(truth) - tp
        precision, recall, f1 = _scores(tp, fp, fn)
        per_page.append({
            "id": str(record.get("id", f"page-{page_number + 1:04d}")),
            "image": str(record["image"]),
            "image_sha256": image_sha256,
            "page_index": page_index if image_path.suffix.lower() == ".pdf" else None,
            "staff_spacing_px": spacing_px,
            "match_tolerance_px": tolerance_px,
            "n_ground_truth": len(truth),
            "n_proposed": len(predicted),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pitch_correct_on_matches": page_pitch_correct,
            "pitch_labelled_matches": page_pitch_total,
            "pitch_accuracy_on_labelled_matches": (
                page_pitch_correct / page_pitch_total if page_pitch_total else None
            ),
            "mean_center_error_px": float(np.mean(center_errors)) if center_errors else None,
            "p95_center_error_px": float(np.percentile(center_errors, 95)) if center_errors else None,
            "mean_center_error_staff_spaces": float(np.mean(center_errors_sp)) if center_errors_sp else None,
            "p95_center_error_staff_spaces": float(np.percentile(center_errors_sp, 95)) if center_errors_sp else None,
            "center_error_sp_values": center_errors_sp,
            "runtime_seconds": scene["runtime_seconds"],
            "capture_diagnostics": scene.get("capture_diagnostics", {}),
            "warnings": scene["warnings"],
        })
        total_truth += len(truth)
        total_proposed += len(predicted)
        all_center_errors.extend(center_errors)
        all_center_errors_sp.extend(center_errors_sp)
        pitch_correct += page_pitch_correct
        pitch_total += page_pitch_total

    tp = sum(page["true_positive"] for page in per_page)
    fp = sum(page["false_positive"] for page in per_page)
    fn = sum(page["false_negative"] for page in per_page)
    precision, recall, f1 = _scores(tp, fp, fn)
    return {
        "schema": "scoretwin-manifest-evaluation-v1",
        "data_status": "depends_on_manifest; inspect each page source before claiming real-photo generalization",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pages_n": len(per_page),
        "ground_truth_noteheads_n": total_truth,
        "proposed_noteheads_n": total_proposed,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_recall_f1_bootstrap_95_ci": _bootstrap_intervals(
            per_page, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "pitch_correct_on_labelled_matches": pitch_correct,
        "pitch_labelled_matches": pitch_total,
        "pitch_accuracy_on_labelled_matches": pitch_correct / pitch_total if pitch_total else None,
        "mean_center_error_px": float(np.mean(all_center_errors)) if all_center_errors else None,
        "p95_center_error_px": float(np.percentile(all_center_errors, 95)) if all_center_errors else None,
        "mean_center_error_staff_spaces": float(np.mean(all_center_errors_sp)) if all_center_errors_sp else None,
        "p95_center_error_staff_spaces": float(np.percentile(all_center_errors_sp, 95)) if all_center_errors_sp else None,
        "tolerance_staff_spaces": tolerance_staff_spaces,
        "software": {
            "scoretwin": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "proposal_model_training_metrics": model.get("training_metrics", {}) if model is not None else {},
        "model_checkpoint_sha256": (
            hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path is not None else None
        ),
        "confidence_note": "Classifier scores and template-fit scores are not calibrated probabilities.",
        "per_page": per_page,
    }
