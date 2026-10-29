"""Component-wise, adversarially inspectable IDSR comparison metrics.

This module deliberately reports a vector of measurements instead of an uncalibrated
single DID scalar. Aggregation weights must be calibrated against human judgments
before the scalar is presented as perceptual identity.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _validate_raster(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError(f"{name} must be a uint8 grayscale or multichannel raster.")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError(f"{name} must have 3 or 4 color channels when multichannel.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def binary_mask_metrics(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> dict:
    """Compute binary mask IoU and F1; empty/empty masks score one by convention."""
    reference = np.asarray(reference_mask) > 0
    candidate = np.asarray(candidate_mask) > 0
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Reference and candidate masks must be same-shape 2D arrays.")
    intersection = int(np.count_nonzero(reference & candidate))
    reference_count = int(np.count_nonzero(reference))
    candidate_count = int(np.count_nonzero(candidate))
    union = int(np.count_nonzero(reference | candidate))
    iou = intersection / union if union else 1.0
    denominator = reference_count + candidate_count
    f1 = (2.0 * intersection / denominator) if denominator else 1.0
    return {
        "reference_foreground_pixels": reference_count,
        "candidate_foreground_pixels": candidate_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(iou),
        "f1": float(f1),
        "empty_mask_convention": "IoU=F1=1 when both masks are empty",
    }


def _match_centers(
    reference_notes: list[dict],
    candidate_notes: list[dict],
    tolerance_px: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for ref_index, reference in enumerate(reference_notes):
        ref_center = reference.get("center")
        if not isinstance(ref_center, (list, tuple)) or len(ref_center) < 2:
            raise ValueError(f"Reference note {ref_index} requires center=[x,y].")
        for cand_index, candidate in enumerate(candidate_notes):
            cand_center = candidate.get("center")
            if not isinstance(cand_center, (list, tuple)) or len(cand_center) < 2:
                raise ValueError(f"Candidate note {cand_index} requires center=[x,y].")
            if int(reference.get("staff_index", 0)) != int(candidate.get("staff_index", 0)):
                continue
            distance = float(np.hypot(
                float(ref_center[0]) - float(cand_center[0]),
                float(ref_center[1]) - float(cand_center[1]),
            ))
            if distance <= tolerance_px:
                candidates.append((distance, ref_index, cand_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_ref: set[int] = set()
    used_candidate: set[int] = set()
    matches = []
    for distance, ref_index, cand_index in candidates:
        if ref_index in used_ref or cand_index in used_candidate:
            continue
        used_ref.add(ref_index)
        used_candidate.add(cand_index)
        matches.append((ref_index, cand_index, distance))
    return matches


def geometry_metrics(
    reference_notes: list[dict],
    candidate_notes: list[dict],
    staff_space_px: float,
    tolerance_staff_spaces: float = 0.3,
) -> dict:
    """One-to-one note-center matching within the same staff and a fixed tolerance."""
    if not np.isfinite(staff_space_px) or staff_space_px <= 0:
        raise ValueError("staff_space_px must be finite and positive.")
    if not np.isfinite(tolerance_staff_spaces) or tolerance_staff_spaces <= 0:
        raise ValueError("tolerance_staff_spaces must be finite and positive.")
    tolerance_px = staff_space_px * tolerance_staff_spaces
    matches = _match_centers(reference_notes, candidate_notes, tolerance_px)
    tp = len(matches)
    fp = len(candidate_notes) - tp
    fn = len(reference_notes) - tp
    precision = tp / len(candidate_notes) if candidate_notes else (1.0 if not reference_notes else 0.0)
    recall = tp / len(reference_notes) if reference_notes else (1.0 if not candidate_notes else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors_px = [distance for _ri, _ci, distance in matches]
    errors_sp = [distance / staff_space_px for distance in errors_px]
    p95_px = float(np.percentile(errors_px, 95)) if errors_px else None
    p95_sp = float(np.percentile(errors_sp, 95)) if errors_sp else None
    return {
        "staff_space_px": float(staff_space_px),
        "tolerance_staff_spaces": float(tolerance_staff_spaces),
        "tolerance_px": float(tolerance_px),
        "reference_note_count": len(reference_notes),
        "candidate_note_count": len(candidate_notes),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_center_error_mean_px": float(np.mean(errors_px)) if errors_px else None,
        "matched_center_error_p95_px": p95_px,
        "matched_center_error_mean_staff_spaces": float(np.mean(errors_sp)) if errors_sp else None,
        "matched_center_error_p95_staff_spaces": p95_sp,
        "matched_symbol_pitch_accuracy": _pitch_accuracy(reference_notes, candidate_notes, matches),
        "matches": [
            {
                "reference_index": ri,
                "candidate_index": ci,
                "center_error_px": float(distance),
                "center_error_staff_spaces": float(distance / staff_space_px),
            }
            for ri, ci, distance in matches
        ],
        "matching_policy": "deterministic greedy one-to-one nearest center, constrained to same staff",
        "caveat": "Precision/recall count unmatched proposals; localization error is conditional on matched pairs.",
    }


def _pitch_accuracy(reference_notes, candidate_notes, matches) -> dict:
    labelled = 0
    correct = 0
    for ref_index, candidate_index, _distance in matches:
        ref_pitch = reference_notes[ref_index].get("pitch")
        candidate_pitch = candidate_notes[candidate_index].get("pitch")
        if ref_pitch is None:
            continue
        labelled += 1
        if candidate_pitch == ref_pitch:
            correct += 1
    return {
        "labelled_matched_notes": labelled,
        "correct_pitch_labels": correct,
        "accuracy": float(correct / labelled) if labelled else None,
        "unlabelled_matches_excluded": len(matches) - labelled,
    }


def relation_metrics(reference_relations: list[dict], candidate_relations: list[dict]) -> dict:
    """Exact typed-edge precision/recall/F1 using relation IDs as anchors."""
    def canonical(relation: dict) -> tuple[str, str, str]:
        return (
            str(relation.get("type", "")),
            str(relation.get("source", "")),
            str(relation.get("target", "")),
        )
    reference = {canonical(item) for item in reference_relations}
    candidate = {canonical(item) for item in candidate_relations}
    intersection = reference & candidate
    tp, fp, fn = len(intersection), len(candidate - reference), len(reference - candidate)
    precision = tp / len(candidate) if candidate else (1.0 if not reference else 0.0)
    recall = tp / len(reference) if reference else (1.0 if not candidate else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_edge_count": len(reference),
        "candidate_edge_count": len(candidate),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "identity_policy": "exact typed edge with source and target IDs; node-ID alignment must be provided by the evaluator",
    }


def photometric_metrics(reference_image: np.ndarray, candidate_image: np.ndarray) -> dict:
    """Compute exactness, MAE, MSE and PSNR between same-shape decoded rasters."""
    reference = _validate_raster(reference_image, "reference_image")
    candidate = _validate_raster(candidate_image, "candidate_image")
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate image shapes must match exactly.")
    difference = reference.astype(np.int16) - candidate.astype(np.int16)
    absolute = np.abs(difference).astype(np.int32)
    mae = float(absolute.mean())
    mse = float(np.square(difference.astype(np.float64)).mean())
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10((255.0 ** 2) / mse))
    if reference.ndim == 2:
        pixel_mismatch = reference != candidate
        pixel_count = int(reference.shape[0] * reference.shape[1])
        channel_count = 1
    else:
        pixel_mismatch = np.any(reference != candidate, axis=2)
        pixel_count = int(reference.shape[0] * reference.shape[1])
        channel_count = int(reference.shape[2])
    return {
        "height_px": int(reference.shape[0]),
        "width_px": int(reference.shape[1]),
        "channel_count": channel_count,
        "pixel_count": pixel_count,
        "channel_value_count": int(reference.size),
        "exact_pixel_fraction": float(1.0 - np.count_nonzero(pixel_mismatch) / pixel_count),
        "mismatched_pixel_count": int(np.count_nonzero(pixel_mismatch)),
        "mean_absolute_channel_error": mae,
        "mean_squared_channel_error": mse,
        "psnr_db": None if mse == 0 else psnr,
        "psnr_infinite_for_exact_match": bool(mse == 0),
        "maximum_absolute_channel_error": int(absolute.max()),
        "identical_array_bytes": bool(reference.tobytes() == candidate.tobytes()),
        "reference_decoded_array_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "candidate_decoded_array_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
    }


def residual_compressibility_metrics(reference_image: np.ndarray, candidate_image: np.ndarray) -> dict:
    """Report a reproducible zlib residual-size proxy, explicitly not an MDL proof."""
    reference = _validate_raster(reference_image, "reference_image")
    candidate = _validate_raster(candidate_image, "candidate_image")
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate image shapes must match exactly.")
    residual = reference.astype(np.int16) - candidate.astype(np.int16)
    raw = np.ascontiguousarray(residual, dtype="<i2").tobytes()
    compressed = zlib.compress(raw, level=9)
    pixels = int(reference.shape[0] * reference.shape[1])
    return {
        "codec": "zlib",
        "codec_level": 9,
        "residual_representation": "signed little-endian int16 per channel",
        "residual_raw_bytes": len(raw),
        "residual_zlib_bytes": len(compressed),
        "residual_raw_bits_per_pixel": float(8 * len(raw) / pixels),
        "residual_zlib_bits_per_pixel": float(8 * len(compressed) / pixels),
        "residual_compression_ratio": float(len(compressed) / len(raw)) if raw else 0.0,
        "residual_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
        "caveat": "Codec-specific compressibility proxy only; not a model-based MDL value and not comparable across codecs or preprocessing without controls.",
    }


def compare_idsr_components(
    reference_image: np.ndarray,
    candidate_image: np.ndarray,
    *,
    staff_space_px: float | None = None,
    reference_notes: list[dict] | None = None,
    candidate_notes: list[dict] | None = None,
    reference_relations: list[dict] | None = None,
    candidate_relations: list[dict] | None = None,
    reference_mask: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
    tolerance_staff_spaces: float = 0.3,
) -> dict:
    """Produce component metrics without pretending their weights are human-calibrated."""
    report: dict[str, Any] = {
        "schema": "scoretwin-idSR-component-metrics-0.1",
        "metrics": {
            "photometric": photometric_metrics(reference_image, candidate_image),
            "residual_compressibility": residual_compressibility_metrics(reference_image, candidate_image),
        },
        "geometry": None,
        "relationships": None,
        "mask": None,
        "aggregate_did": None,
        "aggregate_did_status": (
            "not_computed: scalar weights require preregistered calibration against "
            "human identity judgments; a vector of components is reported instead"
        ),
        "human_calibration": {
            "status": "not_run",
            "recommended_design": "blinded paired page comparison with randomized candidate order and repeated items",
            "primary_measure": "association between component metrics and human identity ratings",
            "inter_rater_agreement": "not measured",
        },
    }
    if (reference_notes is None) != (candidate_notes is None):
        raise ValueError("Provide both reference_notes and candidate_notes, or neither.")
    if reference_notes is not None:
        if staff_space_px is None:
            raise ValueError("staff_space_px is required when computing note geometry metrics.")
        report["geometry"] = geometry_metrics(
            reference_notes,
            candidate_notes or [],
            staff_space_px=staff_space_px,
            tolerance_staff_spaces=tolerance_staff_spaces,
        )
    if (reference_relations is None) != (candidate_relations is None):
        raise ValueError("Provide both reference_relations and candidate_relations, or neither.")
    if reference_relations is not None:
        report["relationships"] = relation_metrics(reference_relations, candidate_relations or [])
    if (reference_mask is None) != (candidate_mask is None):
        raise ValueError("Provide both reference_mask and candidate_mask, or neither.")
    if reference_mask is not None:
        report["mask"] = binary_mask_metrics(reference_mask, candidate_mask)
    return report



def measure_bundle_storage(bundle_dir: str | Path, pixel_count: int) -> dict:
    """Measure stored artifact bytes, explicitly excluding only the immutable source copy.

    This is a product storage measurement, not formal model-based MDL. It includes
    all generated bundle artifacts (including previews and the glTF texture), so
    redundant derived files increase the measured size rather than disappearing.
    The report is returned as a Python object and should be written outside bundle_dir
    to avoid a self-referential file-size measurement.
    """
    root = Path(bundle_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if int(pixel_count) <= 0:
        raise ValueError("pixel_count must be a positive integer.")
    included = []
    excluded_sources = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        entry = {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if path.name.startswith("source-original."):
            excluded_sources.append(entry)
        else:
            included.append(entry)
    total_bytes = sum(item["bytes"] for item in included)
    return {
        "schema": "scoretwin-bundle-storage-0.1",
        "bundle_directory_name": root.name,
        "pixel_count": int(pixel_count),
        "included_file_count": len(included),
        "included_file_bytes": total_bytes,
        "stored_artifact_bits_per_pixel": float(8.0 * total_bytes / int(pixel_count)),
        "included_files": included,
        "excluded_source_copy_count": len(excluded_sources),
        "excluded_source_copy_bytes": sum(item["bytes"] for item in excluded_sources),
        "excluded_source_copies": excluded_sources,
        "exclusion_policy": "Only files named source-original.* are excluded; all other files in the bundle are counted.",
        "status": "storage accounting only; not a model-based minimum-description-length estimate",
        "limitations": [
            "Includes redundant previews and embedded textures, so it measures the current product bundle rather than a minimal code.",
            "Excludes the copied immutable source file to avoid treating literal source duplication as a successful reconstruction code.",
            "Does not compare against a trained generative model or a proven optimal oracle.",
        ],
    }
