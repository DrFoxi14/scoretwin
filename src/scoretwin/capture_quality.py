"""Heuristic capture-quality diagnostics. Flags are not calibrated probabilities."""
from __future__ import annotations

import cv2
import numpy as np


def diagnose_capture_quality(gray: np.ndarray) -> dict:
    """Measure contrast, sharpness and slow background illumination variation."""
    image = np.asarray(gray)
    if image.dtype != np.uint8 or image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty uint8 grayscale image.")
    if min(image.shape) < 16:
        raise ValueError("Image dimensions must be at least 16 pixels.")

    values = image.astype(np.float32)
    p01, p05, p50, p95, p99 = [float(v) for v in np.percentile(values, [1, 5, 50, 95, 99])]
    dynamic_range = p95 - p05
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    laplacian_variance = float(laplacian.var())
    normalized_sharpness = laplacian_variance / ((dynamic_range + 1.0) ** 2)

    bg_values = []
    background_mask = values >= 160
    rows = np.array_split(values, 4, axis=0)
    masks = np.array_split(background_mask, 4, axis=0)
    for row, mask_row in zip(rows, masks):
        tiles = np.array_split(row, 4, axis=1)
        mask_tiles = np.array_split(mask_row, 4, axis=1)
        for tile, mask in zip(tiles, mask_tiles):
            selected = tile[mask]
            if selected.size >= max(20, int(tile.size * 0.02)):
                bg_values.append(float(selected.mean()))
    illumination_spread = (
        float(np.percentile(bg_values, 90) - np.percentile(bg_values, 10))
        if len(bg_values) >= 4 else None
    )

    advisories = []
    if dynamic_range < 24.0:
        advisories.append({"code":"low_dynamic_range","severity":"retake_suggested","action":"Use even lighting and expose so ink and paper are distinct."})
    if normalized_sharpness < 0.012:
        advisories.append({"code":"low_sharpness_suspected","severity":"review_or_retake","action":"Hold still, allow focus to settle, and keep fine note edges in focus."})
    if illumination_spread is not None and illumination_spread > 35.0:
        advisories.append({"code":"uneven_background_illumination","severity":"retake_suggested","action":"Reduce shadows and reflections; light the page more evenly."})
    if p95 < 70.0:
        advisories.append({"code":"page_too_dark_suspected","severity":"retake_suggested","action":"Increase illumination while avoiding motion blur."})
    if p05 > 245.0 and dynamic_range < 10.0:
        advisories.append({"code":"nearly_blank_or_overexposed","severity":"retake_suggested","action":"Check the score is in frame and reduce exposure if symbols have disappeared."})

    return {
        "schema":"scoretwin-capture-quality-0.1",
        "image_shape_px":[int(image.shape[0]),int(image.shape[1])],
        "metrics":{
            "gray_mean":float(values.mean()),
            "gray_standard_deviation":float(values.std()),
            "gray_percentiles":{"p01":p01,"p05":p05,"p50":p50,"p95":p95,"p99":p99},
            "p95_minus_p05_dynamic_range":float(dynamic_range),
            "dark_clipping_fraction_le_4":float(np.mean(image <= 4)),
            "bright_clipping_fraction_ge_251":float(np.mean(image >= 251)),
            "laplacian_variance":laplacian_variance,
            "normalized_laplacian_variance":float(normalized_sharpness),
            "background_grid_p90_minus_p10":illumination_spread
        },
        "advisories":advisories,
        "status":"retake_suggested" if any(a["severity"]=="retake_suggested" for a in advisories) else ("review_or_retake" if advisories else "no_heuristic_warning"),
        "thresholds_calibrated":False,
        "limitations":[
            "A white page naturally has many bright pixels; bright-pixel fraction alone is not a defect.",
            "Sharpness thresholds depend on resolution, print weight and score density.",
            "Stains and dense text can affect the background illumination estimate.",
            "Passing these diagnostics does not imply correct symbol or rhythm recognition."
        ]
    }
