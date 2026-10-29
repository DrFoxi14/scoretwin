"""Run an unlabeled smoke evaluation on supplied photograph or scan pages.

This reports pipeline coverage, runtime and source-byte preservation only. It
does not compute accuracy, DID, IoU against truth, or real-photo generalization.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .benchmark import _bootstrap_mean_ci, _wilson_ci
from .pipeline import process_file
from .proposal import get_default_model, load_model


def evaluate_photo_smoke(
    manifest_path: str | Path,
    model_path: str | Path | None = None,
) -> dict:
    """Analyze declared camera/scan pages and verify lossless source retention.

    Manifest schema::

      {
        "schema": "scoretwin-photo-smoke-v1",
        "pages": [{
          "id": "pair-97-page-0",
          "image": "/path/to/scan.pdf",
          "page_index": 0,
          "capture_type": "camera_photo"
        }]
      }

    Image paths may be absolute or relative to the manifest. Paths are omitted
    from the output so a local report can be shared without exposing its root.
    """
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest_bytes = manifest_file.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object.")
    if manifest.get("schema") != "scoretwin-photo-smoke-v1":
        raise ValueError("Manifest schema must be scoretwin-photo-smoke-v1.")
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Manifest must contain a non-empty pages list.")

    model_sha256 = None
    model_metrics = {}
    if model_path is not None:
        model_file = Path(model_path).expanduser().resolve()
        model_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()
        model_metrics = dict(load_model(model_file).get("training_metrics", {}))
    else:
        model = get_default_model()
        model_sha256 = hashlib.sha256(
            np.asarray(model["weights"], dtype=np.float32).tobytes()
        ).hexdigest()
        model_metrics = dict(model.get("training_metrics", {}))

    rows = []
    for index, entry in enumerate(pages):
        if not isinstance(entry, dict):
            raise ValueError(f"Page entry {index} must be an object.")
        page_id = entry.get("id")
        image_value = entry.get("image")
        capture_type = entry.get("capture_type")
        if not isinstance(page_id, str) or not page_id.strip():
            raise ValueError(f"Page entry {index} needs a non-empty id.")
        if not isinstance(image_value, str) or not image_value.strip():
            raise ValueError(f"Page entry {page_id!r} needs an image path.")
        if capture_type not in {"camera_photo", "scan", "screen_capture", "synthetic"}:
            raise ValueError(
                f"Page entry {page_id!r} needs capture_type: camera_photo, scan, "
                "screen_capture or synthetic."
            )
        image_path = Path(image_value).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_file.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        page_index = int(entry.get("page_index", 0))
        if page_index < 0:
            raise ValueError(f"Page entry {page_id!r} has a negative page_index.")

        source_bytes = image_path.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        expected_sha256 = entry.get("sha256")
        if expected_sha256 is not None and expected_sha256 != source_sha256:
            raise ValueError(f"SHA-256 mismatch for declared page {page_id!r}.")

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="scoretwin-photo-smoke-") as temporary:
            result = process_file(
                image_path,
                Path(temporary) / "bundle",
                page=page_index,
                model_path=model_path,
            )
        wall_seconds = time.perf_counter() - started
        scene = result["scene"]
        rows.append({
            "id": page_id,
            "declared_capture_type": capture_type,
            "page_index": page_index if image_path.suffix.lower() == ".pdf" else None,
            "source_sha256": source_sha256,
            "source_bytes": len(source_bytes),
            "source_bytes_preserved": bool(result["source_hash_verified"]),
            "source_image_shape_px": scene["capture_diagnostics"]["source_image_shape_px"],
            "analysis_image_shape_px": scene["capture_diagnostics"]["analysis_image_shape_px"],
            "staffs_detected": len(scene["staffs"]),
            "raw_proposals": scene["proposal_count"],
            "notehead_proposals": scene["note_count"],
            "unknown_regions": len(scene["unknown_objects"]),
            "estimated_staffline_angle_deg": scene["capture_diagnostics"]["estimated_staffline_angle_deg"],
            "analysis_runtime_seconds": scene["runtime_seconds"],
            "end_to_end_runtime_seconds": wall_seconds,
            "warnings": scene["warnings"],
        })

    source_preserved = sum(row["source_bytes_preserved"] for row in rows)
    pages_with_staff = sum(row["staffs_detected"] > 0 for row in rows)
    pages_with_notes = sum(row["notehead_proposals"] > 0 for row in rows)
    return {
        "schema": "scoretwin-photo-smoke-report-v1",
        "epistemic_status": "MĂSURAT — unlabeled pipeline smoke test; not an accuracy benchmark",
        "data_status": "no_independent_symbolic_or_geometric_ground_truth",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "pages_n": len(rows),
        "pages_source_bytes_preserved_n": source_preserved,
        "source_bytes_preserved_rate": source_preserved / len(rows),
        "source_bytes_preserved_rate_wilson_95_ci": _wilson_ci(source_preserved, len(rows)),
        "sampling_note": (
            "Pages are purposively selected. Intervals below describe page-level variation under "
            "an iid resampling model and do not support population generalization."
        ),
        "coverage": {
            "pages_with_staff_detected": pages_with_staff,
            "page_staff_detection_rate": pages_with_staff / len(rows),
            "page_staff_detection_rate_wilson_95_ci": _wilson_ci(pages_with_staff, len(rows)),
            "pages_with_notehead_proposals": pages_with_notes,
            "page_note_proposal_rate": pages_with_notes / len(rows),
            "page_note_proposal_rate_wilson_95_ci": _wilson_ci(pages_with_notes, len(rows)),
        },
        "runtime_summary": {
            "unit": "end_to_end_seconds_per_page",
            "mean": float(np.mean([row["end_to_end_runtime_seconds"] for row in rows])),
            "mean_page_bootstrap_95_ci": _bootstrap_mean_ci(
                [row["end_to_end_runtime_seconds"] for row in rows], seed=20261029, rounds=2000
            ),
            "bootstrap_unit": "page",
            "bootstrap_rounds": 2000,
            "bootstrap_seed": 20261029,
        },
        "model_sha256": model_sha256,
        "model_training_metrics": model_metrics,
        "software": {
            "scoretwin": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "claims_not_supported": [
            "notehead precision, recall or pitch accuracy",
            "DID, IoU against ground truth, or pixel-level reconstructed identity",
            "generalization beyond the listed pages",
        ],
        "per_page": rows,
    }
