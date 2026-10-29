"""IDSR first vertical slice: analyze, preserve, visualize and export."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .imaging import detect_barlines, detect_staff_systems, detect_stem_candidates, estimate_global_skew_degrees, extract_notehead_candidates, load_grayscale, rectify_global_skew, rectify_page_quadrilateral, refine_notehead_geometry, staff_step_and_pitch
from .gltf_export import build_gltf, validate_embedded_gltf
from .musicxml import build_musicxml
from .proposal import get_default_model, load_model, predict_patch


def _classify_candidates(candidates: list[dict], model: dict) -> list[dict]:
    classified = []
    for candidate in candidates:
        crop = np.asarray(candidate.pop("crop"), dtype=np.uint8)
        scores = predict_patch(model, crop)
        label = max(scores, key=scores.get)
        confidence = float(scores[label])
        if confidence < 0.48:
            label = "unknown"

        contrast_std = float(np.std(crop)) if crop.size else 0.0
        laplacian_variance = (
            float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        )
        if contrast_std < 12.0:
            unknown_reason = "low_local_contrast"
        elif laplacian_variance < 4.0:
            unknown_reason = "low_local_sharpness_or_blur"
        elif label in ("artifact", "unknown"):
            unknown_reason = "non_modeled_mark_or_classifier_uncertain"
        else:
            unknown_reason = None

        item = {
            **candidate,
            "class": label,
            "confidence": round(confidence, 4),
            "class_probabilities": {key: round(float(value), 4) for key, value in scores.items()},
            "image_quality_diagnostics": {
                "crop_grayscale_std": round(contrast_std, 4),
                "crop_laplacian_variance": round(laplacian_variance, 4),
                "thresholds_calibrated": False,
            },
            "unknown_reason": unknown_reason,
        }
        classified.append(item)
    return classified


def analyze_array(gray: np.ndarray, model: dict | None = None) -> dict:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("analyze_array expects a uint8 grayscale image.")
    if min(gray.shape) < 32:
        raise ValueError("Input image is too small to contain a score page.")
    started = time.perf_counter()
    estimated_skew = estimate_global_skew_degrees(gray)
    staffs, binary, staff_mask = detect_staff_systems(gray)
    candidates = extract_notehead_candidates(gray, binary, staff_mask, staffs)
    model_training_metrics: dict = {}
    if candidates:
        proposal_model = model if model is not None else get_default_model()
        model_training_metrics = dict(proposal_model.get("training_metrics", {}))
        classified = _classify_candidates(candidates, proposal_model)
    else:
        classified = []

    # Apply bounded local geometry refinement only to candidates already classified
    # as noteheads. This is analysis-by-synthesis at proposal scale, not a global OMR.
    image_height, image_width = gray.shape
    clean_binary = cv2.subtract(binary, staff_mask) if classified else None
    for item in classified:
        if item["class"] not in ("filled_head", "hollow_head"):
            continue
        staff = staffs[int(item["staff_index"])]
        initial_center = [float(item["center"][0]), float(item["center"][1])]
        fit = refine_notehead_geometry(
            gray, binary, staff_mask, initial_center, float(staff.spacing), item["class"],
            precleaned_binary=clean_binary,
        )
        item["geometry_refinement"] = fit
        if fit["accepted"]:
            center_x, center_y = fit["center"]
            box_width, box_height = [int(v) for v in item["bbox"][2:4]]
            box_x = max(0, min(image_width - box_width, int(round(center_x - box_width / 2.0))))
            box_y = max(0, min(image_height - box_height, int(round(center_y - box_height / 2.0))))
            item["center"] = [float(center_x), float(center_y)]
            item["bbox"] = [box_x, box_y, box_width, box_height]

    staff_objects = []
    for staff in staffs:
        barlines = detect_barlines(binary, staff)
        staff_objects.append({**staff.to_dict(), "barlines": barlines})

    stem_candidates = detect_stem_candidates(
        binary,
        staffs,
        {int(staff["index"]): staff["barlines"] for staff in staff_objects},
    )
    notes = []
    unknown_objects = []
    for item in classified:
        staff_index = item["staff_index"]
        staff = staffs[staff_index]
        cx, cy = item["center"]
        if item["class"] in ("filled_head", "hollow_head"):
            step, letter, octave = staff_step_and_pitch(cy, staff)
            notes.append({
                "id": item["id"],
                "staff_index": staff_index,
                "bbox": item["bbox"],
                "center": item["center"],
                "pitch": {"step": letter, "octave": octave},
                "shape_class": item["class"],
                "diatonic_step_from_bottom_line": step,
                "pitch_confidence": item["confidence"],
                "geometry_refinement": item.get("geometry_refinement"),
                "duration": None,
                "uncertainty": ["key_signature", "accidental", "duration", "voice", "instrument_or_part", "measure_assignment"],
            })
        else:
            unknown_objects.append({
                "id": item["id"],
                "staff_index": staff_index,
                "bbox": item["bbox"],
                "center": item["center"],
                "class": item["class"],
                "confidence": item["confidence"],
                "reason": item.get("unknown_reason") or "Non-note candidate requiring review.",
                "image_quality_diagnostics": item.get("image_quality_diagnostics", {}),
            })

    # Associate long vertical strokes with nearby noteheads. The same stem may belong
    # to multiple vertically stacked heads in a chord. All associations remain proposals.
    for stem in stem_candidates:
        sx, sy = [float(value) for value in stem["center"]]
        bx, by, bw, bh = [float(value) for value in stem["bbox"]]
        spacing = float(staffs[int(stem["staff_index"])].spacing)
        attached = []
        for note in notes:
            if int(note["staff_index"]) != int(stem["staff_index"]):
                continue
            nx, ny, nw, nh = [float(value) for value in note["bbox"]]
            cx, cy = [float(value) for value in note["center"]]
            edge_distance = min(abs(sx - nx), abs(sx - (nx + nw)))
            vertical_intersection = (by <= ny + nh + 0.25 * spacing) and (by + bh >= ny - 0.25 * spacing)
            extends_from_head = (by < cy - 0.25 * spacing) or (by + bh > cy + 0.25 * spacing)
            if edge_distance <= 0.40 * spacing and vertical_intersection and extends_from_head:
                attached.append((edge_distance, note["id"]))
        if attached:
            attached.sort()
            stem["associated_note_ids"] = [note_id for _distance, note_id in attached]
        else:
            stem["associated_note_ids"] = []

    stem_by_note: dict[str, list[str]] = {}
    for stem in stem_candidates:
        if not stem["associated_note_ids"]:
            continue
        for note_id in stem["associated_note_ids"]:
            stem_by_note.setdefault(note_id, []).append(stem["id"])
    for note in notes:
        note["stem_ids"] = stem_by_note.get(note["id"], [])
        note["stem_detection_status"] = "proposal" if note["stem_ids"] else "not_detected_or_unresolved"
        if note["shape_class"] == "hollow_head":
            duration_candidate = "half_note_candidate" if note["stem_ids"] else "whole_note_candidate"
            duration_basis = "hollow_notehead_with_stem" if note["stem_ids"] else "hollow_notehead_without_detected_stem"
        elif note["stem_ids"]:
            duration_candidate = "quarter_or_shorter_candidate"
            duration_basis = "filled_notehead_with_stem; beam_and_flag_count_unknown"
        else:
            duration_candidate = "unresolved"
            duration_basis = "filled_notehead_without_detected_stem"
        note["duration_hypothesis"] = {
            "candidate": duration_candidate,
            "basis": duration_basis,
            "confidence_calibrated": False,
            "used_for_musicxml_duration": False,
        }
        if "stem_attachment" not in note["uncertainty"]:
            note["uncertainty"].append("stem_attachment")

    # Build a provenance-aware scene graph in observed pixel coordinates.
    # This graph is the editable reconstruction hypothesis; it is not a replacement
    # for the source bitmap and does not imply that every visible mark was understood.
    primitives: list[dict] = []
    relations: list[dict] = []
    system_ids: dict[int, str] = {}

    for staff in staff_objects:
        staff_index = int(staff["index"])
        system_id = f"staff-system-{staff_index:03d}"
        system_ids[staff_index] = system_id
        line_ids = []
        primitives.append({
            "id": system_id,
            "kind": "staff_system",
            "geometry": {
                "bbox_px": [0, int(staff["top"]), int(gray.shape[1]), int(staff["bottom"] - staff["top"] + 1)],
                "line_ids": [],
                "spacing_px": float(staff["spacing"]),
            },
            "provenance": {"method": "horizontal_morphology", "status": "estimated"},
            "uncertainty": ["perspective", "curvature", "staff_identity"],
        })
        for line_index, y_value in enumerate(staff["lines"]):
            y = int(y_value)
            y0, y1 = max(0, y - 1), min(staff_mask.shape[0], y + 2)
            row_support = np.any(staff_mask[y0:y1, :] > 0, axis=0)
            xs = np.flatnonzero(row_support)
            x0 = int(xs[0]) if len(xs) else 0
            x1 = int(xs[-1]) if len(xs) else gray.shape[1] - 1
            primitive_id = f"staff{staff_index}-line{line_index}"
            line_ids.append(primitive_id)
            primitives.append({
                "id": primitive_id,
                "kind": "staffline",
                "geometry": {
                    "x0_px": x0,
                    "y0_px": y,
                    "x1_px": x1,
                    "y1_px": y,
                    "thickness_px_estimate": 1.0,
                },
                "provenance": {"method": "horizontal_morphology", "status": "estimated"},
                "uncertainty": ["line_thickness", "local_waviness"],
            })
            relations.append({"type": "member_of", "source": primitive_id, "target": system_id})
        next(p for p in primitives if p["id"] == system_id)["geometry"]["line_ids"] = line_ids
        for bar_index, x_value in enumerate(staff.get("barlines", [])):
            x = int(x_value)
            bar_id = f"staff{staff_index}-bar{bar_index}"
            primitives.append({
                "id": bar_id,
                "kind": "barline",
                "geometry": {
                    "x0_px": x,
                    "y0_px": int(staff["top"]),
                    "x1_px": x,
                    "y1_px": int(staff["bottom"]),
                    "thickness_px_estimate": 1.0,
                },
                "provenance": {"method": "vertical_morphology", "status": "proposal"},
                "uncertainty": ["barline_vs_stem", "measure_boundary"],
            })
            relations.append({"type": "member_of", "source": bar_id, "target": system_id})

    for stem in stem_candidates:
        if not stem["associated_note_ids"]:
            continue
        stem_id = str(stem["id"])
        x, y, box_width, box_height = [float(value) for value in stem["bbox"]]
        cx, cy = [float(value) for value in stem["center"]]
        primitive = {
            "id": stem_id,
            "kind": "stem",
            "geometry": {
                "x0_px": cx,
                "y0_px": y,
                "x1_px": cx,
                "y1_px": y + box_height,
                "bbox_px": [x, y, box_width, box_height],
                "length_px": float(stem["length_px"]),
            },
            "provenance": {
                "method": "vertical_morphology_plus_notehead_proximity",
                "status": "proposal",
            },
            "uncertainty": ["stem_direction", "stem_head_attachment", "beam_or_flag"],
        }
        primitives.append(primitive)
        for note_id in stem["associated_note_ids"]:
            relations.append({"type": "stem_of", "source": stem_id, "target": note_id})

    for note in notes:
        x, y, box_width, box_height = [float(v) for v in note["bbox"]]
        center_x, center_y = [float(v) for v in note["center"]]
        crop = gray[
            max(0, int(y)):min(gray.shape[0], int(y + box_height)),
            max(0, int(x)):min(gray.shape[1], int(x + box_width)),
        ]
        ink_density = float(1.0 - np.mean(crop) / 255.0) if crop.size else 0.0
        primitive = {
            "id": note["id"],
            "kind": "notehead",
            "geometry": {
                "center_x_px": center_x,
                "center_y_px": center_y,
                "radius_x_px": box_width / 2.0,
                "radius_y_px": box_height / 2.0,
                "rotation_rad": 0.0,
                "bbox_px": [x, y, box_width, box_height],
            },
            "appearance": {
                "ink_density_estimate": ink_density,
                "measurement": "one_minus_mean_grayscale_inside_bbox",
                "physical_absorbance_measured": False,
            },
            "symbolic": {
                "pitch_proposal": dict(note["pitch"]),
                "duration": None,
                "stem_ids": list(note.get("stem_ids", [])),
                "duration_hypothesis": dict(note.get("duration_hypothesis", {})),
            },
            "provenance": {
                "method": "component_geometry_plus_synthetic_patch_classifier_plus_local_template_fit",
                "classifier_confidence": float(note["pitch_confidence"]),
                "geometry_fit_score": (
                    note.get("geometry_refinement") or {}
                ).get("fit_score"),
                "geometry_fit_accepted": bool(
                    (note.get("geometry_refinement") or {}).get("accepted", False)
                ),
                "status": "proposal",
            },
            "uncertainty": list(note["uncertainty"]),
        }
        primitives.append(primitive)
        system_id = system_ids.get(int(note["staff_index"]))
        if system_id:
            relations.append({"type": "geometrically_assigned_to_staff", "source": note["id"], "target": system_id})
            staff = staff_objects[int(note["staff_index"])]
            continuous_step = (float(staff["bottom"]) - center_y) / (float(staff["spacing"]) / 2.0)
            note["continuous_staff_step"] = continuous_step
            note["staff_position_offset_steps"] = continuous_step - round(continuous_step)
            bottom_line_id = f"staff{int(note['staff_index'])}-line4"
            relations.append({"type": "pitch_calibrated_by", "source": note["id"], "target": bottom_line_id})
        note["primitive_id"] = note["id"]
        note["appearance"] = primitive["appearance"]

    for item in unknown_objects:
        x, y, box_width, box_height = [int(v) for v in item["bbox"]]
        primitives.append({
            "id": item["id"],
            "kind": "unknown_region",
            "geometry": {
                "bbox_px": [x, y, box_width, box_height],
                "center_x_px": float(item["center"][0]),
                "center_y_px": float(item["center"][1]),
            },
            "provenance": {
                "method": "connected_component_proposal",
                "classifier_confidence": float(item["confidence"]),
                "status": "unresolved",
            },
            "uncertainty": ["symbol_identity", "relation_to_score"],
        })
        system_id = system_ids.get(int(item["staff_index"]))
        if system_id:
            relations.append({"type": "candidate_region_in_staff", "source": item["id"], "target": system_id})

    warnings = []
    if estimated_skew is not None and abs(estimated_skew) > 0.75:
        warnings.append(
            f"Estimated staff-line tilt is {estimated_skew:.2f} degrees; this version does not rectify rotation or perspective."
        )
    if not staffs:
        warnings.append("No five-line staff group detected. Preserve the source and review the page manually.")
    if not notes:
        warnings.append("No noteheads passed the current proposal classifier; the MusicXML will contain no inferred notes.")
    warnings.extend([
        "Pitch proposals assume treble clef and a C-major/no-accidental interpretation; key signatures and accidentals are not yet inferred.",
        "Durations, voice assignments, instrument mapping and full score structure remain unresolved and are marked for review in MusicXML.",
        "The synthetic model's confidence is not calibrated confidence on real manuscripts or photographs.",
        "Geometry-fit scores are local template-correlation values, not probabilities; rejected fits leave the original proposal unchanged.",
    ])
    return {
        "schema": "scoretwin-scene-0.1",
        "image": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "capture_diagnostics": {
            "estimated_staffline_angle_deg": estimated_skew,
            "rotation_rectified": False,
            "perspective_rectified": False,
        },
        "staffs": staff_objects,
        "primitives": primitives,
        "relations": relations,
        "notes": notes,
        "unknown_objects": unknown_objects,
        "proposal_count": len(classified),
        "note_count": len(notes),
        "warnings": warnings,
        "reproducibility": {
            "software": {
                "scoretwin": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "opencv": cv2.__version__,
            },
            "proposal_model": {
                "kind": "linear-softmax-classifier-on-synthetic-16x16-patches",
                "training_metrics": model_training_metrics,
                "confidence_calibrated": False,
                "geometry_refinement": "bounded-local-template-correlation",
            },
        },
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def render_geometry_scene(shape: tuple[int, int], scene: dict, scale: float = 1.0) -> np.ndarray:
    """Render currently modeled primitives on a clean page-sized canvas.

    This is an inspectable structural render, not yet a pixel-identical reconstruction:
    stems, beams, text, slurs, dynamics, paper texture and unknown glyph shapes may be
    absent. The source raster must be retained separately as the lossless identity anchor.
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1 or height * width > 80_000_000:
        raise ValueError("Invalid or excessively large geometry-render dimensions.")
    if not np.isfinite(scale) or scale <= 0 or scale > 1:
        raise ValueError("Geometry-render scale must be in (0, 1]." )
    image = np.full((height, width), 255, dtype=np.uint8)

    for staff in scene.get("staffs", []):
        for y_value in staff.get("lines", []):
            y = int(round(float(y_value) * scale))
            cv2.line(image, (0, y), (width - 1, y), 20, 1, cv2.LINE_AA)
        for x_value in staff.get("barlines", []):
            x = int(round(float(x_value) * scale))
            cv2.line(
                image, (x, int(round(float(staff["top"]) * scale))),
                (x, int(round(float(staff["bottom"]) * scale))),
                20, max(1, int(round(scale))), cv2.LINE_AA,
            )

    for primitive in scene.get("primitives", []):
        if primitive.get("kind") != "stem":
            continue
        geometry = primitive.get("geometry", {})
        x0 = int(round(float(geometry.get("x0_px", 0)) * scale))
        y0 = int(round(float(geometry.get("y0_px", 0)) * scale))
        x1 = int(round(float(geometry.get("x1_px", 0)) * scale))
        y1 = int(round(float(geometry.get("y1_px", 0)) * scale))
        cv2.line(image, (x0, y0), (x1, y1), 20, max(1, int(round(scale))), cv2.LINE_AA)

    for note in scene.get("notes", []):
        cx, cy = [float(value) * scale for value in note["center"]]
        bbox = [float(value) * scale for value in note["bbox"]]
        rx = max(2, int(round(bbox[2] / 2.0)))
        ry = max(2, int(round(bbox[3] / 2.0)))
        fit = note.get("geometry_refinement") or {}
        angle = float(fit.get("rotation_deg", 0.0)) if fit.get("accepted") else 0.0
        class_name = note.get("shape_class", "filled_head")
        thickness = -1 if class_name == "filled_head" else max(1, int(round(ry * 0.35)))
        cv2.ellipse(
            image, (int(round(cx)), int(round(cy))), (rx, ry), angle,
            0, 360, 20, max(1, int(round(thickness * scale))) if thickness > 0 else -1, cv2.LINE_AA,
        )

    # Unknown regions are outlined lightly rather than silently omitted.
    for item in scene.get("unknown_objects", []):
        x, y, box_width, box_height = [int(round(float(value) * scale)) for value in item["bbox"]]
        cv2.rectangle(
            image, (x, y), (x + box_width, y + box_height), 160, max(1, int(round(scale))), cv2.LINE_AA
        )
    return image


def render_overlay(gray: np.ndarray, scene: dict) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for staff in scene.get("staffs", []):
        for y in staff["lines"]:
            cv2.line(overlay, (0, int(y)), (overlay.shape[1] - 1, int(y)), (50, 170, 50), 1, cv2.LINE_AA)
        for x in staff.get("barlines", []):
            cv2.line(overlay, (int(x), int(staff["top"])), (int(x), int(staff["bottom"])), (170, 120, 20), 1, cv2.LINE_AA)

    for note in scene.get("notes", []):
        x, y, width, height = note["bbox"]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (30, 60, 230), 1, cv2.LINE_AA)
        cv2.putText(overlay, f'{note["pitch"]["step"]}{note["pitch"]["octave"]}?',
                    (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 60, 230), 1, cv2.LINE_AA)
    for item in scene.get("unknown_objects", []):
        x, y, width, height = item["bbox"]
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 150, 210), 1, cv2.LINE_AA)
    return overlay


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    page: int = 0,
    model_path: str | Path | None = None,
    rectify_skew: bool = False,
    max_skew_degrees: float = 12.0,
    rectify_corners: list | None = None,
) -> dict:
    """Generate a self-contained analysis bundle and preserve input bytes unchanged."""
    source = Path(input_path)
    output = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_gray = load_grayscale(source, page=page)
    if rectify_skew and rectify_corners is not None:
        raise ValueError("Choose either global-skew rectification or page-corner rectification, not both.")
    if rectify_corners is not None:
        rectification = rectify_page_quadrilateral(source_gray, rectify_corners)
    elif rectify_skew:
        rectification = rectify_global_skew(source_gray, max_abs_skew_degrees=max_skew_degrees)
    else:
        rectification = None
    analysis_gray = rectification.image if rectification is not None else source_gray
    model = load_model(model_path) if model_path else get_default_model()
    scene = analyze_array(analysis_gray, model=model)
    residual_skew = scene["capture_diagnostics"]["estimated_staffline_angle_deg"]
    source_skew = (
        rectification.estimated_skew_degrees
        if rectification is not None and rectification.estimated_skew_degrees is not None
        else estimate_global_skew_degrees(source_gray)
        if rectification is not None
        else residual_skew
    )
    source_to_analysis = (
        rectification.source_to_rectified if rectification is not None else np.eye(3, dtype=np.float64)
    )
    analysis_to_source = (
        rectification.rectified_to_source if rectification is not None else np.eye(3, dtype=np.float64)
    )
    scene["coordinate_frame"] = (
        "analysis_raster_pixel_centers" if rectification is not None else "source_raster_pixel_centers"
    )
    scene["capture_diagnostics"] = {
        **scene["capture_diagnostics"],
        "estimated_staffline_angle_deg": source_skew,
        "analysis_residual_staffline_angle_deg": residual_skew,
        "rotation_rectification_requested": bool(rectify_skew),
        "rotation_rectified": bool(
            rectification is not None
            and rectification.applied
            and rectification.method == "global_rotation"
        ),
        "perspective_rectification_requested": rectify_corners is not None,
        "perspective_rectified": bool(
            rectification is not None and rectification.method == "manual_page_quadrilateral"
        ),
        "rectification_method": rectification.method if rectification is not None else "none",
        "source_image_shape_px": [int(source_gray.shape[0]), int(source_gray.shape[1])],
        "analysis_image_shape_px": [int(analysis_gray.shape[0]), int(analysis_gray.shape[1])],
        "source_to_analysis": source_to_analysis.tolist(),
        "analysis_to_source": analysis_to_source.tolist(),
        "rectification_reason": rectification.reason if rectification is not None else "not_requested",
    }
    if model_path is not None:
        model_file = Path(model_path)
        scene["reproducibility"]["proposal_model"]["checkpoint_path"] = str(model_file)
        scene["reproducibility"]["proposal_model"]["checkpoint_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()

    scene["source"] = {
        "filename": source.name,
        "sha256": source_hash,
        "bytes": len(source_bytes),
        "page_index": page if source.suffix.lower() == ".pdf" else None,
        "preserved_losslessly": True,
    }

    suffix = source.suffix.lower() or ".bin"
    preserved_name = f"source-original{suffix}"
    (output / preserved_name).write_bytes(source_bytes)
    scene["source"]["preserved_file"] = preserved_name

    overlay = render_overlay(analysis_gray, scene)
    geometry_render = render_geometry_scene(analysis_gray.shape, scene)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode diagnostic overlay.")
    (output / "overlay.png").write_bytes(encoded.tobytes())
    render_ok, geometry_encoded = cv2.imencode(".png", geometry_render)
    if not render_ok:
        raise RuntimeError("Could not encode geometry render.")
    (output / "geometry-render.png").write_bytes(geometry_encoded.tobytes())

    # Exact pixel reconstruction by a signed photometric residual.
    # residual_signed = analysis_gray - geometry_render; PNG stores residual + 255
    # as uint16 so every value in [-255, 255] is represented exactly.
    residual_signed = analysis_gray.astype(np.int16) - geometry_render.astype(np.int16)
    residual_offset = (residual_signed.astype(np.int32) + 255).astype(np.uint16)
    residual_path = output / "residual-offset255.png"
    if not cv2.imwrite(str(residual_path), residual_offset):
        raise IOError("Could not write the uint16 photometric residual.")
    stored_residual = cv2.imread(str(residual_path), cv2.IMREAD_UNCHANGED)
    if stored_residual is None or stored_residual.dtype != np.uint16 or stored_residual.shape != analysis_gray.shape:
        raise IOError("The saved photometric residual failed its lossless read-back check.")

    reconstruction_int = (
        geometry_render.astype(np.int32) + stored_residual.astype(np.int32) - 255
    )
    if reconstruction_int.min() < 0 or reconstruction_int.max() > 255:
        raise IOError("The residual reconstruction exceeded the uint8 pixel range.")
    reconstructed = reconstruction_int.astype(np.uint8)
    exact_pixels = bool(np.array_equal(reconstructed, analysis_gray))
    if not exact_pixels:
        raise IOError("Geometry render plus residual does not reconstruct the analysis raster exactly.")
    reconstructed_path = output / "reconstructed-analysis.png"
    if not cv2.imwrite(str(reconstructed_path), reconstructed):
        raise IOError("Could not write the reconstructed analysis raster.")
    reconstructed_readback = cv2.imread(str(reconstructed_path), cv2.IMREAD_GRAYSCALE)
    if reconstructed_readback is None or not np.array_equal(reconstructed_readback, analysis_gray):
        raise IOError("The saved reconstructed image failed its pixel-equality check.")

    identity_reconstruction = {
        "schema": "scoretwin-lossless-residual-0.1",
        "analysis_raster_encoding": "uint8 grayscale",
        "residual_encoding": "uint16 PNG = analysis_pixel - geometry_render_pixel + 255",
        "reconstruction_equation": "analysis_pixel = geometry_render_pixel + residual_uint16 - 255",
        "source_analysis_raster_sha256": hashlib.sha256(analysis_gray.tobytes()).hexdigest(),
        "geometry_render_sha256": hashlib.sha256(geometry_render.tobytes()).hexdigest(),
        "reconstructed_raster_sha256": hashlib.sha256(reconstructed.tobytes()).hexdigest(),
        "residual_file_sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
        "pixel_count": int(analysis_gray.size),
        "mismatched_pixel_count": int(np.count_nonzero(reconstructed != analysis_gray)),
        "maximum_absolute_pixel_error": int(np.max(np.abs(reconstructed.astype(np.int16) - analysis_gray.astype(np.int16)))),
        "exact_pixel_reconstruction_verified": exact_pixels,
        "applies_to": "decoded grayscale analysis raster, including rectification if enabled",
        "does_not_claim": [
            "symbolic music correctness",
            "reconstruction of original color from grayscale",
            "physical ink-depth recovery",
            "lossy compression of the residual",
        ],
        "source_file_bytes_preserved_separately": True,
    }
    identity_report_path = output / "identity-reconstruction.json"
    identity_report_path.write_text(
        json.dumps(identity_reconstruction, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    scene["identity_reconstruction"] = identity_reconstruction

    analysis_raster_path = None
    capture_transform_path = None
    if rectification is not None:
        analysis_raster_path = output / "analysis-raster.png"
        if not cv2.imwrite(str(analysis_raster_path), analysis_gray):
            raise IOError("Could not write the analysis raster.")
        capture_transform_path = output / "capture-transform.json"
        capture_transform_path.write_text(json.dumps({
            "schema": "scoretwin-capture-rectification-0.2",
            "source_sha256": source_hash,
            "source_image_shape_px": [int(source_gray.shape[0]), int(source_gray.shape[1])],
            "analysis_image_shape_px": [int(analysis_gray.shape[0]), int(analysis_gray.shape[1])],
            "source_bytes_preserved": True,
            "rectification": rectification.to_dict(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    texture_ok, texture_encoded = cv2.imencode(".png", analysis_gray)
    if not texture_ok:
        raise RuntimeError("Could not encode the analysis raster for the 3D scene texture.")
    gltf_text = build_gltf(scene, background_png=texture_encoded.tobytes())
    gltf_validation = validate_embedded_gltf(gltf_text)
    gltf_path = output / "scene.gltf"
    gltf_path.write_text(gltf_text, encoding="utf-8")
    scene["exports"] = {
        "gltf_file": gltf_path.name,
        "gltf_validation": gltf_validation,
        "gltf_semantic_depth_is_physical": False,
    }
    (output / "scene.json").write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    musicxml = build_musicxml(scene)
    (output / "provisional.musicxml").write_text(musicxml, encoding="utf-8")

    # Verify the immutable source copy before reporting completion.
    copied_hash = hashlib.sha256((output / preserved_name).read_bytes()).hexdigest()
    if copied_hash != source_hash:
        raise IOError("Preserved source hash does not match the uploaded input.")
    return {
        "output_dir": str(output),
        "scene": scene,
        "musicxml_path": str(output / "provisional.musicxml"),
        "overlay_path": str(output / "overlay.png"),
        "geometry_render_path": str(output / "geometry-render.png"),
        "residual_path": str(residual_path),
        "reconstructed_raster_path": str(reconstructed_path),
        "identity_report_path": str(identity_report_path),
        "identity_reconstruction": identity_reconstruction,
        "gltf_path": str(gltf_path),
        "gltf_validation": gltf_validation,
        "analysis_raster_path": str(analysis_raster_path) if analysis_raster_path else None,
        "capture_transform_path": str(capture_transform_path) if capture_transform_path else None,
        "source_hash_verified": True,
        "source_sha256": source_hash,
    }
