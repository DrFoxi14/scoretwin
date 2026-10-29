"""IDSR first vertical slice: analyze, preserve, visualize and export."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .imaging import detect_barlines, detect_staff_systems, extract_notehead_candidates, load_grayscale, refine_notehead_geometry, staff_step_and_pitch
from .musicxml import build_musicxml
from .proposal import get_default_model, load_model, predict_patch


def _classify_candidates(candidates: list[dict], model: dict) -> list[dict]:
    classified = []
    for candidate in candidates:
        scores = predict_patch(model, candidate.pop("crop"))
        label = max(scores, key=scores.get)
        confidence = float(scores[label])
        if confidence < 0.48:
            label = "unknown"
        item = {**candidate, "class": label, "confidence": round(confidence, 4),
                "class_probabilities": {key: round(float(value), 4) for key, value in scores.items()}}
        classified.append(item)
    return classified


def analyze_array(gray: np.ndarray, model: dict | None = None) -> dict:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("analyze_array expects a uint8 grayscale image.")
    if min(gray.shape) < 32:
        raise ValueError("Input image is too small to contain a score page.")
    started = time.perf_counter()
    staffs, binary, staff_mask = detect_staff_systems(gray)
    candidates = extract_notehead_candidates(gray, binary, staff_mask, staffs)
    classified = _classify_candidates(candidates, model or get_default_model())

    # Apply bounded local geometry refinement only to candidates already classified
    # as noteheads. This is analysis-by-synthesis at proposal scale, not a global OMR.
    image_height, image_width = gray.shape
    for item in classified:
        if item["class"] not in ("filled_head", "hollow_head"):
            continue
        staff = staffs[int(item["staff_index"])]
        initial_center = [float(item["center"][0]), float(item["center"][1])]
        fit = refine_notehead_geometry(
            gray, binary, staff_mask, initial_center, float(staff.spacing), item["class"]
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
                "reason": "Candidate was not classified as either modeled notehead type; it may be non-note ink or an unresolved symbol.",
            })

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
        "staffs": staff_objects,
        "primitives": primitives,
        "relations": relations,
        "notes": notes,
        "unknown_objects": unknown_objects,
        "proposal_count": len(classified),
        "note_count": len(notes),
        "warnings": warnings,
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


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
) -> dict:
    """Generate a self-contained analysis bundle and preserve input bytes unchanged."""
    source = Path(input_path)
    output = Path(output_dir)
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    gray = load_grayscale(source, page=page)
    model = load_model(model_path) if model_path else get_default_model()
    scene = analyze_array(gray, model=model)
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

    overlay = render_overlay(gray, scene)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("Could not encode diagnostic overlay.")
    (output / "overlay.png").write_bytes(encoded.tobytes())

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
        "source_hash_verified": True,
        "source_sha256": source_hash,
    }
