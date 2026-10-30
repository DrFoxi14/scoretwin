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
from .background_and_finger_disentangler import detect_finger_and_thumb_occlusions, detect_page_substrate_quadrilateral, disentangle_background_and_fingers
from .curvilinear_and_world_idsr import extract_dynamic_staff_anchored_idsr
from .imaging import detect_barlines, detect_staff_systems, detect_stem_candidates, estimate_global_skew_degrees, extract_notehead_candidates, load_grayscale, load_pixel_array, rectify_global_skew, rectify_page_quadrilateral, refine_notehead_geometry, staff_step_and_pitch
from .gltf_export import build_gltf, validate_embedded_gltf
from .idsr_engine import compute_mdl_decomposition, estimate_illumination_and_optical_depth, refine_notehead_beer_lambert, render_beer_lambert_scene
from .manifold_traversal import traverse_manifold_voice_graph
from .musicxml import build_musicxml
from .notation import detect_accidentals, detect_beams_and_dots, detect_staff_systems_adaptive, group_staffs_into_systems, infer_chords_voices_and_durations
from .proposal import get_default_model, load_model, predict_patch


def _classify_candidates(candidates: list[dict], model: dict) -> list[dict]:
    classified = []
    for candidate in candidates:
        crop = np.asarray(candidate.pop("crop"), dtype=np.uint8)
        scores = predict_patch(model, crop)
        label = max(scores, key=scores.get)
        confidence = float(scores[label])
        if (candidate.get("corrosion_perforation") or {}).get("is_acid_cellulose_perforation"):
            label = "corroded_filled_head"
            confidence = 0.985
        elif confidence < 0.48:
            label = "unknown"

        contrast_std = float(np.std(crop)) if crop.size else 0.0
        laplacian_variance = (
            float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        )
        if (candidate.get("corrosion_perforation") or {}).get("is_acid_cellulose_perforation"):
            unknown_reason = None
        elif contrast_std < 12.0:
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


def analyze_array(
    gray: np.ndarray,
    model: dict | None = None,
    *,
    adaptive_staffs: bool = False,
    enable_extended_notation: bool = True,
    enable_beer_lambert: bool = True,
    promote_inferred_durations: bool = False,
    detect_fingers: bool = False,
    color_image: np.ndarray | None = None,
) -> dict:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("analyze_array expects a uint8 grayscale image.")
    if min(gray.shape) < 32:
        raise ValueError("Input image is too small to contain a score page.")
    started = time.perf_counter()
    estimated_skew = estimate_global_skew_degrees(gray)
    if adaptive_staffs:
        staffs, binary, staff_mask = detect_staff_systems_adaptive(gray)
    else:
        staffs, binary, staff_mask = detect_staff_systems(gray)

    finger_info = None
    if detect_fingers or color_image is not None:
        img_for_fingers = color_image if color_image is not None else gray
        avg_sp_init = float(np.median([s.spacing for s in staffs])) if staffs else 16.0
        raw_finger = detect_finger_and_thumb_occlusions(img_for_fingers, spacing=avg_sp_init, staff_systems=staffs)
        f_mask = raw_finger.get("finger_occlusion_mask")
        if raw_finger["has_finger_occlusions"] and f_mask is not None:
            # Suppress finger mask from binary ink so holding fingers never produce false notes or barlines
            binary = (binary.astype(bool) & (~f_mask.astype(bool))).astype(np.uint8) * 255
        # JSON-serializable dictionary for scene bundle
        finger_info = {
            "has_finger_occlusions": raw_finger["has_finger_occlusions"],
            "fingers_detected_count": raw_finger["fingers_detected_count"],
            "detected_fingers": raw_finger["detected_fingers"],
            "finger_mask_pixels_count": int(np.count_nonzero(f_mask)) if f_mask is not None else 0,
        }

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
    avg_sp = float(np.median([s.spacing for s in staffs])) if staffs else 16.0
    photometric_info = (
        estimate_illumination_and_optical_depth(gray, spacing=avg_sp)
        if enable_beer_lambert and classified
        else None
    )
    tau_clean = None
    if photometric_info is not None:
        tau_clean = photometric_info["tau"].copy()
        tau_clean[staff_mask > 0] *= 0.25
    for item in classified:
        if item["class"] not in ("filled_head", "hollow_head", "corroded_filled_head"):
            continue
        staff = staffs[int(item["staff_index"])]
        initial_center = [float(item["center"][0]), float(item["center"][1])]
        geom_kind = "filled_head" if item["class"] == "corroded_filled_head" else item["class"]
        fit = refine_notehead_geometry(
            gray, binary, staff_mask, initial_center, float(staff.spacing), geom_kind,
            precleaned_binary=clean_binary,
        )
        item["geometry_refinement"] = fit
        if fit["accepted"] and not (item.get("corrosion_perforation") or {}).get("is_acid_cellulose_perforation"):
            center_x, center_y = fit["center"]
            box_width, box_height = [int(v) for v in item["bbox"][2:4]]
            box_x = max(0, min(image_width - box_width, int(round(center_x - box_width / 2.0))))
            box_y = max(0, min(image_height - box_height, int(round(center_y - box_height / 2.0))))
            item["center"] = [float(center_x), float(center_y)]
            item["bbox"] = [box_x, box_y, box_width, box_height]
        if tau_clean is not None:
            bl_fit = refine_notehead_beer_lambert(
                tau_clean,
                item["center"],
                float(staff.spacing),
                kind="filled_head" if item["class"] == "corroded_filled_head" else item["class"],
            )
            item["beer_lambert_refinement"] = bl_fit

    staff_objects = []
    for staff in staffs:
        barlines = detect_barlines(binary, staff)
        staff_objects.append({**staff.to_dict(), "barlines": barlines})

    systems = group_staffs_into_systems(gray, binary, staffs) if enable_extended_notation else []
    for idx, s_obj in enumerate(staff_objects):
        if systems and idx < len(systems):
            sys_info = systems[idx]
            s_obj["system_index"] = sys_info.get("system_index", idx)
            s_obj["staff_number"] = sys_info.get("staff_number_in_system", 1)
            s_obj["part_id"] = sys_info.get("part_id", f"P{idx + 1}")
            s_obj["part_name"] = sys_info.get("part_name", sys_info.get("voice_name"))
            s_obj["clef"] = sys_info.get("inferred_clef")
            s_obj["voice_name"] = sys_info.get("voice_name")
            s_obj["is_satb_choral"] = sys_info.get("is_satb_choral", False)

    # Reconcile barlines within each multi-staff system to enforce consistent measure bounds
    if systems:
        system_indices = sorted({s.get("system_index", 0) for s in staff_objects})
        for sys_idx in system_indices:
            sys_staves = [s for s in staff_objects if s.get("system_index", 0) == sys_idx]
            if len(sys_staves) > 1:
                all_bars = []
                for s in sys_staves:
                    all_bars.extend(s.get("barlines", []))
                if all_bars:
                    all_bars.sort()
                    med_sp = float(np.median([s.get("spacing", 16.0) for s in sys_staves]))
                    clusters: list[list[int]] = []
                    for b in all_bars:
                        if not clusters or b - clusters[-1][-1] > 1.8 * med_sp:
                            clusters.append([b])
                        else:
                            clusters[-1].append(b)
                    consensus_bars = [int(round(float(np.median(c)))) for c in clusters]
                    for s in sys_staves:
                        s["barlines"] = list(consensus_bars)

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
        if item["class"] in ("filled_head", "hollow_head", "corroded_filled_head"):
            assigned_clef = staff_objects[staff_index].get("clef") if staff_index < len(staff_objects) else None
            step, letter, octave = staff_step_and_pitch(cy, staff, clef=assigned_clef)
            is_corroded = (item["class"] == "corroded_filled_head") or bool(
                (item.get("corrosion_perforation") or {}).get("is_acid_cellulose_perforation")
            )
            notes.append({
                "id": item["id"],
                "staff_index": staff_index,
                "bbox": item["bbox"],
                "center": item["center"],
                "pitch": {"step": letter, "octave": octave},
                "shape_class": "corroded_filled_head" if is_corroded else item["class"],
                "diatonic_step_from_bottom_line": step,
                "pitch_confidence": item["confidence"],
                "geometry_refinement": item.get("geometry_refinement"),
                "beer_lambert_refinement": item.get("beer_lambert_refinement"),
                "handwritten_ductus": item.get("handwritten_ductus"),
                "cluster_type": item.get("cluster_type"),
                "corrosion_perforation": item.get("corrosion_perforation"),
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
    valid_note_ids = {str(n["id"]) for n in notes}
    for stem in stem_candidates:
        stem["associated_note_ids"] = [
            nid for nid in stem.get("associated_note_ids", []) if str(nid) in valid_note_ids
        ]
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

    # Extended notation inference: accidentals, beams, augmentation dots, chords, voices, systems
    accidentals: list[dict] = []
    beams: list[dict] = []
    dots: list[dict] = []
    structure_info: dict = {"chord_count": 0, "chord_relations": [], "voice_relations": []}
    systems: list[dict] = []
    if enable_extended_notation and staffs:
        accidentals = detect_accidentals(gray, binary, staff_mask, staffs, notes)
        beams, dots = detect_beams_and_dots(binary, staff_mask, staffs, notes, stem_candidates)
        structure_info = infer_chords_voices_and_durations(
            staffs,
            notes,
            stem_candidates,
            beams,
            {int(s["index"]): s["barlines"] for s in staff_objects},
        )
        if not systems:
            systems = group_staffs_into_systems(gray, binary, staffs)
        from .historical_clefs import disambiguate_clef_by_syntactic_constraints
        for s_idx, sys_entry in enumerate(systems):
            staff_notes_subset = [n for n in notes if int(n["staff_index"]) == s_idx]
            if staff_notes_subset:
                refined_clef_info = disambiguate_clef_by_syntactic_constraints(staff_notes_subset)
                sys_entry["clef_syntactic_validation"] = {
                    "ledger_penalty": refined_clef_info["evaluations"][0]["ledger_penalty"] if refined_clef_info["evaluations"] else 0,
                    "confidence": refined_clef_info["confidence"],
                }
        for note in notes:
            inferred_type = note.get("inferred_note_type", "quarter")
            type_to_q = {"whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "16th": 0.25}
            inferred_q = type_to_q.get(inferred_type, 1.0) * (1.5 if int(note.get("dots", 0)) > 0 else 1.0)
            note["inferred_duration_type"] = inferred_type
            note["inferred_duration_quarters"] = inferred_q
            if promote_inferred_durations:
                note["duration"] = max(1, int(round(float(inferred_q) * 2)))
                note["duration_hypothesis"]["used_for_musicxml_duration"] = True
                if note.get("inferred_voice") is not None:
                    note["voice"] = int(note["inferred_voice"])
                if note.get("inferred_chord_id") is not None:
                    note["chord_id"] = str(note["inferred_chord_id"])
            else:
                note["duration"] = None
                note["duration_hypothesis"]["used_for_musicxml_duration"] = False
            # Remove accidental/voice uncertainty when explicitly resolved
            if note.get("accidental") is not None and "accidental" in note["uncertainty"]:
                note["uncertainty"].remove("accidental")

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

    valid_note_ids = {str(n["id"]) for n in notes}
    for stem in stem_candidates:
        stem["associated_note_ids"] = [
            nid for nid in stem.get("associated_note_ids", []) if str(nid) in valid_note_ids
        ]
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

    for beam in beams:
        beam_id = str(beam["id"])
        geom = beam["geometry"]
        x0 = float(geom["x0_px"])
        y0 = float(geom["y0_px"])
        x1 = float(geom["x1_px"])
        y1 = float(geom["y1_px"])
        thick = float(geom.get("thickness_px_estimate", 3.0))
        primitives.append({
            "id": beam_id,
            "kind": "beam",
            "staff_index": int(beam["staff_index"]),
            "geometry": {
                "x0_px": x0,
                "y0_px": y0,
                "x1_px": x1,
                "y1_px": y1,
                "bbox_px": [min(x0, x1), min(y0, y1) - thick / 2.0, abs(x1 - x0), abs(y1 - y0) + thick],
                "thickness_px_estimate": thick,
                "beam_count": int(beam.get("beam_count", 1)),
            },
            "provenance": {"method": "horizontal_beam_stroke_extractor", "status": "proposal"},
            "uncertainty": ["beam_slope", "beam_multiplicity"],
        })
        for note_id in beam.get("note_ids", []):
            relations.append({"type": "beam_of", "source": beam_id, "target": str(note_id)})

    for acc in accidentals:
        acc_id = str(acc["id"])
        x, y, aw, ah = [float(v) for v in acc["bbox"]]
        primitives.append({
            "id": acc_id,
            "kind": "accidental",
            "staff_index": int(acc["staff_index"]),
            "geometry": {
                "bbox_px": [x, y, aw, ah],
                "center_x_px": float(acc["center"][0]),
                "center_y_px": float(acc["center"][1]),
            },
            "symbolic": {
                "type": acc["accidental_type"],
                "alter": int(acc["alter"]),
            },
            "provenance": {
                "method": "structural_accidental_classifier",
                "classifier_confidence": float(acc["confidence"]),
                "status": "proposal",
            },
            "uncertainty": ["accidental_scope"],
        })
        if acc.get("target_note_id"):
            relations.append({
                "type": "accidental_of",
                "source": acc_id,
                "target": str(acc["target_note_id"]),
            })

    for dot in dots:
        dot_id = str(dot["id"])
        primitives.append({
            "id": dot_id,
            "kind": "augmentation_dot",
            "staff_index": int(dot["staff_index"]),
            "geometry": {
                "center_x_px": float(dot["center"][0]),
                "center_y_px": float(dot["center"][1]),
                "radius_px": max(1.2, float(dot["bbox"][2]) / 2.0),
                "bbox_px": [float(v) for v in dot["bbox"]],
            },
            "provenance": {"method": "post_notehead_dot_detector", "status": "proposal"},
            "uncertainty": ["staccato_vs_augmentation_dot"],
        })
        if dot.get("target_note_id"):
            relations.append({
                "type": "dot_of",
                "source": dot_id,
                "target": str(dot["target_note_id"]),
            })

    relations.extend(structure_info.get("chord_relations", []))
    relations.extend(structure_info.get("voice_relations", []))

    for note in notes:
        x, y, box_width, box_height = [float(v) for v in note["bbox"]]
        center_x, center_y = [float(v) for v in note["center"]]
        crop = gray[
            max(0, int(y)):min(gray.shape[0], int(y + box_height)),
            max(0, int(x)):min(gray.shape[1], int(x + box_width)),
        ]
        ink_density = float(1.0 - np.mean(crop) / 255.0) if crop.size else 0.0
        bl_info = note.get("beer_lambert_refinement") or {}
        ductus = note.get("handwritten_ductus") or {}
        ductus_angle = float(ductus.get("pen_stroke_angle_deg", 0.0))
        is_corroded = (note.get("shape_class") == "corroded_filled_head") or bool(
            (note.get("corrosion_perforation") or {}).get("is_acid_cellulose_perforation")
        )
        primitive = {
            "id": note["id"],
            "kind": "notehead",
            "geometry": {
                "center_x_px": center_x,
                "center_y_px": center_y,
                "radius_x_px": box_width / 2.0,
                "radius_y_px": box_height / 2.0,
                "rotation_rad": float(np.radians(ductus_angle)) if ductus_angle else 0.0,
                "bbox_px": [x, y, box_width, box_height],
                "cluster_type": note.get("cluster_type"),
                "corrosion_perforation": note.get("corrosion_perforation"),
            },
            "appearance": {
                "ink_density_estimate": ink_density,
                "optical_absorbance": bl_info.get("optical_absorbance", 1.45),
                "measurement": "beer_lambert_log_optical_depth" if bl_info else "one_minus_mean_grayscale_inside_bbox",
                "physical_absorbance_measured": bool(bl_info),
                "handwritten_ductus": ductus if ductus else None,
                "corrosion_perforation": note.get("corrosion_perforation"),
            },
            "symbolic": {
                "pitch_proposal": dict(note["pitch"]),
                "shape_class": "corroded_filled_head" if is_corroded else note.get("shape_class", "filled_head"),
                "intended_symbolic_notehead": "filled_head" if is_corroded else note.get("shape_class", "filled_head"),
                "accidental": note.get("accidental"),
                "voice": note.get("inferred_voice", 1),
                "chord_id": note.get("inferred_chord_id"),
                "duration": note.get("duration"),
                "inferred_duration_type": note.get("inferred_duration_type"),
                "stem_ids": list(note.get("stem_ids", [])),
                "duration_hypothesis": dict(note.get("duration_hypothesis", {})),
            },
            "provenance": {
                "method": "component_geometry_plus_synthetic_patch_classifier_plus_beer_lambert_fit",
                "classifier_confidence": float(note["pitch_confidence"]),
                "geometry_fit_score": (
                    note.get("geometry_refinement") or {}
                ).get("fit_score"),
                "geometry_fit_accepted": bool(
                    (note.get("geometry_refinement") or {}).get("accepted", False)
                ),
                "beer_lambert_accepted": bool(bl_info.get("accepted", False)),
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

    idsr_dynamic_layer = None
    if enable_extended_notation and staffs:
        idsr_dynamic_layer = extract_dynamic_staff_anchored_idsr(
            gray,
            staff_objects,
            staff_mask,
            notes,
            unknown_objects,
            dpi=300.0,
        )
        if finger_info is not None:
            idsr_dynamic_layer["finger_occlusions"] = finger_info

        # Filter note proposals that sit inside detected lyric bands without stems
        if idsr_dynamic_layer.get("dynamic_lyrics_detected"):
            lyric_intervals = [
                (l["staff_index"], l["bbox_px"][1] - 4, l["bbox_px"][1] + l["bbox_px"][3] + 4)
                for l in idsr_dynamic_layer["dynamic_lyrics_detected"]
            ]
            suppressed_ids = set()
            for n in notes:
                cy = float(n["center"][1])
                s_idx = int(n.get("staff_index", 0))
                has_stem = bool(n.get("stem_ids"))
                if not has_stem and any(st == s_idx and y0 <= cy <= y1 for (st, y0, y1) in lyric_intervals):
                    suppressed_ids.add(str(n["id"]))
            if suppressed_ids:
                notes = [n for n in notes if str(n["id"]) not in suppressed_ids]
                primitives = [p for p in primitives if str(p["id"]) not in suppressed_ids]
        for l_idx, lyr in enumerate(idsr_dynamic_layer["dynamic_lyrics_detected"]):
            primitives.append({
                "id": f"dynamic-lyric-{l_idx:04d}",
                "kind": lyr["kind"],
                "staff_index": lyr["staff_index"],
                "geometry": {
                    "bbox_px": lyr["bbox_px"],
                    "center_x_px": lyr["center_px"][0],
                    "center_y_px": lyr["center_px"][1],
                    "exact_x_from_first_bar_cm": lyr["exact_x_from_first_bar_cm"],
                    "exact_y_below_staff_cm": lyr["exact_y_below_staff_cm"],
                    "normalized_or_recentered": False,
                },
                "provenance": {"method": "dynamic_staff_anchored_lyric_extractor", "status": "resolved"},
                "uncertainty": [],
            })
        for d_idx, dyn in enumerate(idsr_dynamic_layer["dynamic_dynamics_detected"]):
            primitives.append({
                "id": f"dynamic-marking-{d_idx:04d}",
                "kind": dyn["kind"],
                "staff_index": dyn["staff_index"],
                "geometry": {
                    "bbox_px": dyn["bbox_px"],
                    "exact_x_from_first_bar_cm": dyn["exact_x_from_first_bar_cm"],
                    "exact_width_cm": dyn["exact_width_cm"],
                },
                "provenance": {"method": "dynamic_staff_anchored_dynamics_extractor", "status": "resolved"},
                "uncertainty": [],
            })
        for v_idx, vlt in enumerate(idsr_dynamic_layer["dynamic_volta_brackets_detected"]):
            primitives.append({
                "id": f"dynamic-volta-{v_idx:04d}",
                "kind": "volta_bracket",
                "staff_index": vlt["staff_index"],
                "geometry": {
                    "volta_number": vlt["volta_number"],
                    "ending_type": vlt["ending_type"],
                    "exact_x_start_cm": vlt["exact_x_start_cm"],
                    "exact_x_end_cm": vlt["exact_x_end_cm"],
                    "bbox_px": vlt.get("bbox_px"),
                    "vx0": vlt.get("vx0"),
                    "vx1": vlt.get("vx1"),
                    "roof_y": vlt.get("roof_y"),
                    "lh_px": vlt.get("lh_px"),
                    "rh_px": vlt.get("rh_px"),
                },
                "provenance": {"method": "dynamic_staff_anchored_volta_extractor", "status": "resolved"},
                "uncertainty": [],
            })

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
                "status": "investigated_in_idsr_dynamic_layer" if idsr_dynamic_layer else "unresolved",
            },
            "uncertainty": ["symbol_identity", "relation_to_score"],
        })
        system_id = system_ids.get(int(item["staff_index"]))
        if system_id:
            relations.append({"type": "candidate_region_in_staff", "source": item["id"], "target": system_id})

    manifold_voice_traversal = None
    if enable_extended_notation and staffs:
        manifold_voice_traversal = traverse_manifold_voice_graph(
            gray,
            staff_objects,
            notes,
            primitives,
            relations,
            unknown_objects,
            binary=binary,
            staff_mask=staff_mask,
            dpi=300.0,
        )
        # Append recovered contextual marks to primitives for explicit rendering
        for rec in manifold_voice_traversal.get("recovered_contextual_marks", []):
            primitives.append({
                "id": rec["id"],
                "kind": rec["classification"],
                "geometry": {
                    "bbox_px": rec["bbox_px"],
                    "exact_x_from_first_bar_cm": rec["exact_x_from_first_bar_cm"],
                    "exact_y_from_staff_top_cm": rec["exact_y_from_staff_top_cm"],
                },
                "provenance": {"method": "manifold_voice_traversal", "status": "resolved"},
                "uncertainty": [],
            })

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
        "Pitch proposals assume treble/bass heuristic clef and local accidental proximity; key signatures are not yet inferred.",
        "Durations, voice assignments, instrument mapping and full score structure remain proposals marked for review in MusicXML.",
        "The synthetic model's confidence is not calibrated confidence on real manuscripts or photographs.",
        "Geometry-fit scores are local template-correlation and Beer-Lambert values; rejected fits leave the original proposal unchanged.",
    ])
    if finger_info and finger_info.get("has_finger_occlusions"):
        warnings.append(
            f"Performer fingers/thumbs detected ({finger_info['fingers_detected_count']} holding borders); zero-hallucination mask applied."
        )
    scene_dict = {
        "schema": "scoretwin-scene-0.1",
        "image": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
        "capture_diagnostics": {
            "estimated_staffline_angle_deg": estimated_skew,
            "rotation_rectified": False,
            "perspective_rectified": False,
        },
        "staffs": staff_objects,
        "systems": systems,
        "primitives": primitives,
        "relations": relations,
        "notes": notes,
        "accidentals": accidentals,
        "beams": beams,
        "dots": dots,
        "chord_count": structure_info.get("chord_count", 0),
        "unknown_objects": unknown_objects,
        "idsr_dynamic_layer": idsr_dynamic_layer,
        "finger_occlusions": finger_info,
        "manifold_voice_traversal": manifold_voice_traversal,
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
                "geometry_refinement": "bounded-local-template-correlation-plus-beer-lambert",
            },
        },
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }
    if enable_beer_lambert and staffs:
        i0_field = photometric_info["i0"] if photometric_info is not None else None
        sigma_px = photometric_info["sensor_noise_sigma_px"] if photometric_info is not None else 2.0
        bl_render = render_beer_lambert_scene(gray.shape, scene_dict, i0_field=i0_field)
        scene_dict["mdl_decomposition"] = compute_mdl_decomposition(
            gray,
            scene_dict,
            hat_i_float64=bl_render["hat_i_float64"],
            sensor_noise_sigma_px=sigma_px,
        )
    return scene_dict


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
        kind = primitive.get("kind")
        geometry = primitive.get("geometry", {})
        if kind == "stem":
            x0 = int(round(float(geometry.get("x0_px", 0)) * scale))
            y0 = int(round(float(geometry.get("y0_px", 0)) * scale))
            x1 = int(round(float(geometry.get("x1_px", 0)) * scale))
            y1 = int(round(float(geometry.get("y1_px", 0)) * scale))
            cv2.line(image, (x0, y0), (x1, y1), 20, max(1, int(round(scale))), cv2.LINE_AA)
        elif kind == "beam":
            x0 = int(round(float(geometry.get("x0_px", 0)) * scale))
            y0 = int(round(float(geometry.get("y0_px", 0)) * scale))
            x1 = int(round(float(geometry.get("x1_px", 0)) * scale))
            y1 = int(round(float(geometry.get("y1_px", 0)) * scale))
            thick = max(2, int(round(float(geometry.get("thickness_px_estimate", 3.0)) * scale)))
            cv2.line(image, (x0, y0), (x1, y1), 20, thick, cv2.LINE_AA)
        elif kind == "augmentation_dot":
            cx = int(round(float(geometry.get("center_x_px", 0)) * scale))
            cy = int(round(float(geometry.get("center_y_px", 0)) * scale))
            r = max(1, int(round(float(geometry.get("radius_px", 2.0)) * scale)))
            cv2.circle(image, (cx, cy), r, 20, -1, cv2.LINE_AA)
        elif kind == "accidental":
            bbox = geometry.get("bbox_px", [0, 0, 4, 8])
            x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
            cv2.rectangle(image, (x, y), (x + max(1, bw), y + max(1, bh)), 40, 1, cv2.LINE_AA)
        elif kind in ("lyric_syllable", "lyric_hyphen"):
            bbox = geometry.get("bbox_px")
            if bbox:
                x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
                if kind == "lyric_hyphen":
                    cy = y + bh // 2
                    cv2.line(image, (x, cy), (x + max(1, bw), cy), 30, max(1, int(round(scale))), cv2.LINE_AA)
                else:
                    cv2.rectangle(image, (x, y), (x + max(1, bw), y + max(1, bh)), 40, -1)
        elif kind in ("dynamic_glyph_mf", "dynamic_glyph_f_or_p", "dynamic_marking"):
            bbox = geometry.get("bbox_px")
            if bbox:
                x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
                cv2.rectangle(image, (x, y), (x + max(1, bw), y + max(1, bh)), 40, max(1, int(round(scale))), cv2.LINE_AA)
        elif kind in ("hairpin_crescendo", "hairpin_diminuendo"):
            bbox = geometry.get("bbox_px")
            if bbox:
                x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
                mid_y = y + bh // 2
                if kind == "hairpin_crescendo":
                    cv2.line(image, (x, mid_y), (x + bw, y), 30, max(1, int(round(scale))), cv2.LINE_AA)
                    cv2.line(image, (x, mid_y), (x + bw, y + bh), 30, max(1, int(round(scale))), cv2.LINE_AA)
                else:
                    cv2.line(image, (x, y), (x + bw, mid_y), 30, max(1, int(round(scale))), cv2.LINE_AA)
                    cv2.line(image, (x, y + bh), (x + bw, mid_y), 30, max(1, int(round(scale))), cv2.LINE_AA)
        elif kind == "volta_bracket":
            vx0 = geometry.get("vx0")
            vx1 = geometry.get("vx1")
            roof_y = geometry.get("roof_y")
            lh_px = geometry.get("lh_px", 10)
            rh_px = geometry.get("rh_px", 10)
            if vx0 is not None and vx1 is not None and roof_y is not None:
                x0 = int(round(float(vx0) * scale))
                x1 = int(round(float(vx1) * scale))
                ry = int(round(float(roof_y) * scale))
                cv2.line(image, (x0, ry), (x1, ry), 25, max(1, int(round(scale))), cv2.LINE_AA)
                cv2.line(image, (x0, ry), (x0, ry + int(round(float(lh_px) * scale))), 25, max(1, int(round(scale))), cv2.LINE_AA)
                if geometry.get("ending_type") == "start_stop":
                    cv2.line(image, (x1, ry), (x1, ry + int(round(float(rh_px) * scale))), 25, max(1, int(round(scale))), cv2.LINE_AA)
        elif kind in ("annotation_circle", "annotation_arrow", "annotation_text"):
            bbox = geometry.get("bbox_px")
            if bbox:
                x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
                cv2.rectangle(image, (x, y), (x + max(1, bw), y + max(1, bh)), 80, max(1, int(round(scale))), cv2.LINE_AA)
        elif kind in ("augmentation_dot_recovered", "staccato_dot_recovered"):
            bbox = geometry.get("bbox_px")
            if bbox:
                x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
                cx = x + max(1, bw // 2)
                cy = y + max(1, bh // 2)
                r = max(1, int(round(max(bw, bh) / 2.0 * scale)))
                cv2.circle(image, (cx, cy), r, 20, -1, cv2.LINE_AA)
        elif kind in ("accidental_stroke_recovered", "local_contextual_ink_trace"):
            bbox = geometry.get("bbox_px")
            if bbox:
                x, y, bw, bh = [int(round(float(v) * scale)) for v in bbox]
                cv2.rectangle(image, (x, y), (x + max(1, bw), y + max(1, bh)), 40, max(1, int(round(scale))), cv2.LINE_AA)

    for note in scene.get("notes", []):
        cx, cy = [float(value) * scale for value in note["center"]]
        bbox = [float(value) * scale for value in note["bbox"]]
        rx = max(2, int(round(bbox[2] / 2.0)))
        ry = max(2, int(round(bbox[3] / 2.0)))
        fit = note.get("geometry_refinement") or {}
        ductus = note.get("handwritten_ductus") or {}
        angle = float(fit.get("rotation_deg", 0.0)) if fit.get("accepted") else float(ductus.get("pen_stroke_angle_deg", 0.0))
        class_name = note.get("shape_class", "filled_head")
        thickness = -1 if class_name in ("filled_head", "corroded_filled_head") else max(1, int(round(ry * 0.35)))
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

    dyn = scene.get("idsr_dynamic_layer") or {}
    for lyr in dyn.get("dynamic_lyrics_detected", []):
        lx, ly, lw, lh = lyr["bbox_px"]
        cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (220, 180, 50), 1, cv2.LINE_AA)
    for dm in dyn.get("dynamic_dynamics_detected", []):
        dx, dy, dw, dh = dm["bbox_px"]
        cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (50, 180, 240), 1, cv2.LINE_AA)
    for vlt in dyn.get("dynamic_volta_brackets_detected", []):
        vx = vlt.get("bbox_px")
        if vx:
            cv2.rectangle(overlay, (vx[0], vx[1]), (vx[0] + vx[2], vx[1] + vx[3]), (200, 80, 180), 1, cv2.LINE_AA)
    return overlay


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    page: int = 0,
    model_path: str | Path | None = None,
    rectify_skew: bool = False,
    max_skew_degrees: float = 12.0,
    rectify_corners: list | None = None,
    auto_substrate: bool = False,
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
    source_pixels = load_pixel_array(source, page=page)
    if source_pixels.shape[:2] != source_gray.shape:
        raise ValueError("Decoded color/grayscale source frames have different dimensions.")
    if rectify_skew and rectify_corners is not None:
        raise ValueError("Choose either global-skew rectification or page-corner rectification, not both.")
    if rectify_corners is not None:
        rectification = rectify_page_quadrilateral(source_gray, rectify_corners)
    elif rectify_skew:
        rectification = rectify_global_skew(source_gray, max_abs_skew_degrees=max_skew_degrees)
    elif auto_substrate:
        sub_res = detect_page_substrate_quadrilateral(source_pixels)
        if not sub_res.get("already_cropped_to_page", True):
            rectification = rectify_page_quadrilateral(source_gray, sub_res["corners_ordered_xy"])
        else:
            rectification = None
    else:
        rectification = None
    analysis_gray = rectification.image if rectification is not None else source_gray
    if rectification is not None and rectification.applied:
        matrix = np.asarray(rectification.source_to_rectified, dtype=np.float64)
        output_size = (int(analysis_gray.shape[1]), int(analysis_gray.shape[0]))
        border_value = 255 if source_pixels.ndim == 2 else (255, 255, 255)
        analysis_pixels = cv2.warpPerspective(
            source_pixels,
            matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
    else:
        analysis_pixels = source_pixels.copy()
    if analysis_pixels.shape[:2] != analysis_gray.shape:
        raise RuntimeError("Color/grayscale analysis rasters do not share one coordinate frame.")
    model = load_model(model_path) if model_path else get_default_model()
    scene = analyze_array(
        analysis_gray,
        model=model,
        adaptive_staffs=True,
        enable_extended_notation=True,
        enable_beer_lambert=True,
        detect_fingers=True,
        color_image=analysis_pixels if analysis_pixels.ndim == 3 else None,
    )
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

    # Exact pixel reconstruction by a signed residual. For color source images the
    # structural render is lifted to BGR, so the residual retains chromatic information
    # that the grayscale OMR detector does not use.
    geometry_pixels = (
        geometry_render
        if analysis_pixels.ndim == 2
        else cv2.cvtColor(geometry_render, cv2.COLOR_GRAY2BGR)
    )
    if geometry_pixels.shape != analysis_pixels.shape:
        raise RuntimeError("Geometry render and analysis pixels do not share channel dimensions.")
    residual_signed = analysis_pixels.astype(np.int16) - geometry_pixels.astype(np.int16)
    residual_offset = (residual_signed.astype(np.int32) + 255).astype(np.uint16)
    residual_path = output / "residual-offset255.png"
    if not cv2.imwrite(str(residual_path), residual_offset):
        raise IOError("Could not write the uint16 photometric residual.")
    stored_residual = cv2.imread(str(residual_path), cv2.IMREAD_UNCHANGED)
    if stored_residual is None or stored_residual.dtype != np.uint16 or stored_residual.shape != analysis_pixels.shape:
        raise IOError("The saved photometric residual failed its lossless read-back check.")

    reconstruction_int = geometry_pixels.astype(np.int32) + stored_residual.astype(np.int32) - 255
    if reconstruction_int.min() < 0 or reconstruction_int.max() > 255:
        raise IOError("The residual reconstruction exceeded the uint8 pixel range.")
    reconstructed = reconstruction_int.astype(np.uint8)
    exact_pixels = bool(np.array_equal(reconstructed, analysis_pixels))
    if not exact_pixels:
        raise IOError("Geometry render plus residual does not reconstruct the analysis raster exactly.")
    reconstructed_path = output / "reconstructed-analysis.png"
    if not cv2.imwrite(str(reconstructed_path), reconstructed):
        raise IOError("Could not write the reconstructed analysis raster.")
    reconstructed_readback = cv2.imread(str(reconstructed_path), cv2.IMREAD_UNCHANGED)
    if reconstructed_readback is None or not np.array_equal(reconstructed_readback, analysis_pixels):
        raise IOError("The saved reconstructed image failed its pixel-equality check.")

    analysis_pixels_path = output / "analysis-raster-pixels.png"
    if not cv2.imwrite(str(analysis_pixels_path), analysis_pixels):
        raise IOError("Could not write the channel-preserving analysis raster.")

    pixel_mismatches = (
        np.any(reconstructed != analysis_pixels, axis=2)
        if analysis_pixels.ndim == 3
        else reconstructed != analysis_pixels
    )
    absolute_error = np.abs(reconstructed.astype(np.int16) - analysis_pixels.astype(np.int16))
    identity_reconstruction = {
        "schema": "scoretwin-lossless-residual-0.1",
        "analysis_raster_encoding": (
            "uint8 grayscale" if analysis_pixels.ndim == 2
            else f"uint8 {analysis_pixels.shape[2]}-channel BGR analysis raster"
        ),
        "residual_encoding": "uint16 PNG = analysis_channel_value - geometry_channel_value + 255",
        "reconstruction_equation": "analysis_channel_value = geometry_channel_value + residual_uint16 - 255",
        "source_analysis_raster_sha256": hashlib.sha256(analysis_pixels.tobytes()).hexdigest(),
        "geometry_render_sha256": hashlib.sha256(geometry_pixels.tobytes()).hexdigest(),
        "reconstructed_raster_sha256": hashlib.sha256(reconstructed.tobytes()).hexdigest(),
        "residual_file_sha256": hashlib.sha256(residual_path.read_bytes()).hexdigest(),
        "pixel_count": int(analysis_pixels.shape[0] * analysis_pixels.shape[1]),
        "channel_value_count": int(analysis_pixels.size),
        "mismatched_pixel_count": int(np.count_nonzero(pixel_mismatches)),
        "maximum_absolute_pixel_error": int(absolute_error.max()),
        "exact_pixel_reconstruction_verified": exact_pixels,
        "applies_to": (
            "decoded grayscale analysis raster" if analysis_pixels.ndim == 2
            else "decoded BGR color analysis raster"
        ) + ", including resampling if rectification is enabled",
        "does_not_claim": [
            "symbolic music correctness",
            "recovery of original compressed source bytes from decoded pixels",
            "recovery of discarded alpha after white compositing",
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

    texture_ok, texture_encoded = cv2.imencode(".png", analysis_pixels)
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
    musicxml = build_musicxml(scene, musescore_compat=True)
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
        "analysis_pixels_path": str(analysis_pixels_path),
        "capture_transform_path": str(capture_transform_path) if capture_transform_path else None,
        "source_hash_verified": True,
        "source_sha256": source_hash,
    }
