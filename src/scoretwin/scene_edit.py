"""Explicit user corrections for ScoreTwin scene proposals.

These operations alter only the editable scene JSON. They never alter the immutable
source file and never claim that a previously unknown glyph was automatically read.
"""

from __future__ import annotations

import math
from copy import deepcopy


def promote_unknown_region(scene: dict, region_id: str) -> dict:
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise ValueError("Unrecognized ScoreTwin scene schema.")
    updated = deepcopy(scene)
    unknown_regions = updated.get("unknown_objects", [])
    region = next((item for item in unknown_regions if str(item.get("id")) == str(region_id)), None)
    if region is None:
        raise ValueError(f"Unknown region '{region_id}' does not exist in this scene.")

    try:
        staff_index = int(region["staff_index"])
        if staff_index < 0 or staff_index >= len(updated["staffs"]):
            raise ValueError("Unknown region refers to a staff index outside the current scene.")
        staff = updated["staffs"][staff_index]
        center = [float(region["center"][0]), float(region["center"][1])]
        bbox = [int(value) for value in region["bbox"]]
        spacing = float(staff["spacing"])
        bottom = float(staff.get("bottom", staff["lines"][-1]))
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Unknown region is missing valid staff, center, or box geometry.") from exc
    if (
        spacing <= 0
        or not all(math.isfinite(value) for value in center)
        or len(bbox) != 4
        or bbox[2] <= 0
        or bbox[3] <= 0
    ):
        raise ValueError("Unknown region has invalid finite geometry.")

    used_ids = {str(note.get("id")) for note in updated.get("notes", [])}
    used_ids.update(str(primitive.get("id")) for primitive in updated.get("primitives", []))
    sequence = 1
    while True:
        note_id = f"manual-note-{sequence:04d}"
        if note_id not in used_ids:
            break
        sequence += 1

    diatonic_step = int(round((bottom - center[1]) / (spacing / 2.0)))
    letters = ["C", "D", "E", "F", "G", "A", "B"]
    absolute_index = 2 + diatonic_step  # treble bottom line is E4
    octave_delta, letter_index = divmod(absolute_index, 7)
    pitch = {"step": letters[letter_index], "octave": 4 + octave_delta}

    uncertainty = [
        "key_signature", "accidental", "duration", "voice",
        "instrument_or_part", "measure_assignment", "stem_attachment",
    ]
    note = {
        "id": note_id,
        "primitive_id": note_id,
        "staff_index": staff_index,
        "bbox": bbox,
        "center": center,
        "pitch": pitch,
        "shape_class": "filled_head",
        "diatonic_step_from_bottom_line": diatonic_step,
        "pitch_confidence": None,
        "geometry_refinement": None,
        "duration": None,
        "voice": 1,
        "duration_hypothesis": {
            "candidate": "unresolved",
            "basis": "user_added_note; duration not inferred",
            "confidence_calibrated": False,
            "used_for_musicxml_duration": False,
        },
        "uncertainty": uncertainty,
        "user_added": True,
        "origin_unknown_region_id": str(region_id),
        "manually_edited_geometry": True,
    }

    primitives = updated.setdefault("primitives", [])
    primitive = next((item for item in primitives if str(item.get("id")) == str(region_id)), None)
    new_geometry = {
        "center_x_px": center[0],
        "center_y_px": center[1],
        "radius_x_px": bbox[2] / 2.0,
        "radius_y_px": bbox[3] / 2.0,
        "rotation_rad": 0.0,
        "bbox_px": list(bbox),
    }
    appearance = {
        "ink_density_estimate": None,
        "measurement": "user_promoted_unknown_region",
        "physical_absorbance_measured": False,
    }
    symbolic = {
        "pitch_proposal": dict(pitch),
        "duration": None,
        "duration_hypothesis": dict(note["duration_hypothesis"]),
        "user_added": True,
        "origin_unknown_region_id": str(region_id),
    }
    provenance = {
        "method": "user_promoted_unknown_region",
        "status": "user_added_proposal",
        "prior_class": region.get("class", "unknown"),
        "prior_reason": region.get("reason", "unclassified mark"),
        "automatic_recognition_claimed": False,
    }
    if primitive is None:
        primitives.append({
            "id": note_id,
            "kind": "notehead",
            "geometry": new_geometry,
            "appearance": appearance,
            "symbolic": symbolic,
            "provenance": provenance,
            "uncertainty": list(uncertainty),
        })
    else:
        primitive["id"] = note_id
        primitive["kind"] = "notehead"
        primitive["geometry"] = new_geometry
        primitive["appearance"] = appearance
        primitive["symbolic"] = symbolic
        primitive["provenance"] = provenance
        primitive["uncertainty"] = list(uncertainty)

    relations = updated.setdefault("relations", [])
    relations[:] = [
        relation for relation in relations
        if not (
            str(relation.get("source")) == str(region_id)
            and relation.get("type") == "candidate_region_in_staff"
        )
    ]
    system_id = f"staff-system-{staff_index:03d}"
    if any(str(item.get("id")) == system_id for item in primitives):
        relations.append({
            "type": "geometrically_assigned_to_staff",
            "source": note_id,
            "target": system_id,
        })
        relations.append({
            "type": "pitch_calibrated_by",
            "source": note_id,
            "target": f"staff{staff_index}-line4",
        })

    updated["notes"] = list(updated.get("notes", [])) + [note]
    updated["unknown_objects"] = [
        item for item in unknown_regions if str(item.get("id")) != str(region_id)
    ]
    updated["note_count"] = len(updated["notes"])
    updated["user_edit_history"] = list(updated.get("user_edit_history", [])) + [{
        "operation": "promote_unknown_region",
        "region_id": str(region_id),
        "created_note_id": note_id,
        "claim": "user-added proposal; automatic recognition not claimed",
    }]
    return updated
