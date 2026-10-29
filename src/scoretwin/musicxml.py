"""Conservative MusicXML serialization for editable scene annotations."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import defaultdict


_STEP_INDEX = {letter: index for index, letter in enumerate("CDEFGAB")}
_ACCIDENTALS = {
    "sharp", "flat", "natural", "double-sharp", "sharp-sharp",
    "flat-flat", "natural-sharp", "natural-flat", "quarter-flat",
    "quarter-sharp", "three-quarters-flat", "three-quarters-sharp",
}
_NOTE_TYPES = {
    "maxima", "long", "breve", "whole", "half", "quarter", "eighth",
    "16th", "32nd", "64th", "128th", "256th", "512th", "1024th",
}


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrs) -> ET.Element:
    element = ET.SubElement(parent, tag, {key.replace("_", "-"): str(value) for key, value in attrs.items()})
    if text is not None:
        element.text = str(text)
    return element


def _as_int(value, field: str, minimum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    return result


def _number_text(value: float | int) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else format(number, ".8g")


def _pitch_rank(note: dict) -> tuple[int, float]:
    pitch = note.get("pitch", {})
    step = pitch.get("step")
    if step not in _STEP_INDEX:
        raise ValueError("Pitch step must be a natural note A-G.")
    octave = _as_int(pitch.get("octave"), "Pitch octave", 0)
    if octave > 9:
        raise ValueError("Pitch octave must be between 0 and 9.")
    alter = float(pitch.get("alter", 0))
    if not math.isfinite(alter) or not -2.0 <= alter <= 2.0:
        raise ValueError("Pitch alter must be a finite value between -2 and 2.")
    return octave * 7 + _STEP_INDEX[step], alter


def _add_attributes(measure: ET.Element, staffs: list[dict], staff_numbers: dict[int, int]) -> None:
    attributes = _sub(measure, "attributes")
    _sub(attributes, "divisions", 1)

    first_staff = staffs[0] if staffs else {}
    fifths = _as_int(first_staff.get("key_fifths", 0), "Key signature fifths")
    if not -7 <= fifths <= 7:
        raise ValueError("Key signature fifths must be between -7 and 7.")
    key = _sub(attributes, "key")
    _sub(key, "fifths", fifths)

    time_signature = first_staff.get("time_signature", {"beats": 4, "beat_type": 4})
    time = _sub(attributes, "time")
    _sub(time, "beats", _as_int(time_signature.get("beats", 4), "Time signature beats", 1))
    _sub(time, "beat-type", _as_int(time_signature.get("beat_type", 4), "Time signature beat type", 1))

    count = max(staff_numbers.values(), default=1)
    if count > 1:
        _sub(attributes, "staves", count)
    by_number = {staff_numbers[index]: staff for index, staff in enumerate(staffs)}
    for number in range(1, count + 1):
        staff = by_number.get(number, {})
        default_clef = {"sign": "F", "line": 4} if count > 1 and number == 2 else {"sign": "G", "line": 2}
        clef = staff.get("clef", default_clef)
        element = _sub(attributes, "clef", number=number) if count > 1 else _sub(attributes, "clef")
        _sub(element, "sign", clef.get("sign", default_clef["sign"]))
        _sub(element, "line", _as_int(clef.get("line", default_clef["line"]), "Clef line", 1))


def _system_index(staff: dict) -> int:
    return _as_int(staff.get("system_index", 0), "System index", 0)


def _staff_top(staff: dict) -> float:
    lines = staff.get("lines") or []
    return float(lines[0]) if lines else float(staff.get("top", 0))


def _review_fields(note: dict) -> list[str]:
    fields = list(note.get("uncertainty", []))
    if note.get("duration") is not None:
        fields = [field for field in fields if field != "duration"]
    if note.get("voice") is not None:
        fields = [field for field in fields if field != "voice"]
    pitch = note.get("pitch", {})
    if note.get("accidental") or pitch.get("accidental"):
        fields = [field for field in fields if field != "accidental"]
    if note.get("measure_index") is not None:
        fields = [field for field in fields if field != "measure_assignment"]
    return sorted(set(fields))


def _voice_number(note: dict, staff_number: int) -> int:
    # When a logical part groups several detected staves, keep unannotated
    # staves on separate provisional voices instead of flattening them in time.
    return _as_int(note.get("voice", staff_number), "Voice number", 1)


def _event_groups(notes: list[dict]) -> tuple[list[list[dict]], bool]:
    has_onset = [note.get("onset") is not None for note in notes]
    if any(has_onset) and not all(has_onset):
        raise ValueError("Either every note in a measure voice needs an onset, or none may specify one.")
    explicit_onsets = all(has_onset) and bool(notes)

    by_chord: dict[str, list[dict]] = defaultdict(list)
    geometry_groups: list[list[dict]] = []
    previous_note = None
    previous_group = None
    spatial_notes = sorted(
        notes,
        key=lambda note: (float(note.get("center", [0])[0]), float(note.get("center", [0, 0])[1])),
    )
    for note in spatial_notes:
        chord_id = note.get("chord_id")
        if chord_id is not None:
            by_chord[str(chord_id)].append(note)
            previous_note = previous_group = None
            continue

        joins_previous = False
        if previous_note is not None and previous_group is not None:
            spacing = float(note.get("_staff_spacing", 12.0))
            previous_center = previous_note.get("center", [0.0, 0.0])
            center = note.get("center", [0.0, 0.0])
            x_gap = abs(float(center[0]) - float(previous_center[0]))
            y_gap = abs(float(center[1]) - float(previous_center[1]))
            previous_pitch = previous_note.get("pitch", {})
            pitch = note.get("pitch", {})
            pitch_differs = (
                pitch.get("step") != previous_pitch.get("step")
                or int(pitch.get("octave", 4)) != int(previous_pitch.get("octave", 4))
            )
            same_staff = note.get("_staff_number") == previous_note.get("_staff_number")
            joins_previous = bool(
                same_staff
                and x_gap <= max(2.0, 0.32 * spacing)
                and y_gap >= 0.38 * spacing
                and pitch_differs
            )
        if joins_previous:
            previous_group.append(note)
            for member in previous_group:
                member["_inferred_chord"] = True
        else:
            previous_group = [note]
            geometry_groups.append(previous_group)
        previous_note = note

    groups = list(by_chord.values()) + geometry_groups
    groups.sort(key=lambda group: min(float(note.get("center", [0])[0]) for note in group))
    if explicit_onsets:
        by_onset: dict[int, list[dict]] = defaultdict(list)
        for group in groups:
            onsets = {_as_int(note["onset"], "Note onset", 0) for note in group}
            if len(onsets) != 1:
                raise ValueError("All notes in a chord must have the same onset.")
            by_onset[next(iter(onsets))].extend(group)
        groups = [by_onset[key] for key in sorted(by_onset)]
    else:
        groups.sort(key=lambda group: min(float(note.get("center", [0])[0]) for note in group))
    for group in groups:
        group.sort(key=_pitch_rank)
    return groups, explicit_onsets


def _write_voice(measure: ET.Element, notes: list[dict], voice_number: int) -> int:
    groups, explicit_onsets = _event_groups(notes)
    cursor = 0
    for group in groups:
        if explicit_onsets:
            start = _as_int(group[0]["onset"], "Note onset", 0)
            if any(_as_int(note["onset"], "Note onset", 0) != start for note in group):
                raise ValueError("Notes in a simultaneous event must have the same onset.")
        else:
            start = cursor
        if start < cursor:
            raise ValueError("Notes in one voice overlap; encode simultaneous notes with a shared onset.")
        if start > cursor:
            forward = _sub(measure, "forward")
            _sub(forward, "duration", start - cursor)
            _sub(forward, "voice", voice_number)
            staff_number = group[0].get("_staff_number")
            if staff_number is not None:
                _sub(forward, "staff", staff_number)

        event_end = start
        for member_index, note in enumerate(group):
            pitch_values = note.get("pitch", {})
            step = pitch_values.get("step")
            rank, alter = _pitch_rank(note)
            octave = _as_int(pitch_values.get("octave"), "Pitch octave", 0)
            if octave > 9:
                raise ValueError("Pitch octave must be between 0 and 9.")
            duration_value = note.get("duration")
            duration = 1 if duration_value is None else _as_int(duration_value, "Note duration", 1)
            note_type = note.get("type", "quarter" if duration_value is None else None)
            if note_type is not None and note_type not in _NOTE_TYPES:
                raise ValueError(f"Unsupported MusicXML note type: {note_type}")

            xml_note = _sub(measure, "note", **({"id": str(note["id"])} if note.get("id") is not None else {}))
            if member_index:
                _sub(xml_note, "chord")
            pitch = _sub(xml_note, "pitch")
            _sub(pitch, "step", step)
            if alter != 0.0 or "alter" in pitch_values:
                _sub(pitch, "alter", _number_text(alter))
            _sub(pitch, "octave", octave)
            _sub(xml_note, "duration", duration)
            _sub(xml_note, "voice", voice_number)
            if note_type is not None:
                _sub(xml_note, "type", note_type)
            accidental = note.get("accidental", pitch_values.get("accidental"))
            if accidental is not None:
                if accidental not in _ACCIDENTALS:
                    raise ValueError(f"Unsupported MusicXML accidental: {accidental}")
                _sub(xml_note, "accidental", accidental)
            staff_number = note.get("_staff_number")
            if staff_number is not None:
                _sub(xml_note, "staff", staff_number)
            notations = _sub(xml_note, "notations")
            review = _review_fields(note)
            message = "IDSR_REVIEW: " + (", ".join(review) if review else "confirm source alignment and interpretation")
            duration_hypothesis = (note.get("duration_hypothesis") or {}).get("candidate", "unresolved")
            message += (
                f" Duration hypothesis: {duration_hypothesis}; "
                "this hypothesis is not used for the exported duration."
            )
            if note.get("_inferred_chord"):
                message += " Chord membership was inferred geometrically and needs review."
            _sub(notations, "other-notation", message, type="single")
            event_end = max(event_end, start + duration)
        cursor = max(cursor, event_end)
    return cursor


def build_musicxml(scene: dict) -> str:
    """Serialize annotated scene events as MusicXML 4.0.

    The exporter preserves explicit chord groups, voice numbers, pitch alters,
    accidentals, durations, onsets, logical part IDs and repeated system indices.
    The image pipeline does not currently infer most of those fields, so absent
    annotations remain review markers rather than guessed notation.
    """
    root = ET.Element("score-partwise", {"version": "4.0"})
    work = _sub(root, "work")
    _sub(work, "work-title", "ScoreTwin provisional transcription")
    identification = _sub(root, "identification")
    miscellaneous = _sub(identification, "miscellaneous")
    _sub(
        miscellaneous,
        "miscellaneous-field",
        "IDSR: pixel proposals are provisional; explicit annotations are serialized and unresolved fields are marked for review.",
        name="idsr-status",
    )

    staffs = scene.get("staffs", [])
    notes = scene.get("notes", [])
    if not staffs:
        part_list = _sub(root, "part-list")
        score_part = _sub(part_list, "score-part", id="P1")
        _sub(score_part, "part-name", "Unresolved score")
        part = _sub(root, "part", id="P1")
        measure = _sub(part, "measure", number=1)
        _add_attributes(measure, [], {})
        return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

    parts: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for staff_index, staff in enumerate(staffs):
        logical_part = str(staff.get("part_id", f"staff-{staff_index + 1}"))
        parts[logical_part].append((staff_index, staff))

    part_list = _sub(root, "part-list")
    part_specs: list[tuple[str, list[tuple[int, dict]]]] = []
    used_part_ids: set[str] = set()
    for part_index, (logical_part, part_staffs) in enumerate(parts.items(), start=1):
        part_id = logical_part if logical_part.startswith("P") else f"P{part_index}"
        if part_id in used_part_ids:
            part_id = f"P{part_index}"
        while part_id in used_part_ids:
            part_id = f"P{part_index + len(used_part_ids)}"
        used_part_ids.add(part_id)
        score_part = _sub(part_list, "score-part", id=part_id)
        name = next((staff.get("part_name") for _, staff in part_staffs if staff.get("part_name")), None)
        _sub(score_part, "part-name", name or f"Detected part {part_index}")
        part_specs.append((part_id, part_staffs))

    for part_id, part_staffs in part_specs:
        systems: dict[int, list[tuple[int, dict]]] = defaultdict(list)
        for staff_index, staff in part_staffs:
            systems[_system_index(staff)].append((staff_index, staff))

        staff_numbers: dict[int, int] = {}
        for system_staffs in systems.values():
            ordered = sorted(system_staffs, key=lambda item: _staff_top(item[1]))
            for order, (staff_index, staff) in enumerate(ordered, start=1):
                staff_numbers[staff_index] = _as_int(staff.get("staff_number", order), "Staff number", 1)

        notes_by_measure: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for note in notes:
            staff_index = _as_int(note.get("staff_index"), "Note staff index", 0)
            if staff_index >= len(staffs):
                raise ValueError(f"Note references missing staff index {staff_index}.")
            if all(staff_index != candidate_index for candidate_index, _ in part_staffs):
                continue
            staff = staffs[staff_index]
            system = _as_int(note.get("system_index", staff.get("system_index", 0)), "System index", 0)
            x = float(note.get("center", [0])[0])
            boundaries = sorted(float(value) for value in staff.get("barlines", []))
            inferred_measure = sum(1 for boundary in boundaries if x > boundary)
            local_measure = _as_int(note.get("measure_index", inferred_measure), "Measure index", 0)
            measure_key = (system, local_measure)
            annotated = dict(note)
            annotated["_staff_number"] = staff_numbers[staff_index]
            annotated["_staff_spacing"] = float(staff.get("spacing", 12.0))
            notes_by_measure[measure_key].append(annotated)

        measure_keys: set[tuple[int, int]] = set(notes_by_measure)
        for system, system_staffs in systems.items():
            count = max((len(staff.get("barlines", [])) + 1 for _, staff in system_staffs), default=1)
            measure_keys.update((system, index) for index in range(count))
        if not measure_keys:
            measure_keys.add((0, 0))
        ordered_measure_keys = sorted(measure_keys)

        part = _sub(root, "part", id=part_id)
        previous_system = None
        for output_measure, measure_key in enumerate(ordered_measure_keys, start=1):
            system, local_measure = measure_key
            measure = _sub(part, "measure", number=output_measure)
            is_new_system = previous_system is not None and system != previous_system
            if is_new_system:
                _sub(measure, "print", new_system="yes")
            if output_measure == 1 or is_new_system:
                part_attributes_staffs = sorted(systems[system], key=lambda item: (staff_numbers[item[0]], _staff_top(item[1])))
                _add_attributes(measure, [staff for _, staff in part_attributes_staffs], {
                    order: staff_numbers[index] for order, (index, staff) in enumerate(part_attributes_staffs)
                })
            previous_system = system

            by_voice: dict[int, list[dict]] = defaultdict(list)
            for note in notes_by_measure.get(measure_key, []):
                staff_number = int(note["_staff_number"])
                voice = _voice_number(note, staff_number)
                by_voice[voice].append(note)
            voices = sorted(by_voice)
            for voice_index, voice in enumerate(voices):
                end = _write_voice(measure, by_voice[voice], voice)
                if voice_index < len(voices) - 1 and end > 0:
                    backup = _sub(measure, "backup")
                    _sub(backup, "duration", end)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")
