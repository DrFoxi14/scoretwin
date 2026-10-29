"""Conservative MusicXML serialization for current, explicitly provisional events."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrs) -> ET.Element:
    element = ET.SubElement(parent, tag, {key.replace("_", "-"): str(value) for key, value in attrs.items()})
    if text is not None:
        element.text = str(text)
    return element


def _add_attributes(measure: ET.Element) -> None:
    attributes = _sub(measure, "attributes")
    _sub(attributes, "divisions", 1)
    key = _sub(attributes, "key")
    _sub(key, "fifths", 0)
    time = _sub(attributes, "time")
    _sub(time, "beats", 4)
    _sub(time, "beat-type", 4)
    clef = _sub(attributes, "clef")
    _sub(clef, "sign", "G")
    _sub(clef, "line", 2)


def build_musicxml(scene: dict) -> str:
    """Export notehead proposals; all inferred durations are marked for review.

    Pitch is a treble-clef diatonic-position proposal. Key signatures, accidentals,
    rhythm, voice assignment, instrument mapping and measure assignment are not
    yet inferred. The XML is a review scaffold, not a verified transcription.
    """
    root = ET.Element("score-partwise", {"version": "4.0"})
    work = _sub(root, "work")
    _sub(work, "work-title", "ScoreTwin provisional transcription")
    identification = _sub(root, "identification")
    miscellaneous = _sub(identification, "miscellaneous")
    _sub(
        miscellaneous,
        "miscellaneous-field",
        "IDSR: pitches are geometric proposals; durations and score structure require review.",
        name="idsr-status",
    )

    staffs = scene.get("staffs", [])
    notes = scene.get("notes", [])
    notes_by_staff: dict[int, list[dict]] = defaultdict(list)
    for note in notes:
        notes_by_staff[int(note["staff_index"])].append(note)

    part_list = _sub(root, "part-list")
    part_ids: list[tuple[int, str]] = []
    for staff_index, _staff in enumerate(staffs):
        part_id = f"P{staff_index + 1}"
        score_part = _sub(part_list, "score-part", id=part_id)
        _sub(score_part, "part-name", f"Detected staff proposal {staff_index + 1}")
        part_ids.append((staff_index, part_id))

    if not staffs:
        score_part = _sub(part_list, "score-part", id="P1")
        _sub(score_part, "part-name", "Unresolved score")
        part = _sub(root, "part", id="P1")
        measure = _sub(part, "measure", number=1)
        _add_attributes(measure)
        return ET.tostring(root, encoding="utf-8", xml_declaration=True).decode("utf-8")

    for staff_index, part_id in part_ids:
        staff = staffs[staff_index]
        staff_notes = sorted(notes_by_staff[staff_index], key=lambda item: item["center"][0])
        boundaries = sorted(float(value) for value in staff.get("barlines", []))
        groups: dict[int, list[dict]] = defaultdict(list)
        for note in staff_notes:
            x = float(note["center"][0])
            measure_index = sum(1 for boundary in boundaries if x > boundary)
            groups[measure_index].append(note)
        measure_count = max(groups.keys(), default=0) + 1

        part = _sub(root, "part", id=part_id)
        for measure_index in range(measure_count):
            measure = _sub(part, "measure", number=measure_index + 1)
            if measure_index == 0:
                _add_attributes(measure)

            measure_notes = sorted(groups.get(measure_index, []), key=lambda item: item["center"][0])
            previous_note: dict | None = None
            for note in measure_notes:
                xml_note = _sub(measure, "note", id=str(note["id"]))
                is_chord_member = False
                if previous_note is not None:
                    x_gap = abs(float(note["center"][0]) - float(previous_note["center"][0]))
                    y_gap = abs(float(note["center"][1]) - float(previous_note["center"][1]))
                    vertical_pitch_diff = (
                        note["pitch"]["step"] != previous_note["pitch"]["step"]
                        or int(note["pitch"]["octave"]) != int(previous_note["pitch"]["octave"])
                    )
                    # Stack noteheads with near-identical x and distinct vertical pitches
                    # are exported as a chord hypothesis. This geometric threshold is
                    # provisional and must be reviewed for dense or handwritten scores.
                    is_chord_member = bool(
                        x_gap <= max(2.0, 0.32 * float(staff.get("spacing", 12.0)))
                        and y_gap >= 0.38 * float(staff.get("spacing", 12.0))
                        and vertical_pitch_diff
                    )
                if is_chord_member:
                    _sub(xml_note, "chord")
                pitch = _sub(xml_note, "pitch")
                _sub(pitch, "step", note["pitch"]["step"])
                _sub(pitch, "octave", note["pitch"]["octave"])
                # Quarter-note duration is a placeholder to keep the file parseable.
                _sub(xml_note, "duration", 1)
                _sub(xml_note, "voice", 1)
                _sub(xml_note, "type", "quarter")
                notations = _sub(xml_note, "notations")
                _sub(
                    notations,
                    "other-notation",
                    "IDSR_REVIEW: duration, accidental, voice and note-to-measure assignment are provisional."
                    + (" Chord membership was inferred geometrically and needs review." if is_chord_member else ""),
                    type="single",
                )

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")
