"""Conservative MusicXML serialization for current, explicitly provisional events."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict


STEP_ORDER = ["C", "D", "E", "F", "G", "A", "B"]


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrs) -> ET.Element:
    element = ET.SubElement(parent, tag, {key.replace("_", "-"): str(value) for key, value in attrs.items()})
    if text is not None:
        element.text = str(text)
    return element


def build_musicxml(scene: dict) -> str:
    """Export notehead proposals; all inferred durations are marked for review.

    Pitch is a treble-clef diatonic-position proposal. Key signatures, accidentals,
    rhythm, voice assignment, instrument mapping and some measure boundaries are
    not yet inferred. The XML is usable as a review scaffold, not a verified score.
    """
    root = ET.Element("score-partwise", {"version": "4.0"})
    work = _sub(root, "work")
    _sub(work, "work-title", "ScoreTwin provisional transcription")
    identification = _sub(root, "identification")
    miscellaneous = _sub(identification, "miscellaneous")
    _sub(miscellaneous, "miscellaneous-field",
         "IDSR: pitches are geometric proposals; durations and score structure require review.",
         name="idsr-status")

    staffs = scene.get("staffs", [])
    notes = scene.get("notes", [])
    notes_by_staff: dict[int, list[dict]] = defaultdict(list)
    for note in notes:
        notes_by_staff[int(note["staff_index"])].append(note)

    part_list = _sub(root, "part-list")
    part_ids: list[tuple[int, str]] = []
    for staff_index, staff in enumerate(staffs):
        part_id = f"P{staff_index + 1}"
        score_part = _sub(part_list, "score-part", id=part_id)
        _sub(score_part, "part-name", f"Detected staff proposal {staff_index + 1}")
        part_ids.append((staff_index, part_id))

    if not staffs:
        score_part = _sub(part_list, "score-part", id="P1")
        _sub(score_part, "part-name", "Unresolved score")
        _sub(root, "part", id="P1")

    for staff_index, part_id in part_ids:
        staff = staffs[staff_index]
        staff_notes = sorted(notes_by_staff[staff_index], key=lambda item: item["center"][0])
        part = _sub(root, "part", id=part_id)
        barlines = sorted(int(value) for value in staff.get("barlines", []))
        boundaries = [float(value) for value in barlines]
        groups: dict[int, list[dict]] = defaultdict(list)
        for note in staff_notes:
            x = float(note["center"][0])
            measure_index = sum(1 for boundary in boundaries if x > boundary)
            groups[measure_index].append(note)
        measure_count = max(groups.keys(), default=0) + 1

        for measure_index in range(measure_count):
            measure = _sub(part, "measure", number=measure_index + 1)
            if measure_index == 0:
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

            for note_index, note in enumerate(groups.get(measure_index, [])):
                xml_note = _sub(measure, "note", id=str(note["id"]))
                pitch = _sub(xml_note, "pitch")
                _sub(pitch, "step", note["pitch"]["step"])
                _sub(pitch, "octave", note["pitch"]["octave"])
                _sub(xml_note, "duration", 1)
                _sub(xml_note, "voice", 1)
                _sub(xml_note, "type", "quarter")
                notations = _sub(xml_note, "notations")
                _sub(notations, "other-notation", "IDSR_REVIEW: duration, accidental, voice and note-to-measure assignment are provisional.",
                     type="single")
                _sub(xml_note, "lyric", None, number=1)
                # Remove empty lyric; its accidental presence could be interpreted as a missing lyric.
                xml_note.remove(xml_note[-1])

    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes.decode("utf-8")
