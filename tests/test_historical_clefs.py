"""Tests for the Universal Historical Clef Engine in IDSR."""

from __future__ import annotations

import cv2
import numpy as np

from scoretwin.historical_clefs import (
    HISTORICAL_CLEF_DEFINITIONS,
    diatonic_step_to_pitch,
    staff_step_and_pitch_with_clef,
    identify_clef_definition,
    detect_clef_glyph_at_staff_start,
    disambiguate_clef_by_syntactic_constraints,
)
from scoretwin.imaging import StaffSystem, staff_step_and_pitch
from scoretwin.musicxml import build_musicxml


def test_ten_canonical_historical_clefs_mathematical_bijections() -> None:
    """Verify that all 10 historical clefs map staff steps to bit-exact diatonic pitches."""
    # 1. French Violin Clef (G on line 1): bottom line is G4
    letter, octv, idx = diatonic_step_to_pitch(0, "G", 1)
    assert (letter, octv) == ("G", 4)
    assert idx == 32

    # 2. Treble Clef (G on line 2): bottom line is E4, line 2 is G4
    assert diatonic_step_to_pitch(0, "G", 2)[:2] == ("E", 4)
    assert diatonic_step_to_pitch(2, "G", 2)[:2] == ("G", 4)

    # 3. Soprano Clef (C on line 1): bottom line is C4 (Middle C)
    assert diatonic_step_to_pitch(0, "C", 1)[:2] == ("C", 4)
    assert diatonic_step_to_pitch(8, "C", 1)[:2] == ("D", 5)  # top line is D5

    # 4. Mezzo-Soprano Clef (C on line 2): line 2 is C4, line 1 is A3
    assert diatonic_step_to_pitch(0, "C", 2)[:2] == ("A", 3)
    assert diatonic_step_to_pitch(2, "C", 2)[:2] == ("C", 4)

    # 5. Alto Clef (C on line 3): line 3 is C4, line 1 is F3
    assert diatonic_step_to_pitch(0, "C", 3)[:2] == ("F", 3)
    assert diatonic_step_to_pitch(4, "C", 3)[:2] == ("C", 4)

    # 6. Tenor Clef (C on line 4): line 4 is C4, line 1 is D3
    assert diatonic_step_to_pitch(0, "C", 4)[:2] == ("D", 3)
    assert diatonic_step_to_pitch(6, "C", 4)[:2] == ("C", 4)

    # 7. Baritone Clef in C (C on line 5): line 5 is C4, line 1 is B2
    assert diatonic_step_to_pitch(0, "C", 5)[:2] == ("B", 2)
    assert diatonic_step_to_pitch(8, "C", 5)[:2] == ("C", 4)

    # 8. Baritone Clef in F (F on line 3): line 3 is F3, line 1 is B2
    assert diatonic_step_to_pitch(0, "F", 3)[:2] == ("B", 2)
    assert diatonic_step_to_pitch(4, "F", 3)[:2] == ("F", 3)

    # 9. Bass Clef (F on line 4): line 4 is F3, line 1 is G2
    assert diatonic_step_to_pitch(0, "F", 4)[:2] == ("G", 2)
    assert diatonic_step_to_pitch(6, "F", 4)[:2] == ("F", 3)

    # 10. Sub-Bass Clef (F on line 5): line 5 is F3, line 1 is E2
    assert diatonic_step_to_pitch(0, "F", 5)[:2] == ("E", 2)
    assert diatonic_step_to_pitch(8, "F", 5)[:2] == ("F", 3)


def test_staff_step_and_pitch_backward_compatibility() -> None:
    """Verify that staff_step_and_pitch defaults to treble clef seamlessly."""
    staff = StaffSystem(
        index=0,
        lines=(100, 116, 132, 148, 164),
        spacing=16.0,
    )
    # Bottom line (cy = 164.0) under default treble is E4
    step, letter, octave = staff_step_and_pitch(164.0, staff)
    assert (step, letter, octave) == (0, "E", 4)

    # Middle line (cy = 132.0, step = 4) under default treble is B4
    step, letter, octave = staff_step_and_pitch(132.0, staff)
    assert (step, letter, octave) == (4, "B", 4)

    # Now pass an Alto Clef (C on line 3): middle line (step = 4) must be C4!
    alto_clef = {"sign": "C", "line": 3}
    step, letter, octave = staff_step_and_pitch(132.0, staff, clef=alto_clef)
    assert (step, letter, octave) == (4, "C", 4)


def test_visual_clef_detection_c_clef_and_g_clef() -> None:
    """Test visual clef glyph detection at staff start for C-clef and G-clef."""
    h, w = 300, 600
    sp = 16.0
    lines = [100.0, 116.0, 132.0, 148.0, 164.0]

    # 1. Synthesize Alto C-clef on line 3 (cy = 132.0)
    img_c = np.full((h, w), 255, dtype=np.uint8)
    for ly in lines:
        cv2.line(img_c, (50, int(ly)), (550, int(ly)), 0, 1)

    # Draw stylized C-clef (two uprights with central cusp on line 3: cy = 132)
    cx_clef = 80
    cv2.rectangle(img_c, (cx_clef - 6, int(lines[0] - 4)), (cx_clef - 3, int(lines[-1] + 4)), 0, -1)
    cv2.rectangle(img_c, (cx_clef + 3, int(lines[0] - 4)), (cx_clef + 6, int(lines[-1] + 4)), 0, -1)
    # Central horizontal crossbars/cusp pointing to line 3
    cv2.rectangle(img_c, (cx_clef - 6, 130), (cx_clef + 6, 134), 0, -1)
    bin_c = (img_c < 128).astype(np.uint8) * 255

    res_c = detect_clef_glyph_at_staff_start(img_c, bin_c, lines, sp, x_start=50)
    assert res_c["detected"] is True
    assert res_c["sign"] == "C"
    assert res_c["line"] == 3
    assert res_c["clef_id"] == "alto"

    # 2. Synthesize tall G-clef (Treble, extends > 5*sp, loop at line 2)
    img_g = np.full((h, w), 255, dtype=np.uint8)
    for ly in lines:
        cv2.line(img_g, (50, int(ly)), (550, int(ly)), 0, 1)

    # Tall vertical spine extending 2*sp above line 1 and 2*sp below line 5
    cv2.line(img_g, (80, int(lines[0] - 2.2 * sp)), (80, int(lines[-1] + 2.2 * sp)), 0, 3)
    # Loop centered on line 2 (y = 148.0)
    cv2.circle(img_g, (80, int(lines[3])), int(1.1 * sp), 0, 2)
    bin_g = (img_g < 128).astype(np.uint8) * 255

    res_g = detect_clef_glyph_at_staff_start(img_g, bin_g, lines, sp, x_start=50)
    assert res_g["detected"] is True
    assert res_g["sign"] == "G"
    assert res_g["line"] in (1, 2)


def test_syntactic_constraint_clef_disambiguation() -> None:
    """Test finite constraint satisfaction selecting the optimal historical clef."""
    # Suppose we have a vocal melody whose steps relative to line 1 are: [2, 3, 4, 5, 4, 3, 2]
    # If treated as Bass Clef (F4), these steps correspond to B2..E3 (deep male voice).
    # If treated as Soprano Clef (C1), these steps correspond to E4..A4 (natural female soprano tessitura).
    notes_soprano = [
        {"diatonic_step_from_bottom_line": s} for s in [1, 2, 3, 4, 5, 4, 3, 2, 1]
    ]
    res = disambiguate_clef_by_syntactic_constraints(notes_soprano)
    assert res["chosen_clef"]["id"] in ("soprano", "treble", "mezzo_soprano")
    assert res["evaluations"][0]["ledger_penalty"] == 0  # 0 ledger lines required


def test_end_to_end_musicxml_with_alto_and_tenor_clefs() -> None:
    """Verify MusicXML export emits valid historical clef tags <sign> and <line>."""
    staffs = [
        {
            "index": 0,
            "top": 100.0,
            "bottom": 164.0,
            "spacing": 16.0,
            "clef": {"sign": "C", "line": 3},  # Alto Clef
        },
        {
            "index": 1,
            "top": 240.0,
            "bottom": 304.0,
            "spacing": 16.0,
            "clef": {"sign": "C", "line": 4},  # Tenor Clef
        },
    ]
    notes = [
        {
            "id": "note-01",
            "staff_index": 0,
            "pitch": {"step": "C", "octave": 4},  # Line 3 in alto clef
            "duration": 2,
            "diatonic_step_from_bottom_line": 4,
            "bbox": [120, 126, 12, 12],
            "center": [126.0, 132.0],
        },
        {
            "id": "note-02",
            "staff_index": 1,
            "pitch": {"step": "C", "octave": 4},  # Line 4 in tenor clef
            "duration": 2,
            "diatonic_step_from_bottom_line": 6,
            "bbox": [120, 250, 12, 12],
            "center": [126.0, 256.0],
        },
    ]
    scene = {
        "schema": "scoretwin-scene-0.1",
        "staffs": staffs,
        "notes": notes,
        "proposal_count": len(notes),
        "note_count": len(notes),
    }

    xml = build_musicxml(scene)
    assert "<sign>C</sign>" in xml
    assert "<line>3</line>" in xml
    assert "<line>4</line>" in xml
    assert "<step>C</step>" in xml
    assert "<octave>4</octave>" in xml
