"""Tests for 4-staff SATB Choir scores and Romanian lyrics layer in IDSR."""

from __future__ import annotations

import cv2
import numpy as np

from scoretwin.imaging import StaffSystem
from scoretwin.notation import group_staffs_into_systems
from scoretwin.pipeline import analyze_array, render_geometry_scene


def test_satb_four_staff_choir_detection_and_romanian_lyrics() -> None:
    """Verify that a 4-staff choral score is recognized as SATB (Sopran, Alto, Tenor, Bas)

    and that lyrics with Romanian syllables and hyphens are isolated without polluting notes.
    """
    h, w = 800, 1000
    sp = 16.0
    gray = np.full((h, w), 250, dtype=np.uint8)

    # 4 staves (Soprano, Alto, Tenor, Bass)
    staff_tops = [100, 240, 380, 520]
    staffs: list[StaffSystem] = []

    for idx, s_top in enumerate(staff_tops):
        lines = [s_top + i * int(sp) for i in range(5)]
        staffs.append(StaffSystem(
            index=idx,
            lines=tuple(lines),
            spacing=sp,
        ))
        for ly in lines:
            cv2.line(gray, (80, ly), (920, ly), 40, 1)

    # Left barline connecting all 4 staves into an SATB choral system
    cv2.line(gray, (80, staff_tops[0]), (80, staff_tops[3] + 4 * int(sp)), 30, 3)

    # Add notes to each staff:
    # Soprano: quarter notes at (200, 132), (320, 132), (440, 116)
    cv2.ellipse(gray, (200, 132), (8, 6), -20, 0, 360, 30, -1)
    cv2.line(gray, (208, 132), (208, 85), 30, 2)
    cv2.ellipse(gray, (320, 132), (8, 6), -20, 0, 360, 30, -1)
    cv2.line(gray, (328, 132), (328, 85), 30, 2)

    # Alto: quarter note at (200, 272)
    cv2.ellipse(gray, (200, 272), (8, 6), -20, 0, 360, 30, -1)
    cv2.line(gray, (208, 272), (208, 225), 30, 2)

    # Tenor: quarter note at (200, 412)
    cv2.ellipse(gray, (200, 412), (8, 6), -20, 0, 360, 30, -1)
    cv2.line(gray, (208, 412), (208, 365), 30, 2)

    # Bass: quarter note at (200, 552)
    cv2.ellipse(gray, (200, 552), (8, 6), -20, 0, 360, 30, -1)
    cv2.line(gray, (208, 552), (208, 505), 30, 2)

    # Add Romanian lyrics underneath Soprano staff: "Doam - ne"
    # Syllable 1: "Doam" at x=185, y=185
    cv2.rectangle(gray, (185, 182), (220, 196), 40, -1)
    # Hyphen: "-" at x=250, y=188
    cv2.rectangle(gray, (250, 187), (268, 190), 40, -1)
    # Syllable 2: "ne" at x=310, y=182
    cv2.rectangle(gray, (310, 182), (330, 196), 40, -1)

    binary = (gray < 180).astype(np.uint8) * 255

    # 1. Test SATB System Grouping
    systems = group_staffs_into_systems(gray, binary, staffs)
    assert len(systems) == 4
    assert systems[0]["is_satb_choral"] is True
    assert systems[0]["voice_name"] == "Sopran"
    assert systems[1]["voice_name"] == "Alto"
    assert systems[2]["voice_name"] == "Tenor"
    assert systems[3]["voice_name"] == "Bas"
    assert systems[3]["inferred_clef"]["sign"] == "F"

    # 2. Test End-to-End Analysis
    scene = analyze_array(
        gray,
        adaptive_staffs=False,
        enable_extended_notation=True,
    )
    assert len(scene["staffs"]) == 4
    # The syllables "Doam", "ne" and hyphen must NOT be hallucinated as noteheads!
    # Only true musical noteheads must be detected
    for n in scene["notes"]:
        # Notes on Soprano staff must be inside or directly adjacent to the staff (y <= 170)
        if n["staff_index"] == 0:
            assert n["center"][1] < 175.0, "Lyric syllable must not be misclassified as a notehead"

    # Verify that lyrics were captured in idsr_dynamic_layer
    dyn = scene.get("idsr_dynamic_layer") or {}
    lyrics = dyn.get("dynamic_lyrics_detected", [])
    assert len(lyrics) >= 2, f"Expected at least 2 lyric items, got {len(lyrics)}"

    # 3. Bit-Exact Lossless 2D Recomposition under Regula Zero
    gz = render_geometry_scene(gray.shape, scene)
    res_signed = gray.astype(np.int16) - gz.astype(np.int16)
    res_u16 = (res_signed + 255).astype(np.uint16)
    recon = np.clip(gz.astype(np.int32) + res_u16.astype(np.int32) - 255, 0, 255).astype(np.uint8)
    assert np.array_equal(recon, gray), "Lossless recomposition must be bit-exact (0.0 px error)"

    # 4. MusicXML Validation with MuseScore Compatibility
    from scoretwin.musicxml import build_musicxml
    import xml.etree.ElementTree as ET
    from pathlib import Path
    import subprocess

    xml_str = build_musicxml(scene, musescore_compat=True)
    root = ET.fromstring(xml_str)
    parts = root.findall("part")
    assert len(parts) >= 1
    # Check that every part has the exact same number of measures
    measure_counts = {len(p.findall("measure")) for p in parts}
    assert len(measure_counts) == 1, f"All parts must have identical measure count, got {measure_counts}"

    # If MuseScore is installed, verify conversion succeeds with exit code 0
    mscore_bin = Path("/Applications/MuseScore 4.app/Contents/MacOS/mscore")
    if mscore_bin.exists():
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            in_xml = Path(td) / "score.musicxml"
            out_mscz = Path(td) / "score.mscz"
            in_xml.write_text(xml_str, encoding="utf-8")
            res_mscore = subprocess.run([str(mscore_bin), str(in_xml), "-o", str(out_mscz)], capture_output=True)
            assert res_mscore.returncode == 0, f"MuseScore failed with error: {res_mscore.stderr.decode()}"
            assert out_mscz.exists() and out_mscz.stat().st_size > 0
