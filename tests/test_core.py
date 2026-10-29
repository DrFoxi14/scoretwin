from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from scoretwin.imaging import detect_staff_systems, extract_notehead_candidates, staff_step_and_pitch
from scoretwin.musicxml import build_musicxml
from scoretwin.pipeline import analyze_array, process_file
from scoretwin.proposal import predict_patch, train_proposal_model


def synthetic_staff() -> np.ndarray:
    image = np.full((250, 440), 255, dtype=np.uint8)
    for y in (70, 82, 94, 106, 118):
        cv2.line(image, (35, y), (405, y), 0, 1, cv2.LINE_AA)
    cv2.ellipse(image, (155, 100), (7, 4), -18, 0, 360, 0, -1, cv2.LINE_AA)
    cv2.ellipse(image, (215, 88), (7, 4), -18, 0, 360, 0, -1, cv2.LINE_AA)
    cv2.line(image, (162, 100), (162, 65), 0, 1, cv2.LINE_AA)
    cv2.line(image, (222, 88), (222, 60), 0, 1, cv2.LINE_AA)
    return image


def test_detects_one_five_line_staff():
    staffs, binary, staff_mask = detect_staff_systems(synthetic_staff())
    assert len(staffs) == 1
    assert len(staffs[0].lines) == 5
    assert 10.0 <= staffs[0].spacing <= 14.0
    assert int((staff_mask > 0).sum()) > 500
    assert binary.shape == staff_mask.shape


def test_pitch_mapping_uses_treble_bottom_line_e4():
    staffs, _, _ = detect_staff_systems(synthetic_staff())
    step, letter, octave = staff_step_and_pitch(float(staffs[0].bottom), staffs[0])
    assert (step, letter, octave) == (0, "E", 4)
    step, letter, octave = staff_step_and_pitch(float(staffs[0].bottom) - staffs[0].spacing, staffs[0])
    assert (step, letter, octave) == (2, "G", 4)


def test_synthetic_proposal_model_returns_normalized_probabilities():
    model, metrics = train_proposal_model(samples_per_class=35, epochs=12, seed=7)
    patch = np.full((24, 24), 255, dtype=np.uint8)
    cv2.ellipse(patch, (12, 12), (6, 4), -15, 0, 360, 0, -1)
    probs = predict_patch(model, patch)
    assert set(probs) == {"filled_head", "hollow_head", "artifact"}
    assert all(0.0 <= value <= 1.0 for value in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert metrics["holdout_n"] >= 60
    assert 0.0 <= metrics["synthetic_holdout_accuracy"] <= 1.0


def test_musicxml_is_well_formed_and_marks_uncertainty():
    scene = {
        "staffs": [{"lines": [10, 20, 30, 40, 50], "spacing": 10, "barlines": [100]}],
        "notes": [{
            "id": "n1",
            "staff_index": 0,
            "center": [70, 40.0],
            "pitch": {"step": "G", "octave": 4},
        }],
    }
    xml = build_musicxml(scene)
    root = ET.fromstring(xml)
    assert root.tag == "score-partwise"
    assert root.attrib["version"] == "4.0"
    assert root.find(".//pitch/step").text == "G"
    assert "IDSR_REVIEW" in root.find(".//other-notation").text


def test_pipeline_creates_bundle_and_preserves_source_bytes(tmp_path):
    image_path = tmp_path / "source.png"
    output_dir = tmp_path / "bundle"
    original = synthetic_staff()
    ok, data = cv2.imencode(".png", original)
    assert ok
    source_bytes = data.tobytes()
    image_path.write_bytes(source_bytes)

    result = process_file(image_path, output_dir)
    preserved_path = output_dir / "source-original.png"
    assert preserved_path.read_bytes() == source_bytes
    expected_hash = hashlib.sha256(source_bytes).hexdigest()
    assert result["source_sha256"] == expected_hash
    assert result["source_hash_verified"] is True
    assert (output_dir / "scene.json").exists()
    assert (output_dir / "overlay.png").exists()
    assert (output_dir / "provisional.musicxml").exists()
    ET.parse(output_dir / "provisional.musicxml")


def test_analysis_has_explicit_unknown_or_no_staff_warning():
    image = np.full((120, 180), 250, dtype=np.uint8)
    scene = analyze_array(image)
    assert scene["staffs"] == []
    assert any("No five-line staff group" in warning for warning in scene["warnings"])
