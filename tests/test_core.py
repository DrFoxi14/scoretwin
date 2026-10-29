from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from scoretwin.benchmark import evaluate_synthetic
from scoretwin.benchmark_manifest import evaluate_manifest
from scoretwin.imaging import detect_barlines, detect_staff_systems, detect_stem_candidates, estimate_global_skew_degrees, extract_notehead_candidates, refine_notehead_geometry, staff_step_and_pitch
from scoretwin.musicxml import build_musicxml
from scoretwin.pipeline import _classify_candidates, analyze_array, process_file
from scoretwin.proposal import load_model, predict_patch, save_model, train_proposal_model
from scoretwin.synthetic import generate_synthetic_page


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


def test_notehead_components_are_proposed_even_across_staff_lines():
    gray, truth = generate_synthetic_page(seed=3, staff_count=2, notes_per_staff=9)
    staffs, binary, staff_mask = detect_staff_systems(gray)
    candidates = extract_notehead_candidates(gray, binary, staff_mask, staffs)
    assert len(staffs) == 2
    assert len(candidates) >= len(truth) - 2
    assert all(candidate["bbox"][2] > 0 and candidate["bbox"][3] > 0 for candidate in candidates)
    assert all("crop" in candidate for candidate in candidates)


def test_barline_detector_does_not_confuse_stems_with_barline():
    gray = synthetic_staff()
    staffs, binary, _staff_mask = detect_staff_systems(gray)
    assert detect_barlines(binary, staffs[0]) == []

    image_with_barline = gray.copy()
    cv2.line(
        image_with_barline,
        (300, staffs[0].top),
        (300, staffs[0].bottom),
        0,
        1,
        cv2.LINE_AA,
    )
    _staffs, binary_with_barline, _mask = detect_staff_systems(image_with_barline)
    detected = detect_barlines(binary_with_barline, staffs[0])
    assert any(abs(x - 300) <= 1 for x in detected)


def test_detects_long_vertical_stem_candidates():
    gray = synthetic_staff()
    staffs, binary, _staff_mask = detect_staff_systems(gray)
    candidates = detect_stem_candidates(binary, staffs, {0: []})
    assert len(candidates) >= 2
    assert all(candidate["length_px"] >= 2.0 * staffs[candidate["staff_index"]].spacing for candidate in candidates)
    assert all(candidate["bbox"][2] <= max(3, round(0.42 * staffs[candidate["staff_index"]].spacing)) for candidate in candidates)


def test_local_template_refinement_reduces_synthetic_center_error():
    gray, truth = generate_synthetic_page(seed=3, staff_count=2, notes_per_staff=9)
    staffs, binary, staff_mask = detect_staff_systems(gray)
    rng = np.random.default_rng(456)
    errors_before = []
    errors_after = []

    for item in truth:
        true_x, true_y = item["center"]
        initial = [true_x + rng.normal(0, 2.1), true_y + rng.normal(0, 2.1)]
        before = float(np.hypot(initial[0] - true_x, initial[1] - true_y))
        staff = staffs[item["staff_index"]]
        result = refine_notehead_geometry(
            gray, binary, staff_mask, initial, staff.spacing, item["head_type"]
        )
        after_x, after_y = result["center"]
        after = float(np.hypot(after_x - true_x, after_y - true_y))
        errors_before.append(before)
        errors_after.append(after)
        assert result["accepted"]
        assert result["fit_score"] is not None

    assert len(errors_before) == 18
    assert float(np.mean(errors_after)) < 0.7 * float(np.mean(errors_before))


def test_skew_diagnostic_flags_a_rotated_synthetic_page():
    gray, _truth = generate_synthetic_page(seed=3, staff_count=2, notes_per_staff=9)
    assert abs(estimate_global_skew_degrees(gray) or 0.0) < 0.5

    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 3.0, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (width, height), borderValue=255)
    angle = estimate_global_skew_degrees(rotated)
    assert angle is not None
    assert abs(angle) >= 2.4


def test_pitch_mapping_uses_treble_bottom_line_e4():
    staffs, _, _ = detect_staff_systems(synthetic_staff())
    step, letter, octave = staff_step_and_pitch(float(staffs[0].bottom), staffs[0])
    assert (step, letter, octave) == (0, "E", 4)
    step, letter, octave = staff_step_and_pitch(float(staffs[0].bottom) - staffs[0].spacing, staffs[0])
    assert (step, letter, octave) == (2, "G", 4)


def test_uncertainty_diagnostics_separate_low_contrast_from_unknown_marks():
    model = {
        "weights": np.zeros((16 * 16 + 1, 3), dtype=np.float32),
        "labels": ("filled_head", "hollow_head", "artifact"),
    }
    candidates = [{
        "id": "candidate-blank",
        "bbox": [1, 1, 16, 16],
        "center": [9.0, 9.0],
        "staff_index": 0,
        "crop": np.full((24, 24), 255, dtype=np.uint8),
    }]
    result = _classify_candidates(candidates, model)[0]
    assert result["class"] == "unknown"
    assert result["unknown_reason"] == "low_local_contrast"
    assert result["image_quality_diagnostics"]["thresholds_calibrated"] is False


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


def test_proposal_checkpoint_retains_reproducibility_metadata(tmp_path):
    model, metrics = train_proposal_model(samples_per_class=30, epochs=8, seed=91)
    model_path = tmp_path / "proposal.npz"
    save_model(model, model_path)
    restored = load_model(model_path)
    assert restored["training_metrics"]["seed"] == 91
    assert restored["training_metrics"]["train_n"] == metrics["train_n"]
    assert restored["training_metrics"]["synthetic_holdout_accuracy"] == metrics["synthetic_holdout_accuracy"]


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


def test_musicxml_serializes_annotated_chords_voices_accidentals_and_systems():
    staffs = [
        {"part_id": "piano", "part_name": "Piano", "system_index": 0, "staff_number": 1,
         "lines": [10, 20, 30, 40, 50], "barlines": [], "clef": {"sign": "G", "line": 2}},
        {"part_id": "piano", "part_name": "Piano", "system_index": 0, "staff_number": 2,
         "lines": [70, 80, 90, 100, 110], "barlines": [], "clef": {"sign": "F", "line": 4}},
        {"part_id": "piano", "part_name": "Piano", "system_index": 1, "staff_number": 1,
         "lines": [140, 150, 160, 170, 180], "barlines": [], "clef": {"sign": "G", "line": 2}},
        {"part_id": "piano", "part_name": "Piano", "system_index": 1, "staff_number": 2,
         "lines": [200, 210, 220, 230, 240], "barlines": [], "clef": {"sign": "F", "line": 4}},
    ]
    notes = [
        {"id": "n1", "staff_index": 0, "center": [50, 40], "pitch": {"step": "D", "octave": 4},
         "voice": 1, "onset": 0, "duration": 2, "chord_id": "c1", "uncertainty": []},
        {"id": "n2", "staff_index": 0, "center": [50, 30], "pitch": {"step": "F", "octave": 4, "alter": 1},
         "accidental": "sharp", "voice": 1, "onset": 0, "duration": 2, "chord_id": "c1", "uncertainty": []},
        {"id": "n3", "staff_index": 1, "center": [50, 100], "pitch": {"step": "C", "octave": 3},
         "voice": 2, "onset": 0, "duration": 4, "uncertainty": []},
        {"id": "n4", "staff_index": 2, "center": [50, 170], "pitch": {"step": "E", "octave": 4},
         "voice": 1, "onset": 0, "duration": 1, "measure_index": 0, "uncertainty": []},
    ]
    root = ET.fromstring(build_musicxml({"staffs": staffs, "notes": notes}))
    parts = root.findall("./part")
    measures = parts[0].findall("./measure")
    assert len(parts) == 1
    assert len(measures) == 2
    assert measures[0].find("./attributes/staves").text == "2"
    assert measures[1].find("./print").attrib["new-system"] == "yes"
    assert len(measures[0].findall("./note/chord")) == 1
    assert measures[0].find(".//pitch/alter").text == "1"
    assert measures[0].find(".//accidental").text == "sharp"
    assert {item.text for item in measures[0].findall("./note/voice")} == {"1", "2"}
    assert any(item.text == "2" for item in measures[0].findall("./note/staff"))
    assert measures[0].find("./backup/duration").text == "2"


def test_musicxml_exports_stacked_noteheads_as_review_chords():
    scene = {
        "staffs": [{"lines": [10, 20, 30, 40, 50], "spacing": 10, "barlines": []}],
        "notes": [
            {"id": "n1", "staff_index": 0, "center": [50.0, 40.0], "pitch": {"step": "E", "octave": 4}},
            {"id": "n2", "staff_index": 0, "center": [51.0, 30.0], "pitch": {"step": "G", "octave": 4}},
            {"id": "n3", "staff_index": 0, "center": [75.0, 40.0], "pitch": {"step": "E", "octave": 4}},
        ],
    }
    root = ET.fromstring(build_musicxml(scene))
    notes = root.findall("./part/measure/note")
    assert len(notes) == 3
    assert notes[0].find("chord") is None
    assert notes[1].find("chord") is not None
    assert notes[2].find("chord") is None
    marker_text = notes[1].find("./notations/other-notation").text
    assert "Chord membership was inferred geometrically" in marker_text


def test_musicxml_duration_hypothesis_is_not_mistaken_for_exported_duration():
    scene = {
        "staffs": [{"lines": [10, 20, 30, 40, 50], "spacing": 10, "barlines": []}],
        "notes": [{
            "id": "n1",
            "staff_index": 0,
            "center": [70.0, 40.0],
            "pitch": {"step": "E", "octave": 4},
            "duration_hypothesis": {
                "candidate": "half_note_candidate",
                "confidence_calibrated": False,
                "used_for_musicxml_duration": False,
            },
        }],
    }
    root = ET.fromstring(build_musicxml(scene))
    note = root.find("./part/measure/note")
    assert note.find("type").text == "quarter"
    marker_text = note.find("./notations/other-notation").text
    assert "Duration hypothesis: half_note_candidate" in marker_text
    assert "not used for the exported duration" in marker_text


def test_empty_scene_still_has_a_measure():
    root = ET.fromstring(build_musicxml({"staffs": [], "notes": []}))
    assert root.find("./part/measure") is not None


def test_pipeline_creates_bundle_and_preserves_source_bytes(tmp_path):
    image_path = tmp_path / "source.png"
    output_dir = tmp_path / "bundle"
    original = synthetic_staff()
    ok, data = cv2.imencode(".png", original)
    assert ok
    source_bytes = data.tobytes()
    image_path.write_bytes(source_bytes)

    result = process_file(image_path, output_dir)
    saved_scene = __import__("json").loads((output_dir / "scene.json").read_text(encoding="utf-8"))
    assert saved_scene["schema"] == "scoretwin-scene-0.1"
    assert any(primitive["kind"] == "staffline" for primitive in saved_scene["primitives"])
    assert any(relation["type"] == "member_of" for relation in saved_scene["relations"])
    assert saved_scene["source"]["sha256"] == result["source_sha256"]
    assert saved_scene["source"]["preserved_losslessly"] is True
    preserved_path = output_dir / "source-original.png"
    assert preserved_path.read_bytes() == source_bytes
    expected_hash = hashlib.sha256(source_bytes).hexdigest()
    assert result["source_sha256"] == expected_hash
    assert result["source_hash_verified"] is True
    assert (output_dir / "scene.json").exists()
    assert (output_dir / "overlay.png").exists()
    assert (output_dir / "geometry-render.png").exists()
    geometry_render = cv2.imread(str(output_dir / "geometry-render.png"), cv2.IMREAD_GRAYSCALE)
    assert geometry_render is not None and geometry_render.shape == original.shape
    assert int(geometry_render.min()) < 255
    assert (output_dir / "provisional.musicxml").exists()
    ET.parse(output_dir / "provisional.musicxml")


def test_pipeline_links_detected_stems_to_noteheads():
    model = {
        "weights": np.zeros((16 * 16 + 1, 3), dtype=np.float32),
        "labels": ("filled_head", "hollow_head", "artifact"),
    }
    model["weights"][0, 0] = 6.0
    scene = analyze_array(synthetic_staff(), model=model)
    assert scene["note_count"] >= 2
    stem_primitives = [primitive for primitive in scene["primitives"] if primitive["kind"] == "stem"]
    stem_relations = [relation for relation in scene["relations"] if relation["type"] == "stem_of"]
    assert len(stem_primitives) >= 2
    assert len(stem_relations) >= 2
    stem_note_ids = {relation["target"] for relation in stem_relations}
    assert stem_note_ids.issubset({note["id"] for note in scene["notes"]})
    for note in scene["notes"]:
        assert note["duration_hypothesis"]["confidence_calibrated"] is False
        assert note["duration_hypothesis"]["used_for_musicxml_duration"] is False


def test_analysis_has_explicit_unknown_or_no_staff_warning():
    image = np.full((120, 180), 250, dtype=np.uint8)
    scene = analyze_array(image)
    assert scene["staffs"] == []
    assert any("No five-line staff group" in warning for warning in scene["warnings"])


def test_annotated_manifest_evaluator_reports_page_bootstrap_intervals(tmp_path):
    image, truth = generate_synthetic_page(seed=7, staff_count=1, notes_per_staff=5)
    image_path = tmp_path / "annotated-page.png"
    assert cv2.imwrite(str(image_path), image)
    manifest = {
        "schema": "scoretwin-annotated-pages-v1",
        "pages": [{
            "id": "synthetic-regression-page",
            "image": image_path.name,
            "staff_spacing_px": 16.0,
            "notes": truth,
        }],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    result = evaluate_manifest(manifest_path, bootstrap_samples=100, bootstrap_seed=4)
    assert result["pages_n"] == 1
    assert result["ground_truth_noteheads_n"] == 5
    intervals = result["precision_recall_f1_bootstrap_95_ci"]
    assert set(intervals) == {
        "precision", "recall", "f1", "mean_center_error_sp", "p95_center_error_sp"
    }
    assert intervals["f1"]["bootstrap_samples_valid"] == 100
    assert intervals["mean_center_error_sp"]["bootstrap_samples_requested"] == 100
    assert result["per_page"][0]["image_sha256"]
    assert "mean_center_error_staff_spaces" in result["per_page"][0]
    assert result["software"]["opencv"] == cv2.__version__
    assert result["mean_center_error_staff_spaces"] is not None
    assert result["p95_center_error_staff_spaces"] is not None
    intervals = result["precision_recall_f1_bootstrap_95_ci"]
    assert intervals["mean_center_error_sp"]["lower_95"] is not None
    assert intervals["p95_center_error_sp"]["upper_95"] is not None
    assert result["per_page"][0]["p95_center_error_staff_spaces"] is not None


def test_scene_graph_records_geometry_provenance_and_pitch_calibration():
    gray, _truth = generate_synthetic_page(seed=28, staff_count=1, notes_per_staff=5)
    original = gray.copy()
    scene = analyze_array(gray)
    kinds = {primitive["kind"] for primitive in scene["primitives"]}
    assert scene["schema"] == "scoretwin-scene-0.1"
    assert {"staff_system", "staffline", "notehead"}.issubset(kinds)
    assert any(relation["type"] == "pitch_calibrated_by" for relation in scene["relations"])
    assert all("provenance" in primitive and "uncertainty" in primitive for primitive in scene["primitives"])
    assert scene["reproducibility"]["software"]["scoretwin"]
    assert scene["reproducibility"]["proposal_model"]["training_metrics"]["seed"] == 19
    assert np.array_equal(gray, original)


def test_synthetic_benchmark_schema_is_reproducible():
    report = evaluate_synthetic(
        pages=3,
        seed=44,
        staff_count=1,
        notes_per_staff=5,
        noise_sigma=2.0,
        blur_sigma=0.35,
    )
    assert report["benchmark"] == "synthetic-notehead-proposal-v2"
    assert report["measurements"]["n_ground_truth_noteheads"] == 15
    assert report["protocol"]["pages"] == 3
    assert report["protocol"]["matching_tolerance_staff_spaces"] == 0.3
    assert 0.0 <= report["measurements"]["micro_precision"] <= 1.0
    assert 0.0 <= report["measurements"]["micro_recall"] <= 1.0
    assert len(report["measurements"]["macro_page_f1_ci95_bootstrap"]) == 2
    assert len(report["per_page"]) == 3
    assert all("p95_center_error_staff_spaces" in row for row in report["per_page"])
    assert isinstance(report["runtime_environment"]["runtime_meets_all_project_requirements"], bool)
    assert report["software"]["scoretwin"]
    assert report["proposal_model_training_metrics"]["seed"] == 19
