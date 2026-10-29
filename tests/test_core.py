from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from scoretwin.benchmark import evaluate_perspective_capture, evaluate_synthetic
from scoretwin.benchmark_manifest import evaluate_manifest
from scoretwin.benchmark_photo_smoke import evaluate_photo_smoke
from scoretwin.calibration import evaluate_human_calibration, recommended_study_design
from scoretwin.capture_quality import diagnose_capture_quality
from scoretwin.gltf_export import build_gltf, validate_embedded_gltf
from scoretwin.metrics import binary_mask_metrics, compare_idsr_components, geometry_metrics, mask_attack_sensitivity, measure_bundle_storage, photometric_metrics, relation_metrics, residual_compressibility_metrics
from scoretwin.imaging import detect_barlines, detect_staff_systems, detect_stem_candidates, estimate_global_skew_degrees, extract_notehead_candidates, load_grayscale, load_pixel_array, rectify_global_skew, rectify_page_quadrilateral, refine_notehead_geometry, staff_step_and_pitch, transform_points_xy
from scoretwin.musicxml import build_musicxml
from scoretwin.pipeline import _classify_candidates, analyze_array, process_file
from scoretwin.proposal import load_model, predict_patch, save_model, train_proposal_model
from scoretwin.synthetic import generate_synthetic_page
from scoretwin.web import PAGE as WEB_PAGE, analyze as analyze_web_upload, export_gltf


def synthetic_staff() -> np.ndarray:
    image = np.full((250, 440), 255, dtype=np.uint8)
    for y in (70, 82, 94, 106, 118):
        cv2.line(image, (35, y), (405, y), 0, 1, cv2.LINE_AA)
    cv2.ellipse(image, (155, 100), (7, 4), -18, 0, 360, 0, -1, cv2.LINE_AA)
    cv2.ellipse(image, (215, 88), (7, 4), -18, 0, 360, 0, -1, cv2.LINE_AA)
    cv2.line(image, (162, 100), (162, 65), 0, 1, cv2.LINE_AA)
    cv2.line(image, (222, 88), (222, 60), 0, 1, cv2.LINE_AA)
    return image


def test_capture_quality_flags_low_contrast_and_blur_without_claiming_calibration():
    sharp, _truth = generate_synthetic_page(seed=17, staff_count=2, notes_per_staff=9)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.2)
    low_contrast = np.clip(sharp.astype(np.float32) * 0.05 + 120.0, 0, 255).astype(np.uint8)

    sharp_report = diagnose_capture_quality(sharp)
    blur_report = diagnose_capture_quality(blurred)
    contrast_report = diagnose_capture_quality(low_contrast)

    assert sharp_report["thresholds_calibrated"] is False
    assert sharp_report["metrics"]["normalized_laplacian_variance"] > blur_report["metrics"]["normalized_laplacian_variance"]
    assert "low_dynamic_range" in {item["code"] for item in contrast_report["advisories"]}
    assert contrast_report["status"] == "retake_suggested"
    with pytest.raises(ValueError, match="uint8 grayscale"):
        diagnose_capture_quality(np.zeros((32, 32, 3), dtype=np.uint8))


def test_scene_and_browser_expose_capture_quality_diagnostics():
    scene = analyze_array(synthetic_staff())
    assert scene["capture_quality"]["schema"] == "scoretwin-capture-quality-0.1"
    assert scene["capture_quality"]["thresholds_calibrated"] is False
    assert scene["capture_diagnostics"]["estimated_staffline_angle_deg"] is not None
    assert 'id="capture-quality-summary"' in WEB_PAGE
    assert "qualityAdvisories.map" in WEB_PAGE
    assert 'id="live-camera"' in WEB_PAGE
    assert 'id="capture-frame"' in WEB_PAGE
    assert "navigator.mediaDevices.getUserMedia" in WEB_PAGE
    assert "liveCameraStream.getTracks().forEach(track=>track.stop())" in WEB_PAGE
    assert "liveCapturedFile=new File([selected.blob]" in WEB_PAGE
    assert "const file=activeFile()" in WEB_PAGE


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


def test_global_skew_rectification_retains_source_coordinate_transform():
    gray, truth = generate_synthetic_page(seed=38, staff_count=1, notes_per_staff=5)
    height, width = gray.shape
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), 3.0, 1.0)
    rotated = cv2.warpAffine(gray, rotation, (width, height), borderValue=255)
    original = rotated.copy()

    result = rectify_global_skew(rotated)

    assert result.applied
    assert result.reason == "global_rotation_rectified"
    assert result.estimated_skew_degrees is not None
    assert abs(result.estimated_skew_degrees) >= 2.4
    assert abs(estimate_global_skew_degrees(result.image) or 0.0) < 0.5
    assert result.image.shape[0] >= rotated.shape[0]
    assert result.image.shape[1] >= rotated.shape[1]
    assert np.array_equal(rotated, original)

    points = np.asarray([[0.0, 0.0], [width - 1.0, height - 1.0], truth[0]["center"]])
    rectified_points = transform_points_xy(points, result.source_to_rectified)
    recovered_points = transform_points_xy(rectified_points, result.rectified_to_source)
    assert np.max(np.abs(recovered_points - points)) < 1e-8
    metadata = result.to_dict()
    assert metadata["preserves_source_pixels"] is False
    assert metadata["coordinate_convention"].startswith("pixel centers")


def test_global_skew_rectification_returns_identity_when_no_lines_are_found():
    blank = np.full((120, 180), 255, dtype=np.uint8)
    result = rectify_global_skew(blank)
    assert not result.applied
    assert result.reason == "no_dominant_near_horizontal_lines"
    assert np.array_equal(result.image, blank)
    assert np.array_equal(result.source_to_rectified, np.eye(3))


def test_manual_page_quadrilateral_rectification_is_reversible_and_keeps_source():
    page = np.full((180, 260), 255, dtype=np.uint8)
    cv2.line(page, (30, 40), (230, 40), 0, 2)
    cv2.line(page, (30, 80), (230, 80), 0, 2)
    source_corners = np.asarray([[20, 15], [235, 28], [250, 160], [8, 168]], dtype=np.float32)
    reference_corners = np.asarray([[0, 0], [259, 0], [259, 179], [0, 179]], dtype=np.float32)
    page_to_photo = cv2.getPerspectiveTransform(reference_corners, source_corners)
    photographed = cv2.warpPerspective(page, page_to_photo, (260, 180), borderValue=255)
    original = photographed.copy()

    result = rectify_page_quadrilateral(photographed, source_corners)

    assert result.applied
    assert result.method == "manual_page_quadrilateral"
    assert result.reason == "manual_page_quadrilateral_rectified"
    assert result.image.shape[0] >= 130
    assert result.image.shape[1] >= 230
    assert np.array_equal(photographed, original)
    mapped_corners = transform_points_xy(source_corners, result.source_to_rectified)
    expected = np.asarray(
        [[0, 0], [result.image.shape[1] - 1, 0],
         [result.image.shape[1] - 1, result.image.shape[0] - 1], [0, result.image.shape[0] - 1]],
        dtype=np.float64,
    )
    assert np.max(np.abs(mapped_corners - expected)) < 1e-3
    points = np.asarray([[31.0, 43.0], [124.0, 88.0], [227.0, 152.0]])
    roundtrip = transform_points_xy(
        transform_points_xy(points, result.source_to_rectified), result.rectified_to_source
    )
    assert np.max(np.abs(roundtrip - points)) < 1e-7


def test_manual_page_quadrilateral_rejects_invalid_corner_order_and_bounds():
    gray = np.full((100, 120), 255, dtype=np.uint8)
    with np.testing.assert_raises_regex(ValueError, "strictly convex"):
        rectify_page_quadrilateral(gray, [[5, 5], [100, 80], [100, 5], [5, 80]])
    with np.testing.assert_raises_regex(ValueError, "clockwise"):
        rectify_page_quadrilateral(gray, [[5, 5], [5, 90], [100, 90], [100, 5]])
    with np.testing.assert_raises_regex(ValueError, "inside the source image"):
        rectify_page_quadrilateral(gray, [[-1, 0], [100, 0], [100, 90], [0, 90]])


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


def test_human_calibration_reports_rank_correlation_kappa_and_repeat_consistency():
    pairs = []
    repeated_indices = {4, 14, 24}
    for index in range(30):
        rating = int(round(1 + index * 6 / 29))
        repeat_group = f"repeat-{index}" if index in repeated_indices else None
        pair = {
            "id": f"pair-{index:02d}",
            "independence_group": f"source-piece-{index:02d}",
            "repeat_group": repeat_group,
            "ratings": {f"rater-{rater}": rating for rater in range(5)},
            "metrics": {
                "mask_iou": 0.2 + 0.8 * index / 29,
                "center_error_sp": 1.0 - 0.95 * index / 29,
            },
        }
        pairs.append(pair)
        if index in repeated_indices:
            repeated = dict(pair)
            repeated["id"] = f"repeat-copy-{index:02d}"
            pairs.append(repeated)

    study = {
        "schema": "scoretwin-human-calibration-v1",
        "rating_scale": {
            "min": 1,
            "max": 7,
            "anchors": {"1": "clearly different", "7": "visually identical"},
        },
        "metric_directions": {"mask_iou": "higher", "center_error_sp": "lower"},
        "study_design": {"blinded": True, "randomized_order": True},
        "pairs": pairs,
    }
    report = evaluate_human_calibration(study, bootstrap_samples=100, bootstrap_seed=8)
    assert report["pair_count"] == 33
    assert report["independence_group_count"] == 30
    assert report["rater_count"] == 5
    assert report["total_rating_count"] == 165
    assert report["metric_human_associations"]["mask_iou"]["n_pairs"] == 30
    assert report["metric_human_associations"]["center_error_sp"]["n_pairs"] == 30
    assert report["metric_human_associations"]["mask_iou"]["spearman_rho"] > 0.98
    assert report["metric_human_associations"]["center_error_sp"]["spearman_rho"] > 0.98
    assert report["metric_human_associations"]["mask_iou"]["bootstrap_95_ci"]["bootstrap_samples_requested"] == 100
    assert report["inter_rater_agreement"]["mean_pairwise_quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert report["within_rater_repeat_consistency"]["exact_repeat_agreement_rate"] == pytest.approx(1.0)
    assert report["scalar_did_weights"] is None
    assert recommended_study_design()["planned_unique_pairs_with_20_percent_exclusion_margin"] >= 50


def test_checked_in_human_calibration_example_is_explicitly_not_real_data():
    root = Path(__file__).resolve().parents[1]
    example_path = root / "schemas" / "scoretwin-human-calibration-example.json"
    protocol_path = root / "docs" / "HUMAN_CALIBRATION_PROTOCOL.md"
    assert example_path.is_file()
    assert protocol_path.is_file()
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert example["schema"] == "scoretwin-human-calibration-v1"
    assert example["study_design"]["example_not_data"] is True
    assert any(pair.get("repeat_group") for pair in example["pairs"])
    report = evaluate_human_calibration(example, bootstrap_samples=100, bootstrap_seed=17)
    assert report["data_status"].startswith("human ratings supplied")
    assert report["pair_count"] == 2
    assert report["scalar_did_weights"] is None


def test_human_calibration_rejects_invalid_schema_ratings_and_directions():
    with pytest.raises(ValueError, match="schema"):
        evaluate_human_calibration({"schema": "wrong"})
    valid = {
        "schema": "scoretwin-human-calibration-v1",
        "rating_scale": {"min": 1, "max": 7},
        "metric_directions": {"mask_iou": "maybe"},
        "pairs": [{"id": "pair-1", "ratings": {"r1": 5}, "metrics": {"mask_iou": 0.8}}],
    }
    with pytest.raises(ValueError, match="higher.*lower"):
        evaluate_human_calibration(valid)


def test_component_metrics_report_exact_photo_geometry_masks_and_relations():
    reference = np.full((40, 60, 3), 255, dtype=np.uint8)
    reference[10:13, 5:45] = [0, 0, 0]
    candidate = reference.copy()

    photo = photometric_metrics(reference, candidate)
    assert photo["exact_pixel_fraction"] == 1.0
    assert photo["mismatched_pixel_count"] == 0
    assert photo["maximum_absolute_channel_error"] == 0
    assert photo["psnr_db"] is None
    assert photo["psnr_infinite_for_exact_match"] is True

    compression = residual_compressibility_metrics(reference, candidate)
    assert compression["residual_raw_bytes"] == reference.size * 2
    assert compression["residual_zlib_bits_per_pixel"] < compression["residual_raw_bits_per_pixel"]

    geometry = geometry_metrics(
        [
            {"id": "r1", "staff_index": 0, "center": [10.0, 20.0], "pitch": {"step": "E", "octave": 4}},
            {"id": "r2", "staff_index": 0, "center": [30.0, 20.0], "pitch": {"step": "G", "octave": 4}},
        ],
        [
            {"id": "p1", "staff_index": 0, "center": [11.0, 20.0], "pitch": {"step": "E", "octave": 4}},
            {"id": "extra", "staff_index": 0, "center": [50.0, 20.0], "pitch": {"step": "C", "octave": 5}},
        ],
        staff_space_px=10.0,
        tolerance_staff_spaces=0.3,
    )
    assert geometry["true_positive"] == 1
    assert geometry["false_positive"] == 1
    assert geometry["false_negative"] == 1
    assert geometry["f1"] == 0.5
    assert geometry["matched_center_error_mean_staff_spaces"] == 0.1
    assert geometry["matched_symbol_pitch_accuracy"]["accuracy"] == 1.0

    relation_result = relation_metrics(
        [{"type": "member_of", "source": "a", "target": "s"}],
        [{"type": "member_of", "source": "a", "target": "s"},
         {"type": "stem_of", "source": "b", "target": "a"}],
    )
    assert relation_result["true_positive"] == 1
    assert relation_result["false_positive"] == 1
    assert relation_result["f1"] == pytest.approx(2 / 3)

    mask_ref = np.zeros((20, 20), dtype=np.uint8)
    mask_ref[4:12, 3:10] = 255
    mask_candidate = mask_ref.copy()
    mask_candidate[15, 15] = 255
    mask_result = binary_mask_metrics(mask_ref, mask_candidate)
    assert mask_result["iou"] < 1.0
    assert mask_result["f1"] < 1.0

    report = compare_idsr_components(reference, candidate, reference_mask=mask_ref, candidate_mask=mask_ref)
    assert report["aggregate_did"] is None
    assert "weights require preregistered calibration" in report["aggregate_did_status"]
    assert report["metrics"]["photometric"]["exact_pixel_fraction"] == 1.0


def test_mask_redteam_shows_iou_can_hide_a_missing_small_symbol():
    reference = np.zeros((128, 128), dtype=np.uint8)
    reference[20:80, 20:100] = 255
    reference[95:99, 95:99] = 255  # a disconnected small glyph proxy

    report = mask_attack_sensitivity(reference, reference)
    assert report["baseline_iou"] == 1.0
    assert "dilation" in report["attacks"]["one_pixel_dilation"] or report["attacks"]["one_pixel_dilation"]["iou"] < 1.0
    missing_small_object = report["attacks"]["remove_smallest_component"]
    assert missing_small_object["changed_candidate_pixels"] == 16
    assert 0.99 < missing_small_object["iou"] < 1.0
    assert missing_small_object["foreground_component_count"] == 1
    assert any("missing small symbol" in detail for detail in report["interpretation"])
    assert any("per-object notehead/stem precision and recall" in defense for defense in report["defenses"])


def test_bundle_storage_accounting_counts_all_artifacts_except_source_copy(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "source-original.png").write_bytes(b"immutable-source")
    (bundle / "scene.json").write_bytes(b'{"schema":"test"}')
    (bundle / "residual-offset255.png").write_bytes(b"residual")
    (bundle / "scene.gltf").write_bytes(b"mesh-plus-texture")

    report = measure_bundle_storage(bundle, pixel_count=100)
    included_bytes = len(b'{"schema":"test"}') + len(b"residual") + len(b"mesh-plus-texture")
    assert report["included_file_count"] == 3
    assert report["included_file_bytes"] == included_bytes
    assert report["stored_artifact_bits_per_pixel"] == 8 * included_bytes / 100
    assert report["excluded_source_copy_count"] == 1
    assert report["excluded_source_copy_bytes"] == len(b"immutable-source")
    assert "not a model-based minimum-description-length" in report["status"]
    assert {item["path"] for item in report["included_files"]} == {
        "scene.json", "residual-offset255.png", "scene.gltf"
    }


def test_component_metrics_reject_shape_mismatch_and_uncalibrated_geometry():
    with pytest.raises(ValueError, match="shapes must match"):
        photometric_metrics(np.zeros((8, 8), dtype=np.uint8), np.zeros((9, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="staff_space_px"):
        compare_idsr_components(
            np.zeros((8, 8), dtype=np.uint8),
            np.zeros((8, 8), dtype=np.uint8),
            reference_notes=[],
            candidate_notes=[],
        )


def test_gltf_export_is_self_contained_and_preserves_scene_metadata():
    scene = {
        "schema": "scoretwin-scene-0.1",
        "coordinate_frame": "source_raster_pixel_centers",
        "image": {"width": 100, "height": 80},
        "source": {"sha256": "example-source-hash"},
        "capture_diagnostics": {"perspective_rectified": False},
        "staffs": [],
        "notes": [{
            "id": "note-1",
            "staff_index": 0,
            "center": [30.0, 30.0],
            "bbox": [26, 27, 8, 6],
            "shape_class": "hollow_head",
            "pitch": {"step": "G", "octave": 4},
            "voice": 2,
            "duration_hypothesis": {"candidate": "half_note_candidate"},
            "stem_ids": ["stem-1"],
        }],
        "primitives": [
            {
                "id": "system-0", "kind": "staff_system",
                "geometry": {"bbox_px": [0, 10, 100, 48]},
                "provenance": {"method": "test"},
                "uncertainty": [],
            },
            {
                "id": "line-0", "kind": "staffline",
                "geometry": {"x0_px": 0, "y0_px": 10, "x1_px": 99, "y1_px": 10, "thickness_px_estimate": 1},
                "provenance": {"method": "test"},
                "uncertainty": ["line_thickness"],
            },
            {
                "id": "stem-1", "kind": "stem",
                "geometry": {"x0_px": 34, "y0_px": 10, "x1_px": 34, "y1_px": 30, "thickness_px_estimate": 1},
                "provenance": {"method": "test"},
                "uncertainty": ["stem_direction"],
            },
            {
                "id": "note-1", "kind": "notehead",
                "geometry": {"center_x_px": 30, "center_y_px": 30, "radius_x_px": 4, "radius_y_px": 3, "rotation_rad": 0},
                "appearance": {"physical_absorbance_measured": False},
                "symbolic": {"pitch_proposal": {"step": "G", "octave": 4}},
                "provenance": {"method": "test"},
                "uncertainty": ["duration"],
            },
            {
                "id": "unknown-1", "kind": "unknown_region",
                "geometry": {"bbox_px": [60, 20, 8, 12]},
                "provenance": {"status": "unresolved"},
                "uncertainty": ["symbol_identity"],
            },
        ],
        "relations": [
            {"type": "member_of", "source": "line-0", "target": "system-0"},
            {"type": "stem_of", "source": "stem-1", "target": "note-1"},
        ],
    }

    serialized = build_gltf(scene, layer_separation=7.5)
    validation = validate_embedded_gltf(serialized)
    gltf = __import__("json").loads(serialized)
    assert validation["valid"] is True
    assert validation["nodes"] == 4
    assert validation["meshes"] == 4
    assert validation["buffer_bytes"] > 0
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["extras"]["source"]["sha256"] == "example-source-hash"
    assert gltf["extras"]["depth_semantics"].startswith("Z offsets are visualization")
    note_node = next(node for node in gltf["nodes"] if node["name"] == "note-1")
    assert note_node["extras"]["idsr"]["uncertainty"] == ["duration"]
    assert note_node["extras"]["idsr"]["note_metadata"]["voice"] == 2
    assert note_node["extras"]["idsr"]["semantic_layer_z"] > 7.5
    stem_node = next(node for node in gltf["nodes"] if node["name"] == "stem-1")
    assert stem_node["extras"]["idsr"]["relations"]["stem_of"] == ["note-1"]
    assert stem_node["extras"]["idsr"]["semantic_layer_z"] == note_node["extras"]["idsr"]["semantic_layer_z"]
    line_node = next(node for node in gltf["nodes"] if node["name"] == "line-0")
    assert line_node["extras"]["idsr"]["relations"]["outgoing"][0]["type"] == "member_of"
    unknown_node = next(node for node in gltf["nodes"] if node["name"] == "unknown-1")
    unknown_mesh = gltf["meshes"][unknown_node["mesh"]]["primitives"][0]
    assert gltf["accessors"][unknown_mesh["attributes"]["POSITION"]]["count"] == 16


def test_gltf_export_embeds_a_page_texture_under_the_symbolic_geometry():
    image = np.full((64, 96), 255, dtype=np.uint8)
    cv2.line(image, (0, 20), (95, 20), 0, 1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    scene = {
        "schema": "scoretwin-scene-0.1",
        "coordinate_frame": "source_raster_pixel_centers",
        "image": {"width": 96, "height": 64},
        "source": {"sha256": "original-byte-hash"},
        "primitives": [{
            "id": "line-0",
            "kind": "staffline",
            "geometry": {"x0_px": 0, "y0_px": 20, "x1_px": 95, "y1_px": 20},
            "provenance": {"status": "estimated"},
            "uncertainty": [],
        }],
        "notes": [],
        "relations": [],
    }
    gltf_text = build_gltf(scene, background_png=encoded.tobytes())
    report = validate_embedded_gltf(gltf_text)
    gltf = __import__("json").loads(gltf_text)
    assert report["valid"] is True
    assert report["has_background_texture"] is True
    assert report["background_texture_resolution_px"] == [96, 64]
    assert gltf["extras"]["background_texture_embedded"] is True
    assert gltf["extras"]["background_texture_resampled_to_analysis_frame"] is False
    assert gltf["extensionsUsed"] == ["KHR_materials_unlit"]
    assert gltf["images"][0]["mimeType"] == "image/png"
    assert gltf["images"][0]["uri"].startswith("data:image/png;base64,")
    plane = next(node for node in gltf["nodes"] if node["name"] == "analysis-raster-plane")
    plane_primitive = gltf["meshes"][plane["mesh"]]["primitives"][0]
    assert plane_primitive["attributes"]["TEXCOORD_0"] < len(gltf["accessors"])
    assert plane["extras"]["idsr"]["semantic_layer_z"] < 0


def test_browser_gltf_endpoint_exports_texture_without_polluting_scene_json():
    image = np.full((32, 48), 255, dtype=np.uint8)
    cv2.line(image, (0, 12), (47, 12), 0, 1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    scene = {
        "schema": "scoretwin-scene-0.1",
        "image": {"width": 48, "height": 32},
        "source": {"sha256": "source-hash"},
        "primitives": [{
            "id": "staff-line",
            "kind": "staffline",
            "geometry": {"x0_px": 0, "y0_px": 12, "x1_px": 47, "y1_px": 12},
        }],
        "notes": [],
        "relations": [],
    }
    payload = {
        "scene": scene,
        "background_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }
    result = export_gltf(payload)
    report = validate_embedded_gltf(result["gltf"])
    gltf = __import__("json").loads(result["gltf"])
    assert report["has_background_texture"] is True
    assert gltf["extras"]["source"]["sha256"] == "source-hash"
    assert "background_png_base64" not in gltf
    assert "backgroundPreviewPngBase64=result.source_preview_png_base64" in WEB_PAGE
    assert "background_png_base64:backgroundPreviewPngBase64" in WEB_PAGE


def test_browser_analysis_returns_residual_proof_and_downloadable_png():
    original = synthetic_staff()
    ok, encoded = cv2.imencode(".png", original)
    assert ok

    class MemoryUpload:
        filename = "synthetic-score.png"

        async def read(self) -> bytes:
            return encoded.tobytes()

    result = asyncio.run(analyze_web_upload(MemoryUpload(), page=0, corners_json=None, capture_metadata_json=None))
    proof = result["identity_reconstruction"]
    assert proof["exact_pixel_reconstruction_verified"] is True
    assert proof["mismatched_pixel_count"] == 0
    assert proof["maximum_absolute_pixel_error"] == 0
    assert result["residual_download_available"] is True
    assert result["residual_png_base64"]
    assert result["identity_report_json"]
    assert 'id="download-residual"' in WEB_PAGE
    assert 'id="download-identity-report"' in WEB_PAGE
    assert "download-residual').addEventListener" in WEB_PAGE

    residual_bytes = base64.b64decode(result["residual_png_base64"], validate=True)
    residual = cv2.imdecode(np.frombuffer(residual_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    source_bytes = base64.b64decode(result["source_preview_png_base64"], validate=True)
    source_preview = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    geometry_bytes = base64.b64decode(result["geometry_render_png_base64"], validate=True)
    geometry_render = cv2.imdecode(np.frombuffer(geometry_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert residual.dtype == np.uint16
    assert source_preview.shape == geometry_render.shape == residual.shape == original.shape
    recomposed = geometry_render.astype(np.int32) + residual.astype(np.int32) - 255
    assert np.array_equal(recomposed.astype(np.uint8), source_preview)
    assert np.array_equal(source_preview, original)


def test_gltf_export_handles_an_empty_scene_with_a_valid_buffer():
    gltf_text = build_gltf({
        "schema": "scoretwin-scene-0.1",
        "image": {"width": 100, "height": 80},
        "primitives": [],
        "notes": [],
    })
    report = validate_embedded_gltf(gltf_text)
    assert report["valid"] is True
    assert report["nodes"] == 0
    assert report["meshes"] == 0
    assert report["buffer_bytes"] == 4


def test_gltf_export_rejects_nonfinite_geometry_and_bad_scenes():
    scene = {
        "schema": "scoretwin-scene-0.1",
        "image": {"width": 100, "height": 80},
        "primitives": [{
            "id": "bad-line",
            "kind": "staffline",
            "geometry": {"x0_px": float("nan"), "y0_px": 10, "x1_px": 99, "y1_px": 10},
        }],
        "notes": [],
    }
    with __import__("pytest").raises(ValueError, match="finite"):
        build_gltf(scene)
    with __import__("pytest").raises(ValueError, match="Unrecognized"):
        build_gltf({"schema": "wrong", "image": {"width": 1, "height": 1}})


def test_browser_analysis_validates_and_preserves_live_burst_metadata():
    image = synthetic_staff()
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    class MemoryUpload:
        filename = "burst-best.jpg"

        async def read(self) -> bytes:
            return encoded.tobytes()

    metadata = {
        "method": "live_camera_burst_best_single_frame",
        "frames_requested": 3,
        "frames_encoded": 2,
        "selection_metric": "client-supplied-this-is-ignored",
        "selection_metric_calibrated": False,
        "frame_scores": [
            {"rank": 1, "contrast": 41.2, "edge_energy": 3.2, "selection_score": 3.2, "width_px": 440, "height_px": 250},
            {"rank": 2, "contrast": 39.1, "edge_energy": 2.7, "selection_score": 2.7, "width_px": 440, "height_px": 250},
        ],
        "selected_width_px": 440,
        "selected_height_px": 250,
        "fusion_performed": False,
    }
    result = asyncio.run(analyze_web_upload(
        MemoryUpload(),
        page=0,
        corners_json=None,
        capture_metadata_json=json.dumps(metadata),
    ))
    captured = result["source"]["browser_capture"]
    assert captured["method"] == "live_camera_burst_best_single_frame"
    assert captured["frames_requested"] == 3
    assert captured["frames_encoded"] == 2
    assert captured["selection_metric_calibrated"] is False
    assert captured["frame_scores"][0]["edge_energy"] == pytest.approx(3.2)
    assert captured["fusion_performed"] is False
    assert "client-reported" in captured["provenance"]
    assert result["summary"]["browser_capture_method"] == captured["method"]
    assert "frame.score.edge_energy" in WEB_PAGE
    assert "frames_requested:3" in WEB_PAGE
    assert "fusion_performed:false" in WEB_PAGE
    assert "capture_metadata_json',JSON.stringify(liveCaptureMetadata)" in WEB_PAGE


def test_browser_analysis_rejects_malformed_live_burst_metadata():
    image = synthetic_staff()
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    class MemoryUpload:
        filename = "burst-best.jpg"

        async def read(self) -> bytes:
            return encoded.tobytes()

    invalid = {
        "method": "pretend_super_resolution",
        "frames_requested": 3,
        "frames_encoded": 1,
        "frame_scores": [{"rank":1,"contrast":1,"edge_energy":1,"selection_score":1,"width_px":440,"height_px":250}],
        "selected_width_px": 440,
        "selected_height_px": 250,
    }
    with pytest.raises(Exception, match="Unknown browser capture method"):
        asyncio.run(analyze_web_upload(
            MemoryUpload(),
            page=0,
            corners_json=None,
            capture_metadata_json=json.dumps(invalid),
        ))


def test_browser_analysis_preserves_color_channels_in_residual_download():
    gray = synthetic_staff()
    color_bgr = np.stack([
        gray,
        np.clip(gray.astype(np.int16) * 3 // 4 + 25, 0, 255).astype(np.uint8),
        np.clip(gray.astype(np.int16) // 2 + 60, 0, 255).astype(np.uint8),
    ], axis=2)
    ok, encoded = cv2.imencode(".png", color_bgr)
    assert ok

    class MemoryColorUpload:
        filename = "color-score.png"

        async def read(self) -> bytes:
            return encoded.tobytes()

    result = asyncio.run(analyze_web_upload(MemoryColorUpload(), page=0, corners_json=None, capture_metadata_json=None))
    assert result["identity_reconstruction"]["analysis_raster_encoding"] == "uint8 3-channel BGR analysis raster"
    assert result["identity_reconstruction"]["exact_pixel_reconstruction_verified"] is True
    assert result["residual_download_available"] is True

    residual_bytes = base64.b64decode(result["residual_png_base64"], validate=True)
    residual = cv2.imdecode(np.frombuffer(residual_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    source_bytes = base64.b64decode(result["source_preview_png_base64"], validate=True)
    source = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    geometry_bytes = base64.b64decode(result["geometry_render_png_base64"], validate=True)
    geometry = cv2.imdecode(np.frombuffer(geometry_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert source.shape == residual.shape == color_bgr.shape
    assert residual.dtype == np.uint16
    recomposed = cv2.cvtColor(geometry, cv2.COLOR_GRAY2BGR).astype(np.int32) + residual.astype(np.int32) - 255
    assert np.array_equal(source, color_bgr)
    assert np.array_equal(recomposed.astype(np.uint8), color_bgr)


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
    assert (output_dir / "residual-offset255.png").exists()
    assert (output_dir / "reconstructed-analysis.png").exists()
    assert (output_dir / "identity-reconstruction.json").exists()
    geometry_pixels = cv2.imread(str(output_dir / "geometry-render.png"), cv2.IMREAD_GRAYSCALE)
    residual_pixels = cv2.imread(str(output_dir / "residual-offset255.png"), cv2.IMREAD_UNCHANGED)
    reconstructed_pixels = cv2.imread(str(output_dir / "reconstructed-analysis.png"), cv2.IMREAD_GRAYSCALE)
    assert residual_pixels.dtype == np.uint16
    assert residual_pixels.shape == original.shape
    reconstructed_math = geometry_pixels.astype(np.int32) + residual_pixels.astype(np.int32) - 255
    assert reconstructed_math.min() >= 0 and reconstructed_math.max() <= 255
    assert np.array_equal(reconstructed_math.astype(np.uint8), original)
    assert np.array_equal(reconstructed_pixels, original)
    identity_report = __import__("json").loads((output_dir / "identity-reconstruction.json").read_text(encoding="utf-8"))
    assert identity_report["exact_pixel_reconstruction_verified"] is True
    assert identity_report["mismatched_pixel_count"] == 0
    assert identity_report["maximum_absolute_pixel_error"] == 0
    assert saved_scene["identity_reconstruction"]["residual_file_sha256"]
    assert (output_dir / "scene.gltf").exists()
    gltf_text = (output_dir / "scene.gltf").read_text(encoding="utf-8")
    gltf_validation = validate_embedded_gltf(gltf_text)
    assert gltf_validation["valid"] is True
    assert result["gltf_path"] == str(output_dir / "scene.gltf")
    assert result["gltf_validation"]["nodes"] == gltf_validation["nodes"]
    assert gltf_validation["has_background_texture"] is True
    assert gltf_validation["background_texture_resolution_px"] == [original.shape[1], original.shape[0]]
    geometry_render = cv2.imread(str(output_dir / "geometry-render.png"), cv2.IMREAD_GRAYSCALE)
    assert geometry_render is not None and geometry_render.shape == original.shape
    assert int(geometry_render.min()) < 255
    assert (output_dir / "provisional.musicxml").exists()
    ET.parse(output_dir / "provisional.musicxml")


def test_color_source_residual_reconstructs_every_bgr_channel(tmp_path):
    gray = synthetic_staff()
    channels = [
        gray,
        np.clip(gray.astype(np.int16) * 3 // 4 + 25, 0, 255).astype(np.uint8),
        np.clip(gray.astype(np.int16) // 2 + 60, 0, 255).astype(np.uint8),
    ]
    color_bgr = np.stack(channels, axis=2)
    source_path = tmp_path / "color-score.png"
    assert cv2.imwrite(str(source_path), color_bgr)
    source_bytes = source_path.read_bytes()
    loaded_pixels = load_pixel_array(source_path)
    assert loaded_pixels.shape == color_bgr.shape
    assert np.array_equal(loaded_pixels, color_bgr)

    output_dir = tmp_path / "color-bundle"
    result = process_file(source_path, output_dir)
    analysis_pixels = cv2.imread(result["analysis_pixels_path"], cv2.IMREAD_UNCHANGED)
    geometry = cv2.imread(result["geometry_render_path"], cv2.IMREAD_GRAYSCALE)
    residual = cv2.imread(result["residual_path"], cv2.IMREAD_UNCHANGED)
    reconstructed = cv2.imread(result["reconstructed_raster_path"], cv2.IMREAD_UNCHANGED)
    report = result["identity_reconstruction"]

    assert result["source_hash_verified"] is True
    assert (output_dir / "source-original.png").read_bytes() == source_bytes
    assert analysis_pixels.shape == color_bgr.shape
    assert np.array_equal(analysis_pixels, color_bgr)
    assert residual.dtype == np.uint16 and residual.shape == color_bgr.shape
    geometry_bgr = cv2.cvtColor(geometry, cv2.COLOR_GRAY2BGR)
    recomposed = geometry_bgr.astype(np.int32) + residual.astype(np.int32) - 255
    assert recomposed.min() >= 0 and recomposed.max() <= 255
    assert np.array_equal(recomposed.astype(np.uint8), color_bgr)
    assert np.array_equal(reconstructed, color_bgr)
    assert report["analysis_raster_encoding"] == "uint8 3-channel BGR analysis raster"
    assert report["exact_pixel_reconstruction_verified"] is True
    assert report["mismatched_pixel_count"] == 0
    assert report["channel_value_count"] == int(color_bgr.size)
    assert report["maximum_absolute_pixel_error"] == 0
    gltf = __import__("json").loads((output_dir / "scene.gltf").read_text(encoding="utf-8"))
    assert gltf["extras"]["background_texture_embedded"] is True
    assert gltf["extras"]["identity_reconstruction"]["analysis_raster_encoding"] == report["analysis_raster_encoding"]


def test_pipeline_can_analyze_rectified_raster_and_keep_source_coordinates(tmp_path):
    gray, _truth = generate_synthetic_page(seed=52, staff_count=1, notes_per_staff=5)
    height, width = gray.shape
    rotation = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), 3.0, 1.0)
    rotated = cv2.warpAffine(gray, rotation, (width, height), borderValue=255)
    input_path = tmp_path / "rotated-source.png"
    assert cv2.imwrite(str(input_path), rotated)
    original_bytes = input_path.read_bytes()
    output_dir = tmp_path / "rectified-bundle"

    result = process_file(input_path, output_dir, rectify_skew=True)

    scene = result["scene"]
    diagnostic = scene["capture_diagnostics"]
    assert result["source_hash_verified"] is True
    assert (output_dir / scene["source"]["preserved_file"]).read_bytes() == original_bytes
    assert scene["coordinate_frame"] == "analysis_raster_pixel_centers"
    assert diagnostic["rotation_rectification_requested"] is True
    assert diagnostic["rotation_rectified"] is True
    assert abs(diagnostic["analysis_residual_staffline_angle_deg"] or 0.0) < 0.5
    assert diagnostic["source_image_shape_px"] == list(rotated.shape)
    assert diagnostic["analysis_image_shape_px"] == [scene["image"]["height"], scene["image"]["width"]]
    assert (output_dir / "analysis-raster.png").exists()
    assert (output_dir / "capture-transform.json").exists()


def test_pipeline_can_analyze_manual_page_quadrilateral_and_record_homography(tmp_path):
    gray, _truth = generate_synthetic_page(seed=53, staff_count=1, notes_per_staff=5)
    height, width = gray.shape
    input_path = tmp_path / "page.png"
    assert cv2.imwrite(str(input_path), gray)
    output_dir = tmp_path / "analysis-perspective"
    corners = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]

    result = process_file(input_path, output_dir, rectify_corners=corners)

    diagnostics = result["scene"]["capture_diagnostics"]
    assert diagnostics["perspective_rectification_requested"]
    assert diagnostics["perspective_rectified"]
    assert not diagnostics["rotation_rectified"]
    assert diagnostics["rectification_method"] == "manual_page_quadrilateral"
    assert result["source_hash_verified"]
    capture = __import__("json").loads((output_dir / "capture-transform.json").read_text(encoding="utf-8"))
    assert capture["rectification"]["method"] == "manual_page_quadrilateral"
    assert capture["source_bytes_preserved"]


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


def test_perspective_capture_benchmark_keeps_pairs_and_reports_both_modes():
    report = evaluate_perspective_capture(
        pages=2,
        seed=91,
        staff_count=1,
        notes_per_staff=5,
        noise_sigma=0.0,
        blur_sigma=0.0,
        bootstrap_rounds=100,
    )
    assert report["benchmark"] == "synthetic-registered-perspective-capture-v1"
    assert report["not_did"] is True
    assert report["data_status"] == "synthetic_registered_pairs_no_real_photographs"
    assert report["protocol"]["corner_ground_truth"].startswith("exact page quadrilateral")
    assert set(report["measurements"]) == {
        "source_reference", "projective_capture", "known_corner_rectified"
    }
    assert report["measurements"]["source_reference"]["pages_n"] == 2
    assert report["measurements"]["projective_capture"]["pages_n"] == 2
    assert report["measurements"]["known_corner_rectified"]["pages_n"] == 2
    assert report["measurements"]["projective_capture"]["pages_with_any_staff_detected"] == 0
    assert report["measurements"]["known_corner_rectified"]["pages_with_any_staff_detected"] == 2
    assert len(report["per_page"]["known_corner_rectified"]) == 2
    assert len(report["capture_pairs"]) == 2
    assert len(report["capture_pairs"][0]["projective_raster_sha256"]) == 64
    assert len(report["capture_pairs"][0]["source_to_projective_homography"]) == 3


def test_photo_smoke_reports_source_identity_without_claiming_accuracy(tmp_path):
    gray, _truth = generate_synthetic_page(seed=34, staff_count=1, notes_per_staff=5)
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), gray)
    manifest_path = tmp_path / "photo-manifest.json"
    manifest_path.write_text(__import__("json").dumps({
        "schema": "scoretwin-photo-smoke-v1",
        "pages": [{
            "id": "smoke-page",
            "image": image_path.name,
            "capture_type": "synthetic",
        }],
    }), encoding="utf-8")

    report = evaluate_photo_smoke(manifest_path)

    assert report["pages_n"] == 1
    assert report["pages_source_bytes_preserved_n"] == 1
    assert report["source_bytes_preserved_rate"] == 1.0
    assert len(report["source_bytes_preserved_rate_wilson_95_ci"]) == 2
    assert report["data_status"] == "no_independent_symbolic_or_geometric_ground_truth"
    assert report["claims_not_supported"]
    assert "accuracy benchmark" in report["epistemic_status"]
    assert "source_path" not in report["per_page"][0]
    assert report["per_page"][0]["source_bytes_preserved"] is True


def test_image_only_color_pdf_preserves_native_bgr_samples(tmp_path):
    import fitz

    gray, _truth = generate_synthetic_page(seed=42, staff_count=1, notes_per_staff=5)
    color = np.stack([
        gray,
        np.clip(gray.astype(np.int16) * 3 // 4 + 25, 0, 255).astype(np.uint8),
        np.clip(gray.astype(np.int16) // 2 + 60, 0, 255).astype(np.uint8),
    ], axis=2)
    ok, encoded = cv2.imencode(".png", color)
    assert ok
    document = fitz.open()
    page = document.new_page(width=color.shape[1], height=color.shape[0])
    page.insert_image(page.rect, stream=encoded.tobytes())
    pdf_path = tmp_path / "full-page-color-raster.pdf"
    document.save(pdf_path)
    document.close()

    loaded = load_pixel_array(pdf_path, page=0, dpi=300)

    assert loaded.shape == color.shape
    assert np.array_equal(loaded, color)


def test_image_only_pdf_uses_native_page_raster_without_dpi_upsampling(tmp_path):
    import fitz

    gray, _truth = generate_synthetic_page(seed=42, staff_count=1, notes_per_staff=5)
    ok, encoded = cv2.imencode(".png", gray)
    assert ok
    document = fitz.open()
    page = document.new_page(width=gray.shape[1], height=gray.shape[0])
    page.insert_image(page.rect, stream=encoded.tobytes())
    pdf_path = tmp_path / "full-page-raster.pdf"
    document.save(pdf_path)
    document.close()

    loaded = load_grayscale(pdf_path, page=0, dpi=300)

    assert loaded.shape == gray.shape
    assert np.array_equal(loaded, gray)
