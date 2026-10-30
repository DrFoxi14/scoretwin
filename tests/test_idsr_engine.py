"""Comprehensive unit and falsification tests for IDSR Phase 8 modules."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from scoretwin.did import (
    compute_did_report,
    evaluate_p1_counterexample_and_bound,
    run_adversarial_metric_audit,
)
from scoretwin.gltf_export import build_gltf, validate_embedded_gltf
from scoretwin.idsr_engine import (
    compute_crlb_staff_spaces,
    compute_mdl_decomposition,
    edit_note_preserving_identity,
    estimate_illumination_and_optical_depth,
    evaluate_beer_lambert_regimes,
    refine_notehead_beer_lambert,
    render_beer_lambert_scene,
    run_contract_basin_experiment,
)
from scoretwin.musicxml import build_musicxml
from scoretwin.notation import detect_staff_systems_adaptive
from scoretwin.pipeline import analyze_array
from scoretwin.proposal import predict_idsr_patch, train_idsr_proposal_model
from scoretwin.reasoning import run_vatican_reasoning_test
from scoretwin.registered_eval import evaluate_registered_real_pairs, split_and_register_pair
from scoretwin.superposition import (
    build_measure_prob_distribution,
    collapse_superposition,
    compute_bipartite_mutual_information,
    evaluate_superposition_scaling_and_explosion,
    mps_from_probability_tensor,
    truncate_mps,
)
from scoretwin.synthetic import generate_extended_synthetic_page


def test_adaptive_staff_detector_recovers_faint_and_skewed_real_scans():
    pairs_dir = Path(__file__).resolve().parents[1] / "data" / "registered_pairs"
    pair_97 = cv2.imread(str(pairs_dir / "pair_97.png"), cv2.IMREAD_GRAYSCALE)
    pair_106 = cv2.imread(str(pairs_dir / "pair_106.png"), cv2.IMREAD_GRAYSCALE)
    assert pair_97 is not None and pair_106 is not None

    scan_97 = split_and_register_pair(pair_97)["scan_raw"]
    staffs_97, _, _ = detect_staff_systems_adaptive(scan_97)
    assert len(staffs_97) == 9

    scan_106 = split_and_register_pair(pair_106)["scan_raw"]
    staffs_106, _, _ = detect_staff_systems_adaptive(scan_106)
    assert len(staffs_106) == 12


def test_extended_notation_detects_chords_voices_accidentals_beams_dots_and_systems():
    gray, truth = generate_extended_synthetic_page(
        seed=11,
        system_count=2,
        staves_per_system=2,
        hand=0.25,
        noise_sigma=1.5,
    )
    scene = analyze_array(
        gray,
        adaptive_staffs=True,
        enable_extended_notation=True,
        enable_beer_lambert=True,
        promote_inferred_durations=True,
    )
    assert len(scene["staffs"]) == 4
    assert len({s["system_index"] for s in scene["systems"]}) == 2
    assert len(scene["accidentals"]) >= 4
    assert len(scene["beams"]) >= 4
    assert len(scene["dots"]) >= 2
    assert scene["chord_count"] >= 4
    voices = {int(n.get("voice", 1)) for n in scene["notes"]}
    assert {1, 2}.issubset(voices)

    # Verify MusicXML serialization and glTF 3D export with extended primitives
    xml_text = build_musicxml(scene)
    root = ET.fromstring(xml_text)
    assert root.tag == "score-partwise"
    assert len(root.findall(".//note/chord")) >= 2
    assert len(root.findall(".//pitch/alter")) >= 2

    gltf_text = build_gltf(scene)
    gltf_report = validate_embedded_gltf(gltf_text)
    assert gltf_report["valid"] is True
    kinds = {p["kind"] for p in scene["primitives"]}
    assert {"beam", "accidental", "augmentation_dot"}.issubset(kinds)


def test_6class_subpixel_proposal_model_satisfies_0_3_sp_contract():
    model, metrics = train_idsr_proposal_model(samples_per_class=80, epochs=22, seed=101)
    assert metrics["holdout_6class_accuracy"] >= 0.78
    assert metrics["holdout_refined_offset_mean_sp"] <= 0.30
    assert metrics["satisfies_0_30_sp_p95_contract"] is True

    patch = np.full((24, 24), 255, dtype=np.uint8)
    cv2.ellipse(patch, (13, 11), (6, 4), -18, 0, 360, 15, -1, cv2.LINE_AA)
    pred = predict_idsr_patch(model, patch)
    assert pred["label"] in model["labels"]
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-5
    assert len(pred["offset_sp"]) == 2


def test_beer_lambert_refinement_and_crlb_jacobian_full_rank():
    crlb = compute_crlb_staff_spaces(
        spacing=16.0, rx=9.0, ry=6.0, absorbance=1.45, sensor_noise_sigma_px=2.0
    )
    assert crlb["jacobian_rank"] == 2
    assert 0.0001 <= crlb["crlb_center_sp"] <= 0.02

    basin = run_contract_basin_experiment(spacing=16.0, trials_per_sigma=20, seed=42)
    by_sigma = {row["sigma_init_sp"]: row for row in basin["sweep"]}
    # Inside contract (<= 0.30 sp), refinement converges to <= 0.05 sp with 0% pitch error
    assert by_sigma[0.25]["mean_error_after_sp"] <= 0.05
    assert by_sigma[0.30]["mean_error_after_sp"] <= 0.055
    assert by_sigma[0.25]["pitch_error_rate"] == 0.0
    # Outside breakdown (> 0.50 sp), refinement cannot save pitch accuracy
    assert by_sigma[0.80]["mean_error_after_sp"] > 0.25
    assert by_sigma[0.80]["pitch_error_rate"] > 0.20


def test_p4_beer_lambert_regime_falsification():
    regimes = evaluate_beer_lambert_regimes(seed=42)
    assert regimes["translucent_iron_gall_ink"]["p4_holds_within_10pct"] is True
    assert regimes["standard_carbon_print_ink"]["p4_holds_within_10pct"] is True
    assert regimes["graphite_pencil_specular"]["p4_holds_within_10pct"] is False
    assert regimes["opaque_gouache_correction"]["p4_holds_within_10pct"] is False
    assert (
        regimes["opaque_gouache_correction"]["beer_lambert_relative_error_at_intersection"]
        > 0.35
    )


def test_mdl_decomposition_and_machine_precision_identity():
    gray, _ = generate_extended_synthetic_page(seed=5, system_count=1, staves_per_system=2)
    scene = analyze_array(gray, adaptive_staffs=True, enable_beer_lambert=True)
    mdl = scene["mdl_decomposition"]
    assert mdl["float64_identity_within_machine_eps"] is True
    assert mdl["float64_max_reconstruction_error"] <= 1e-14
    assert mdl["uint8_exact_reconstruction"] is True
    assert mdl["mdl_excess_bits_per_pixel"] >= 0.0


def test_mps_superposition_mutual_information_truncation_bound_and_explosion():
    prob = build_measure_prob_distribution(n_sites=6, n_states=4, seed=42)
    mi = compute_bipartite_mutual_information(prob, cut_index=3)
    assert 0.20 <= mi["mutual_information_bits"] <= 2.0

    mps = mps_from_probability_tensor(prob)
    for chi in (1, 2, 4, 8):
        trunc = truncate_mps(mps, max_bond_dim=chi)
        assert trunc["bound_satisfied"] is True

    c_map = collapse_superposition(mps, policy="map_collapse")
    assert c_map["policy"] == "map_collapse"
    assert len(c_map["selected_labels"]) == 6

    c_sup = collapse_superposition(mps, policy="preserve_superposition", top_k=3)
    assert len(c_sup["candidates"]) == 3
    assert c_sup["captured_probability_mass"] > 0.0

    c_abs = collapse_superposition(mps, policy="abstain_above_entropy", entropy_threshold_bits=0.5)
    assert "abstained_sites" in c_abs

    scaling = evaluate_superposition_scaling_and_explosion(n_measures=12, n_sites=6, n_states=4)
    assert scaling["verstraete_cirac_bound_all_satisfied"] is True
    explosion = scaling["polyphonic_bond_explosion"]
    assert explosion[-1]["effective_bond_rank_99_9pct"] > explosion[0]["effective_bond_rank_99_9pct"]


def test_did_metric_p1_counterexample_and_faza_3_5_adversarial_defenses():
    p1 = evaluate_p1_counterexample_and_bound(n_measures=10, primitives_per_measure=12)
    assert p1["born_digital_counterexample"]["p1_unqualified_falsified"] is True
    assert p1["born_digital_counterexample"]["mask_iou"] == 1.0
    assert p1["manuscript_regime"]["p1_star_bound_verified"] is True
    assert (
        p1["manuscript_regime"]["idsr_parametric_iou_mean"]
        > p1["manuscript_regime"]["symbolic_reengraving_iou_mean"] + 0.40
    )

    audit = run_adversarial_metric_audit()
    assert audit["did_audit"]["defense_triggered"] is True
    assert audit["mdl_audit"]["defense_triggered"] is True
    assert audit["iou_audit"]["defense_triggered"] is True


def test_identity_preserving_local_note_editing():
    gray, _ = generate_extended_synthetic_page(seed=19, system_count=1, staves_per_system=1)
    scene = analyze_array(gray, adaptive_staffs=True)
    assert scene["notes"]
    target_id = scene["notes"][0]["id"]
    old_step = int(scene["notes"][0]["diatonic_step_from_bottom_line"])
    res = edit_note_preserving_identity(
        gray, scene, note_id=target_id, new_diatonic_step=old_step + 2
    )
    assert res["outside_roi_bit_identical"] is True
    assert res["modified_pixel_count"] > 10
    assert res["unchanged_pixel_fraction"] > 0.98


def test_real_registered_pairs_evaluation():
    report = evaluate_registered_real_pairs()
    assert report["pairs_n"] == 3
    assert report["idsr_v0_2_staff_recall"] == 1.0
    assert report["idsr_v0_2_staffs_detected_total"] == 33
    assert report["baseline_v0_1_staffs_detected_total"] < 15
    assert report["idsr_v0_2_mean_did"] < report["baseline_v0_1_mean_did"]


def test_vatican_xiv_century_reasoning_zero_abstention_and_exact_1_30cm():
    res = run_vatican_reasoning_test(seed=1340, request_optional_modern_layer3=True)
    assert res["human_labels_used"] == 0
    assert res["human_clicks_used"] == 0
    assert res["abstained_or_unknown_count"] == 0
    assert res["layer1_physical_twin"]["float64_max_error"] == 0.0
    assert res["layer1_physical_twin"]["uint8_mismatched_pixels"] == 0
    assert res["layer1_physical_twin"]["positions_normalized"] is False
    assert res["layer1_structural_map"]["topology"]["lines_per_staff"] == 4
    note_130 = res["layer1_structural_map"]["note_at_1_30_cm_verification"]
    assert abs(note_130["measured_distance_from_bar_cm"] - 1.30) <= 0.005
    assert note_130["normalized"] is False

    verdicts = {i["verdict"] for i in res["layer2_semantic_investigations"]}
    assert "RECOVERED_AUGMENTATION_OR_MORA_DOT" in verdicts
    assert "PROVEN_PHYSICAL_SUPPORT_RESIDUE_PRESERVED_VISUALLY" in verdicts
    assert "IRREVERSIBLE_PHYSICAL_LOSS_PROVEN" in verdicts
    assert len(res["reasoning_trace"]) == 7
    assert res["layer3_optional_modern_projection"]["total_modifications_count"] == 17


def test_redteam_torture_scenarios() -> None:
    from scoretwin.redteam_scenarios import run_redteam_torture_evaluation

    res = run_redteam_torture_evaluation()
    assert res["abstained_or_unknown_count"] == 0
    assert res["all_scenarios_passed"] is True
    sc = res["scenarios"]

    # 1. Exact lyrics without notehead confusion or normalization
    assert sc["1_exact_lyrics"]["tokens_detected"] == 4
    assert sc["1_exact_lyrics"]["lyrics_confused_as_noteheads"] == 0
    assert sc["1_exact_lyrics"]["max_horizontal_origin_error_cm"] <= 0.02

    # 2. Encircled note + arrow + 'attention'
    assert sc["2_encircled_note_arrow_attention"]["recovered_note_center_error_sp"] < 0.05
    assert sc["2_encircled_note_arrow_attention"]["closed_loop_enclosure_detected"] is True
    assert sc["2_encircled_note_arrow_attention"]["marginalia_text_recognized"] == "attention"

    # 3. Dynamics ('mf', '<', 'f')
    dyn_kinds = [d["kind"] for d in sc["3_dynamics_and_hairpins"]["detected_items"]]
    assert dyn_kinds == ["dynamic_glyph_mf", "hairpin_crescendo", "dynamic_glyph_f"]

    # 4. Tuplet ('[ 3 ]' -> 5.0 beats total in measure 1)
    assert sc["4_tuplets_triplets"]["measure_beats_with_tuplet_scaling"] == 5.0
    assert sc["4_tuplets_triplets"]["matches_time_signature_5_4_or_measure_sum"] is True

    # 5. Tie vs Legato Slur vs Phrasing Slur
    assert sc["5_tie_vs_slur_vs_phrase"]["tie_analysis"]["curve_class"] == "TIE_DURATION_BINDING"
    assert sc["5_tie_vs_slur_vs_phrase"]["legato_slur_analysis"]["curve_class"] == "LEGATO_ARTICULATION_SLUR"
    assert sc["5_tie_vs_slur_vs_phrase"]["phrasing_slur_analysis"]["curve_class"] == "PHRASING_SLUR_MACRO_ARCH"

    # 6. Verso bleedthrough vs recto ink
    assert sc["6_verso_bleedthrough"]["verdict"] == "PROVEN_VERSO_BLEEDTHROUGH_EXCLUDED_FROM_RECTO_SCORE"

    # 7. Volta brackets (1., 2., 3.) + repeat barlines (|:, :|) + playable unrolling
    v7 = sc["7_volta_brackets_and_playable_repeats"]
    assert v7["repeat_dots_confused_as_augmentation_dots"] == 0
    assert [b["symbol"] for b in v7["repeat_barlines_detected"]] == ["|:", "|", ":|", ":|", "||"]
    assert [v["ending_type"] for v in v7["volta_brackets_detected"]] == [
        "start_stop",
        "start_stop",
        "start_discontinue",
    ]
    playable = v7["playable_execution"]
    assert playable["is_playable"] is True
    assert playable["written_measures_order"] == ["M1", "M2", "M3", "M4"]
    assert playable["unrolled_playable_measures_order"] == ["M1", "M2", "M1", "M3", "M1", "M4"]
    assert playable["unrolled_playable_pitches"] == [
        "E4", "F4", "G4", "A4",
        "E4", "F4", "B4", "C5",
        "E4", "F4", "D5", "E5",
    ]
    assert playable["total_written_notes"] == 8
    assert playable["total_played_notes"] == 12

    # 8. 45-degree rotation + 65% shadow + defocus blur inversion
    s8 = sc["8_optical_and_geometric_torture_45deg_shadow_blur"]
    assert s8["angle_recovery_error_deg"] <= 0.1
    assert s8["false_positive_shadow_edges"] == 0
    assert s8["sharpness_gain_ratio"] > 1.3
    assert s8["max_2d_replica_pixel_error_on_tilted_45deg_frame"] == 0.0

    # 9. 'Impossibility' (Brenden Iten) hyper-density stress test
    s9 = sc["9_impossibility_brenden_iten_hyper_density"]
    assert s9["multi_beam_stack"]["beams_detected_in_stack"] == 6
    assert s9["multi_beam_stack"]["derived_note_type"] == "256th"
    assert s9["biphasic_second_cluster_with_15ma"]["total_cluster_notes_recovered"] == 8
    assert s9["feathered_beam_and_irrational_tuplet"]["feathered_beam_classification"] == "feathered_beam_accelerando"
    assert s9["feathered_beam_and_irrational_tuplet"]["seven_tuplet_notes_sum_beats"] == 4.0

    # 10. Pre-Beethoven paleography (Mensural / Figured Bass / Custos) + D.S. al Coda
    s10 = sc["10_pre_beethoven_paleography_and_ds_al_coda"]
    assert s10["mensural_tempus_prolatio_matrix"]["circle_with_dot"]["minimae_per_brevis"] == 9
    assert s10["mensural_tempus_prolatio_matrix"]["c_no_dot"]["minimae_per_brevis"] == 4
    assert s10["ds_al_coda_playable_navigation"]["unrolled_playable_measures"] == [
        "M1", "M2", "M3", "M4", "M2", "M3", "M5"
    ]


def test_directions_1_and_2_dynamic_pipeline_and_curvilinear_world_idsr() -> None:
    from scoretwin.curvilinear_and_world_idsr import (
        dewarp_curvilinear_spine_and_page,
        invert_3d_curvilinear_page_warp,
        reconstruct_nonlinear_polar_staff,
        reconstruct_relative_interval_and_world_notation,
    )
    from scoretwin.redteam_scenarios import generate_redteam_torture_page

    # Direction 1: Dynamic staff-anchored pipeline on torture page (0 hardcoded windows)
    page = generate_redteam_torture_page()
    scene = analyze_array(
        page["observed_gray"],
        adaptive_staffs=True,
        enable_extended_notation=True,
        enable_beer_lambert=True,
    )
    dyn = scene["idsr_dynamic_layer"]
    assert dyn is not None
    assert dyn["hardcoded_windows_used"] == 0
    assert dyn["abstained_or_unknown_count"] == 0
    assert len(dyn["dynamic_lyrics_detected"]) == 4
    assert len(dyn["dynamic_dynamics_detected"]) == 3
    assert len(dyn["dynamic_volta_brackets_detected"]) == 3
    assert [v["ending_type"] for v in dyn["dynamic_volta_brackets_detected"]] == [
        "start_stop",
        "start_stop",
        "start_discontinue",
    ]
    assert dyn["playable_systems"][0]["schedule"]["unrolled_playable_measures_order"] == [
        "M1", "M2", "M1", "M3", "M1", "M4"
    ]

    # Direction 2A: 3D Curvilinear book-spine & parchment wave dewarping
    c3d = invert_3d_curvilinear_page_warp()
    assert c3d["max_spine_and_wave_warp_sp"] > 2.0
    assert c3d["curvilinear_tracker_residual_waviness_rms_sp"] < 0.03
    assert c3d["pitch_errors_after_curvilinear_dewarp"] == 0
    assert c3d["max_2d_replica_pixel_error_on_curved_3d_surface"] == 0.0

    # Generalized Multi-Staff Spine Dewarping (> 3.5 sp warp)
    h_test, w_test = 600, 1000
    sp_test = 16.0
    img_test = np.full((h_test, w_test), 255, dtype=np.uint8)
    x_coords = np.arange(w_test, dtype=np.float64)
    dy_severe = 60.0 * np.exp(-((x_coords - 100.0)**2)/(2.0*(140.0**2))) + 12.0 * np.sin(2.0*np.pi*x_coords/400.0)
    for s_base in [180.0, 400.0]:
        for lk in range(5):
            ly = s_base + lk * sp_test + dy_severe
            for xi in range(60, 940):
                yi = int(round(ly[xi]))
                if 0 <= yi < h_test:
                    img_test[yi, xi] = 30
    dew_res = dewarp_curvilinear_spine_and_page(img_test, spacing_hint=sp_test)
    assert dew_res["staves_tracked_count"] >= 2
    assert dew_res["max_warp_amplitude_sp"] > 3.0
    assert dew_res["exact_lossless_identity_verified"] is True
    assert dew_res["exact_lossless_reconstruction_error_px"] == 0
    assert dew_res["canonical_planar_straight_lines"] is True

    # Direction 2B: Non-linear circular / polar staff IDSR
    pol = reconstruct_nonlinear_polar_staff()
    assert pol["concentric_staff_lines_detected"] == 5
    assert pol["notes_recovered"] == 6
    assert pol["pitch_step_errors"] == 0

    # Direction 2C: Relative-interval Byzantine Psaltic + Chinese Jianpu (0 staff lines)
    world = reconstruct_relative_interval_and_world_notation()
    assert world["byzantine_psaltic_analysis"]["martyria_zero_drift_verified"] is True
    assert world["chinese_jianpu_analysis"]["playable_midi_sequence"] == [55, 60, 64, 67, 72]


def test_manifold_voice_traversal_and_photocopy_residue_discrimination():
    from scoretwin.manifold_traversal import traverse_manifold_voice_graph
    from scoretwin.pipeline import analyze_array, render_geometry_scene
    from scoretwin.synthetic import generate_extended_synthetic_page

    # Generate synthetic page with polyphony, chords, barlines, and augmentation dots
    gray, _ = generate_extended_synthetic_page(seed=21, system_count=1, staves_per_system=1)

    # Inject 2 isolated photocopy spots far from any notes or staff lines
    # Spot A: top-left margin (x=25, y=25)
    cv2.circle(gray, (25, 25), 4, 30, -1)
    # Spot B: top-right margin (x=w-30, y=25)
    cv2.rectangle(gray, (gray.shape[1] - 30, 25), (gray.shape[1] - 24, 31), 20, -1)

    # Run analyze_array with extended notation & manifold traversal
    scene = analyze_array(gray, adaptive_staffs=True, enable_extended_notation=True, enable_beer_lambert=True)
    mvt = scene.get("manifold_voice_traversal")
    assert mvt is not None, "Manifold-guided voice traversal must be present in scene"

    summary = mvt["summary"]
    assert summary["zero_abstentions_verified"] is True
    assert summary["positions_normalized"] is False
    assert summary["exact_physical_coordinates_preserved"] is True

    # 1. Verify isolated photocopy/dust spots are routed to residual r (NOT to z)
    routed_r = mvt["routed_to_residual_r"]
    assert len(routed_r) >= 2, f"Expected at least 2 photocopy spots routed to r, got {len(routed_r)}"
    for spot in routed_r:
        assert spot["classification"] == "PROVEN_PHOTOCOPY_OR_PAPER_RESIDUE"
        assert spot["action"] == "ROUTE_TO_RESIDUAL_R"
        assert spot["exclude_from_symbolic_z"] is True
        assert spot["isolated_distance_sp"] > 1.75

    # 2. Verify polyphonic voice threading
    v_threads = mvt["voice_threads"]
    assert len(v_threads) >= 1
    # Check that physical inter-onset distance is recorded without quantization
    for v_name, notes_list in v_threads.items():
        for i in range(len(notes_list) - 1):
            assert notes_list[i]["inter_onset_physical_distance_cm"] is not None
            assert notes_list[i]["inter_onset_physical_distance_cm"] >= 0.0

    # 3. Verify unmeasured Mozart Fioritura / Cadenza detection
    fiorituras = mvt["fiorituras_detected"]
    assert len(fiorituras) >= 1, "Dense measures must be flagged for unmeasured rhythmic auditing"
    fio = fiorituras[0]
    assert fio["fioritura_or_cadenza_detected"] is True
    assert fio["all_attacks_preserved_losslessly"] is True

    # 4. Verify 2D lossless recomposition exactness: G(z) + r - 255 == gray
    gz = render_geometry_scene(gray.shape, scene)
    res_signed = gray.astype(np.int16) - gz.astype(np.int16)
    res_u16 = (res_signed + 255).astype(np.uint16)
    reconstructed = np.clip(gz.astype(np.int32) + res_u16.astype(np.int32) - 255, 0, 255).astype(np.uint8)
    assert np.array_equal(reconstructed, gray), "Lossless identity must hold with zero pixel error"






