"""Command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from .benchmark import evaluate_perspective_capture, evaluate_synthetic
from .benchmark_manifest import evaluate_manifest
from .benchmark_photo_smoke import evaluate_photo_smoke
from .gltf_export import validate_embedded_gltf
from .idsr_engine import edit_note_preserving_identity
from .imaging import load_grayscale, rectify_global_skew, rectify_page_quadrilateral
from .pipeline import analyze_array, process_file
from .proposal import save_model, train_proposal_model
from .reasoning import run_vatican_reasoning_test
from .registered_eval import run_all_idsr_experiments
from .synthetic import generate_synthetic_page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scoretwin", description="ScoreTwin / IDSR research prototype")
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Analyze an image or PDF page")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--out", type=Path, default=Path("scoretwin-run"))
    analyze.add_argument("--page", type=int, default=0)
    analyze.add_argument("--model", type=Path, default=None)
    analyze.add_argument("--rectify-skew", action="store_true", help="Use a reversible global-skew analysis raster")
    analyze.add_argument("--max-skew-degrees", type=float, default=12.0)
    analyze.add_argument(
        "--corners-json",
        default=None,
        help='Manual page corners as JSON: [[x_tl,y_tl],[x_tr,y_tr],[x_br,y_br],[x_bl,y_bl]] in source pixels',
    )

    train = commands.add_parser("train-model", help="Train the synthetic patch proposal model")
    train.add_argument("--out", type=Path, default=Path("models/proposal.npz"))
    train.add_argument("--samples-per-class", type=int, default=400)
    train.add_argument("--epochs", type=int, default=45)
    train.add_argument("--seed", type=int, default=19)

    demo = commands.add_parser("demo", help="Generate and analyze a reproducible synthetic score page")
    demo.add_argument("--out", type=Path, default=Path("scoretwin-demo"))
    demo.add_argument("--seed", type=int, default=3)
    demo.add_argument("--staff-count", type=int, default=2)
    demo.add_argument("--notes-per-staff", type=int, default=9)

    benchmark = commands.add_parser("evaluate-synthetic", help="Evaluate note proposals on synthetic pages")
    benchmark.add_argument("--pages", type=int, default=10)
    benchmark.add_argument("--seed", type=int, default=20261029)
    benchmark.add_argument("--staff-count", type=int, default=2)
    benchmark.add_argument("--notes-per-staff", type=int, default=9)
    benchmark.add_argument("--noise-sigma", type=float, default=0.0)
    benchmark.add_argument("--blur-sigma", type=float, default=0.0)
    benchmark.add_argument("--contract-staff-spaces", type=float, default=0.3)
    benchmark.add_argument("--bootstrap-rounds", type=int, default=2000)
    benchmark.add_argument("--out", type=Path, default=Path("synthetic-evaluation.json"))

    perspective = commands.add_parser(
        "evaluate-perspective",
        help="Compare notehead proposals before and after known-corner projective correction",
    )
    perspective.add_argument("--pages", type=int, default=20)
    perspective.add_argument("--seed", type=int, default=20261029)
    perspective.add_argument("--staff-count", type=int, default=2)
    perspective.add_argument("--notes-per-staff", type=int, default=9)
    perspective.add_argument("--spacing", type=int, default=16)
    perspective.add_argument("--noise-sigma", type=float, default=2.0)
    perspective.add_argument("--blur-sigma", type=float, default=0.35)
    perspective.add_argument("--contract-staff-spaces", type=float, default=0.3)
    perspective.add_argument("--bootstrap-rounds", type=int, default=2000)
    perspective.add_argument("--out", type=Path, default=Path("perspective-evaluation.json"))

    manifest = commands.add_parser("evaluate-manifest", help="Evaluate notehead proposals on annotated external pages")
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--model", type=Path, default=None)
    manifest.add_argument("--out", type=Path, default=Path("manifest-evaluation.json"))
    manifest.add_argument("--tolerance-sp", type=float, default=0.35)
    manifest.add_argument("--bootstrap-samples", type=int, default=1000)
    manifest.add_argument("--bootstrap-seed", type=int, default=20261029)

    photo_smoke = commands.add_parser(
        "evaluate-photo-smoke",
        help="Run an unlabeled smoke evaluation on supplied camera/scan pages",
    )
    photo_smoke.add_argument("--manifest", type=Path, required=True)
    photo_smoke.add_argument("--model", type=Path, default=None)
    photo_smoke.add_argument("--out", type=Path, default=Path("photo-smoke-evaluation.json"))

    rectify = commands.add_parser(
        "rectify-preview",
        help="Create an optional global-skew preview with reversible source coordinates",
    )
    rectify.add_argument("input", type=Path)
    rectify.add_argument("--out", type=Path, default=Path("capture-rectification"))
    rectify.add_argument("--page", type=int, default=0)
    rectify.add_argument("--dpi", type=int, default=200)
    rectify.add_argument("--max-skew-degrees", type=float, default=12.0)
    rectify.add_argument(
        "--corners-json",
        default=None,
        help='Manual page corners as JSON: [[x_tl,y_tl],[x_tr,y_tr],[x_br,y_br],[x_bl,y_bl]] in source pixels',
    )

    validate_gltf = commands.add_parser("validate-gltf", help="Validate a self-contained glTF 2.0 scene")
    validate_gltf.add_argument("input", type=Path, help="Path to a ScoreTwin .gltf JSON file")

    eval_idsr = commands.add_parser(
        "evaluate-idsr",
        help="Run the complete reproducible IDSR experimental suite (Phases -1, 0, 3, 3.5, 8)",
    )
    eval_idsr.add_argument(
        "--out",
        type=Path,
        default=Path("results/idsr-phase8-experiments-2026-10-29.json"),
    )

    edit_cmd = commands.add_parser(
        "edit-note",
        help="Edit a single note's pitch while preserving outside-ROI pixels bit-for-bit",
    )
    edit_cmd.add_argument("input", type=Path)
    edit_cmd.add_argument("--note-id", default=None)
    edit_cmd.add_argument("--new-step", type=int, default=4)
    edit_cmd.add_argument("--out", type=Path, default=Path("edited-score.png"))

    vatican_cmd = commands.add_parser(
        "vatican-test",
        help="Run zero-shot ScoreTwin-Reasoning with decolorize_and_investigate on 14th-century manuscript",
    )
    vatican_cmd.add_argument("--seed", type=int, default=1340)
    vatican_cmd.add_argument("--modern-layer3", action="store_true", default=True)
    vatican_cmd.add_argument(
        "--out",
        type=Path,
        default=Path("results/vatican-test-xiv-2026-10-30.json"),
    )

    serve = commands.add_parser("serve", help="Run the local browser interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "analyze":
        result = process_file(
            args.input,
            args.out,
            page=args.page,
            model_path=args.model,
            rectify_skew=args.rectify_skew,
            max_skew_degrees=args.max_skew_degrees,
            rectify_corners=json.loads(args.corners_json) if args.corners_json else None,
        )
        print(json.dumps({
            "output_dir": result["output_dir"],
            "source_sha256": result["source_sha256"],
            "source_hash_verified": result["source_hash_verified"],
            "staff_count": len(result["scene"]["staffs"]),
            "note_count": result["scene"]["note_count"],
            "capture_diagnostics": result["scene"]["capture_diagnostics"],
            "warnings": result["scene"]["warnings"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "train-model":
        model, metrics = train_proposal_model(args.samples_per_class, args.epochs, args.seed)
        save_model(model, args.out)
        print(json.dumps({"model_path": str(args.out), **metrics}, indent=2, ensure_ascii=False))
    elif args.command == "demo":
        args.out.mkdir(parents=True, exist_ok=True)
        image_path = args.out / "demo-score.png"
        gray, truth = generate_synthetic_page(
            seed=args.seed,
            staff_count=args.staff_count,
            notes_per_staff=args.notes_per_staff,
        )
        if not cv2.imwrite(str(image_path), gray):
            raise IOError(f"Could not write demo image to {image_path}")
        (args.out / "ground-truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
        result = process_file(image_path, args.out / "analysis")
        print(json.dumps({
            "demo_image": str(image_path),
            "analysis_dir": result["output_dir"],
            "ground_truth_notes": len(truth),
            "proposed_notes": result["scene"]["note_count"],
            "source_hash_verified": result["source_hash_verified"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "evaluate-synthetic":
        result = evaluate_synthetic(
            pages=args.pages,
            seed=args.seed,
            staff_count=args.staff_count,
            notes_per_staff=args.notes_per_staff,
            noise_sigma=args.noise_sigma,
            blur_sigma=args.blur_sigma,
            contract_staff_spaces=args.contract_staff_spaces,
            bootstrap_rounds=args.bootstrap_rounds,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report_path": str(args.out), **{k: v for k, v in result.items() if k != "per_page"}},
                         indent=2, ensure_ascii=False))
    elif args.command == "evaluate-perspective":
        result = evaluate_perspective_capture(
            pages=args.pages,
            seed=args.seed,
            staff_count=args.staff_count,
            notes_per_staff=args.notes_per_staff,
            spacing=args.spacing,
            noise_sigma=args.noise_sigma,
            blur_sigma=args.blur_sigma,
            contract_staff_spaces=args.contract_staff_spaces,
            bootstrap_rounds=args.bootstrap_rounds,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report_path": str(args.out), "measurements": result["measurements"]},
                         indent=2, ensure_ascii=False))
    elif args.command == "evaluate-manifest":
        result = evaluate_manifest(
            args.manifest,
            model_path=args.model,
            tolerance_staff_spaces=args.tolerance_sp,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "report_path": str(args.out),
            "pages_n": result["pages_n"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "confidence_intervals": result["precision_recall_f1_bootstrap_95_ci"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "evaluate-photo-smoke":
        result = evaluate_photo_smoke(args.manifest, model_path=args.model)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "report_path": str(args.out),
            "pages_n": result["pages_n"],
            "pages_source_bytes_preserved_n": result["pages_source_bytes_preserved_n"],
            "data_status": result["data_status"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "rectify-preview":
        source_bytes = args.input.read_bytes()
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        gray = load_grayscale(args.input, page=args.page, dpi=args.dpi)
        if args.corners_json:
            rectification = rectify_page_quadrilateral(gray, json.loads(args.corners_json))
        else:
            rectification = rectify_global_skew(gray, max_abs_skew_degrees=args.max_skew_degrees)
        args.out.mkdir(parents=True, exist_ok=True)
        source_copy = args.out / "source" / args.input.name
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        source_copy.write_bytes(source_bytes)
        copied_source_sha256 = hashlib.sha256(source_copy.read_bytes()).hexdigest()
        rectified_path = args.out / "rectified-preview.png"
        if not cv2.imwrite(str(rectified_path), rectification.image):
            raise IOError(f"Could not write rectified preview to {rectified_path}")
        metadata = {
            "schema": "scoretwin-capture-rectification-0.2",
            "source_file_name": args.input.name,
            "source_sha256": source_sha256,
            "source_copy_sha256": copied_source_sha256,
            "source_bytes_preserved": source_sha256 == copied_source_sha256,
            "source_page_index": args.page if args.input.suffix.lower() == ".pdf" else None,
            "source_image_shape_px": [int(gray.shape[0]), int(gray.shape[1])],
            "rectified_image_shape_px": [
                int(rectification.image.shape[0]), int(rectification.image.shape[1])
            ],
            "rectification": rectification.to_dict(),
            "rectified_preview_file": rectified_path.name,
        }
        metadata_path = args.out / "capture-transform.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "output_dir": str(args.out),
            "metadata_file": str(metadata_path),
            "rectified_preview_file": str(rectified_path),
            "source_bytes_preserved": metadata["source_bytes_preserved"],
            "estimated_skew_degrees": rectification.estimated_skew_degrees,
            "applied_correction_degrees": rectification.applied_correction_degrees,
            "rectification_applied": rectification.applied,
            "rectification_method": rectification.method,
            "reason": rectification.reason,
        }, indent=2, ensure_ascii=False))
    elif args.command == "validate-gltf":
        report = validate_embedded_gltf(args.input.read_text(encoding="utf-8"))
        print(json.dumps({"input": str(args.input), **report}, indent=2, ensure_ascii=False))
    elif args.command == "evaluate-idsr":
        report = run_all_idsr_experiments(output_json_path=args.out)
        print(json.dumps({
            "report_path": str(args.out),
            "reproducibility": report["reproducibility"],
            "real_pairs_mean_did_v0_1": report["faza_8_real_registered_pairs"]["baseline_v0_1_mean_did"],
            "real_pairs_mean_did_v0_2": report["faza_8_real_registered_pairs"]["idsr_v0_2_mean_did"],
            "real_pairs_staff_recall_v0_2": report["faza_8_real_registered_pairs"]["idsr_v0_2_staff_recall"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "edit-note":
        gray = load_grayscale(args.input)
        scene = analyze_array(gray, adaptive_staffs=True)
        if not scene["notes"]:
            raise ValueError("No notes detected in input image.")
        target_id = args.note_id or scene["notes"][0]["id"]
        edit_res = edit_note_preserving_identity(
            gray, scene, note_id=target_id, new_diatonic_step=args.new_step
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), edit_res["edited_image"])
        print(json.dumps({
            "edited_image_path": str(args.out),
            "note_id": target_id,
            "outside_roi_bit_identical": edit_res["outside_roi_bit_identical"],
            "unchanged_pixel_fraction": edit_res["unchanged_pixel_fraction"],
            "modified_pixel_count": edit_res["modified_pixel_count"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "vatican-test":
        report = run_vatican_reasoning_test(
            seed=args.seed,
            request_optional_modern_layer3=bool(args.modern_layer3),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({
            "report_path": str(args.out),
            "page_sha256": report["page_sha256"],
            "abstained_or_unknown_count": report["abstained_or_unknown_count"],
            "layer1_physical_twin": report["layer1_physical_twin"],
            "note_at_1_30_cm_verification": report["layer1_structural_map"]["note_at_1_30_cm_verification"],
        }, indent=2, ensure_ascii=False))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("scoretwin.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
