"""Command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from .benchmark import evaluate_perspective_capture, evaluate_synthetic
from .benchmark_manifest import evaluate_manifest
from .imaging import load_grayscale, rectify_global_skew, rectify_page_quadrilateral
from .pipeline import process_file
from .proposal import save_model, train_proposal_model
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
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("scoretwin.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
