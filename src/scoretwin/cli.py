"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .benchmark import evaluate_synthetic
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
    benchmark.add_argument("--out", type=Path, default=Path("synthetic-evaluation.json"))

    serve = commands.add_parser("serve", help="Run the local browser interface")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "analyze":
        result = process_file(args.input, args.out, page=args.page, model_path=args.model)
        print(json.dumps({
            "output_dir": result["output_dir"],
            "source_sha256": result["source_sha256"],
            "source_hash_verified": result["source_hash_verified"],
            "staff_count": len(result["scene"]["staffs"]),
            "note_count": result["scene"]["note_count"],
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
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"report_path": str(args.out), **{k: v for k, v in result.items() if k != "per_page"}},
                         indent=2, ensure_ascii=False))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("scoretwin.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
