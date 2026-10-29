"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import process_file
from .proposal import save_model, train_proposal_model


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
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("scoretwin.web:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
