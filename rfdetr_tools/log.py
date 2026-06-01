"""Logging utilities."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _print_metrics_summary(metrics_csv: Path, *, tail: int = 5) -> None:
    if not metrics_csv.is_file():
        print(f"No metrics file at {metrics_csv}")
        return

    with metrics_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"Metrics file is empty: {metrics_csv}")
        return

    print(f"Metrics CSV: {metrics_csv}")
    print(f"Rows logged: {len(rows)}")

    key_metrics = [
        "val/mAP_50_95",
        "val/mAP_50",
        "val/F1",
        "val/precision",
        "val/recall",
    ]
    last = rows[-1]
    print("Latest metrics:")
    for key in key_metrics:
        if key in last and last[key]:
            print(f"  {key}: {last[key]}")

    if len(rows) > 1:
        print(f"\nLast {min(tail, len(rows))} epochs:")
        for row in rows[-tail:]:
            epoch = row.get("epoch", "?")
            map_score = row.get("val/mAP_50_95", "n/a")
            print(f"  epoch {epoch}: val/mAP_50_95={map_score}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("log", help="Inspect metrics or launch TensorBoard.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training output directory.")
    parser.add_argument("--tensorboard", action="store_true", help="Launch TensorBoard for the run.")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--summary", action="store_true", help="Print a metrics.csv summary.")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    metrics_csv = output_dir / "metrics.csv"

    if args.summary or not args.tensorboard:
        _print_metrics_summary(metrics_csv)

    if not args.tensorboard:
        if not args.summary:
            print("\nUse --summary to print metrics.csv or --tensorboard to launch TensorBoard.")
        return

    try:
        import tensorboard  # noqa: F401
    except ModuleNotFoundError:
        print("TensorBoard is not installed. Install with: uv pip install tensorboard", file=sys.stderr)
        raise SystemExit(1)

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(output_dir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    print(f"Launching TensorBoard at http://{args.host}:{args.port}")
    raise SystemExit(subprocess.call(cmd))
