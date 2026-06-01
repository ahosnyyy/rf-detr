"""Unified CLI entry point."""

from __future__ import annotations

import argparse

from rfdetr_tools import download_checkpoints as download_checkpoints_cmd
from rfdetr_tools import eval as eval_cmd
from rfdetr_tools import export as export_cmd
from rfdetr_tools import fit_gpu as fit_gpu_cmd
from rfdetr_tools import infer as infer_cmd
from rfdetr_tools import log as log_cmd
from rfdetr_tools import train as train_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfdetr-tools",
        description="Train, evaluate, export, infer, and monitor RF-DETR fine-tuning runs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_cmd.add_parser(subparsers)
    fit_gpu_cmd.add_parser(subparsers)
    download_checkpoints_cmd.add_parser(subparsers)
    eval_cmd.add_parser(subparsers)
    export_cmd.add_parser(subparsers)
    infer_cmd.add_parser(subparsers)
    log_cmd.add_parser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
