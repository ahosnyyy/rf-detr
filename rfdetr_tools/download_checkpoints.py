"""Download pretrained RF-DETR weights into the project checkpoints directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from rfdetr_tools.checkpoints import (
    VARIANT_DEFAULT_WEIGHTS,
    checkpoints_dir,
    ensure_pretrained_weights,
    ensure_variant_weights,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download-checkpoints",
        help="Download pretrained weights into the project checkpoints/ directory.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weight filename to download (e.g. rf-detr-small.pth).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(VARIANT_DEFAULT_WEIGHTS),
        help="Download the default pretrained weights for a model variant.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download default weights for all supported open-source variants.",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Force re-download even if the file already exists.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    targets: list[str] = []
    if args.all:
        targets.extend(sorted(set(VARIANT_DEFAULT_WEIGHTS.values())))
    elif args.weights is not None:
        targets.append(Path(args.weights).name)
    elif args.model is not None:
        path = ensure_variant_weights(args.model, redownload=args.redownload)
        if path is None:
            raise ValueError(f"No default pretrained weights registered for {args.model!r}.")
        print(f"Ready: {path}")
        return
    else:
        # Default: RFDETRSmall weights
        path = ensure_variant_weights("RFDETRSmall", redownload=args.redownload)
        print(f"Ready: {path}")
        return

    ckpt_dir = checkpoints_dir()
    print(f"Checkpoints directory: {ckpt_dir}")
    for name in targets:
        path = ensure_pretrained_weights(name, redownload=args.redownload)
        print(f"Ready: {path}")
