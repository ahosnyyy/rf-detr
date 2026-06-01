"""Helpers for running rfdetr_tools scripts directly (python rfdetr_tools/train.py)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


def ensure_project_root() -> None:
    """Put the repo root on sys.path when a script is run as a file."""
    if __package__ in (None, ""):
        root = Path(__file__).resolve().parent.parent
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def script_main(
    configure_parser: Callable[[argparse.ArgumentParser], None],
    *,
    description: str | None = None,
) -> None:
    ensure_project_root()
    parser = argparse.ArgumentParser(description=description)
    configure_parser(parser)
    args = parser.parse_args()
    args.func(args)
