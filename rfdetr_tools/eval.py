"""Evaluation command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rfdetr import RFDETR
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

from rfdetr_tools.checkpoints import resolve_checkpoint
from rfdetr_tools.config import load_run_context, validate_dataset_dir


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Run COCO validation or test metrics on a checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--train-output-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Override dataset directory from the run.")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

    checkpoint = resolve_checkpoint(
        checkpoint=args.checkpoint,
        train_output_dir=args.train_output_dir,
    )
    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else None
    if dataset_dir is not None:
        validate_dataset_dir(dataset_dir)

    wrapper = RFDETR.from_checkpoint(checkpoint)
    model_config, train_config, model_name = load_run_context(checkpoint, dataset_dir=dataset_dir)
    if dataset_dir is None:
        validate_dataset_dir(Path(train_config.dataset_dir))

    print(f"Checkpoint: {checkpoint}")
    print(f"Model: {model_name}")
    print(f"Dataset: {train_config.dataset_dir}")
    print(f"Split: {args.split}")

    module = RFDETRModelModule(model_config, train_config)
    module.model = wrapper.model.model

    datamodule = RFDETRDataModule(model_config, train_config)

    accelerator, devices = RFDETR._resolve_trainer_device_kwargs(args.device)
    trainer_kwargs: dict[str, object] = {
        "logger": False,
    }
    if accelerator is not None:
        trainer_kwargs["accelerator"] = accelerator
    if devices is not None:
        trainer_kwargs["devices"] = devices

    trainer = build_trainer(train_config, model_config, **trainer_kwargs)
    if args.split == "valid":
        trainer.validate(module, datamodule)
    else:
        trainer.test(module, datamodule)
