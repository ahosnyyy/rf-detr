"""Training command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rfdetr_tools.config import (
    build_train_kwargs,
    load_yaml_config,
    prepare_model_kwargs,
    validate_dataset_dir,
)


def _check_logger(name: str, package: str, install_hint: str) -> bool:
    try:
        __import__(package)
        return True
    except ModuleNotFoundError:
        print(f"Warning: {name} is not installed; continuing without it.\n{install_hint}", file=sys.stderr)
        return False


def _print_run_info(output_dir: Path, *, tensorboard: bool, progress_bar: str | None) -> None:
    print(f"Output directory: {output_dir}")
    if tensorboard:
        print(f"TensorBoard: rfdetr-tools log --output-dir {output_dir.as_posix()}")
    if progress_bar is None:
        print("Terminal progress: disabled")
    else:
        print(f"Terminal progress: {progress_bar}")
    print(f"Metrics CSV: {output_dir / 'metrics.csv'}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Fine-tune an RF-DETR model.")
    parser.add_argument("--config", type=Path, help="YAML training config file.")
    parser.add_argument("--model", type=str, default=None, help="Model variant (e.g. RFDETRSmall).")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="RF-DETR COCO dataset directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-encoder", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--aug-preset", type=str, default=None, choices=("conservative", "aggressive", "aerial", "industrial"))
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--run-test", action="store_true", help="Run test split evaluation after training.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--progress-bar", type=str, default=None, choices=("tqdm", "rich", "none"))
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.set_defaults(func=run)


def _merge_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config is not None:
        config.update(load_yaml_config(args.config.resolve()))

    cli_overrides = {
        "model": args.model,
        "dataset_dir": str(args.dataset_dir) if args.dataset_dir else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "device": args.device,
        "resume": str(args.resume.resolve()) if args.resume else None,
        "aug_preset": args.aug_preset,
        "wandb": True if args.wandb else None,
        "mlflow": True if args.mlflow else None,
        "project": args.project,
        "run": args.run,
        "gradient_checkpointing": True if args.gradient_checkpointing else None,
        "devices": args.devices,
        "run_test": True if args.run_test else None,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    if args.no_early_stopping:
        config["early_stopping"] = False
    if args.no_tensorboard:
        config["tensorboard"] = False
    if args.progress_bar == "none":
        config["progress_bar"] = None
    elif args.progress_bar is not None:
        config["progress_bar"] = args.progress_bar

    if "dataset_dir" not in config:
        raise ValueError("dataset_dir is required (config file or --dataset-dir).")
    if "output_dir" not in config:
        raise ValueError("output_dir is required (config file or --output-dir).")

    return config


def run(args: argparse.Namespace) -> None:
    config = _merge_config(args)
    dataset_dir = Path(config["dataset_dir"]).resolve()
    validate_dataset_dir(dataset_dir)

    output_dir = Path(config["output_dir"]).resolve()
    use_tensorboard = config.get("tensorboard", True)
    if use_tensorboard:
        use_tensorboard = _check_logger("tensorboard", "tensorboard", "Install with: uv pip install tensorboard")
    config["tensorboard"] = use_tensorboard

    if config.get("wandb"):
        if not _check_logger("wandb", "wandb", 'Install with: uv pip install "rfdetr[loggers]"'):
            config["wandb"] = False

    if config.get("mlflow"):
        if not _check_logger("mlflow", "mlflow", 'Install with: uv pip install "rfdetr[loggers]"'):
            config["mlflow"] = False

    progress_bar = config.get("progress_bar", "tqdm")
    model_cls, model_kwargs, train_kwargs = build_train_kwargs(config)
    model_kwargs = prepare_model_kwargs(model_cls.__name__, model_kwargs)
    print(f"Model: {model_cls.__name__}")
    if model_kwargs:
        print(f"Model options: {model_kwargs}")
    _print_run_info(output_dir, tensorboard=use_tensorboard, progress_bar=progress_bar)

    model = model_cls(**model_kwargs)
    model.train(**train_kwargs)
