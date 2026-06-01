"""Export command."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from rfdetr import RFDETR

from rfdetr_tools.checkpoints import default_export_dir, resolve_checkpoint

if TYPE_CHECKING:
    from rfdetr.detr import RFDETR as RFDETRModel


def _check_onnx_deps() -> bool:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ModuleNotFoundError:
        print(
            "Warning: ONNX export requires rfdetr[onnx]. Install with:\n"
            "  pip install \"rfdetr[onnx]\"",
            file=sys.stderr,
        )
        return False
    return True


def export_onnx(
    model: RFDETRModel,
    *,
    export_dir: Path,
    batch_size: int,
    dynamic_batch: bool,
    opset_version: int,
    sample_image: Path | None,
    class_names: list[str],
    task_name: str | None,
) -> Path:
    notes: dict[str, object] = {"classes": class_names}
    if task_name:
        notes["task"] = task_name
    onnx_path = model.export(
        output_dir=str(export_dir),
        infer_dir=str(sample_image) if sample_image is not None else None,
        batch_size=batch_size,
        dynamic_batch=dynamic_batch,
        opset_version=opset_version,
        notes=notes,
    )
    return Path(onnx_path)


def export_torchscript(
    model: RFDETRModel,
    *,
    export_dir: Path,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    output_name: str | None = None,
) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    variant = getattr(model, "size", "rfdetr")
    filename = output_name or f"{variant}.ts.pt"
    output_path = export_dir / filename

    inference_model = deepcopy(model.model.model)
    inference_model.eval()
    inference_model.export()
    inference_model = inference_model.to(device=device, dtype=dtype)

    resolution = model.model.resolution
    channels = model.model_config.num_channels
    example = torch.randn(
        batch_size,
        channels,
        resolution,
        resolution,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        traced = torch.jit.trace(inference_model, example)

    traced.save(str(output_path))
    return output_path


def write_metadata(
    *,
    export_dir: Path,
    checkpoint: Path,
    variant: str,
    class_names: list[str],
    resolution: int,
    batch_size: int,
    dynamic_batch: bool,
    task_name: str | None,
    onnx_path: Path | None,
    torchscript_path: Path | None,
) -> Path:
    metadata = {
        "checkpoint": str(checkpoint),
        "variant": variant,
        "class_names": class_names,
        "resolution": resolution,
        "batch_size": batch_size,
        "dynamic_batch": dynamic_batch,
        "onnx_model": str(onnx_path) if onnx_path is not None else None,
        "torchscript_model": str(torchscript_path) if torchscript_path is not None else None,
    }
    if task_name:
        metadata["task"] = task_name
    metadata_path = export_dir / "export_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata_path


def configure_parser(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--train-output-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--format", choices=("onnx", "torchscript", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu"))
    parser.add_argument("--sample-image", type=Path, default=None)
    parser.add_argument("--torchscript-name", type=str, default=None)
    parser.set_defaults(func=run)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export", help="Export a checkpoint to ONNX and/or TorchScript.")
    configure_parser(parser)


def run(args: argparse.Namespace) -> None:
    checkpoint = resolve_checkpoint(
        checkpoint=args.checkpoint,
        train_output_dir=args.train_output_dir,
    )
    export_dir = (args.output_dir or default_export_dir(checkpoint)).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    task_name = args.task_name
    if task_name is None and args.train_output_dir is not None:
        task_name = args.train_output_dir.resolve().name

    print(f"Loading checkpoint: {checkpoint}")
    model = RFDETR.from_checkpoint(checkpoint)
    class_names = model.class_names
    resolution = model.model.resolution
    variant = getattr(model, "size", "rfdetr")
    print(f"Variant: {variant}")
    print(f"Model resolution: {resolution}x{resolution}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")

    onnx_path: Path | None = None
    torchscript_path: Path | None = None

    if args.format in ("onnx", "all"):
        if not _check_onnx_deps():
            if args.format == "onnx":
                raise SystemExit(1)
            print("Skipping ONNX export.")
        else:
            print("Exporting ONNX...")
            onnx_path = export_onnx(
                model,
                export_dir=export_dir,
                batch_size=args.batch_size,
                dynamic_batch=args.dynamic_batch,
                opset_version=args.opset_version,
                sample_image=args.sample_image,
                class_names=class_names,
                task_name=task_name,
            )
            print(f"ONNX model: {onnx_path}")

    if args.format in ("torchscript", "all"):
        device = torch.device(args.device)
        print(f"Exporting TorchScript on {device}...")
        torchscript_path = export_torchscript(
            model,
            export_dir=export_dir,
            batch_size=args.batch_size,
            device=device,
            dtype=torch.float32,
            output_name=args.torchscript_name,
        )
        print(f"TorchScript model: {torchscript_path}")

    metadata_path = write_metadata(
        export_dir=export_dir,
        checkpoint=checkpoint,
        variant=variant,
        class_names=class_names,
        resolution=resolution,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        task_name=task_name,
        onnx_path=onnx_path,
        torchscript_path=torchscript_path,
    )
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    from rfdetr_tools._run import script_main

    script_main(configure_parser, description="Export a checkpoint to ONNX and/or TorchScript.")
