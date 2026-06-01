"""Inference command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from rfdetr import RFDETR

from rfdetr_tools.checkpoints import resolve_checkpoint
from rfdetr_tools.inference import (
    annotate_image,
    collect_image_paths,
    decode_raw_outputs,
    load_export_metadata,
    preprocess_image,
    resolve_metadata_path,
)


def _resolve_pytorch_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return resolve_checkpoint(checkpoint=args.checkpoint, train_output_dir=args.train_output_dir)
    if args.train_output_dir is not None:
        return resolve_checkpoint(checkpoint=None, train_output_dir=args.train_output_dir)
    raise ValueError("PyTorch inference requires --checkpoint or --train-output-dir.")


def _resolve_export_model_path(model_path: Path | None, metadata_path: Path, key: str) -> Path:
    if model_path is not None:
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return model_path

    metadata = load_export_metadata(metadata_path)
    exported = metadata.get(key)
    if not exported:
        raise FileNotFoundError(f"No {key} entry in {metadata_path}. Run export first or pass --model.")
    resolved = Path(exported).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model from metadata not found: {resolved}")
    return resolved


def run_pytorch(args: argparse.Namespace, image_paths: list[Path]) -> None:
    checkpoint = _resolve_pytorch_checkpoint(args)
    print(f"Checkpoint: {checkpoint}")
    model = RFDETR.from_checkpoint(checkpoint)
    class_names = model.class_names

    for image_path in image_paths:
        detections = model.predict(str(image_path), threshold=args.threshold)
        if isinstance(detections, list):
            detections = detections[0]
        _report_and_save(image_path, detections, class_names, args)


def run_onnx(args: argparse.Namespace, image_paths: list[Path]) -> None:
    try:
        from rfdetr.export._onnx.inference import _create_onnx_session, _run_inference
    except ImportError as exc:
        raise ImportError(
            'ONNX inference requires rfdetr[onnx]. Install with: uv pip install "rfdetr[onnx]"'
        ) from exc

    metadata_path = resolve_metadata_path(args.metadata, args.model)
    model_path = _resolve_export_model_path(args.model, metadata_path, "onnx_model")
    class_names = load_export_metadata(metadata_path).get("class_names", [])
    print(f"ONNX model: {model_path}")
    session = _create_onnx_session(model_path)

    for image_path in image_paths:
        detections, pil_img = _run_inference(session, image_path, threshold=args.threshold)
        _report_and_save(image_path, detections, class_names, args, pil_img=pil_img)


def run_torchscript(args: argparse.Namespace, image_paths: list[Path]) -> None:
    metadata_path = resolve_metadata_path(args.metadata, args.model)
    metadata = load_export_metadata(metadata_path)
    model_path = _resolve_export_model_path(args.model, metadata_path, "torchscript_model")
    class_names = metadata.get("class_names", [])
    resolution = int(metadata.get("resolution", 512))
    batch_size = int(metadata.get("batch_size", 1))
    device = torch.device(args.device)

    print(f"TorchScript model: {model_path}")
    print(f"Device: {device}")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    if batch_size != 1:
        raise ValueError("TorchScript inference currently supports batch_size=1 only.")

    for image_path in image_paths:
        inp_tensor, pil_img = preprocess_image(image_path, width=resolution, height=resolution)
        with torch.no_grad():
            outputs = model(torch.from_numpy(inp_tensor).to(device))

        if isinstance(outputs, (tuple, list)):
            boxes = outputs[0][0].cpu().numpy()
            logits = outputs[1][0].cpu().numpy()
        elif isinstance(outputs, dict):
            boxes = outputs["pred_boxes"][0].cpu().numpy()
            logits = outputs["pred_logits"][0].cpu().numpy()
        else:
            raise TypeError(f"Unexpected TorchScript output type: {type(outputs)!r}")

        detections = decode_raw_outputs(
            boxes,
            logits,
            threshold=args.threshold,
            orig_size_wh=pil_img.size,
        )
        _report_and_save(image_path, detections, class_names, args, pil_img=pil_img)


def _resolve_output_path(image_path: Path, args: argparse.Namespace) -> Path | None:
    if args.output is None:
        return None
    output = args.output.resolve()
    image_paths: list[Path] = args._image_paths
    if output.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        if len(image_paths) > 1:
            raise ValueError("Pass an output directory when running on multiple images.")
        return output
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{image_path.stem}_pred{image_path.suffix}"


def _report_and_save(
    image_path: Path,
    detections: object,
    class_names: list[str],
    args: argparse.Namespace,
    *,
    pil_img: Image.Image | None = None,
) -> None:
    if not args.no_display:
        print(f"\n{image_path.name}: {len(detections)} detections above {args.threshold:.2f}")
        for class_id, confidence, box in zip(
            detections.class_id,
            detections.confidence,
            detections.xyxy,
            strict=True,
        ):
            name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            x1, y1, x2, y2 = box
            print(f"  {name}: {confidence:.3f} @ [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

    output_path = _resolve_output_path(image_path, args)
    if output_path is None:
        return

    if pil_img is None:
        pil_img = Image.open(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = annotate_image(pil_img, detections, class_names)
    Image.fromarray(annotated).save(output_path)
    print(f"Saved: {output_path}")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("infer", help="Run detection on image(s) with PyTorch, ONNX, or TorchScript.")
    parser.add_argument("input", type=Path, help="Input image or directory.")
    parser.add_argument("--backend", choices=("pytorch", "onnx", "torchscript"), default="pytorch")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-output-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None, help="Exported model path for ONNX/TorchScript.")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=None, help="Output image or directory for annotated results.")
    parser.add_argument("--no-display", action="store_true")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    try:
        image_paths = collect_image_paths(args.input.resolve())
        args._image_paths = image_paths

        if args.backend == "pytorch":
            run_pytorch(args, image_paths)
        elif args.backend == "onnx":
            run_onnx(args, image_paths)
        else:
            run_torchscript(args, image_paths)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
