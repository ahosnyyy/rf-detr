"""Shared preprocessing, decoding, and visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from PIL import Image as PILImage

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_export_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_metadata_path(
    metadata: Path | None,
    model_path: Path | None,
) -> Path:
    if metadata is not None:
        metadata = metadata.resolve()
        if not metadata.is_file():
            raise FileNotFoundError(f"Export metadata not found: {metadata}")
        return metadata

    if model_path is not None:
        candidate = model_path.resolve().parent / "export_metadata.json"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"No export_metadata.json beside {model_path}. Pass --metadata or run export first."
        )

    raise ValueError("Pass --metadata or --model to locate export metadata.")


def preprocess_image(
    image_path: Path,
    *,
    width: int,
    height: int,
    channels: int = 3,
) -> tuple[np.ndarray, PILImage.Image]:
    pil_mode = "L" if channels == 1 else "RGB"
    mean = np.array([IMAGENET_MEAN[i % 3] for i in range(channels)], dtype=np.float32)
    std = np.array([IMAGENET_STD[i % 3] for i in range(channels)], dtype=np.float32)

    pil_img = PILImage.open(image_path)
    arr = (
        np.array(
            pil_img.convert(pil_mode).resize((width, height), PILImage.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis].astype(np.float32), pil_img


def decode_raw_outputs(
    boxes_cwh: np.ndarray,
    logits: np.ndarray,
    *,
    threshold: float,
    orig_size_wh: tuple[int, int],
) -> sv.Detections:
    logits = logits[:, :-1]
    one = np.asarray(1, dtype=logits.dtype)
    scores_all = one / (one + np.exp(-logits.clip(-88, 88)))
    scores = scores_all.max(axis=-1)
    cls = scores_all.argmax(axis=-1)
    keep = scores > threshold

    cx, cy, bw, bh = boxes_cwh[keep].T
    ow, oh = orig_size_wh
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)

    return sv.Detections(
        xyxy=xyxy,
        confidence=scores[keep],
        class_id=cls[keep].astype(int),
    )


def annotate_image(
    pil_img: PILImage.Image,
    detections: sv.Detections,
    class_names: list[str],
) -> np.ndarray:
    labels = [
        f"{class_names[class_id] if class_id < len(class_names) else class_id} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence, strict=True)
    ]
    annotated = np.array(pil_img.convert("RGB"))
    return sv.BoxAnnotator().annotate(
        sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT).annotate(
            annotated,
            detections,
            labels=labels,
        ),
        detections,
    )


def collect_image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")
    images = sorted(
        p.resolve()
        for p in path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return images
