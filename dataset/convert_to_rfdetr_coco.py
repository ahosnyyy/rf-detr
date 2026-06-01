#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Convert a local dataset to RF-DETR COCO layout.

RF-DETR auto-detects COCO datasets when ``train/_annotations.coco.json`` exists.
Each split directory (``train``, ``valid``, ``test``) holds images next to its
annotation file. See:
https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/

Supported inputs:
- RF-DETR / Roboflow COCO (validated and optionally copied)
- YOLO (``data.yaml`` + ``train/images/``) via ``supervision``
- Generic COCO exports (``val`` renamed to ``valid``, images moved from subfolders)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "valid": ("valid", "val", "validation"),
    "test": ("test",),
}
RF_SPLIT_NAMES = ("train", "valid", "test")
ANNOTATIONS_NAME = "_annotations.coco.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LinkMode = Literal["copy", "hardlink", "symlink"]


@dataclass(frozen=True)
class DatasetFormat:
    """Detected source dataset format."""

    name: str


@dataclass(frozen=True)
class ConversionStats:
    """Summary counters for one split."""

    split: str
    images: int
    annotations: int
    categories: int


def _find_split_dir(dataset_dir: Path, split: str) -> Path | None:
    """Return the first existing directory for a logical split name."""
    for alias in SPLIT_ALIASES[split]:
        candidate = dataset_dir / alias
        if candidate.is_dir():
            return candidate
    return None


def _is_rfdetr_coco(dataset_dir: Path) -> bool:
    """Return True when the dataset already matches RF-DETR COCO detection rules."""
    train_annotations = dataset_dir / "train" / ANNOTATIONS_NAME
    return train_annotations.is_file()


def _is_yolo(dataset_dir: Path) -> bool:
    """Return True when the dataset looks like YOLO format."""
    yaml_paths = list(dataset_dir.glob("data.yaml")) + list(dataset_dir.glob("data.yml"))
    if not yaml_paths:
        return False
    train_images = dataset_dir / "train" / "images"
    return train_images.is_dir()


def detect_format(dataset_dir: Path) -> DatasetFormat:
    """Detect whether the input is RF-DETR COCO or YOLO."""
    if _is_rfdetr_coco(dataset_dir):
        return DatasetFormat(name="rfdetr_coco")
    if _is_yolo(dataset_dir):
        return DatasetFormat(name="yolo")
    raise ValueError(
        f"Could not detect dataset format in {dataset_dir}. "
        f"Expected RF-DETR COCO (train/{ANNOTATIONS_NAME}) or YOLO (data.yaml + train/images/)."
    )


def _resolve_image_path(split_dir: Path, file_name: str) -> Path | None:
    """Locate an image referenced by COCO ``file_name`` within a split directory."""
    direct = split_dir / file_name
    if direct.is_file():
        return direct

    basename = Path(file_name).name
    for candidate in (split_dir / basename, split_dir / "images" / basename, split_dir / "images" / file_name):
        if candidate.is_file():
            return candidate
    return None


def _link_file(source: Path, destination: Path, mode: LinkMode) -> None:
    """Copy or link ``source`` to ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    if mode == "copy":
        shutil.copy2(source, destination)
        return

    try:
        if mode == "hardlink":
            source.link_to(destination)
        else:
            destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _load_coco(path: Path) -> dict[str, Any]:
    """Load a COCO JSON annotation file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_coco(path: Path, payload: dict[str, Any]) -> None:
    """Write a COCO JSON annotation file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _normalize_coco_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure required top-level keys exist for RF-DETR training."""
    normalized = dict(payload)
    normalized.setdefault("info", {"description": "Converted for RF-DETR training", "version": "1.0"})
    normalized.setdefault("licenses", [])
    normalized.setdefault("images", [])
    normalized.setdefault("categories", [])
    normalized.setdefault("annotations", [])
    return normalized


def _validate_split(split_dir: Path, payload: dict[str, Any]) -> list[str]:
    """Return validation errors for one COCO split."""
    errors: list[str] = []
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])

    if not images:
        errors.append(f"{split_dir.name}: no images listed in {ANNOTATIONS_NAME}")
    if not categories:
        errors.append(f"{split_dir.name}: no categories listed in {ANNOTATIONS_NAME}")

    image_ids = {image["id"] for image in images}
    category_ids = {category["id"] for category in categories}
    for image in images:
        resolved = _resolve_image_path(split_dir, image["file_name"])
        if resolved is None:
            errors.append(f"{split_dir.name}: missing image file {image['file_name']}")

    for annotation in annotations:
        if annotation.get("image_id") not in image_ids:
            errors.append(f"{split_dir.name}: annotation {annotation.get('id')} has unknown image_id")
        if annotation.get("category_id") not in category_ids:
            errors.append(
                f"{split_dir.name}: annotation {annotation.get('id')} has unknown category_id "
                f"{annotation.get('category_id')}"
            )

    return errors


def _iter_rfdetr_coco_splits(input_dir: Path) -> tuple[list[ConversionStats], list[str]]:
    """Load and validate RF-DETR COCO splits without writing output."""
    stats: list[ConversionStats] = []
    errors: list[str] = []

    for split in RF_SPLIT_NAMES:
        split_dir = _find_split_dir(input_dir, split)
        if split_dir is None:
            continue

        annotations_path = split_dir / ANNOTATIONS_NAME
        if not annotations_path.is_file():
            alt_names = sorted(split_dir.glob("*.json"))
            if len(alt_names) == 1:
                annotations_path = alt_names[0]
            else:
                errors.append(f"{split}: missing {ANNOTATIONS_NAME}")
                continue

        payload = _normalize_coco_payload(_load_coco(annotations_path))
        errors.extend(_validate_split(split_dir, payload))
        stats.append(
            ConversionStats(
                split=split,
                images=len(payload["images"]),
                annotations=len(payload["annotations"]),
                categories=len(payload["categories"]),
            )
        )

    if not any(item.split == "train" for item in stats):
        errors.append("Training split is required. Expected train/ with annotations.")

    return stats, errors


def validate_rfdetr_coco(input_dir: Path) -> list[ConversionStats]:
    """Validate an in-place RF-DETR COCO dataset."""
    stats, errors = _iter_rfdetr_coco_splits(input_dir)
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return stats


def convert_rfdetr_coco(
    input_dir: Path,
    output_dir: Path,
    *,
    link_mode: LinkMode,
    overwrite: bool,
) -> list[ConversionStats]:
    """Validate and materialize an RF-DETR COCO dataset."""
    stats, errors = _iter_rfdetr_coco_splits(input_dir)
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    for split in RF_SPLIT_NAMES:
        split_dir = _find_split_dir(input_dir, split)
        if split_dir is None:
            continue

        annotations_path = split_dir / ANNOTATIONS_NAME
        if not annotations_path.is_file():
            alt_names = sorted(split_dir.glob("*.json"))
            annotations_path = alt_names[0]

        payload = _normalize_coco_payload(_load_coco(annotations_path))
        out_split_dir = output_dir / split
        _write_coco(out_split_dir / ANNOTATIONS_NAME, payload)

        for image in payload["images"]:
            source = _resolve_image_path(split_dir, image["file_name"])
            if source is None:
                continue
            destination = out_split_dir / Path(image["file_name"]).name
            _link_file(source, destination, link_mode)

    return stats


def _parse_yaml_paths(data: dict[str, Any], dataset_dir: Path) -> dict[str, Path | None]:
    """Resolve YOLO split image directories from ``data.yaml``."""
    resolved: dict[str, Path | None] = {"train": None, "valid": None, "test": None}
    key_aliases = {
        "train": ("train",),
        "valid": ("val", "valid", "validation"),
        "test": ("test",),
    }

    for split, aliases in key_aliases.items():
        for alias in aliases:
            raw = data.get(alias)
            if raw is None:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = (dataset_dir / path).resolve()
            resolved[split] = path
            break

        if resolved[split] is None:
            resolved[split] = _find_split_dir(dataset_dir, split)
            if resolved[split] is not None:
                images_dir = resolved[split] / "images"
                resolved[split] = images_dir if images_dir.is_dir() else resolved[split]

    return resolved


def convert_yolo(
    input_dir: Path,
    output_dir: Path,
    *,
    link_mode: LinkMode,
    overwrite: bool,
) -> list[ConversionStats]:
    """Convert a YOLO dataset to RF-DETR COCO layout using supervision."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read data.yaml for YOLO conversion.") from exc

    try:
        import supervision as sv
    except ImportError as exc:
        raise ImportError(
            "supervision is required for YOLO conversion. Install RF-DETR with: pip install rfdetr"
        ) from exc

    yaml_path = next(path for path in (input_dir / "data.yaml", input_dir / "data.yml") if path.is_file())
    with yaml_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    split_paths = _parse_yaml_paths(data, input_dir)
    if split_paths["train"] is None:
        raise ValueError("Could not resolve YOLO train split from data.yaml or train/images/.")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    stats: list[ConversionStats] = []

    for split, images_dir in split_paths.items():
        if images_dir is None or not images_dir.is_dir():
            continue

        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            labels_dir = images_dir.parent.parent / "labels"
        if not labels_dir.is_dir():
            raise ValueError(f"Could not find labels directory for split '{split}' near {images_dir}")

        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(labels_dir),
            data_yaml_path=str(yaml_path),
        )

        out_split_dir = output_dir / split
        out_images_dir = out_split_dir
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_annotations = out_split_dir / ANNOTATIONS_NAME

        dataset.as_coco(
            images_directory_path=str(out_images_dir),
            annotations_path=str(out_annotations),
        )

        payload = _normalize_coco_payload(_load_coco(out_annotations))
        for image in payload["images"]:
            source = out_split_dir / image["file_name"]
            if not source.is_file():
                source = out_images_dir / "images" / image["file_name"]
            if source.is_file() and source.parent != out_split_dir:
                destination = out_split_dir / Path(image["file_name"]).name
                _link_file(source, destination, link_mode)
                image["file_name"] = destination.name

        _write_coco(out_annotations, payload)
        stats.append(
            ConversionStats(
                split=split,
                images=len(payload["images"]),
                annotations=len(payload["annotations"]),
                categories=len(payload["categories"]),
            )
        )

    if not stats:
        raise ValueError("No YOLO splits were converted.")

    return stats


def merge_test_into_valid(dataset_dir: Path, *, remove_test: bool = True) -> ConversionStats:
    """Merge the test split into valid, remapping COCO ids and moving image files."""
    valid_dir = _find_split_dir(dataset_dir, "valid")
    test_dir = _find_split_dir(dataset_dir, "test")
    if valid_dir is None:
        raise ValueError(f"Missing valid split in {dataset_dir}")
    if test_dir is None:
        raise ValueError(f"Missing test split in {dataset_dir}")

    valid_annotations = valid_dir / ANNOTATIONS_NAME
    test_annotations = test_dir / ANNOTATIONS_NAME
    if not valid_annotations.is_file() or not test_annotations.is_file():
        raise ValueError(f"Both valid/ and test/ must contain {ANNOTATIONS_NAME}")

    valid_payload = _normalize_coco_payload(_load_coco(valid_annotations))
    test_payload = _normalize_coco_payload(_load_coco(test_annotations))

    valid_names = {image["file_name"] for image in valid_payload["images"]}
    test_names = {image["file_name"] for image in test_payload["images"]}
    overlap = valid_names & test_names
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(f"Duplicate file_name between valid and test: {sample}")

    next_image_id = max((image["id"] for image in valid_payload["images"]), default=-1) + 1
    next_annotation_id = max((annotation["id"] for annotation in valid_payload["annotations"]), default=0) + 1

    image_id_map: dict[int, int] = {}
    for image in test_payload["images"]:
        image_id_map[image["id"]] = next_image_id
        image["id"] = next_image_id
        next_image_id += 1
        valid_payload["images"].append(image)

        source = _resolve_image_path(test_dir, image["file_name"])
        if source is None:
            raise ValueError(f"Missing test image file: {image['file_name']}")
        destination = valid_dir / source.name
        if destination.exists():
            raise ValueError(f"Image already exists in valid/: {destination.name}")
        shutil.move(str(source), str(destination))

    for annotation in test_payload["annotations"]:
        old_image_id = annotation["image_id"]
        if old_image_id not in image_id_map:
            raise ValueError(f"Test annotation {annotation['id']} references unknown image_id {old_image_id}")
        annotation["id"] = next_annotation_id
        annotation["image_id"] = image_id_map[old_image_id]
        next_annotation_id += 1
        valid_payload["annotations"].append(annotation)

    _write_coco(valid_annotations, valid_payload)

    if remove_test:
        shutil.rmtree(test_dir)

    return ConversionStats(
        split="valid",
        images=len(valid_payload["images"]),
        annotations=len(valid_payload["annotations"]),
        categories=len(valid_payload["categories"]),
    )


def _print_stats(input_dir: Path, output_dir: Path, dataset_format: DatasetFormat, stats: Iterable[ConversionStats]) -> None:
    """Print a human-readable conversion summary."""
    print(f"Input:  {input_dir.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Format: {dataset_format.name}")
    print("")
    print("Splits:")
    for item in stats:
        print(
            f"  {item.split:5s}  images={item.images:5d}  "
            f"annotations={item.annotations:5d}  categories={item.categories}"
        )
    print("")
    print("Ready for RF-DETR training, e.g.:")
    print(f"  model.train(dataset_dir=r'{output_dir.as_posix()}')")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Convert a dataset to RF-DETR COCO format.")
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the source dataset directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory. Defaults to <input_dir>_rfdetr_coco next to the input.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How to place image files in the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an RF-DETR COCO dataset in place without writing a copy.",
    )
    parser.add_argument(
        "--merge-test-into-valid",
        action="store_true",
        help="Merge test/ into valid/ in place (remaps COCO ids, moves images, deletes test/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the conversion CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    dataset_format = detect_format(input_dir)

    if args.merge_test_into_valid:
        if dataset_format.name != "rfdetr_coco":
            parser.error("--merge-test-into-valid requires an RF-DETR COCO input dataset.")
        merged = merge_test_into_valid(input_dir)
        print(f"Merged test into valid in {input_dir.resolve()}")
        print(
            f"  valid  images={merged.images:5d}  "
            f"annotations={merged.annotations:5d}  categories={merged.categories}"
        )
        return 0

    if args.validate_only:
        if dataset_format.name != "rfdetr_coco":
            parser.error("--validate-only requires an RF-DETR COCO input dataset.")
        stats = validate_rfdetr_coco(input_dir)
        _print_stats(input_dir, input_dir, dataset_format, stats)
        print("Validation passed.")
        return 0

    output_dir = args.output_dir.resolve() if args.output_dir else input_dir.with_name(f"{input_dir.name}_rfdetr_coco")

    if dataset_format.name == "rfdetr_coco":
        stats = convert_rfdetr_coco(
            input_dir,
            output_dir,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )
    else:
        stats = convert_yolo(
            input_dir,
            output_dir,
            link_mode=args.link_mode,
            overwrite=args.overwrite,
        )

    _print_stats(input_dir, output_dir, dataset_format, stats)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
