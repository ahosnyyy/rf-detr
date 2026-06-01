"""Checkpoint paths: pretrained weights and fine-tuned run artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from rfdetr.assets.model_weights import download_pretrain_weights

CHECKPOINTS_ENV_VAR = "RFDETR_CHECKPOINTS_DIR"

VARIANT_DEFAULT_WEIGHTS: dict[str, str] = {
    "RFDETRNano": "rf-detr-nano.pth",
    "RFDETRSmall": "rf-detr-small.pth",
    "RFDETRMedium": "rf-detr-medium.pth",
    "RFDETRLarge": "rf-detr-large-2026.pth",
}

BEST_CHECKPOINT_NAMES = (
    "checkpoint_best_total.pth",
    "checkpoint_best_regular.pth",
    "checkpoint_best_ema.pth",
)


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "rfdetr_tools").is_dir() and (path / "requirements.txt").is_file():
            return path
    return start


def checkpoints_dir(start: Path | None = None) -> Path:
    override = os.environ.get(CHECKPOINTS_ENV_VAR)
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = find_project_root(start) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_pretrained_weights(
    weights: str | Path,
    *,
    start: Path | None = None,
) -> Path:
    """Map a weight name or path to an absolute file under the project checkpoints dir."""
    weights_path = Path(weights).expanduser()
    if weights_path.is_absolute():
        return weights_path.resolve()

    if weights_path.parent != Path("."):
        # e.g. checkpoints/rf-detr-small.pth
        return (find_project_root(start) / weights_path).resolve()

    # Bare filename -> checkpoints/<filename>
    return (checkpoints_dir(start) / weights_path.name).resolve()


def ensure_pretrained_weights(
    weights: str | Path,
    *,
    start: Path | None = None,
    redownload: bool = False,
) -> Path:
    """Download pretrained weights into the project checkpoints dir if missing."""
    destination = resolve_pretrained_weights(weights, start=start)
    destination.parent.mkdir(parents=True, exist_ok=True)
    download_pretrain_weights(str(destination), redownload=redownload)
    if not destination.is_file():
        raise FileNotFoundError(f"Pretrained weights not available after download: {destination}")
    return destination


def default_weights_for_variant(variant: str) -> str | None:
    return VARIANT_DEFAULT_WEIGHTS.get(variant)


def ensure_variant_weights(
    variant: str,
    *,
    start: Path | None = None,
    redownload: bool = False,
) -> Path | None:
    filename = default_weights_for_variant(variant)
    if filename is None:
        return None
    return ensure_pretrained_weights(filename, start=start, redownload=redownload)


def find_best_checkpoint(train_output_dir: Path) -> Path:
    """Return the best available fine-tuned checkpoint in a training output directory."""
    train_output_dir = train_output_dir.resolve()
    for name in BEST_CHECKPOINT_NAMES:
        candidate = train_output_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No best checkpoint found in {train_output_dir}. Expected one of: "
        f"{', '.join(BEST_CHECKPOINT_NAMES)}"
    )


def resolve_checkpoint(
    *,
    checkpoint: Path | None,
    train_output_dir: Path | None,
) -> Path:
    """Resolve an explicit checkpoint path or discover one under a train output dir."""
    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        if checkpoint.is_file():
            return checkpoint
        if checkpoint.parent.is_dir():
            discovered = find_best_checkpoint(checkpoint.parent)
            print(f"Checkpoint not found at {checkpoint}; using {discovered}")
            return discovered
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if train_output_dir is not None:
        return find_best_checkpoint(train_output_dir)

    raise ValueError("Pass --checkpoint or --train-output-dir.")


def default_export_dir(checkpoint: Path) -> Path:
    return checkpoint.parent / "exported"
