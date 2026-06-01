"""Config loading and model variant resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from rfdetr import (
    RFDETR,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSmall,
)
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE, AUG_AERIAL, AUG_CONSERVATIVE, AUG_INDUSTRIAL

from rfdetr_tools.checkpoints import (
    default_weights_for_variant,
    ensure_pretrained_weights,
    resolve_pretrained_weights,
)

MODEL_VARIANTS: dict[str, type[RFDETR]] = {
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRMedium": RFDETRMedium,
    "RFDETRLarge": RFDETRLarge,
}

AUG_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": AUG_CONSERVATIVE,
    "aggressive": AUG_AGGRESSIVE,
    "aerial": AUG_AERIAL,
    "industrial": AUG_INDUSTRIAL,
}


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def resolve_model_class(name: str) -> type[RFDETR]:
    if name not in MODEL_VARIANTS:
        options = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown model variant {name!r}. Choose from: {options}")
    return MODEL_VARIANTS[name]


def _parse_model_section(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Parse model variant and ModelConfig kwargs from flat or nested YAML."""
    model_section = config.pop("model", "RFDETRSmall")
    if isinstance(model_section, str):
        return model_section, {}
    if isinstance(model_section, dict):
        model_section = dict(model_section)
        variant = model_section.pop("variant", model_section.pop("name", "RFDETRSmall"))
        return str(variant), model_section
    raise ValueError("model must be a variant name (e.g. RFDETRSmall) or a mapping with variant/name.")


def build_train_kwargs(config: dict[str, Any]) -> tuple[type[RFDETR], dict[str, Any], dict[str, Any]]:
    """Split a config dict into model class, model kwargs, and train kwargs."""
    config = dict(config)
    model_name, model_section_kwargs = _parse_model_section(config)
    model_cls = resolve_model_class(model_name)

    model_kwargs: dict[str, Any] = dict(model_section_kwargs)
    for key in ("gradient_checkpointing", "num_classes", "pretrain_weights", "resolution"):
        if key in config:
            model_kwargs[key] = config.pop(key)

    aug_preset = config.pop("aug_preset", None)
    if aug_preset is not None:
        if aug_preset not in AUG_PRESETS:
            options = ", ".join(sorted(AUG_PRESETS))
            raise ValueError(f"Unknown aug_preset {aug_preset!r}. Choose from: {options}")
        config["aug_config"] = AUG_PRESETS[aug_preset]

    if "devices" in config and config["devices"] is not None:
        config["devices"] = str(config["devices"])

    return model_cls, model_kwargs, config


def prepare_model_kwargs(variant: str, model_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve and auto-download pretrained weights into checkpoints/."""
    prepared = dict(model_kwargs)
    weights = prepared.get("pretrain_weights")
    if weights is None:
        weights = default_weights_for_variant(variant)
    if weights is None:
        return prepared

    destination = ensure_pretrained_weights(weights)
    prepared["pretrain_weights"] = str(destination)
    return prepared


def validate_dataset_dir(dataset_dir: Path) -> None:
    annotations = dataset_dir / "train" / "_annotations.coco.json"
    if not annotations.is_file():
        raise FileNotFoundError(f"Expected COCO annotations at {annotations}")


def _train_config_from_checkpoint(
    checkpoint: Path,
    *,
    dataset_dir: Path | None = None,
) -> TrainConfig:
    training_cfg_path = checkpoint.parent / "training_config.json"
    if training_cfg_path.is_file():
        saved = json.loads(training_cfg_path.read_text(encoding="utf-8"))
        train_config = TrainConfig(**saved["train_config"])
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = ckpt["args"]
        if hasattr(args, "model_dump"):
            args_dict = args.model_dump()
        elif isinstance(args, dict):
            args_dict = dict(args)
        else:
            args_dict = vars(args)
        train_config = TrainConfig(**args_dict)

    updates: dict[str, Any] = {"output_dir": str(checkpoint.parent.resolve())}
    if dataset_dir is not None:
        updates["dataset_dir"] = str(dataset_dir.resolve())
    return train_config.model_copy(update=updates)


def load_run_context(
    checkpoint: Path,
    *,
    dataset_dir: Path | None = None,
) -> tuple[ModelConfig, TrainConfig, str]:
    """Load aligned model/train configs for evaluation or analysis."""
    wrapper = RFDETR.from_checkpoint(checkpoint)
    model_config = wrapper.model_config
    train_config = _train_config_from_checkpoint(checkpoint, dataset_dir=dataset_dir)
    return model_config, train_config, type(wrapper).__name__
