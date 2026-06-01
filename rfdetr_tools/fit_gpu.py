"""GPU fit sweep: probe batch size and related settings for the current hardware."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from rfdetr_tools.config import build_train_kwargs, load_yaml_config, prepare_model_kwargs, validate_dataset_dir


@dataclass
class ProbeScenario:
    gradient_checkpointing: bool
    target_effective: int
    batch_size: int
    grad_accum_steps: int
    effective_batch: int
    probe_resolution: int
    device_name: str
    ok: bool
    error: str | None = None


def _prepare_probe_context(config: dict[str, Any], dataset_dir: Path):
    from rfdetr.detr import _ensure_model_on_device

    model_cls, model_kwargs, train_kwargs = build_train_kwargs(dict(config))
    model_kwargs = prepare_model_kwargs(model_cls.__name__, model_kwargs)
    model = model_cls(**model_kwargs)
    model._align_num_classes_from_dataset(str(dataset_dir))
    train_config = model.get_train_config(**train_kwargs)
    _ensure_model_on_device(model.model)
    return model, model.model_config, train_config


def _probe_resolution(model_config, train_config) -> int:
    from rfdetr.datasets.coco import compute_multi_scale_scales

    multi_scale = getattr(train_config, "multi_scale", False)
    do_random_resize = getattr(train_config, "do_random_resize_via_padding", False)
    if multi_scale and not do_random_resize:
        scales = compute_multi_scale_scales(
            model_config.resolution,
            getattr(train_config, "expanded_scales", True),
            model_config.patch_size,
            model_config.num_windows,
        )
        return max(scales) if scales else model_config.resolution
    return model_config.resolution


def probe_scenario(
    base_config: dict[str, Any],
    dataset_dir: Path,
    *,
    gradient_checkpointing: bool,
    target_effective: int,
    safety_margin: float,
    max_micro_batch: int,
) -> ProbeScenario:
    from rfdetr.training.auto_batch import resolve_auto_batch_config

    config = dict(base_config)
    model_section = config.get("model")
    if isinstance(model_section, dict):
        model_section = dict(model_section)
        model_section["gradient_checkpointing"] = gradient_checkpointing
        config["model"] = model_section
    else:
        config["gradient_checkpointing"] = gradient_checkpointing

    config["batch_size"] = "auto"
    config["auto_batch_target_effective"] = target_effective

    model = None
    try:
        model, model_config, train_config = _prepare_probe_context(config, dataset_dir)
        result = resolve_auto_batch_config(
            model_context=model.model,
            model_config=model_config,
            train_config=train_config.model_copy(update={"batch_size": "auto"}),
            safety_margin=safety_margin,
            max_micro_batch=max_micro_batch,
        )
        return ProbeScenario(
            gradient_checkpointing=gradient_checkpointing,
            target_effective=target_effective,
            batch_size=result.safe_micro_batch,
            grad_accum_steps=result.recommended_grad_accum_steps,
            effective_batch=result.effective_batch_size,
            probe_resolution=_probe_resolution(model_config, train_config),
            device_name=result.device_name,
            ok=True,
        )
    except Exception as exc:
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return ProbeScenario(
            gradient_checkpointing=gradient_checkpointing,
            target_effective=target_effective,
            batch_size=0,
            grad_accum_steps=0,
            effective_batch=0,
            probe_resolution=0,
            device_name=device_name,
            ok=False,
            error=str(exc),
        )
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def rank_scenarios(scenarios: list[ProbeScenario]) -> list[ProbeScenario]:
    ok = [s for s in scenarios if s.ok]
    return sorted(
        ok,
        key=lambda s: (
            -s.batch_size,
            s.grad_accum_steps,
            s.gradient_checkpointing,
            -abs(s.effective_batch - s.target_effective),
        ),
    )


def _merge_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config is not None:
        config.update(load_yaml_config(args.config.resolve()))
    if args.dataset_dir is not None:
        config["dataset_dir"] = str(args.dataset_dir.resolve())
    if args.model is not None:
        config["model"] = args.model
    if "dataset_dir" not in config:
        raise ValueError("dataset_dir is required (config file or --dataset-dir).")
    return config


def _apply_best_to_config(base: dict[str, Any], best: ProbeScenario) -> dict[str, Any]:
    updated = dict(base)
    model_section = updated.get("model")
    if isinstance(model_section, dict):
        model_section = dict(model_section)
        model_section["gradient_checkpointing"] = best.gradient_checkpointing
        updated["model"] = model_section
    else:
        updated["gradient_checkpointing"] = best.gradient_checkpointing

    updated["batch_size"] = best.batch_size
    updated["grad_accum_steps"] = best.grad_accum_steps
    updated["auto_batch_target_effective"] = best.target_effective
    return updated


def _print_results(scenarios: list[ProbeScenario], best: ProbeScenario | None) -> None:
    print("\nGPU fit sweep results")
    print("=" * 72)
    for scenario in scenarios:
        status = "OK" if scenario.ok else "FAIL"
        gc = "gc-on" if scenario.gradient_checkpointing else "gc-off"
        if scenario.ok:
            print(
                f"[{status}] target={scenario.target_effective:>2}  {gc}  "
                f"batch={scenario.batch_size} x accum={scenario.grad_accum_steps} "
                f"-> effective={scenario.effective_batch}  "
                f"(probe {scenario.probe_resolution}px)"
            )
        else:
            print(f"[{status}] target={scenario.target_effective:>2}  {gc}  {scenario.error}")

    if best is None:
        print("\nNo feasible configuration found. Try a smaller model or lower resolution.")
        return

    gc_label = "enabled" if best.gradient_checkpointing else "disabled"
    print("\nRecommended settings")
    print("-" * 72)
    print(f"Device:              {best.device_name}")
    print(f"Probe resolution:    {best.probe_resolution}px (worst-case train step)")
    print(f"batch_size:          {best.batch_size}")
    print(f"grad_accum_steps:    {best.grad_accum_steps}")
    print(f"effective batch:     {best.effective_batch} (target {best.target_effective})")
    print(f"gradient_checkpointing: {gc_label}")
    print("\nSuggested YAML snippet:")
    print(yaml.safe_dump(
        {
            "model": {
                "variant": "RFDETRSmall",
                "gradient_checkpointing": best.gradient_checkpointing,
            },
            "batch_size": best.batch_size,
            "grad_accum_steps": best.grad_accum_steps,
            "auto_batch_target_effective": best.target_effective,
        },
        default_flow_style=False,
    ).strip())


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "fit-gpu",
        help="Sweep batch size / grad accumulation and find GPU-safe training settings.",
    )
    parser.add_argument("--config", type=Path, help="Base YAML config (model, dataset, aug, etc.).")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--model", type=str, default=None, help="Override model variant from config.")
    parser.add_argument(
        "--targets",
        type=str,
        default="8,16,32",
        help="Comma-separated effective batch sizes to try (per device).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("auto", "on", "off"),
        default="auto",
        help="Sweep gc off/on (auto), or fix to on/off.",
    )
    parser.add_argument("--safety-margin", type=float, default=0.9, help="Fraction of max probed batch to use.")
    parser.add_argument("--max-micro-batch", type=int, default=128, help="Upper bound for batch probe.")
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        help="Write a config YAML with the best settings merged into the base config.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write full sweep results as JSON.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Start training immediately with the best settings.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        print("fit-gpu requires CUDA. No GPU detected.", file=sys.stderr)
        raise SystemExit(1)

    config = _merge_config(args)
    dataset_dir = Path(config["dataset_dir"]).resolve()
    validate_dataset_dir(dataset_dir)

    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    if not targets or any(t < 1 for t in targets):
        raise ValueError("--targets must be a comma-separated list of positive integers.")

    if args.gradient_checkpointing == "auto":
        gc_values = [False, True]
    else:
        gc_values = [args.gradient_checkpointing == "on"]

    model_cls, model_kwargs, _ = build_train_kwargs(dict(config))
    print(f"Model: {model_cls.__name__}")
    print(f"Dataset: {dataset_dir}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Sweeping targets={targets}, gradient_checkpointing={gc_values}")

    scenarios: list[ProbeScenario] = []
    for target in targets:
        for gc in gc_values:
            print(f"\nProbing target_effective={target}, gradient_checkpointing={gc} ...")
            scenario = probe_scenario(
                config,
                dataset_dir,
                gradient_checkpointing=gc,
                target_effective=target,
                safety_margin=args.safety_margin,
                max_micro_batch=args.max_micro_batch,
            )
            scenarios.append(scenario)

    ranked = rank_scenarios(scenarios)
    best = ranked[0] if ranked else None
    _print_results(scenarios, best)

    if args.output_json is not None:
        payload = {
            "best": asdict(best) if best else None,
            "scenarios": [asdict(s) for s in scenarios],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.output_json.resolve()}")

    if best is None:
        raise SystemExit(1)

    merged = _apply_best_to_config(config, best)
    if args.output_config is not None:
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        print(f"Wrote config: {args.output_config.resolve()}")

    if args.train:
        if "output_dir" not in merged:
            raise ValueError("output_dir is required in config to --train after fit-gpu.")
        print("\nStarting training with recommended settings...")
        model_cls, model_kwargs, train_kwargs = build_train_kwargs(merged)
        train_kwargs["device"] = "cuda"
        model = model_cls(**model_kwargs)
        model.train(**train_kwargs)
