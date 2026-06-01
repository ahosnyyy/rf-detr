# RF-DETR Project

Fine-tune, evaluate, export, and deploy [RF-DETR](https://rfdetr.roboflow.com/latest/) object detection models on custom COCO-format datasets.

This repo wraps the `rfdetr` Python package with a unified CLI for the full workflow: **train → eval → log → export → infer**.

## Project layout

```text
rf-detr/
├── checkpoints/              # Pretrained COCO weights (auto-downloaded, gitignored)
├── configs/                  # YAML experiment configs (template + project examples)
├── dataset/                  # COCO datasets + conversion utilities
├── output/                   # Training runs (checkpoints, logs, exports)
├── rfdetr_tools/             # CLI package
│   ├── cli.py                # Entry point (rfdetr-tools)
│   ├── checkpoints.py        # Pretrained weights + fine-tuned checkpoint discovery
│   ├── config.py             # YAML loading and run context
│   ├── train.py
│   ├── fit_gpu.py            # GPU batch-size sweep
│   ├── download_checkpoints.py
│   ├── eval.py
│   ├── export.py
│   ├── infer.py
│   └── log.py
├── pyproject.toml
└── README.md
```

## Prerequisites

- **Python 3.10–3.13**
- **NVIDIA GPU** recommended for training
- **[uv](https://docs.astral.sh/uv/)** package manager

## Install

From the project root:

```powershell
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1

# PyTorch with CUDA first
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# RF-DETR + this project's CLI
uv pip install "rfdetr[train]" "rfdetr[onnx]" tensorboard
uv pip install -e .
```

Pretrained COCO weights are stored in `checkpoints/` (auto-downloaded on first train). To fetch them upfront:

```powershell
uv run rfdetr-tools download-checkpoints
uv run rfdetr-tools download-checkpoints --model RFDETRSmall
uv run rfdetr-tools download-checkpoints --all
```

Override the directory with `RFDETR_CHECKPOINTS_DIR` if needed.

Optional logging extras:

```powershell
uv pip install "rfdetr[loggers]"   # Weights & Biases, MLflow
```

Verify:

```powershell
uv run rfdetr-tools --help
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## Dataset format

RF-DETR expects COCO detection layout:

```text
dataset/my_task/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/                         # optional
    ├── _annotations.coco.json
    └── *.jpg
```

Validate or convert:

```powershell
python dataset/convert_to_rfdetr_coco.py dataset/my_task --validate-only
```

## Configuration

Experiments are defined in YAML under `configs/`. Example: `configs/egyptian_id_small.yaml`.

```yaml
model:
  variant: RFDETRSmall
  gradient_checkpointing: false

dataset_dir: dataset/egyptian_id
output_dir: output/egyptian_id_small
epochs: 50
batch_size: 4
grad_accum_steps: 4
aug_preset: aggressive
```

`model` accepts either a variant name (`model: RFDETRSmall`) or a mapping with `variant` plus optional model kwargs (`gradient_checkpointing`, `resolution`, `num_classes`, `pretrain_weights`). See `configs/template.yaml`.

CLI flags override config file values.

## Commands

All commands work via `uv run rfdetr-tools <command>` or `python -m rfdetr_tools <command>`.

### Fit GPU (batch size sweep)

Probe your GPU with RF-DETR's built-in memory profiler. Tries combinations of **effective batch target** (8, 16, 32) and **gradient checkpointing** (off/on), then recommends the fastest config that fits:

```powershell
uv run rfdetr-tools fit-gpu --config configs/egyptian_id_small.yaml

# Custom targets, write fitted config, then train
uv run rfdetr-tools fit-gpu --config configs/egyptian_id_small.yaml --targets 8,16,32 --output-config configs/egyptian_id_small_fitted.yaml
uv run rfdetr-tools train --config configs/egyptian_id_small_fitted.yaml

# Probe and train in one step
uv run rfdetr-tools fit-gpu --config configs/egyptian_id_small.yaml --train
```

You can also set `batch_size: auto` in YAML — training probes once at startup (same profiler, but `fit-gpu` compares more settings up front).

### Train

```powershell
# From config file
uv run rfdetr-tools train --config configs/egyptian_id_small.yaml

# Override settings
uv run rfdetr-tools train --config configs/egyptian_id_small.yaml --batch-size 2 --gradient-checkpointing

# Without config file
uv run rfdetr-tools train --dataset-dir dataset/egyptian_id --output-dir output/egyptian_id_small --model RFDETRSmall --aug-preset aggressive

# Resume
uv run rfdetr-tools train --config configs/egyptian_id_small.yaml --resume output/egyptian_id_small/checkpoint_best_total.pth

# W&B logging
uv run rfdetr-tools train --config configs/egyptian_id_small.yaml --wandb --project my-project --run exp-01

# Mirror console output to train.log
uv run rfdetr-tools train --config configs/egyptian_id_small.yaml --log-file
```

Or in YAML:

```yaml
log_file: train.log   # or true for the same default name
```

**RTX 3050 (4 GB) tips:** run `fit-gpu` first, or keep effective batch at 16 (`batch_size × grad_accum_steps`). If OOM, add `--gradient-checkpointing` or lower `--batch-size`.

Outputs per run:

| Artifact | Description |
|----------|-------------|
| `checkpoint_best_*.pth` | Best weights (prefers `checkpoint_best_total`, then `regular`, then `ema`) |
| `metrics.csv` | Per-epoch train/val metrics (always written by RF-DETR) |
| `train.log` | Console output (optional; set `log_file:` in config or `--log-file`) |
| `training_config.json` | Full reproducibility config |
| TensorBoard events | Under the output directory |

### Eval

Run COCO mAP on a saved checkpoint:

```powershell
uv run rfdetr-tools eval --train-output-dir output/egyptian_id_small
uv run rfdetr-tools eval --checkpoint output/egyptian_id_small/checkpoint_best_regular.pth --split valid
uv run rfdetr-tools eval --train-output-dir output/egyptian_id_small --split test
```

### Log

Inspect metrics, tail the console log, or launch TensorBoard:

```powershell
uv run rfdetr-tools log --output-dir output/egyptian_id_small --summary
uv run rfdetr-tools log --output-dir output/egyptian_id_small --summary --tail-log 30
uv run rfdetr-tools log --output-dir output/egyptian_id_small --tensorboard --port 6006
```

### Export

Export best checkpoint to ONNX and TorchScript:

```powershell
uv run rfdetr-tools export --train-output-dir output/egyptian_id_small
uv run rfdetr-tools export --checkpoint path/to/checkpoint.pth --format onnx
```

Writes to `<run-dir>/exported/`:

- `rfdetr-small.onnx`
- `rfdetr-small.ts.pt`
- `export_metadata.json`

### Infer

Run detection on image(s):

```powershell
# PyTorch (checkpoint)
uv run rfdetr-tools infer image.jpg --backend pytorch --train-output-dir output/egyptian_id_small --output result.jpg

# ONNX
uv run rfdetr-tools infer image.jpg --backend onnx --model output/egyptian_id_small/exported/rfdetr-small.onnx --output result.jpg

# TorchScript
uv run rfdetr-tools infer image.jpg --backend torchscript --model output/egyptian_id_small/exported/rfdetr-small.ts.pt --output result.jpg

# Directory of images
uv run rfdetr-tools infer dataset/egyptian_id/valid --backend onnx --model output/egyptian_id_small/exported/rfdetr-small.onnx --output output/predictions/
```

## End-to-end workflow

```powershell
# Optional: probe GPU and write a fitted config
uv run rfdetr-tools fit-gpu --config configs/egyptian_id_small.yaml --output-config configs/egyptian_id_small_fitted.yaml

uv run rfdetr-tools train --config configs/egyptian_id_small.yaml
uv run rfdetr-tools log --output-dir output/egyptian_id_small --summary
uv run rfdetr-tools eval --train-output-dir output/egyptian_id_small
uv run rfdetr-tools export --train-output-dir output/egyptian_id_small
uv run rfdetr-tools infer dataset/egyptian_id/valid/sample.jpg --backend onnx --model output/egyptian_id_small/exported/rfdetr-small.onnx --output output/pred.jpg
```

The Egyptian ID dataset in this repo is an **example project**. For a new task, copy `configs/template.yaml`, point `dataset_dir` at your COCO dataset, and set `output_dir` to a new run folder.

## Suggested next steps

These are optional improvements worth considering as the project grows:

| Area | Suggestion |
|------|------------|
| **New experiments** | Copy `configs/template.yaml` instead of editing the Egyptian ID config in place |
| **GPU tuning** | Run `fit-gpu` before long training jobs; commit the generated `*_fitted.yaml` |
| **Experiment tracking** | Install `rfdetr[loggers]` and use `--wandb` / `--project` on train |
| **Dataset splits** | Add a held-out `test/` split if you only have `train/` and `valid/` today |
| **CI** | GitHub Actions job: `uv sync`, lint, and a smoke test (`rfdetr-tools --help`, config parse) |
| **Quality gates** | Add `ruff` + `pre-commit` for formatting and import order |
| **Tests** | Small pytest suite for `config.py` and `checkpoints.resolve_checkpoint()` |
| **Deployment** | ONNX Runtime or TensorRT benchmarks; document latency vs PyTorch |
| **Reproducibility** | Pin `rfdetr`, `torch`, and CUDA in `uv.lock`; tag releases with export metadata |
| **Windows eval** | If metric tables still fail to print, set `$env:PYTHONUTF8=1` before `eval` |

## Model variants

Set `model:` in config or `--model` on the CLI:

| Variant | Typical use |
|---------|-------------|
| `RFDETRNano` | Edge / low latency |
| `RFDETRSmall` | Default balance (512×512) |
| `RFDETRMedium` | Higher accuracy |
| `RFDETRLarge` | Best accuracy, more VRAM |

## Augmentation presets

Set `aug_preset:` in config or `--aug-preset` on CLI:

- `conservative` — light augmentations
- `aggressive` — strong augmentations (default in Egyptian ID config)
- `aerial` — aerial imagery
- `industrial` — industrial scenes

## Troubleshooting

### CUDA not available

Reinstall PyTorch from the CUDA index:

```powershell
uv pip uninstall torch torchvision
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Out of memory

- `--gradient-checkpointing`
- Lower `--batch-size`, raise `--grad-accum-steps` to keep effective batch at 16
- Use `RFDETRNano` or `RFDETRSmall`

### Fresh reinstall

```powershell
Remove-Item -Recurse -Force .venv
uv venv --python 3.11 .venv
.venv\Scripts\Activate.ps1
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "rfdetr[train]" "rfdetr[onnx]" tensorboard
uv pip install -e .
```

## References

- [RF-DETR documentation](https://rfdetr.roboflow.com/latest/)
- [Training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/)
- [Export guide](https://rfdetr.roboflow.com/latest/learn/export/)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
