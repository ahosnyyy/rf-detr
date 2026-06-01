# RF-DETR Project

Fine-tune, evaluate, export, and deploy [RF-DETR](https://rfdetr.roboflow.com/latest/) object detection models on custom COCO-format datasets.

This repo provides scripts under `rfdetr_tools/` for the full workflow: **train → eval → log → export → infer**. Run them from the project root with `python -m rfdetr_tools` — nothing is installed as a package.

## Project layout

```text
rf-detr/
├── checkpoints/              # Pretrained COCO weights (auto-downloaded, gitignored)
├── configs/                  # YAML experiment configs
├── dataset/                  # COCO datasets + conversion utilities
├── output/                   # Training runs (checkpoints, logs, exports)
├── rfdetr_tools/             # CLI scripts (run via python -m rfdetr_tools)
│   ├── cli.py
│   ├── checkpoints.py
│   ├── config.py
│   ├── train.py
│   ├── fit_gpu.py
│   ├── download_checkpoints.py
│   ├── eval.py
│   ├── export.py
│   ├── infer.py
│   └── log.py
├── requirements.txt          # pyyaml, tensorboard (local tooling only)
└── README.md
```

## Prerequisites

- **Python 3.10–3.13**
- **NVIDIA GPU** recommended for training
- **pip** and **venv**

## Install

All commands assume you are in the **project root** (`rf-detr/`).

**Linux / macOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA (pick the index that matches your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# RF-DETR + local script deps
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt

python -m rfdetr_tools --help
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt
python -m rfdetr_tools --help
```

Pretrained weights go in `checkpoints/` (auto-downloaded on first train). Prefetch:

```bash
python -m rfdetr_tools download-checkpoints --model RFDETRLarge
python -m rfdetr_tools download-checkpoints --all
```

Override the directory with `RFDETR_CHECKPOINTS_DIR` if needed.

Optional experiment tracking:

```bash
pip install "rfdetr[loggers]"   # Weights & Biases, MLflow
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

```bash
python dataset/convert_to_rfdetr_coco.py dataset/my_task --validate-only
```

## Configuration

Experiments are defined in YAML under `configs/`. Example: `configs/egyptian_id_small.yaml`. See `configs/template.yaml` for RTX 5090 / RFDETRLarge defaults.

```yaml
model:
  variant: RFDETRSmall
  resolution: 512
  gradient_checkpointing: false

dataset_dir: dataset/egyptian_id
output_dir: output/egyptian_id_small
epochs: 50
batch_size: 4
grad_accum_steps: 4
aug_preset: aggressive
log_file: train.log
```

CLI flags override config file values.

## Commands

Run from the project root:

```bash
python -m rfdetr_tools <command> [options]
```

### Fit GPU (batch size sweep)

```bash
python -m rfdetr_tools fit-gpu --config configs/egyptian_id_small.yaml

python -m rfdetr_tools fit-gpu --config configs/egyptian_id_small.yaml \
  --targets 8,16,32 --output-config configs/egyptian_id_small_fitted.yaml

python -m rfdetr_tools fit-gpu --config configs/egyptian_id_small.yaml --train
```

### Train

```bash
python -m rfdetr_tools train --config configs/egyptian_id_small.yaml

python -m rfdetr_tools train --config configs/egyptian_id_small.yaml \
  --batch-size 2 --gradient-checkpointing

python -m rfdetr_tools train --config configs/egyptian_id_small.yaml --log-file

python -m rfdetr_tools train --config configs/egyptian_id_small.yaml \
  --wandb --project my-project --run exp-01
```

Outputs per run:

| Artifact | Description |
|----------|-------------|
| `checkpoint_best_*.pth` | Best weights |
| `metrics.csv` | Per-epoch metrics (always written) |
| `train.log` | Console log (if `log_file` set) |
| `training_config.json` | Full reproducibility config |
| TensorBoard events | Under the output directory |

### Eval

```bash
python -m rfdetr_tools eval --train-output-dir output/egyptian_id_small
python -m rfdetr_tools eval --checkpoint output/egyptian_id_small/checkpoint_best_regular.pth --split valid
```

### Log

```bash
python -m rfdetr_tools log --output-dir output/egyptian_id_small --summary
python -m rfdetr_tools log --output-dir output/egyptian_id_small --summary --tail-log 30
python -m rfdetr_tools log --output-dir output/egyptian_id_small --tensorboard --port 6006
```

### Export

```bash
python -m rfdetr_tools export --train-output-dir output/egyptian_id_small
python -m rfdetr_tools export --checkpoint path/to/checkpoint.pth --format onnx
```

### Infer

```bash
python -m rfdetr_tools infer image.jpg --backend pytorch \
  --train-output-dir output/egyptian_id_small --output result.jpg

python -m rfdetr_tools infer image.jpg --backend onnx \
  --model output/egyptian_id_small/exported/rfdetr-small.onnx --output result.jpg
```

## End-to-end workflow

```bash
python -m rfdetr_tools fit-gpu --config configs/template.yaml --output-config configs/my_task_fitted.yaml
python -m rfdetr_tools train --config configs/template.yaml
python -m rfdetr_tools log --output-dir output/my_task_large --summary
python -m rfdetr_tools eval --train-output-dir output/my_task_large
python -m rfdetr_tools export --train-output-dir output/my_task_large
```

Copy `configs/template.yaml` for new tasks and point `dataset_dir` at your COCO dataset.

## Model variants

| Variant | Default resolution |
|---------|-------------------|
| `RFDETRNano` | 384 |
| `RFDETRSmall` | 512 |
| `RFDETRMedium` | 576 |
| `RFDETRLarge` | 704 |

## Augmentation presets

`conservative`, `aggressive`, `aerial`, `industrial` — set via `aug_preset:` in config or `--aug-preset` on CLI.

## Troubleshooting

### CUDA not available

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Out of memory

- `--gradient-checkpointing`
- Lower `--batch-size`, raise `--grad-accum-steps`
- Run `fit-gpu` first
- Use a smaller variant (`RFDETRSmall` / `RFDETRNano`)

### Fresh reinstall (Linux)

```bash
cd /path/to/rf-detr
rm -rf .venv

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt

python -m rfdetr_tools download-checkpoints --model RFDETRLarge   # optional
python -m rfdetr_tools --help
```

### Windows eval encoding

If metric tables fail to print: `$env:PYTHONUTF8=1` before `eval`.

## References

- [RF-DETR documentation](https://rfdetr.roboflow.com/latest/)
- [Training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/)
- [Export guide](https://rfdetr.roboflow.com/latest/learn/export/)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
