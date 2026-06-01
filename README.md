# RF-DETR Project

Fine-tune, evaluate, export, and deploy [RF-DETR](https://rfdetr.roboflow.com/latest/) object detection models on custom COCO-format datasets.

**Workflow:** train → eval → log → export → infer

Run scripts directly from the project root — no package install required:

```bash
python rfdetr_tools/train.py --config configs/template.yaml
```

## Project layout

```text
rf-detr/
├── checkpoints/              # Pretrained COCO weights (auto-downloaded, gitignored)
├── configs/                  # YAML experiment configs
│   ├── template.yaml         # RTX 5090 / RFDETRLarge starter config
│   └── egyptian_id_small.yaml
├── dataset/                  # COCO datasets + conversion utilities
├── output/                   # Training runs (checkpoints, logs, exports)
├── rfdetr_tools/             # Runnable scripts + shared helpers
│   ├── train.py
│   ├── fit_gpu.py
│   ├── download_checkpoints.py
│   ├── eval.py
│   ├── export.py
│   ├── infer.py
│   └── log.py
├── requirements.txt          # pyyaml, tensorboard (local script deps)
└── README.md
```

## Prerequisites

- Python **3.10–3.13**
- NVIDIA GPU recommended for training
- `pip` and `venv`

## Install

Always run commands from the **project root** (`rf-detr/`).

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch with CUDA — pick the index that matches your driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# RF-DETR + local script dependencies
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt

python rfdetr_tools/train.py --help
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt
python rfdetr_tools/train.py --help
```

### Pretrained weights

Weights are stored in `checkpoints/` and auto-download on first train. To prefetch:

```bash
python rfdetr_tools/download_checkpoints.py --model RFDETRLarge
python rfdetr_tools/download_checkpoints.py --model RFDETRSmall
python rfdetr_tools/download_checkpoints.py --all
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

Validate an existing dataset:

```bash
python dataset/convert_to_rfdetr_coco.py dataset/my_task --validate-only
```

## Configuration

Experiments are YAML files under `configs/`.

- **`configs/template.yaml`** — RFDETRLarge, tuned for RTX 5090 (32 GB)
- **`configs/egyptian_id_small.yaml`** — example project (Egyptian ID, RFDETRSmall)

```yaml
model:
  variant: RFDETRLarge
  resolution: 704              # must be divisible by 32
  gradient_checkpointing: false
  pretrain_weights: rf-detr-large-2026.pth

dataset_dir: dataset/my_task
output_dir: output/my_task_large

epochs: 100                    # max; early stopping usually stops sooner
batch_size: 16
grad_accum_steps: 2
auto_batch_target_effective: 32  # batch_size × grad_accum_steps

early_stopping: true
early_stopping_patience: 15
skip_best_epochs: 3

aug_preset: aggressive         # conservative | aggressive | aerial | industrial
tensorboard: true
progress_bar: rich             # rich | tqdm | omit to disable
log_file: train.log            # optional console log mirror
```

`model` accepts either a variant name (`model: RFDETRSmall`) or a mapping with `variant` plus optional kwargs (`resolution`, `pretrain_weights`, `gradient_checkpointing`, …). CLI flags override YAML values.

### Model variants

| Variant | Default resolution | Pretrained weights |
|---------|-------------------|-------------------|
| `RFDETRNano` | 384 | `rf-detr-nano.pth` |
| `RFDETRSmall` | 512 | `rf-detr-small.pth` |
| `RFDETRMedium` | 576 | `rf-detr-medium.pth` |
| `RFDETRLarge` | 704 | `rf-detr-large-2026.pth` |

## Scripts

| Script | Purpose |
|--------|---------|
| `rfdetr_tools/train.py` | Fine-tune on a COCO dataset |
| `rfdetr_tools/fit_gpu.py` | Probe GPU memory; recommend batch size |
| `rfdetr_tools/download_checkpoints.py` | Download pretrained weights |
| `rfdetr_tools/eval.py` | COCO mAP on valid/test split |
| `rfdetr_tools/log.py` | Print metrics summary or launch TensorBoard |
| `rfdetr_tools/export.py` | Export ONNX + TorchScript |
| `rfdetr_tools/infer.py` | Run detection (PyTorch / ONNX / TorchScript) |

Alternative combined entry point (same options): `python -m rfdetr_tools train …`

### Fit GPU

Probe effective batch targets and gradient checkpointing before a long run:

```bash
python rfdetr_tools/fit_gpu.py --config configs/template.yaml

python rfdetr_tools/fit_gpu.py --config configs/template.yaml \
  --targets 16,32,64 --output-config configs/my_task_fitted.yaml

python rfdetr_tools/fit_gpu.py --config configs/egyptian_id_small.yaml --train
```

### Train

```bash
python rfdetr_tools/train.py --config configs/template.yaml

python rfdetr_tools/train.py --config configs/egyptian_id_small.yaml \
  --batch-size 2 --gradient-checkpointing

python rfdetr_tools/train.py --config configs/egyptian_id_small.yaml --log-file

python rfdetr_tools/train.py --config configs/egyptian_id_small.yaml \
  --wandb --project my-project --run exp-01
```

**Training outputs** (under `output_dir`):

| File | Description |
|------|-------------|
| `checkpoint_best_*.pth` | Best weights (`total` → `regular` → `ema`) |
| `metrics.csv` | Per-epoch train/val metrics (always written) |
| `train.log` | Console output (if `log_file` set) |
| `training_config.json` | Full run config for reproducibility |
| TensorBoard events | Scalars for charts |

### Eval

```bash
python rfdetr_tools/eval.py --train-output-dir output/egyptian_id_small
python rfdetr_tools/eval.py --checkpoint output/egyptian_id_small/checkpoint_best_regular.pth --split valid
python rfdetr_tools/eval.py --train-output-dir output/egyptian_id_small --split test
```

### Log

```bash
python rfdetr_tools/log.py --output-dir output/egyptian_id_small --summary
python rfdetr_tools/log.py --output-dir output/egyptian_id_small --summary --tail-log 30
python rfdetr_tools/log.py --output-dir output/egyptian_id_small --tensorboard --port 6006
```

### Export

```bash
python rfdetr_tools/export.py --train-output-dir output/egyptian_id_small
python rfdetr_tools/export.py --checkpoint path/to/checkpoint.pth --format onnx
```

Writes to `<run-dir>/exported/`:

- `rfdetr-*.onnx`
- `rfdetr-*.ts.pt`
- `export_metadata.json`

### Infer

```bash
# PyTorch checkpoint
python rfdetr_tools/infer.py image.jpg --backend pytorch \
  --train-output-dir output/egyptian_id_small --output result.jpg

# ONNX
python rfdetr_tools/infer.py image.jpg --backend onnx \
  --model output/egyptian_id_small/exported/rfdetr-small.onnx --output result.jpg

# Directory of images
python rfdetr_tools/infer.py dataset/egyptian_id/valid --backend onnx \
  --model output/egyptian_id_small/exported/rfdetr-small.onnx \
  --output output/predictions/
```

## End-to-end example

**New task (5090 / Large):**

```bash
python rfdetr_tools/fit_gpu.py --config configs/template.yaml --output-config configs/my_task_fitted.yaml
python rfdetr_tools/train.py --config configs/my_task_fitted.yaml
python rfdetr_tools/log.py --output-dir output/my_task_large --summary
python rfdetr_tools/eval.py --train-output-dir output/my_task_large
python rfdetr_tools/export.py --train-output-dir output/my_task_large
```

**Egyptian ID example** (included in repo):

```bash
python rfdetr_tools/fit_gpu.py --config configs/egyptian_id_small.yaml --output-config configs/egyptian_id_small_fitted.yaml
python rfdetr_tools/train.py --config configs/egyptian_id_small_fitted.yaml
python rfdetr_tools/export.py --train-output-dir output/egyptian_id_small
python rfdetr_tools/infer.py dataset/egyptian_id/valid/sample.jpg --backend onnx \
  --model output/egyptian_id_small/exported/rfdetr-small.onnx --output output/pred.jpg
```

Copy `configs/template.yaml` for new datasets; keep the Egyptian ID config as a reference only.

## Troubleshooting

### CUDA not available

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Use the [PyTorch install selector](https://pytorch.org/get-started/locally/) if `cu128` does not match your driver.

### Out of memory

1. Run `fit_gpu.py` and use the generated `*_fitted.yaml`
2. `--gradient-checkpointing` on train
3. Lower `--batch-size`, raise `--grad-accum-steps` to keep effective batch ~16–32
4. Use a smaller variant (`RFDETRSmall`, `RFDETRNano`)

### Progress bar / eval on Windows

- Use `progress_bar: tqdm` in config (safer on Windows terminals)
- If eval metric tables fail to print: `$env:PYTHONUTF8=1`

### Fresh reinstall

**Linux:**

```bash
cd /path/to/rf-detr
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt
python rfdetr_tools/download_checkpoints.py --model RFDETRLarge   # optional
python rfdetr_tools/train.py --help
```

**Windows:**

```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "rfdetr[train]" "rfdetr[onnx]"
pip install -r requirements.txt
python rfdetr_tools/train.py --help
```

## References

- [RF-DETR documentation](https://rfdetr.roboflow.com/latest/)
- [Training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/)
- [Export guide](https://rfdetr.roboflow.com/latest/learn/export/)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
