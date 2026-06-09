# ABM-AI — Agent-Based Model Deep Learning Pipeline

Predicts and generates cell population dynamics from ABM simulation parameters
using five deep learning models spanning regression, deterministic image generation,
and conditional GAN image synthesis.

Dataset: ~1543 samples · Images: 1000×1000 px (zero-padded to 1024×1024)

---

## Quick start

### 1 — Sort and split data
```bash
python pipeline.py data.txt --images /path/to/images \
    --out data/processed \
    --val_split 0.15 --test_split 0.15
```

### 2 — Train all models
```bash
# Single machine (sequential):
bash scripts/train_all.sh

# HPC cluster (all 5 models in parallel via SLURM):
sbatch scripts/train_all_slurm.sh
```

### 3 — Run inference on test set
```bash
bash scripts/infer_all.sh          # single machine
sbatch scripts/infer_all_slurm.sh  # SLURM cluster
```

### 4 — Evaluate all models
```bash
bash scripts/eval_all.sh           # single machine
sbatch scripts/eval_all_slurm.sh   # SLURM cluster
```

---

## Models

| Name | Output 1 | Output 2 | Stochastic | Params |
|------|----------|----------|------------|--------|
| `mlp` | Numerical predictions | — | No | ~9k |
| `imreg` | Generated image | — | No | ~736k |
| `cgan` | Generated image | — | Yes (noise) | ~1.7M |
| `mmimreg` | Generated image | Numerical predictions | No | ~740k |
| `mmcgan` | Generated image | Numerical predictions | Yes (noise) | ~1.7M |

Model sizes are tuned for ~1080 training samples (see [CLAUDE.md](CLAUDE.md) for rationale).

---

## Project structure

```
ABM/
├── pipeline.py          # Data sorting + train/val/test split CLI
├── train.py             # Training entry point (all 5 models)
├── inference.py         # Inference entry point (all 5 models)
├── evaluation.py        # Metrics, scatter plots, image grids
├── utils.py             # Metrics, checkpointing, seeding
│
├── datasets/
│   ├── abm_dataset.py   # PyTorch Dataset (numerical + images)
│   └── transforms.py    # MinMaxScaler, StandardScaler, ImageTransform (+pad mode)
│
├── models/
│   ├── base_models.py   # Shared: ConvUpGenerator, PatchDiscriminator, RegressionHead
│   ├── mlp.py           # MLP regression
│   ├── imreg.py         # Deterministic image generator
│   ├── cgan.py          # Conditional GAN
│   ├── mmimreg.py       # Multimodal: image + numerical (deterministic)
│   └── mmcgan.py        # Multimodal: image + numerical (GAN)
│
├── trainer/
│   └── base_trainer.py  # Universal training loop (regression + GAN)
│
├── scripts/
│   ├── train_all.sh          # Train all 5 models sequentially (single machine)
│   ├── train_all_slurm.sh    # Train all 5 models in parallel (SLURM array)
│   ├── infer_all.sh          # Inference on all models (sequential)
│   ├── infer_all_slurm.sh    # Inference on all models (SLURM array)
│   ├── eval_all.sh           # Evaluate all models (sequential)
│   └── eval_all_slurm.sh     # Evaluate all models (SLURM array)
│
└── tests/
    ├── conftest.py
    ├── test_datasets.py
    └── test_models.py
```

---

## Data columns

| Role | Col | Description |
|------|-----|-------------|
| Input | 1 | Cell diffusion rate |
| Input | 3 | Cellcycle time mean |
| Output | 5 | Population size |
| Output | 7 | Number of proliferating cells |
| Output | 8 | Number of quiescent cells |
| Output | 12 | Diameter (outer limits, µm) |
| Output | 13 | Extension in x (µm) |
| Output | 14 | Extension in y (µm) |

Image filename pattern: `{population_name}_raymg000001.png`
Native image size: **1000×1000 px** — zero-padded to **1024×1024** during training (12 px per side, no information loss).

---

## Running tests

```bash
pip install pytest torch torchvision pillow numpy
pytest tests/ -v
```
