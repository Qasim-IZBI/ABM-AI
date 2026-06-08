# ABM — Agent-Based Model Deep Learning Pipeline

Predicts cell population dynamics from ABM simulation parameters using deep learning.

## Quick start

### 1 — Sort data
```bash
python pipeline.py data1.txt data2.txt --images /path/to/images --out data/processed
```

### 2 — Train
```bash
python train.py --data data/processed --model mlp --epochs 100 --out results/run1
```

### 3 — Inference
```bash
python inference.py --ckpt results/run1/checkpoints/epoch_0100.pt \
                    --data data/processed --out predictions/
```

## Project structure

```
ABM/
├── pipeline.py          # Data sorting CLI: text files + images → .npy arrays
├── train.py             # Training entry point
├── inference.py         # Inference entry point
├── utils.py             # Metrics, checkpointing, seeding
├── datasets/
│   ├── abm_dataset.py   # PyTorch Dataset (numerical + images)
│   └── transforms.py    # MinMaxScaler, StandardScaler, ImageTransform
├── models/
│   └── mlp.py           # MLP regression model
├── trainer/
│   └── base_trainer.py  # Training loop with auto-resume
└── tests/
    ├── conftest.py
    └── test_datasets.py
```

## Inputs and outputs

| Role | Columns | Description |
|------|---------|-------------|
| Input | Line 2 | Default cell diffusion rate |
| Input | Line 4 | Cellcycle time mean |
| Output | Line 6 | Population size |
| Output | Line 8 | Number of proliferating cells |
| Output | Line 9 | Number of quiescent cells |
| Output | Line 13 | Diameter (outer limits) |
| Output | Line 14 | Extension in x |
| Output | Line 15 | Extension in y |

## Running tests

```bash
pip install pytest torch torchvision pillow numpy
pytest tests/
```
