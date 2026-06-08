# CLAUDE.md — ABM Project Guide

## Project overview

This repo trains deep learning models to predict cell population outputs from
Agent-Based Model (ABM) simulation parameters. The pipeline has three stages:

1. **Data sorting** (`pipeline.py`) — parses tab-separated simulation logs and
   matches them to corresponding microscopy images.
2. **Training** (`train.py`) — trains a regression model on the processed data.
3. **Inference** (`inference.py`) — runs a trained model on new data and saves
   predictions.

---

## Data format

Raw data lives in `ABM_DATA/`. Each row in the `.txt` file is one ABM simulation
run with 15 tab-separated columns (Windows line endings `\r\n`, trailing tab):

```
col 0   Population name          e.g. B5_T2_1_2
col 1   Cell diffusion rate      (varies — used as INPUT)
col 2   Cell-cell adhesion       (constant 1.0 — excluded)
col 3   Cellcycle time mean      (varies — used as INPUT)
col 4   Cellcycle time SD        (constant 0.083 — excluded)
col 5   Population size          <- OUTPUT
col 6   Time (days)              (near-constant — excluded)
col 7   n_proliferating          <- OUTPUT
col 8   n_quiescent              <- OUTPUT
col 9   n_dead                   (always 0 — excluded)
col 10  Radius of gyration       (excluded — col 12 preferred)
col 11  Diameter (gyration)      (excluded)
col 12  Diameter (outer limits)  <- OUTPUT
col 13  Extension in x           <- OUTPUT
col 14  Extension in y           <- OUTPUT
```

Image filename pattern: `{population_name}_raymg000001.png`
e.g. `B5_T2_1_2_raymg000001.png`

---

## Running the pipeline

### Sort data (required before training)

```bash
# Single text file, images in a separate directory:
python pipeline.py ABM_DATA/abm.ai-training_all_data_numerical.txt \
    --images /path/to/images --out data/processed

# Multiple text files, multiple image directories:
python pipeline.py data1.txt data2.txt \
    --images /imgs/batch1 /imgs/batch2 --out data/processed

# Load images into numpy array (256×256 RGB):
python pipeline.py data1.txt --images /imgs --load-images --size 256 256 --out data/processed
```

Outputs written to `--out`:
- `inputs.npy`       shape `(N, 2)`
- `labels.npy`       shape `(N, 6)`
- `image_paths.txt`  one resolved path per row

### Train

```bash
# MLP, numerical inputs only:
python train.py --data data/processed --model mlp --epochs 100

# Custom architecture:
python train.py --data data/processed --model mlp \
    --hidden_dims 512 512 256 128 --dropout 0.2 --activation gelu \
    --lr 3e-4 --batch_size 32 --epochs 200 --out results/run2

# Auto-resume from latest checkpoint (just re-run the same command):
python train.py --data data/processed --model mlp --out results/run1
```

### Inference

```bash
python inference.py \
    --ckpt results/run1/checkpoints/epoch_0100.pt \
    --data data/processed \
    --out predictions/run1
```

Saves `predictions.npy` (shape `N × 6`) and prints per-output MAE.

---

## Model interface

Every model must expose:

```python
model.forward(x: Tensor) -> Tensor   # (B, n_inputs) -> (B, n_outputs)
model.config_dict() -> dict           # JSON-safe config for checkpointing
```

Config is stored as a `@dataclass` (see `models/mlp.py` for the pattern).
Add new models to `models/__init__.py` MODELS dict; they become available
as `--model <name>` in `train.py`.

---

## Checkpoint format

```json
{
  "epoch":     int,
  "model":     state_dict,
  "optimizer": state_dict,
  "config":    { model config dict }
}
```

Checkpoints are saved every `--save_every` epochs as
`<out>/checkpoints/epoch_NNNN.pt`. `BaseTrainer` auto-resumes from the
highest-numbered checkpoint found in that directory.

---

## Key conventions

- **Normalisation**: inputs and labels are scaled with `MinMaxScaler` fitted on
  the training split only. The scaler is re-fitted every run; if you need to
  apply the same scaler at inference, save it with `pickle` after training.
- **Images**: normalised to `[-1, 1]` by `ImageTransform`. Training applies
  random horizontal/vertical flips; inference does not.
- **Device**: auto-detected (`cuda` if available, else `cpu`).
- **Reproducibility**: pass `--seed` (default 42).
- **Excluded columns**: cell-cell adhesion (col 2) and cellcycle SD (col 4) are
  constant across the dataset and excluded. `n_dead` (col 9) is always 0.

---

## Tests

```bash
pip install pytest torch torchvision pillow numpy
pytest tests/ -v
```

Tests use an in-memory temporary dataset (no real files needed).

---

## Adding a new model

1. Create `models/<name>.py` with a `<Name>Config` dataclass and `<Name>(nn.Module)`.
2. Implement `forward(x)` and `config_dict()`.
3. Register in `models/__init__.py`: add to `MODELS` dict.
4. The new model is immediately available as `--model <name>` in `train.py`.
