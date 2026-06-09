# CLAUDE.md — ABM-AI Project Guide

## Project overview

This repo trains five deep learning models to predict or generate cell population
outputs from Agent-Based Model (ABM) simulation parameters. The full pipeline is:

1. **Sort + split** (`pipeline.py`) — parses tab-separated simulation logs, matches
   images, and writes fixed train/val/test splits.
2. **Train** (`train.py`) — trains any of the five model architectures.
3. **Inference** (`inference.py`) — runs a trained model on new data (no labels needed).
4. **Evaluate** (`evaluation.py`) — computes metrics and saves plots against ground truth.

Batch shell scripts in `scripts/` cover steps 2–4 for both sequential (single machine)
and parallel (SLURM cluster) execution.

---

## Dataset facts

- **Total samples**: ~1543 · **Train**: ~1080 · **Val**: ~231 · **Test**: ~231
- **Inputs**: 2 numerical parameters per simulation run
- **Outputs**: 6 numerical values + 1000×1000 px RGB image
- **Model sizes** are deliberately small (9k–1.7M params) to avoid overfitting ~1080 samples.
  See the *Model sizing* section for the full rationale.

---

## Data format

Raw data lives in `ABM_DATA/`. Each row is one ABM simulation run with 15
tab-separated columns (Windows line endings `\r\n`, trailing tab):

```
col 0   Population name          e.g. B5_T2_1_2
col 1   Cell diffusion rate      (varies — INPUT)
col 2   Cell-cell adhesion       (constant 1.0 — excluded)
col 3   Cellcycle time mean      (varies — INPUT)
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

Image filename pattern: `{population_name}_rayimg000001.png`
Native image dimensions: **1000×1000 px**

---

## Image resolution — padding vs resizing

Native images are 1000×1000 px. 1000 is not a power of 2, so `ConvUpGenerator`
(which requires `log2(size / 4)` to be an integer) cannot use it directly.

**Recommended approach — zero-padding to 1024:**
- Adds 12 px of zeros symmetrically on each side: 1000 → 1024
- **No pixel information is lost**
- Pass `--pad_images` to `train.py`, `inference.py`, `evaluation.py`

```
 Original 1000×1000          Padded 1024×1024
┌──────────────────┐       ┌──────────────────────┐
│                  │  →    │ 12px black border    │
│   actual image   │       │ ┌──────────────────┐ │
│                  │       │ │   actual image   │ │
└──────────────────┘       │ └──────────────────┘ │
                           │ 12px black border    │
                           └──────────────────────┘
```

**Alternative — resize to 512:**
- Halves linear resolution (26% of native pixels)
- Omit `--pad_images` and pass `--image_size 512`
- Use when GPU memory is very limited

| `--image_size` | Approach | Resolution | GPU mem / batch |
|---|---|---|---|
| **1024** (default) | `--pad_images` | **100% native** | ~200 MB |
| 512 | resize | 26% native | ~50 MB |
| 256 | resize | 6.5% native | ~13 MB |

---

## Step 1 — Sort and split data (`pipeline.py`)

```bash
# Recommended: fixed train/val/test split (70 / 15 / 15)
python pipeline.py ABM_DATA/abm.ai-training_all_data_numerical.txt \
    --images /path/to/images \
    --out data/processed \
    --val_split 0.15 --test_split 0.15

# Multiple text files, multiple image directories:
python pipeline.py data1.txt data2.txt \
    --images /imgs/batch1 /imgs/batch2 \
    --out data/processed \
    --val_split 0.15 --test_split 0.15

# Also pre-load images as numpy arrays with zero-padding:
python pipeline.py data.txt --images /imgs \
    --load-images --size 1024 1024 --pad \
    --out data/processed --val_split 0.15 --test_split 0.15

# No split — flat output (backwards compatible):
python pipeline.py data.txt --images /imgs --out data/processed
```

**With splitting**, outputs are written to three subdirectories:
```
data/processed/
├── train/   inputs.npy  labels.npy  image_paths.txt   (~70 %)
├── val/     inputs.npy  labels.npy  image_paths.txt   (~15 %)
└── test/    inputs.npy  labels.npy  image_paths.txt   (~15 %)
```

`train.py` auto-detects these subdirectories and uses the pre-split data.
The test set is never touched during training.

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--val_split` | 0 | Fraction for validation |
| `--test_split` | 0 | Fraction for test set |
| `--seed` | 42 | Shuffle seed for reproducibility |
| `--load-images` | off | Pre-load all images into `images.npy` |
| `--size W H` | 1024 1024 | Target size for `--load-images` |
| `--pad` | off | Zero-pad instead of resize for `--load-images` |

---

## Step 2 — Train (`train.py`)

`train.py` accepts all five model types via `--model`. It auto-detects whether
`--data` contains pre-split subdirectories or a flat directory.

**Always pass `--pad_images` when working with 1000×1000 native images.**

### MLP (numerical regression)
```bash
python train.py --data data/processed --model mlp \
    --epochs 200 --batch_size 64 --lr 1e-3 \
    --hidden_dims 128 64 --dropout 0.2 --activation relu \
    --out results/mlp
```

### imreg (deterministic image generator)
```bash
python train.py --data data/processed --model imreg \
    --epochs 300 --batch_size 8 --lr 2e-4 \
    --image_size 1024 --pad_images --ngf 32 \
    --lambda_l1 1.0 --lambda_mse 1.0 \
    --out results/imreg
```

### cgan (conditional GAN)
```bash
python train.py --data data/processed --model cgan \
    --epochs 300 --batch_size 4 --lr 2e-4 \
    --image_size 1024 --pad_images --ngf 32 --ndf 32 --noise_dim 64 \
    --lambda_l1 10.0 \
    --out results/cgan
```

### mmimreg (multimodal deterministic: image + numerical)
```bash
python train.py --data data/processed --model mmimreg \
    --epochs 300 --batch_size 8 --lr 2e-4 \
    --image_size 1024 --pad_images --ngf 32 --hidden_dims 64 64 \
    --lambda_l1 1.0 --lambda_mse 1.0 --lambda_reg 1.0 \
    --out results/mmimreg
```

### mmcgan (multimodal GAN: image + numerical)
```bash
python train.py --data data/processed --model mmcgan \
    --epochs 400 --batch_size 4 --lr 2e-4 \
    --image_size 1024 --pad_images --ngf 32 --ndf 32 --noise_dim 64 \
    --hidden_dims 64 64 --lambda_l1 10.0 --lambda_reg 1.0 \
    --out results/mmcgan
```

**Auto-resume**: re-run the same command — `BaseTrainer` picks up the latest
checkpoint in `<out>/checkpoints/` automatically.

**Scalers saved**: `train.py` writes `input_scaler.pkl` and (for numerical models)
`label_scaler.pkl` to `<out>/`. These are loaded automatically by `inference.py`
and `evaluation.py` to denormalise outputs into real units.

---

## Step 3 — Inference (`inference.py`)

Handles all five model types. Loads scalers automatically.

```bash
# Numerical model:
python inference.py \
    --ckpt results/mlp/checkpoints/epoch_0200.pt \
    --data data/processed/test --out inference/mlp

# Image model with padding:
python inference.py \
    --ckpt results/cgan/checkpoints/epoch_0300.pt \
    --data data/processed/test --out inference/cgan \
    --image_size 1024 --pad_images --save_images
```

**Outputs by model type:**

| Model | `predictions.npy` | `images.npy` | `images/*.png` |
|-------|-------------------|--------------|----------------|
| mlp | ✓ real units | — | — |
| imreg / cgan | — | ✓ | ✓ (if `--save_images`) |
| mmimreg / mmcgan | ✓ real units | ✓ | ✓ (if `--save_images`) |

---

## Step 4 — Evaluation (`evaluation.py`)

Requires ground-truth labels. Loads scalers automatically.

```bash
# Any model:
python evaluation.py \
    --ckpt results/mmimreg/checkpoints/epoch_0300.pt \
    --data data/processed/test --out eval/mmimreg \
    --image_size 1024 --pad_images

# Headless server (no matplotlib / plots):
python evaluation.py \
    --ckpt results/mmcgan/checkpoints/epoch_0400.pt \
    --data data/processed/test --out eval/mmcgan \
    --image_size 1024 --pad_images --no_plots
```

**Outputs:**

| File | Numerical models | Image models |
|------|-----------------|--------------|
| `metrics.json` | MSE, MAE, R² (normalised + real units) | PSNR, L1, SSIM |
| `predictions.csv` | per-sample predicted vs actual | — |
| `scatter.png` | predicted vs actual per output | — |
| `image_grid.png` | — | real (top) vs generated (bottom) |

SSIM requires `pip install scikit-image`. Scatter plots require `matplotlib`.

---

## Batch scripts

Six scripts in `scripts/` cover training, inference, and evaluation. Each script
comes in two variants:

### `*_all.sh` — sequential, single machine

Runs models **one after another** on the machine where you launch it.

```bash
bash scripts/train_all.sh
bash scripts/infer_all.sh
bash scripts/eval_all.sh
```

All images are handled with `--pad_images` by default. Configuration via env vars:

```bash
# Common overrides (all *_all.sh scripts):
DATA=/my/data          # processed data directory
RESULTS_ROOT=results   # where training outputs live
INFER_ROOT=inference   # where inference writes
EVAL_ROOT=eval         # where evaluation writes
SPLIT=test             # which split to use (train|val|test)
MODELS="mlp cgan"      # subset of models to run
CONDA_ENV=abm          # conda environment name ("" to skip activation)
NO_PLOTS=1             # skip matplotlib output (eval_all.sh only)
SAVE_IMAGES=1          # write per-sample PNGs (infer_all.sh only)

# Example:
DATA=/my/data MODELS="mlp mmimreg" bash scripts/train_all.sh
```

### `*_all_slurm.sh` — parallel, SLURM cluster

Submits all 5 models as **independent parallel jobs** to a SLURM scheduler.
Each job gets its own dedicated GPU; all 5 finish in the time of the slowest one.

```bash
# Edit DATA / RESULTS_ROOT / CONDA_ENV at the top of each script first, then:
sbatch scripts/train_all_slurm.sh    # --array=0-4, one GPU per model
sbatch scripts/infer_all_slurm.sh
sbatch scripts/eval_all_slurm.sh

# Single model by array index:
sbatch --array=0 scripts/train_all_slurm.sh   # mlp only
sbatch --array=2 scripts/train_all_slurm.sh   # cgan only
```

**Array index mapping** (all SLURM scripts): `0=mlp  1=imreg  2=cgan  3=mmimreg  4=mmcgan`

| | `*_all.sh` | `*_all_slurm.sh` |
|---|---|---|
| Where it runs | Your current machine | SLURM cluster |
| Models run | One at a time | All 5 simultaneously |
| Wall time | Sum of all 5 | Longest single model |
| Survives disconnect | No | Yes |
| Requires SLURM | No | Yes |

---

## Model sizing rationale

With ~1080 training samples and only 2 input dimensions, large models overfit badly.
Defaults were chosen to keep the samples-per-parameter ratio reasonable:

| Model | Params | Samples/param |
|-------|--------|---------------|
| mlp | 9k | 0.12× |
| imreg | 736k | 0.0015× |
| cgan | 1.7M | 0.0006× |
| mmimreg | 740k | 0.0015× |
| mmcgan | 1.7M | 0.0006× |

The image model ratios look low but are expected: each sample contains 1024×1024×3 ≈ 3M
output values, and the 2D input space constrains what the model needs to learn.

**Default hyperparameters (tuned for this dataset):**

| Model | Key defaults |
|-------|-------------|
| mlp | `hidden_dims=[128,64]`, `dropout=0.2` |
| imreg / mmimreg | `ngf=32`, `image_size=1024` |
| cgan / mmcgan | `ngf=32`, `ndf=32`, `noise_dim=64`, `image_size=1024` |
| mm* reg heads | `hidden_dims=[64,64]` |

---

## Model architectures

### MLP (`models/mlp.py`)
```
Input  : (B, 2)  — normalised [diffusion_rate, cellcycle_time_mean]
FC(2→128) → ReLU → Dropout(0.2) → FC(128→64) → ReLU → Dropout(0.2) → FC(64→6)
Output : (B, 6)  — predicted simulation outputs
```
Config: `MLPConfig(n_inputs, n_outputs, hidden_dims=[128,64], dropout=0.2, activation="relu")`

### imreg (`models/imreg.py`)
```
Input  : (B, 2)  — condition
FC(2 → ngf×8×4×4) → reshape(ngf×8, 4, 4)
7× ConvTranspose2d up-blocks (4→8→…→512→1024) → tanh
Output : (B, 3, 1024, 1024)
```
Config: `ImageRegressorConfig(n_inputs, image_size=1024, ngf=32, lambda_l1=1.0, lambda_mse=1.0)`

### cgan (`models/cgan.py`)
```
Generator  : cat(condition, noise) → FC → 7× ConvTranspose2d → (3, 1024, 1024)
Discriminator : 70×70 PatchGAN — condition broadcast spatially, spectral norm + instance norm
Loss G     : LSGAN adversarial  +  λ_l1 × L1(fake, real)
Loss D     : LSGAN real/fake (averaged)
```
Config: `CGANConfig(n_inputs, noise_dim=64, image_size=1024, ngf=32, ndf=32, lambda_l1=10.0)`

### mmimreg (`models/mmimreg.py`)
```
Image branch     : ConvUpGenerator(condition) → (3, 1024, 1024)   [no noise]
Numerical branch : RegressionHead(condition)  → (n_outputs,)
Loss : λ_l1 × L1(img)  +  λ_mse × MSE(img)  +  λ_reg × MSE(numerical)
```
Config: `MultiModalImRegConfig(n_inputs, n_outputs, image_size=1024, ngf=32, hidden_dims=[64,64], lambda_l1=1.0, lambda_mse=1.0, lambda_reg=1.0)`

### mmcgan (`models/mmcgan.py`)
```
G  : ConvUpGenerator(cat(condition, noise)) → (3, 1024, 1024)
R  : RegressionHead(condition) → (n_outputs,)   [deterministic — no noise]
D  : PatchDiscriminator(image, condition)
Loss G : LSGAN_adv  +  λ_l1 × L1(fake, real)  +  λ_reg × MSE(ŷ, y)
Loss D : LSGAN real/fake
```
Config: `MultiModalCGANConfig(n_inputs, n_outputs, noise_dim=64, image_size=1024, ngf=32, ndf=32, hidden_dims=[64,64], lambda_l1=10.0, lambda_reg=1.0)`

### Shared components (`models/base_models.py`)
- `ConvUpGenerator(in_channels, image_size, ngf)` — used by mmimreg and mmcgan
- `PatchDiscriminator(n_cond, ndf)` — used by mmcgan
- `RegressionHead(in_features, n_outputs, hidden_dims)` — used by mmimreg and mmcgan

---

## Training interface (`trainer/base_trainer.py`)

`BaseTrainer` detects the model type automatically from its methods:

**Regression mode** (MLP): model exposes only `forward(x)`.
```
Single AdamW optimizer — MSE loss — logs loss + R²
Validation loop runs every epoch
```

**Generator-only mode** (imreg, mmimreg): model has `generator_parameters()` and
`compute_generator_loss(batch)` but NOT `discriminator_parameters()`.
```
Single Adam optimizer (betas=0.5, 0.999)
No discriminator step
```

**Full GAN mode** (cgan, mmcgan): model also has `discriminator_parameters()` and
`compute_discriminator_loss(batch, visuals)`.
```
Two Adam optimizers — alternating G and D steps each batch
```

---

## Checkpoint format

```json
{
  "epoch":       int,
  "model":       state_dict,
  "optimizer":   state_dict,       // opt_G for GAN models
  "optimizer_d": state_dict,       // only present for cgan / mmcgan
  "config":      { model config }  // full dataclass as dict, including model_name
}
```

Saved every `--save_every` epochs as `<out>/checkpoints/epoch_NNNN.pt`.
`BaseTrainer` auto-resumes from the highest-numbered checkpoint found.

Also saved adjacent to the checkpoints directory:
- `<out>/input_scaler.pkl` — MinMaxScaler fitted on training inputs
- `<out>/label_scaler.pkl` — MinMaxScaler fitted on training labels (numerical models only)

---

## Key conventions

- **Normalisation**: inputs scaled to `[0, 1]` with `MinMaxScaler` fitted on the
  training split only. Scalers are persisted to disk and loaded automatically by
  `inference.py` and `evaluation.py` to report metrics in real units.
- **Images**: normalised to `[-1, 1]` by `ImageTransform`. Training applies random
  horizontal/vertical flips; inference/evaluation does not.
- **Padding**: native images are 1000×1000 px. Pass `--pad_images` to zero-pad to
  1024×1024 (12 px per side). This preserves all pixel information. Do not mix
  `--pad_images` and resize modes between training and inference — they must match.
- **Data splits**: run `pipeline.py --val_split 0.15 --test_split 0.15` once with
  `--seed 42` (default). All subsequent runs see identical splits. The test set is
  held out until final evaluation.
- **Device**: auto-detected (`cuda` if available, else `cpu`).
- **Reproducibility**: `--seed 42` in `train.py` (default). Pipeline shuffling uses
  `--seed 42` (default).
- **Excluded columns**: cell-cell adhesion (col 2, constant 1.0), cellcycle time SD
  (col 4, constant 0.083), n_dead (col 9, always 0) — all excluded.

---

## Tests

```bash
pip install pytest torch torchvision pillow numpy
pytest tests/ -v
```

37 tests covering dataset loading, scaler roundtrips, padding transform correctness,
and forward passes + training interfaces for all five models.
Tests use in-memory temporary data — no real files needed.

---

## Adding a new model

### Regression model (numerical output only)
1. Create `models/<name>.py` with a `<Name>Config` dataclass and `<Name>(nn.Module)`.
2. Implement `forward(x) -> Tensor` and `config_dict() -> dict`.
3. Register in `models/__init__.py` MODELS dict and add to `NUMERICAL_PREDICTION_MODELS`.
4. `train.py` and `inference.py` pick it up automatically via `build_model`.

### Image-generation model (no discriminator)
1. Same as above, but also implement:
   - `generator_parameters()` → iterable of parameters
   - `compute_generator_loss(batch) → (loss, logs, visuals)`
   - `generate(condition) → image tensor`
2. Register and add to `IMAGE_GENERATION_MODELS`.

### GAN model (generator + discriminator)
1. Implement all of the above, plus:
   - `discriminator_parameters()` → iterable of parameters
   - `compute_discriminator_loss(batch, visuals) → (loss, logs)`
2. `BaseTrainer` detects the discriminator automatically and sets up dual optimizers.

### Multimodal model (image + numerical)
1. Combine both patterns: implement `generate()`, `predict()`, `forward()` returning
   `(image, numerical)`, and add to **both** `IMAGE_GENERATION_MODELS` and
   `NUMERICAL_PREDICTION_MODELS`.
