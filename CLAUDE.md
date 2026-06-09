# CLAUDE.md — ABM-AI Project Guide

## Project overview

This repo trains five deep learning models to predict or generate cell population
outputs from Agent-Based Model (ABM) simulation parameters. The full pipeline is:

1. **Sort + split** (`pipeline.py`) — parses tab-separated simulation logs, matches
   images, and writes fixed train/val/test splits.
2. **Train** (`train.py`) — trains any of the five model architectures.
3. **Inference** (`inference.py`) — runs a trained model on new data (no labels needed).
4. **Evaluate** (`evaluation.py`) — computes metrics and saves plots against ground truth.

Batch shell scripts in `scripts/` automate steps 2–4 across all models sequentially
or as SLURM array jobs.

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

Image filename pattern: `{population_name}_raymg000001.png`

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

# No split (flat output — old behaviour):
python pipeline.py data.txt --images /imgs --out data/processed

# Also load images as a numpy array (256×256 RGB):
python pipeline.py data.txt --images /imgs --load-images --size 256 256 \
    --out data/processed --val_split 0.15 --test_split 0.15
```

**With splitting**, outputs are written to three subdirectories:
```
data/processed/
├── train/   inputs.npy  labels.npy  image_paths.txt   (~70 %)
├── val/     inputs.npy  labels.npy  image_paths.txt   (~15 %)
└── test/    inputs.npy  labels.npy  image_paths.txt   (~15 %)
```

`train.py` automatically detects these subdirectories and uses the correct splits.
The test set is never touched during training.

**Without splitting**, all three files are written flat into `--out`.

---

## Step 2 — Train (`train.py`)

`train.py` accepts all five model types via `--model`. It auto-detects whether
`--data` contains pre-split subdirectories (from `pipeline.py`) or a flat directory.

### MLP (numerical regression)
```bash
python train.py --data data/processed --model mlp \
    --epochs 200 --batch_size 64 --lr 1e-3 \
    --hidden_dims 512 256 128 --dropout 0.1 --activation relu \
    --out results/mlp
```

### imreg (deterministic image generator)
```bash
python train.py --data data/processed --model imreg \
    --epochs 300 --batch_size 32 --lr 2e-4 \
    --image_size 256 --ngf 64 \
    --lambda_l1 1.0 --lambda_mse 1.0 \
    --out results/imreg
```

### cgan (conditional GAN)
```bash
python train.py --data data/processed --model cgan \
    --epochs 300 --batch_size 16 --lr 2e-4 \
    --image_size 256 --ngf 64 --ndf 64 --noise_dim 128 \
    --lambda_l1 10.0 \
    --out results/cgan
```

### mmimreg (multimodal deterministic: image + numerical)
```bash
python train.py --data data/processed --model mmimreg \
    --epochs 300 --batch_size 32 --lr 2e-4 \
    --image_size 256 --ngf 64 --hidden_dims 256 128 \
    --lambda_l1 1.0 --lambda_mse 1.0 --lambda_reg 1.0 \
    --out results/mmimreg
```

### mmcgan (multimodal GAN: image + numerical)
```bash
python train.py --data data/processed --model mmcgan \
    --epochs 400 --batch_size 16 --lr 2e-4 \
    --image_size 256 --ngf 64 --ndf 64 --noise_dim 128 \
    --hidden_dims 256 128 --lambda_l1 10.0 --lambda_reg 1.0 \
    --out results/mmcgan
```

**Auto-resume**: re-run the same command — `BaseTrainer` picks up the latest
checkpoint in `<out>/checkpoints/` automatically.

**Scalers saved**: `train.py` writes `input_scaler.pkl` and (for numerical models)
`label_scaler.pkl` to `<out>/`. These are loaded by `inference.py` and
`evaluation.py` to denormalise outputs into real units.

---

## Step 3 — Inference (`inference.py`)

Handles all five model types. Loads scalers automatically from the same directory
as the checkpoint.

```bash
# Any model — same command:
python inference.py \
    --ckpt results/mlp/checkpoints/epoch_0200.pt \
    --data data/processed/test \
    --out  inference/mlp

# Also save individual PNGs for image models:
python inference.py \
    --ckpt results/cgan/checkpoints/epoch_0300.pt \
    --data data/processed/test \
    --out  inference/cgan --save_images
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
python evaluation.py \
    --ckpt results/mlp/checkpoints/epoch_0200.pt \
    --data data/processed/test \
    --out  eval/mlp

# Headless server (no matplotlib):
python evaluation.py \
    --ckpt results/mmcgan/checkpoints/epoch_0400.pt \
    --data data/processed/test \
    --out  eval/mmcgan --no_plots
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

All scripts share the same env-var overrides and automatically find the latest
checkpoint in each model's `results/<model>/checkpoints/` directory.

### Sequential (single machine)

```bash
# Train all 5 models:
bash scripts/train_all.sh

# Override paths and subset of models:
DATA=/my/data OUT_ROOT=/my/results MODELS="mlp mmimreg" bash scripts/train_all.sh

# Inference on test split:
bash scripts/infer_all.sh

# Evaluation on test split (headless):
NO_PLOTS=1 bash scripts/eval_all.sh

# Common env vars (all scripts):
#   DATA          processed data directory
#   RESULTS_ROOT  where training outputs live   (default: results)
#   INFER_ROOT    where inference writes         (default: inference)
#   EVAL_ROOT     where evaluation writes        (default: eval)
#   SPLIT         which split to run on          (default: test)
#   MODELS        space-separated model list
#   CONDA_ENV     conda env name                 (default: abm)
```

### SLURM (cluster)

```bash
# Edit DATA / RESULTS_ROOT / CONDA_ENV at the top of each script, then:
sbatch scripts/train_all_slurm.sh    # array 0-4, one GPU per model
sbatch scripts/infer_all_slurm.sh
sbatch scripts/eval_all_slurm.sh

# Single model:
sbatch --array=2 scripts/train_all_slurm.sh   # cgan only
```

**Array index mapping** (all SLURM scripts): `0=mlp  1=imreg  2=cgan  3=mmimreg  4=mmcgan`

---

## Model architectures

### MLP (`models/mlp.py`)
```
Input  : (B, 2)  — normalised [diffusion_rate, cellcycle_time_mean]
FC → ReLU → Dropout  (repeated for each hidden layer)
Output : (B, 6)  — predicted simulation outputs
```
Config: `MLPConfig(n_inputs, n_outputs, hidden_dims, dropout, activation)`

### imreg (`models/imreg.py`)
```
Input  : (B, 2)  — condition
FC → reshape → (ngf×8, 4, 4)
6× ConvTranspose2d up-blocks → (3, 256, 256)  tanh
```
Config: `ImageRegressorConfig(n_inputs, image_size, ngf, lambda_l1, lambda_mse)`

### cgan (`models/cgan.py`)
```
Generator  : cat(condition, noise) → FC → ConvTranspose2d × 6 → (3, 256, 256)
Discriminator : 70×70 PatchGAN, condition broadcast spatially, spectral norm
Loss G     : LSGAN adversarial  +  λ_l1 × L1(fake, real)
Loss D     : LSGAN real/fake
```
Config: `CGANConfig(n_inputs, noise_dim, image_size, ngf, ndf, lambda_l1)`

### mmimreg (`models/mmimreg.py`)
```
Image branch   : ConvUpGenerator(condition) → (3, H, W)
Numerical branch : RegressionHead(condition) → (n_outputs,)
Loss : λ_l1 × L1  +  λ_mse × MSE (image)  +  λ_reg × MSE (numerical)
```
Config: `MultiModalImRegConfig(n_inputs, n_outputs, image_size, ngf, hidden_dims, lambda_l1, lambda_mse, lambda_reg)`

### mmcgan (`models/mmcgan.py`)
```
G  : ConvUpGenerator(cat(condition, noise)) → (3, H, W)
R  : RegressionHead(condition) → (n_outputs,)     [deterministic — no noise]
D  : PatchDiscriminator(image, condition)
Loss G : LSGAN_adv  +  λ_l1 × L1(fake, real)  +  λ_reg × MSE(ŷ, y)
Loss D : LSGAN real/fake
```
Config: `MultiModalCGANConfig(n_inputs, n_outputs, noise_dim, image_size, ngf, ndf, hidden_dims, lambda_l1, lambda_reg)`

### Shared components (`models/base_models.py`)
- `ConvUpGenerator(in_channels, image_size, ngf)` — used by mmimreg and mmcgan
- `PatchDiscriminator(n_cond, ndf)` — used by mmcgan
- `RegressionHead(in_features, n_outputs, hidden_dims)` — used by mmimreg and mmcgan

---

## Training interface (`trainer/base_trainer.py`)

`BaseTrainer` detects the model type automatically:

**Regression mode** (MLP): model exposes only `forward(x)`.
```
Single AdamW optimizer — MSE loss — logs loss + R²
```

**Generator-only mode** (imreg, mmimreg): model exposes `generator_parameters()`
and `compute_generator_loss(batch)` but NOT `discriminator_parameters()`.
```
Single Adam optimizer (betas 0.5, 0.999) — no discriminator step
```

**Full GAN mode** (cgan, mmcgan): model also exposes `discriminator_parameters()`
and `compute_discriminator_loss(batch, visuals)`.
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

Adjacent to checkpoints, `train.py` also saves:
- `<out>/input_scaler.pkl` — MinMaxScaler fitted on training inputs
- `<out>/label_scaler.pkl` — MinMaxScaler fitted on training labels (numerical models only)

---

## Key conventions

- **Normalisation**: inputs scaled to `[0, 1]` with `MinMaxScaler` fitted on the
  training split only. Scalers are persisted to disk and loaded automatically by
  `inference.py` and `evaluation.py` for real-unit output.
- **Images**: normalised to `[-1, 1]` by `ImageTransform`. Training applies random
  horizontal/vertical flips; inference/evaluation does not.
- **Data splits**: run `pipeline.py --val_split 0.15 --test_split 0.15` once. All
  subsequent runs use the same fixed split. The test set is held out until final
  evaluation.
- **Device**: auto-detected (`cuda` if available, else `cpu`).
- **Reproducibility**: pass `--seed` to `train.py` (default 42); pass `--seed` to
  `pipeline.py` for reproducible shuffling (default 42).
- **Excluded columns**: cell-cell adhesion (col 2, constant 1.0), cellcycle SD
  (col 4, constant 0.083), n_dead (col 9, always 0) are all excluded from inputs
  and outputs.

---

## Tests

```bash
pip install pytest torch torchvision pillow numpy
pytest tests/ -v
```

37 tests covering dataset loading, scaler roundtrips, and forward passes + training
interfaces for all five models. Tests use in-memory temporary data — no real files needed.

---

## Adding a new model

### Regression model (numerical output only)
1. Create `models/<name>.py` with a `<Name>Config` dataclass and `<Name>(nn.Module)`.
2. Implement `forward(x) -> Tensor` and `config_dict() -> dict`.
3. Register in `models/__init__.py` `MODELS` dict and add to `NUMERICAL_PREDICTION_MODELS`.
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
