#!/usr/bin/env python3
"""
ABM training entry point.

Usage examples
--------------
# MLP on numerical inputs:
  python train.py --data /path/to/processed --model mlp --epochs 100

# cGAN — generates images from simulation parameters (requires --images in pipeline.py):
  python train.py --data /path/to/processed --model cgan --epochs 200 \\
      --lr 2e-4 --batch_size 16 --out results/cgan_run1

# Custom MLP architecture:
  python train.py --data /path/to/processed --model mlp \\
      --hidden_dims 512 512 256 --dropout 0.2 --activation gelu --out results/mlp_run2

# Resume automatically from the latest checkpoint in --out:
  python train.py --data /path/to/processed --model mlp --out results/mlp_run1
"""

import argparse
import os
import pickle

import numpy as np
from torch.utils.data import DataLoader, random_split

from datasets import ABMDataset, MinMaxScaler, ImageTransform
from models import build_model, MODELS, IMAGE_GENERATION_MODELS, NUMERICAL_PREDICTION_MODELS
from trainer import BaseTrainer
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a deep learning model on ABM simulation data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data", required=True, metavar="DIR",
                   help="Directory produced by pipeline.py "
                        "(inputs.npy, labels.npy, image_paths.txt)")
    p.add_argument("--load_images", action="store_true",
                   help="Load images per sample. Automatically enabled for cgan.")
    p.add_argument("--image_size", type=int, default=1024,
                   help="Target image size in pixels — must be a power of 2 "
                        "(default: 1024). Native images are 1000×1000; use "
                        "--pad_images to zero-pad to 1024 with no information loss.")
    p.add_argument("--pad_images", action="store_true",
                   help="Zero-pad images to image_size instead of resizing. "
                        "Preserves all pixel information (12 px padding per side "
                        "for 1000×1000 → 1024×1024).")
    p.add_argument("--val_split", type=float, default=0.15,
                   help="Fraction of data used for validation (default: 0.15)")

    # ── Model — shared ────────────────────────────────────────────────────────
    p.add_argument("--model", default="mlp", choices=list(MODELS),
                   help="Model architecture (default: mlp)")

    # ── Model — MLP-specific ──────────────────────────────────────────────────
    p.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 64],
                   help="Hidden layer sizes (MLP, default: 256 256 128)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", default="relu", choices=["relu", "gelu", "silu"])

    # ── Model — cGAN-specific ─────────────────────────────────────────────────
    p.add_argument("--noise_dim", type=int, default=128,
                   help="Latent noise dimension (cgan, default: 128)")
    p.add_argument("--ngf", type=int, default=64,
                   help="Generator base channels (cgan, default: 64)")
    p.add_argument("--ndf", type=int, default=64,
                   help="Discriminator base channels (cgan, default: 64)")
    p.add_argument("--lambda_l1", type=float, default=10.0,
                   help="L1 pixel loss weight (cgan default: 10.0 / imreg default: 1.0)")
    p.add_argument("--lambda_mse", type=float, default=1.0,
                   help="MSE pixel loss weight (imreg / mmimreg, default: 1.0)")
    p.add_argument("--lambda_reg", type=float, default=1.0,
                   help="Numerical regression MSE weight (mmcgan / mmimreg, default: 1.0)")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="AdamW weight decay (regression models only)")
    p.add_argument("--seed", type=int, default=42)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--out", default="results", metavar="DIR",
                   help="Output directory for checkpoints and logs (default: results)")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--log_every", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Image-generating models always need images — the image IS the target
    if args.model in IMAGE_GENERATION_MODELS and not args.load_images:
        print(f"Note: --load_images automatically enabled for {args.model}.")
        args.load_images = True

    # ── Dataset ──────────────────────────────────────────────────────────────
    img_tf   = ImageTransform(args.image_size, train=True,  pad=args.pad_images) if args.load_images else None
    img_tf_v = ImageTransform(args.image_size, train=False, pad=args.pad_images) if args.load_images else None

    # Detect whether pipeline.py produced pre-split subdirectories
    pre_split = os.path.isdir(os.path.join(args.data, "train"))

    if pre_split:
        train_ds = ABMDataset(os.path.join(args.data, "train"),
                              load_images=args.load_images, image_transform=img_tf)
        val_path = os.path.join(args.data, "val")
        val_ds   = ABMDataset(val_path, load_images=args.load_images,
                              image_transform=img_tf_v) if os.path.isdir(val_path) else None
        # Scalers fitted on train split only
        input_scaler = MinMaxScaler().fit(train_ds.raw_inputs())
        train_ds.input_transform = input_scaler
        if val_ds:
            val_ds.input_transform = input_scaler
        train_loader_ds = train_ds
        n_train = len(train_ds)
        n_val   = len(val_ds) if val_ds else 0
    else:
        full_ds  = ABMDataset(args.data, load_images=args.load_images, image_transform=img_tf)
        n_val    = max(1, int(len(full_ds) * args.val_split))
        n_train  = len(full_ds) - n_val
        train_split, val_split = random_split(full_ds, [n_train, n_val])
        input_scaler = MinMaxScaler().fit(full_ds.raw_inputs()[train_split.indices])
        full_ds.input_transform = input_scaler
        train_ds = train_split
        val_ds   = val_split
        train_loader_ds = full_ds

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "input_scaler.pkl"), "wb") as f:
        pickle.dump(input_scaler, f)

    if args.model in NUMERICAL_PREDICTION_MODELS:
        raw_labels = (train_ds.raw_labels() if pre_split
                      else train_loader_ds.raw_labels()[train_ds.indices])
        label_scaler = MinMaxScaler().fit(raw_labels)
        if pre_split:
            train_ds.label_transform = label_scaler
            if val_ds:
                val_ds.label_transform = label_scaler
        else:
            train_loader_ds.label_transform = label_scaler
        with open(os.path.join(args.out, "label_scaler.pkl"), "wb") as f:
            pickle.dump(label_scaler, f)

    # train_ds is always an ABMDataset in the pre-split path;
    # in the random-split path full_ds is the ABMDataset (train_ds is a Subset).
    ref_ds = train_ds if pre_split else full_ds

    print(f"Train: {n_train} samples   Val: {n_val} samples"
          + ("  [pre-split]" if pre_split else "  [random split]"))
    print(f"Inputs : {ref_ds.n_inputs}  {ref_ds.input_names}")
    if args.model in IMAGE_GENERATION_MODELS:
        print(f"Output : RGB image ({args.image_size}×{args.image_size})")
    if args.model in NUMERICAL_PREDICTION_MODELS:
        print(f"Output : numerical {ref_ds.n_outputs}  {ref_ds.output_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = (DataLoader(val_ds, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
                    if val_ds is not None else None)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        args.model,
        n_inputs=ref_ds.n_inputs,
        n_outputs=ref_ds.n_outputs,
        # MLP args (ignored by cGAN config)
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
        # cGAN args (ignored by MLP config)
        noise_dim=args.noise_dim,
        image_size=args.image_size,
        ngf=args.ngf,
        ndf=args.ndf,
        lambda_l1=args.lambda_l1,
        lambda_mse=args.lambda_mse,
        lambda_reg=args.lambda_reg,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {args.model}  ({n_params:,} parameters)")

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if (args.model == "mlp" and val_loader is not None) else None,
        lr=args.lr,
        weight_decay=args.weight_decay,
        out_dir=args.out,
    )
    trainer.train(epochs=args.epochs, save_every=args.save_every, log_every=args.log_every)


if __name__ == "__main__":
    main()
