#!/usr/bin/env python3
"""
ABM inference entry point — runs a trained model on new processed data.

Usage examples
--------------
# Predict from a checkpoint and save results:
  python inference.py --ckpt results/checkpoints/epoch_0100.pt \\
                      --data /path/to/processed --out predictions/

# With images (must match training setup):
  python inference.py --ckpt results/checkpoints/epoch_0100.pt \\
                      --data /path/to/processed --load_images --out predictions/
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import ABMDataset, MinMaxScaler, ImageTransform
from models import build_model
from utils import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a trained ABM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", required=True, metavar="PATH",
                   help="Path to a .pt checkpoint file")
    p.add_argument("--data", required=True, metavar="DIR",
                   help="Directory with inputs.npy, labels.npy, image_paths.txt")
    p.add_argument("--load_images", action="store_true")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out", default="predictions", metavar="DIR",
                   help="Directory to write predictions.npy (default: predictions)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────────
    raw = torch.load(args.ckpt, map_location="cpu")
    cfg = raw.get("config", {})

    model = build_model(
        name=cfg.get("model_name", "mlp"),
        n_inputs=cfg["n_inputs"],
        n_outputs=cfg["n_outputs"],
        **{k: cfg[k] for k in ("hidden_dims", "dropout", "activation") if k in cfg},
    )
    load_checkpoint(args.ckpt, model)
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ── Dataset ──────────────────────────────────────────────────────────────────
    img_tf = ImageTransform(args.image_size, train=False) if args.load_images else None
    ds = ABMDataset(args.data, load_images=args.load_images, image_transform=img_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Running inference on {len(ds)} samples ...")

    # ── Predict ──────────────────────────────────────────────────────────────────
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch["labels"].numpy())

    predictions = np.concatenate(all_preds,   axis=0)
    targets      = np.concatenate(all_targets, axis=0)

    # ── Save ─────────────────────────────────────────────────────────────────────
    pred_path = os.path.join(args.out, "predictions.npy")
    np.save(pred_path, predictions)
    print(f"Saved predictions.npy  shape={predictions.shape}")

    # Per-output MAE summary
    mae = np.mean(np.abs(predictions - targets), axis=0)
    print("\nPer-output MAE (in normalised units):")
    for name, err in zip(ds.output_names, mae):
        print(f"  {name:<25}: {err:.4f}")


if __name__ == "__main__":
    main()
