#!/usr/bin/env python3
"""
ABM inference entry point — runs a trained model on new data (no ground truth needed).

Handles all five model types automatically:
  mlp     → saves predictions.npy  (numerical, real units)
  imreg   → saves images/          (generated PNGs)
  cgan    → saves images/          (generated PNGs, sampled with random noise)
  mmimreg → saves predictions.npy + images/
  mmcgan  → saves predictions.npy + images/

Usage examples
--------------
  python inference.py --ckpt results/mlp/checkpoints/epoch_0200.pt \\
                      --data data/processed/test --out inference/mlp

  python inference.py --ckpt results/cgan/checkpoints/epoch_0300.pt \\
                      --data data/processed/test --out inference/cgan

  python inference.py --ckpt results/mmcgan/checkpoints/epoch_0400.pt \\
                      --data data/processed/test --out inference/mmcgan
"""

import argparse
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from datasets import ABMDataset, MinMaxScaler, ImageTransform
from models import build_model, IMAGE_GENERATION_MODELS, NUMERICAL_PREDICTION_MODELS
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
                   help="Data directory (inputs.npy, labels.npy, image_paths.txt)")
    p.add_argument("--out", default="inference", metavar="DIR",
                   help="Directory to write outputs (default: inference)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--image_size", type=int, default=512,
                   help="Image resize, must match training (default: 512)")
    p.add_argument("--save_images", action="store_true",
                   help="Save each generated image as an individual PNG")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str):
    raw  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = dict(raw.get("config", {}))
    name = cfg.pop("model_name", "mlp")
    n_in = cfg.pop("n_inputs")
    n_out= cfg.pop("n_outputs", 0)
    model = build_model(name, n_inputs=n_in, n_outputs=n_out, **cfg)
    model.load_state_dict(raw["model"])
    return model, name


def load_scalers(ckpt_path: str):
    root = os.path.normpath(os.path.join(os.path.dirname(ckpt_path), ".."))

    def _load(fname):
        path = os.path.join(root, fname)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    return _load("input_scaler.pkl"), _load("label_scaler.pkl")


def save_png(tensor: torch.Tensor, path: str):
    """Save (3, H, W) tensor in [-1, 1] as PNG."""
    arr = ((tensor.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    Image.fromarray(arr.transpose(1, 2, 0)).save(path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, model_name = load_model(args.ckpt)
    model.to(device).eval()
    is_img = model_name in IMAGE_GENERATION_MODELS
    is_num = model_name in NUMERICAL_PREDICTION_MODELS
    print(f"Model   : {model_name}")
    print(f"Checkpoint: {args.ckpt}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    input_scaler, label_scaler = load_scalers(args.ckpt)
    if input_scaler is None:
        print("[WARN] input_scaler.pkl not found — fitting on inference data.")

    # ── Dataset ───────────────────────────────────────────────────────────────
    img_tf = ImageTransform(args.image_size, train=False) if is_img else None
    ds = ABMDataset(args.data, load_images=is_img, image_transform=img_tf)

    if input_scaler is None:
        input_scaler = MinMaxScaler().fit(ds.raw_inputs())
    ds.input_transform = input_scaler
    # Note: we do NOT apply label_transform here — inference data may have no labels

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"Samples : {len(ds)}")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_num_preds = []
    all_images    = []

    img_dir = os.path.join(args.out, "images")
    if is_img and args.save_images:
        os.makedirs(img_dir, exist_ok=True)

    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)

            if model_name == "mlp":
                all_num_preds.append(model(x).cpu())

            elif model_name in ("imreg", "cgan"):
                fake = model.generate(x).cpu()
                all_images.append(fake)

            elif model_name in ("mmimreg", "mmcgan"):
                fake, pred_num = model(x)
                all_images.append(fake.cpu())
                all_num_preds.append(pred_num.cpu())

            # Optionally write individual PNGs
            if is_img and args.save_images:
                for img_tensor in all_images[-1]:
                    save_png(img_tensor, os.path.join(img_dir, f"{sample_idx:05d}.png"))
                    sample_idx += 1

    # ── Save numerical predictions (denormalised to real units) ───────────────
    if all_num_preds:
        preds_norm = torch.cat(all_num_preds)
        if label_scaler is not None:
            preds_real = label_scaler.inverse_transform(preds_norm)
        else:
            preds_real = preds_norm
            print("[WARN] label_scaler.pkl not found — saving normalised predictions.")

        pred_path = os.path.join(args.out, "predictions.npy")
        np.save(pred_path, preds_real.numpy())
        print(f"Saved   : {pred_path}  shape={preds_real.shape}")

        # Human-readable per-output summary
        print(f"\nPrediction ranges (real units):")
        for i, name in enumerate(ds.output_names):
            col = preds_real[:, i]
            print(f"  {name:<25}  min={col.min():.2f}  max={col.max():.2f}  "
                  f"mean={col.mean():.2f}")

    # ── Save images ───────────────────────────────────────────────────────────
    if all_images:
        images = torch.cat(all_images)
        img_npy = os.path.join(args.out, "images.npy")
        np.save(img_npy, ((images.clamp(-1, 1) + 1) / 2 * 255).byte().numpy())
        print(f"Saved   : {img_npy}  shape={images.shape}")
        if args.save_images:
            print(f"Saved   : {img_dir}/  ({len(images)} PNGs)")


if __name__ == "__main__":
    main()
