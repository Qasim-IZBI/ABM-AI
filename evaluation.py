#!/usr/bin/env python3
"""
ABM evaluation script — computes metrics and saves visualisations for any trained model.

Metrics
-------
  Numerical outputs   : MSE, MAE, R² — per output column and averaged
                        reported in both normalised [0, 1] and original units
  Image outputs       : PSNR, L1 pixel error, SSIM (if scikit-image installed)

Outputs written to --out
------------------------
  metrics.json        — all scalar metrics
  predictions.csv     — per-sample predicted vs actual  (numerical models)
  scatter.png         — predicted vs actual scatter      (numerical models)
  image_grid.png      — real (top) vs generated (bottom) (image models)

Usage examples
--------------
  python evaluation.py --ckpt results/mlp/checkpoints/epoch_0200.pt \\
                       --data /path/to/processed --out eval/mlp

  python evaluation.py --ckpt results/mmcgan/checkpoints/epoch_0400.pt \\
                       --data /path/to/processed --out eval/mmcgan

  # Headless / no matplotlib:
  python evaluation.py --ckpt results/cgan/checkpoints/epoch_0300.pt \\
                       --data /path/to/processed --out eval/cgan --no_plots
"""

import argparse
import csv
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from datasets import ABMDataset, MinMaxScaler, ImageTransform
from models import build_model, IMAGE_GENERATION_MODELS, NUMERICAL_PREDICTION_MODELS
from utils import load_checkpoint

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained ABM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ckpt", required=True, metavar="PATH",
                   help="Path to checkpoint .pt file")
    p.add_argument("--data", required=True, metavar="DIR",
                   help="Processed data directory (inputs.npy, labels.npy, image_paths.txt)")
    p.add_argument("--out", default="eval", metavar="DIR",
                   help="Directory for evaluation outputs (default: eval)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_images", type=int, default=8,
                   help="Number of image pairs to show in the grid (default: 8)")
    p.add_argument("--image_size", type=int, default=1024,
                   help="Image size, must match training (default: 1024)")
    p.add_argument("--pad_images", action="store_true",
                   help="Zero-pad images to image_size instead of resizing (must match training)")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip all plot/image outputs (useful on headless servers)")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str):
    """Reconstruct model from checkpoint and return (model, model_name)."""
    raw  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = dict(raw.get("config", {}))
    name = cfg.pop("model_name", "mlp")
    n_in = cfg.pop("n_inputs")
    n_out= cfg.pop("n_outputs", 0)
    model = build_model(name, n_inputs=n_in, n_outputs=n_out, **cfg)
    model.load_state_dict(raw["model"])
    return model, name


# ── Scaler loading ────────────────────────────────────────────────────────────

def load_scalers(ckpt_path: str):
    """
    Look for scalers saved by train.py one directory above the checkpoints folder.
    E.g. results/mlp/checkpoints/epoch_0200.pt → looks in results/mlp/
    Returns (input_scaler, label_scaler); either may be None if not found.
    """
    root = os.path.normpath(os.path.join(os.path.dirname(ckpt_path), ".."))

    def _load(fname):
        path = os.path.join(root, fname)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    return _load("input_scaler.pkl"), _load("label_scaler.pkl")


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_output_metrics(preds: torch.Tensor, targets: torch.Tensor,
                       names: list) -> dict:
    """MSE, MAE, R² per output column plus mean across columns."""
    out = {}
    for i, name in enumerate(names):
        p, t = preds[:, i].float(), targets[:, i].float()
        mse  = F.mse_loss(p, t).item()
        mae  = torch.mean(torch.abs(p - t)).item()
        ss_res = torch.sum((t - p) ** 2).item()
        ss_tot = torch.sum((t - t.mean()) ** 2).item()
        r2   = 1.0 - ss_res / max(ss_tot, 1e-8)
        out[name] = {"mse": round(mse, 6), "mae": round(mae, 6), "r2": round(r2, 4)}

    out["mean"] = {
        k: round(float(np.mean([v[k] for v in out.values()])), 6)
        for k in ("mse", "mae", "r2")
    }
    return out


def image_metrics(fakes: torch.Tensor, reals: torch.Tensor) -> dict:
    """PSNR, L1 and optionally SSIM for (B, 3, H, W) tensors in [-1, 1]."""
    mse_val = F.mse_loss(fakes, reals).item()
    psnr    = 20 * math.log10(2.0 / math.sqrt(max(mse_val, 1e-10)))
    l1      = F.l1_loss(fakes, reals).item()
    out     = {"psnr": round(psnr, 4), "l1": round(l1, 6)}

    if HAS_SKIMAGE:
        # Convert to [0, 1] numpy for skimage
        f_np = ((fakes.clamp(-1, 1) + 1) / 2).numpy().transpose(0, 2, 3, 1)  # (B,H,W,3)
        r_np = ((reals.clamp(-1, 1) + 1) / 2).numpy().transpose(0, 2, 3, 1)
        ssim_scores = [
            _ssim(r_np[i], f_np[i], data_range=1.0, channel_axis=-1)
            for i in range(len(f_np))
        ]
        out["ssim"] = round(float(np.mean(ssim_scores)), 4)

    return out


# ── Output helpers ────────────────────────────────────────────────────────────

def save_csv(preds: torch.Tensor, targets: torch.Tensor,
             names: list, out_dir: str):
    path = os.path.join(out_dir, "predictions.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [f"pred_{n}" for n in names] + [f"true_{n}" for n in names]
        )
        for p, t in zip(preds.numpy(), targets.numpy()):
            writer.writerow([*p.tolist(), *t.tolist()])
    return path


def scatter_plot(preds: np.ndarray, targets: np.ndarray,
                 names: list, path: str):
    if not HAS_MPL:
        print("[INFO] matplotlib not available — skipping scatter.png")
        return
    n = len(names)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()

    for i, name in enumerate(names):
        ax = axes[i]
        ax.scatter(targets[:, i], preds[:, i], alpha=0.25, s=8, color="steelblue")
        lo = min(targets[:, i].min(), preds[:, i].min())
        hi = max(targets[:, i].max(), preds[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="ideal")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Predicted vs Actual (real units)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    """(3, H, W) tensor in [-1, 1] → PIL RGB image."""
    arr = ((tensor.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return Image.fromarray(arr.transpose(1, 2, 0))


def save_image_grid(fakes: torch.Tensor, reals: torch.Tensor,
                    path: str, n: int = 8):
    """Save a grid with real images on top and generated images on the bottom."""
    n   = min(n, len(fakes))
    row = [(_to_pil(reals[i]), _to_pil(fakes[i])) for i in range(n)]
    W, H = row[0][0].size
    grid = Image.new("RGB", (n * W, 2 * H), color=(20, 20, 20))
    for col, (real, fake) in enumerate(row):
        grid.paste(real, (col * W, 0))
        grid.paste(fake, (col * W, H))

    # Add labels using PIL draw if available
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid)
        draw.text((4, 4), "Real",      fill=(255, 255, 100))
        draw.text((4, H + 4), "Generated", fill=(100, 255, 100))
    except Exception:
        pass

    grid.save(path)


# ── Summary printing ──────────────────────────────────────────────────────────

def print_summary(results: dict):
    print()
    print(f"{'='*56}")
    print(f"  Model    : {results['model']}")
    print(f"  Samples  : {results['n_samples']}")
    print(f"{'='*56}")

    if "numerical_raw_units" in results:
        print("\n  Numerical outputs (real units):")
        print(f"  {'Output':<25} {'MSE':>12} {'MAE':>12} {'R²':>8}")
        print(f"  {'-'*60}")
        for name, m in results["numerical_raw_units"].items():
            tag = "  " if name != "mean" else "→ "
            print(f"  {tag}{name:<23} {m['mse']:>12.2f} {m['mae']:>12.2f} {m['r2']:>8.4f}")

    if "image" in results:
        im = results["image"]
        print(f"\n  Image quality:")
        print(f"    PSNR : {im['psnr']:.2f} dB")
        print(f"    L1   : {im['l1']:.4f}")
        if "ssim" in im:
            print(f"    SSIM : {im['ssim']:.4f}")

    print(f"{'='*56}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model, model_name = load_model(args.ckpt)
    model.to(device).eval()
    is_img = model_name in IMAGE_GENERATION_MODELS
    is_num = model_name in NUMERICAL_PREDICTION_MODELS
    print(f"Loaded  : {model_name}  from {args.ckpt}")

    # ── Scalers ───────────────────────────────────────────────────────────────
    input_scaler, label_scaler = load_scalers(args.ckpt)

    if input_scaler is None:
        print("[WARN] input_scaler.pkl not found — fitting on eval data (approximate).")

    if is_num and label_scaler is None:
        print("[WARN] label_scaler.pkl not found — fitting on eval data (approximate).")

    # ── Dataset ───────────────────────────────────────────────────────────────
    img_tf = ImageTransform(args.image_size, train=False, pad=args.pad_images) if is_img else None
    ds = ABMDataset(args.data, load_images=is_img, image_transform=img_tf)

    if input_scaler is None:
        input_scaler = MinMaxScaler().fit(ds.raw_inputs())
    ds.input_transform = input_scaler

    if is_num:
        if label_scaler is None:
            label_scaler = MinMaxScaler().fit(ds.raw_labels())
        ds.label_transform = label_scaler

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f"Dataset : {len(ds)} samples")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds_num, all_tgts_num = [], []
    all_fakes,     all_reals    = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device)

            if model_name == "mlp":
                all_preds_num.append(model(x).cpu())
                all_tgts_num.append(batch["labels"])

            elif model_name in ("imreg", "cgan"):
                all_fakes.append(model.generate(x).cpu())
                all_reals.append(batch["image"])

            elif model_name in ("mmimreg", "mmcgan"):
                fake, pred_num = model(x)
                all_fakes.append(fake.cpu())
                all_reals.append(batch["image"])
                all_preds_num.append(pred_num.cpu())
                all_tgts_num.append(batch["labels"])

    # ── Compute metrics ────────────────────────────────────────────────────────
    results = {"model": model_name, "n_samples": len(ds)}

    if all_preds_num:
        preds_norm = torch.cat(all_preds_num)
        tgts_norm  = torch.cat(all_tgts_num)

        # Normalised-space metrics
        results["numerical_normalised"] = per_output_metrics(
            preds_norm, tgts_norm, ds.output_names
        )

        # Real-unit metrics (inverse transform both sides)
        preds_raw = label_scaler.inverse_transform(preds_norm)
        tgts_raw  = label_scaler.inverse_transform(tgts_norm)
        results["numerical_raw_units"] = per_output_metrics(
            preds_raw, tgts_raw, ds.output_names
        )

        csv_path = save_csv(preds_raw, tgts_raw, ds.output_names, args.out)
        print(f"Saved   : {csv_path}")

        if not args.no_plots:
            scat_path = os.path.join(args.out, "scatter.png")
            scatter_plot(preds_raw.numpy(), tgts_raw.numpy(),
                         ds.output_names, scat_path)
            if HAS_MPL:
                print(f"Saved   : {scat_path}")

    if all_fakes:
        fakes = torch.cat(all_fakes)
        reals = torch.cat(all_reals)
        results["image"] = image_metrics(fakes, reals)

        if not args.no_plots:
            grid_path = os.path.join(args.out, "image_grid.png")
            save_image_grid(fakes, reals, grid_path, n=args.n_images)
            print(f"Saved   : {grid_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(args.out, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved   : {json_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
