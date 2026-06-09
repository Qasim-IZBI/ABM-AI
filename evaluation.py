#!/usr/bin/env python3
"""
ABM evaluation — scores pre-computed inference outputs against ground truth.

Reads the outputs written by inference.py (predictions.npy, images.npy) and
compares them to the ground truth in the data directory. The model is NOT
reloaded — no checkpoint or GPU needed.

Outputs written to --out
------------------------
  metrics.json        — all scalar metrics
  predictions.csv     — per-sample predicted vs actual  (numerical outputs)
  scatter.png         — predicted vs actual scatter      (numerical outputs)
  image_grid.png      — real (top) vs generated (bottom) (image outputs)

Usage examples
--------------
  python evaluation.py \\
      --inference_dir inference/mlp \\
      --data data/processed/test \\
      --out eval/mlp

  python evaluation.py \\
      --inference_dir inference/mmcgan \\
      --data data/processed/test \\
      --out eval/mmcgan \\
      --pad_images --no_plots
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets import ABMDataset, ImageTransform

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
        description="Evaluate pre-computed inference outputs against ground truth.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--inference_dir", required=True, metavar="DIR",
                   help="Directory written by inference.py "
                        "(must contain predictions.npy and/or images.npy)")
    p.add_argument("--data", required=True, metavar="DIR",
                   help="Ground-truth data directory "
                        "(inputs.npy, labels.npy, image_paths.txt)")
    p.add_argument("--out", default="eval", metavar="DIR",
                   help="Directory for evaluation outputs (default: eval)")
    p.add_argument("--n_images", type=int, default=8,
                   help="Number of image pairs in the grid (default: 8)")
    p.add_argument("--image_size", type=int, default=1024,
                   help="Image size used at inference — needed to load real images "
                        "for comparison (default: 1024)")
    p.add_argument("--pad_images", action="store_true",
                   help="Zero-pad real images to image_size instead of resizing "
                        "(must match how inference.py was run)")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip scatter.png and image_grid.png")
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

def per_output_metrics(preds: torch.Tensor, targets: torch.Tensor,
                       names: list) -> dict:
    """MSE, MAE, R² per output column plus mean across columns."""
    out = {}
    for i, name in enumerate(names):
        p, t = preds[:, i].float(), targets[:, i].float()
        mse    = F.mse_loss(p, t).item()
        mae    = torch.mean(torch.abs(p - t)).item()
        ss_res = torch.sum((t - p) ** 2).item()
        ss_tot = torch.sum((t - t.mean()) ** 2).item()
        r2     = 1.0 - ss_res / max(ss_tot, 1e-8)
        out[name] = {"mse": round(mse, 6), "mae": round(mae, 6), "r2": round(r2, 4)}

    out["mean"] = {
        k: round(float(np.mean([v[k] for v in out.values()])), 6)
        for k in ("mse", "mae", "r2")
    }
    return out


def image_metrics(fakes: torch.Tensor, reals: torch.Tensor) -> dict:
    """PSNR, L1 and optionally SSIM. Both tensors must be in [-1, 1]."""
    mse_val = F.mse_loss(fakes, reals).item()
    psnr    = 20 * math.log10(2.0 / math.sqrt(max(mse_val, 1e-10)))
    l1      = F.l1_loss(fakes, reals).item()
    out     = {"psnr": round(psnr, 4), "l1": round(l1, 6)}

    if HAS_SKIMAGE:
        f_np = ((fakes.clamp(-1, 1) + 1) / 2).numpy().transpose(0, 2, 3, 1)
        r_np = ((reals.clamp(-1, 1) + 1) / 2).numpy().transpose(0, 2, 3, 1)
        out["ssim"] = round(float(np.mean([
            _ssim(r_np[i], f_np[i], data_range=1.0, channel_axis=-1)
            for i in range(len(f_np))
        ])), 4)

    return out


# ── Output helpers ────────────────────────────────────────────────────────────

def save_csv(preds: torch.Tensor, targets: torch.Tensor,
             names: list, out_dir: str) -> str:
    path = os.path.join(out_dir, "predictions.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"pred_{n}" for n in names] + [f"true_{n}" for n in names])
        for p, t in zip(preds.numpy(), targets.numpy()):
            writer.writerow([*p.tolist(), *t.tolist()])
    return path


def scatter_plot(preds: np.ndarray, targets: np.ndarray,
                 names: list, path: str):
    if not HAS_MPL:
        print("[INFO] matplotlib not available — skipping scatter.png")
        return
    cols = 3
    rows = math.ceil(len(names) / cols)
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
    arr = ((tensor.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return Image.fromarray(arr.transpose(1, 2, 0))


def save_image_grid(fakes: torch.Tensor, reals: torch.Tensor,
                    path: str, n: int = 8):
    n   = min(n, len(fakes))
    row = [(_to_pil(reals[i]), _to_pil(fakes[i])) for i in range(n)]
    W, H = row[0][0].size
    grid = Image.new("RGB", (n * W, 2 * H), color=(20, 20, 20))
    for col, (real, fake) in enumerate(row):
        grid.paste(real, (col * W, 0))
        grid.paste(fake, (col * W, H))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid)
        draw.text((4, 4),      "Real",      fill=(255, 255, 100))
        draw.text((4, H + 4),  "Generated", fill=(100, 255, 100))
    except Exception:
        pass
    grid.save(path)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print()
    print(f"{'='*56}")
    print(f"  Run      : {results['run']}")
    print(f"  Samples  : {results['n_samples']}")
    print(f"{'='*56}")

    if "numerical" in results:
        print("\n  Numerical outputs (real units):")
        print(f"  {'Output':<25} {'MSE':>12} {'MAE':>12} {'R²':>8}")
        print(f"  {'-'*60}")
        for name, m in results["numerical"].items():
            tag = "→ " if name == "mean" else "  "
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
    os.makedirs(args.out, exist_ok=True)

    run_name = os.path.basename(os.path.normpath(args.inference_dir))

    # ── Discover what inference produced ──────────────────────────────────────
    pred_path = os.path.join(args.inference_dir, "predictions.npy")
    img_path  = os.path.join(args.inference_dir, "images.npy")

    has_num = os.path.isfile(pred_path)
    has_img = os.path.isfile(img_path)

    if not has_num and not has_img:
        sys.exit(
            f"ERROR: neither predictions.npy nor images.npy found in "
            f"{args.inference_dir}.\nRun inference.py first."
        )

    print(f"Run          : {run_name}")
    print(f"Inference dir: {args.inference_dir}")
    print(f"Data dir     : {args.data}")
    if has_num:
        print(f"Found        : predictions.npy")
    if has_img:
        print(f"Found        : images.npy")

    results = {"run": run_name}

    # ── Load sample indices written by inference.py ───────────────────────────
    # When images are missing, inference skips those samples. sample_indices.npy
    # records which rows of the original dataset were actually processed so we
    # can select the matching rows from labels.npy and the real image dataset.
    idx_path = os.path.join(args.inference_dir, "sample_indices.npy")
    if os.path.isfile(idx_path):
        sample_indices = np.load(idx_path)
        print(f"Sample indices: {len(sample_indices)} of "
              f"{np.load(os.path.join(args.data, 'labels.npy')).shape[0]} total")
    else:
        sample_indices = None
        print("[INFO] sample_indices.npy not found — assuming all rows match")

    # ── Numerical evaluation ──────────────────────────────────────────────────
    if has_num:
        # Inference already outputs real units — compare directly to raw labels
        preds_real = torch.tensor(np.load(pred_path), dtype=torch.float32)
        all_labels = np.load(os.path.join(args.data, "labels.npy"))

        if sample_indices is not None:
            labels_raw = torch.tensor(all_labels[sample_indices], dtype=torch.float32)
        else:
            labels_raw = torch.tensor(all_labels, dtype=torch.float32)

        if preds_real.shape[0] != labels_raw.shape[0]:
            sys.exit(
                f"ERROR: predictions.npy has {preds_real.shape[0]} rows but "
                f"selected labels have {labels_raw.shape[0]} rows.\n"
                f"Re-run inference.py to regenerate sample_indices.npy."
            )

        # Output names from the dataset (no images needed)
        ds_meta = ABMDataset(args.data, load_images=False)
        names   = ds_meta.output_names
        n_samples = preds_real.shape[0]

        results["n_samples"] = n_samples
        results["numerical"] = per_output_metrics(preds_real, labels_raw, names)

        csv_path = save_csv(preds_real, labels_raw, names, args.out)
        print(f"Saved   : {csv_path}")

        if not args.no_plots:
            scat_path = os.path.join(args.out, "scatter.png")
            scatter_plot(preds_real.numpy(), labels_raw.numpy(), names, scat_path)
            if HAS_MPL:
                print(f"Saved   : {scat_path}")

    # ── Image evaluation ──────────────────────────────────────────────────────
    if has_img:
        # Generated images saved by inference.py as uint8 (0–255), shape (N,3,H,W)
        fakes_u8 = np.load(img_path)
        fakes    = torch.tensor(fakes_u8, dtype=torch.float32) / 127.5 - 1.0

        # Load real images — use sample_indices to load only the rows that
        # inference actually processed (skipping any with missing images).
        img_tf = ImageTransform(args.image_size, train=False, pad=args.pad_images)
        ds_img = ABMDataset(args.data, load_images=True, image_transform=img_tf)

        if sample_indices is not None:
            # Use sample_indices to load only the rows inference actually processed.
            idx_in_ds = [ds_img._indices.index(i) for i in sample_indices
                         if i in ds_img._indices]
            reals = torch.stack([ds_img[j]["image"] for j in idx_in_ds])
            fakes = fakes[:len(reals)]
        else:
            n = min(len(ds_img), fakes.shape[0])
            if len(ds_img) != fakes.shape[0]:
                print(f"[WARN] images.npy has {fakes.shape[0]} frames but dataset has "
                      f"{len(ds_img)} — truncating to {n}")
            reals = torch.stack([ds_img[i]["image"] for i in range(n)])
            fakes = fakes[:n]

        results.setdefault("n_samples", fakes.shape[0])
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
