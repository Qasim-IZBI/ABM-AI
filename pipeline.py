#!/usr/bin/env python3
"""
ABM data pipeline — sorts numerical text files and images into
aligned inputs/labels arrays ready for deep-learning training.

Usage examples
--------------
# Minimal — images searched in the same folder as the text file:
  python pipeline.py data1.txt data2.txt

# Explicit image directories (searched in order):
  python pipeline.py data1.txt data2.txt --images /img/dir1 /img/dir2

# Load images into a numpy array (resized to 256x256):
  python pipeline.py data1.txt --images /imgs --load-images --size 256 256

# Custom output directory:
  python pipeline.py data1.txt --images /imgs --out /results
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

# ── Column layout (0-based, 15 columns per row) ────────────────────────────────
# 0:  Population name
# 1:  Default cell diffusion rate        <- INPUT
# 2:  Default cell-cell adhesion         (constant — excluded)
# 3:  Cellcycle time mean                <- INPUT
# 4:  Cellcycle time SD                  (constant — excluded)
# 5:  Population size                    <- OUTPUT
# 6:  Time (days)                        (near-constant — excluded)
# 7:  Number of proliferating cells      <- OUTPUT
# 8:  Number of quiescent cells          <- OUTPUT
# 9:  Number of dead cells               (always 0 — excluded)
# 10: Radius of gyration                 (redundant — excluded)
# 11: Diameter (radius of gyration)      (excluded)
# 12: Diameter (outer limits)            <- OUTPUT
# 13: Extension in x                     <- OUTPUT
# 14: Extension in y                     <- OUTPUT

INPUT_COLS   = [1, 3]
OUTPUT_COLS  = [5, 7, 8, 12, 13, 14]
INPUT_NAMES  = ["diffusion_rate", "cellcycle_time_mean"]
OUTPUT_NAMES = ["population_size", "n_proliferating", "n_quiescent",
                "diameter_outer_limits", "extension_x", "extension_y"]

IMAGE_SUFFIX = "_raymg000001.png"


def parse_args():
    p = argparse.ArgumentParser(
        description="Sort ABM text files and images into training arrays.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("data", nargs="+", metavar="TEXT_FILE",
                   help="One or more tab-separated data .txt files")
    p.add_argument("--images", nargs="+", metavar="DIR", default=[],
                   help="Directories to search for images (searched in order)")
    p.add_argument("--load-images", action="store_true",
                   help="Load every image into memory and save images.npy")
    p.add_argument("--size", nargs=2, type=int, default=[256, 256],
                   metavar=("W", "H"),
                   help="Resize target when --load-images is set (default: 256 256)")
    p.add_argument("--out", metavar="DIR", default=".",
                   help="Output directory (default: current directory)")
    return p.parse_args()


def load_text_file(path):
    """Parse one tab-separated data file, return list of row dicts."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            parts = raw.strip().split("\t")
            parts = [p for p in parts if p]   # drop trailing empty token
            if len(parts) < 15:
                print(f"  [SKIP] {os.path.basename(path)}:{lineno} — "
                      f"only {len(parts)} columns")
                continue
            rows.append(parts)
    return rows


def find_image(name, search_dirs):
    """Return the first existing path for <name>_raymg000001.png, or None."""
    filename = f"{name}{IMAGE_SUFFIX}"
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ── 1. Load all text files ─────────────────────────────────────────────────
    all_rows = []
    seen_names = {}   # name -> first source file (for duplicate detection)

    for txt in args.data:
        if not os.path.isfile(txt):
            sys.exit(f"ERROR: data file not found: {txt}")
        print(f"Reading {txt} ...", end=" ", flush=True)
        rows = load_text_file(txt)
        print(f"{len(rows)} rows")

        for r in rows:
            name = r[0]
            if name in seen_names:
                print(f"  [WARN] duplicate population '{name}' "
                      f"(first seen in {seen_names[name]}, now in {txt})")
            else:
                seen_names[name] = os.path.basename(txt)
            all_rows.append(r)

    if not all_rows:
        sys.exit("ERROR: no valid rows found across all input files.")

    # ── 2. Build numerical arrays ──────────────────────────────────────────────
    inputs = np.array([[float(r[c]) for c in INPUT_COLS]  for r in all_rows],
                      dtype=np.float32)
    labels = np.array([[float(r[c]) for c in OUTPUT_COLS] for r in all_rows],
                      dtype=np.float32)

    # ── 3. Resolve image paths ─────────────────────────────────────────────────
    # If no --images given, search next to each text file
    search_dirs = args.images if args.images else [os.path.dirname(os.path.abspath(t))
                                                    for t in args.data]
    search_dirs = list(dict.fromkeys(search_dirs))   # deduplicate, preserve order

    image_paths = []
    missing = []
    for r in all_rows:
        p = find_image(r[0], search_dirs)
        image_paths.append(p or "MISSING")
        if p is None:
            missing.append(r[0])

    found_count = len(all_rows) - len(missing)

    # ── 4. Print summary ───────────────────────────────────────────────────────
    print()
    print(f"{'Samples':<20}: {len(all_rows)}")
    print(f"{'Input columns':<20}: {INPUT_NAMES}")
    print(f"{'Output columns':<20}: {OUTPUT_NAMES}")
    print(f"{'Images found':<20}: {found_count} / {len(all_rows)}")
    if missing:
        print(f"{'Images missing':<20}: {len(missing)}")
        for name in missing[:5]:
            print(f"    {name}{IMAGE_SUFFIX}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    # ── 5. Save outputs ────────────────────────────────────────────────────────
    np.save(os.path.join(args.out, "inputs.npy"), inputs)
    np.save(os.path.join(args.out, "labels.npy"), labels)

    paths_file = os.path.join(args.out, "image_paths.txt")
    with open(paths_file, "w") as f:
        f.write("\n".join(image_paths))

    print()
    print(f"Saved inputs.npy        shape={inputs.shape}")
    print(f"Saved labels.npy        shape={labels.shape}")
    print(f"Saved image_paths.txt   ({len(image_paths)} entries)")

    # ── 6. Optionally load all images into numpy array ─────────────────────────
    if args.load_images:
        if missing:
            print(f"\nWARNING: {len(missing)} images missing — "
                  "cannot build images.npy. Fix missing paths first.")
        else:
            w, h = args.size
            print(f"\nLoading {len(all_rows)} images at {w}x{h} ...", flush=True)
            imgs = []
            for i, p in enumerate(image_paths):
                img = Image.open(p).convert("RGB").resize((w, h))
                imgs.append(np.array(img, dtype=np.uint8))
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(image_paths)}", flush=True)
            images = np.stack(imgs)
            out_path = os.path.join(args.out, "images.npy")
            np.save(out_path, images)
            print(f"Saved images.npy        shape={images.shape}")


if __name__ == "__main__":
    main()
