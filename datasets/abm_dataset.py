import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# Names matching the columns extracted by pipeline.py
INPUT_NAMES  = ["diffusion_rate", "cellcycle_time_mean"]
OUTPUT_NAMES = ["population_size", "n_proliferating", "n_quiescent",
                "diameter_outer_limits", "extension_x", "extension_y"]


class ABMDataset(Dataset):
    """
    Loads processed ABM data from a directory produced by pipeline.py:

        inputs.npy       — shape (N, 2)   numerical input parameters
        labels.npy       — shape (N, 6)   simulation output targets
        image_paths.txt  — N image paths  (one per line; "MISSING" if absent)

    Args:
        data_dir:         path to directory containing the three files above
        load_images:      if True, load the .png image for each sample
        image_transform:  callable applied to PIL.Image before returning
        input_transform:  callable applied to the input tensor
        label_transform:  callable applied to the label tensor
    """

    def __init__(
        self,
        data_dir: str,
        load_images: bool = True,
        image_transform=None,
        input_transform=None,
        label_transform=None,
    ):
        self.load_images = load_images
        self.image_transform = image_transform
        self.input_transform = input_transform
        self.label_transform = label_transform

        inputs_path = os.path.join(data_dir, "inputs.npy")
        labels_path = os.path.join(data_dir, "labels.npy")
        paths_file  = os.path.join(data_dir, "image_paths.txt")

        for p in (inputs_path, labels_path, paths_file):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Expected file not found: {p}")

        self._inputs = np.load(inputs_path)
        self._labels = np.load(labels_path)

        with open(paths_file) as f:
            self._image_paths = [line.strip() for line in f]

        n = len(self._inputs)
        assert len(self._labels) == n and len(self._image_paths) == n, \
            "inputs.npy, labels.npy and image_paths.txt must have the same length"

        if load_images:
            valid = [
                i for i, p in enumerate(self._image_paths)
                if p != "MISSING" and os.path.isfile(p)
            ]
            n_missing = n - len(valid)
            if n_missing:
                print(f"[ABMDataset] {n_missing}/{n} images missing — those samples skipped")
                # Show the first missing path so the user knows where to look
                first_missing = next(
                    p for p in self._image_paths if p == "MISSING" or not os.path.isfile(p)
                )
                print(f"[ABMDataset] First missing path: {first_missing}")
            if len(valid) == 0:
                raise RuntimeError(
                    f"No images found in {data_dir}.\n"
                    f"  All {n} paths in image_paths.txt are missing or unresolvable.\n"
                    f"  First entry: {self._image_paths[0]}\n"
                    f"  Re-run pipeline.py with --images pointing to the correct directory, "
                    f"or train without images (numerical-only models do not need them)."
                )
            self._indices = valid
        else:
            self._indices = list(range(n))

    # ── properties ──────────────────────────────────────────────────────────────

    @property
    def n_inputs(self) -> int:
        return self._inputs.shape[1]

    @property
    def n_outputs(self) -> int:
        return self._labels.shape[1]

    @property
    def input_names(self):
        return INPUT_NAMES

    @property
    def output_names(self):
        return OUTPUT_NAMES

    # ── raw numpy arrays (for fitting scalers on training split) ────────────────

    def raw_inputs(self) -> np.ndarray:
        return self._inputs[self._indices]

    def raw_labels(self) -> np.ndarray:
        return self._labels[self._indices]

    # ── Dataset interface ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        i = self._indices[idx]

        x = torch.tensor(self._inputs[i], dtype=torch.float32)
        y = torch.tensor(self._labels[i], dtype=torch.float32)

        if self.input_transform is not None:
            x = self.input_transform(x)
        if self.label_transform is not None:
            y = self.label_transform(y)

        sample = {"inputs": x, "labels": y}

        if self.load_images:
            img = Image.open(self._image_paths[i]).convert("RGB")
            if self.image_transform is not None:
                img = self.image_transform(img)
            else:
                img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            sample["image"] = img

        return sample
