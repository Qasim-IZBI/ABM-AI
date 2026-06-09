import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class MinMaxScaler:
    """Fit on training data, apply to any split. Scales to [0, 1]."""

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, x: np.ndarray):
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
        return self

    def transform(self, t: torch.Tensor) -> torch.Tensor:
        mn = torch.tensor(self.min, dtype=t.dtype)
        mx = torch.tensor(self.max, dtype=t.dtype)
        denom = (mx - mn).clamp(min=1e-8)
        return (t - mn) / denom

    def inverse_transform(self, t: torch.Tensor) -> torch.Tensor:
        mn = torch.tensor(self.min, dtype=t.dtype)
        mx = torch.tensor(self.max, dtype=t.dtype)
        return t * (mx - mn) + mn

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.transform(t)


class StandardScaler:
    """Fit on training data, apply to any split. Zero mean, unit variance."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        return self

    def transform(self, t: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor(self.mean, dtype=t.dtype)
        sigma = torch.tensor(self.std, dtype=t.dtype).clamp(min=1e-8)
        return (t - mu) / sigma

    def inverse_transform(self, t: torch.Tensor) -> torch.Tensor:
        mu = torch.tensor(self.mean, dtype=t.dtype)
        sigma = torch.tensor(self.std, dtype=t.dtype)
        return t * sigma + mu

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return self.transform(t)


class _PadToSquare:
    """
    Symmetrically zero-pad a PIL image to (size × size).

    If either dimension exceeds size the image is proportionally scaled down
    first so padding can always bring it to exactly size × size.
    Native ABM images are 1000×1000; padding to 1024 adds 12 px on each
    side with no information loss.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Scale down if the source is larger than the target (rare edge case)
        if w > self.size or h > self.size:
            scale = self.size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
            w, h = img.size

        pad_w = self.size - w
        pad_h = self.size - h
        left   = pad_w // 2
        right  = pad_w - left
        top    = pad_h // 2
        bottom = pad_h - top

        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(img, (left, top))
        return canvas


def ImageTransform(image_size: int = 1024, train: bool = True, pad: bool = False):
    """
    Build a torchvision transform pipeline for RGB images.

    Args:
        image_size: Target spatial size (H = W). Must be a power of 2.
        train:      Apply random horizontal/vertical flips when True.
        pad:        If True, zero-pad the image to image_size × image_size
                    instead of resizing. Preserves all pixel information.
                    Use this for 1000×1000 → 1024×1024 (adds 12 px per side).
                    If False (default), the image is bilinearly resized.
    """
    if pad:
        ops = [_PadToSquare(image_size)]
    else:
        ops = [T.Resize((image_size, image_size))]

    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]

    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # → [-1, 1]
    ]
    return T.Compose(ops)
