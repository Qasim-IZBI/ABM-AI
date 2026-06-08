import numpy as np
import torch
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


def ImageTransform(image_size: int = 256, train: bool = True):
    """Returns a torchvision transform pipeline for RGB images."""
    ops = [T.Resize((image_size, image_size))]
    if train:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    ops += [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ]
    return T.Compose(ops)
