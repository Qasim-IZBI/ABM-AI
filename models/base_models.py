"""
Shared building blocks used by the multimodal models (mmimreg, mmcgan).

cgan.py and imreg.py are kept self-contained for backwards compatibility.
New models should build from these primitives.
"""
import math
from typing import List

import torch
import torch.nn as nn


class ConvUpGenerator(nn.Module):
    """
    Deterministic convolutional image generator.

    Maps a flat vector directly to an RGB image via a learned FC projection
    followed by a stack of ConvTranspose2d up-blocks.

    Input : (B, in_channels)   — condition, or cat(condition, noise)
    Output: (B, 3, image_size, image_size)  in [-1, 1]
    """

    def __init__(self, in_channels: int, image_size: int, ngf: int):
        super().__init__()
        self._ngf = ngf
        n_up = int(math.log2(image_size // 4))

        self.fc = nn.Linear(in_channels, ngf * 8 * 4 * 4)

        ch = [ngf * 8]
        for _ in range(n_up - 1):
            ch.append(max(ngf, ch[-1] // 2))

        blocks = []
        for i in range(n_up - 1):
            blocks.append(self._up_block(ch[i], ch[i + 1]))
        blocks.append(nn.Sequential(
            nn.ConvTranspose2d(ch[-1], 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        ))
        self.up_blocks = nn.Sequential(*blocks)

    @staticmethod
    def _up_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_blocks(self.fc(x).view(x.size(0), self._ngf * 8, 4, 4))


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator with spectral norm + instance norm.

    The condition vector is broadcast spatially and concatenated to the image
    so every patch receives conditioning information.

    Input : image (B, 3, H, W) + condition (B, n_cond)
    Output: (B, 1, ~30, ~30)  — patch real/fake scores (not sigmoided)
    """

    def __init__(self, n_cond: int, ndf: int):
        super().__init__()
        in_c = 3 + n_cond

        def conv_block(ic, oc, stride, norm=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(ic, oc, kernel_size=4, stride=stride, padding=1, bias=False)
            )]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv_block(in_c,  ndf,   stride=2, norm=False),   # 256 → 128
            conv_block(ndf,   ndf*2, stride=2),                # 128 → 64
            conv_block(ndf*2, ndf*4, stride=2),                # 64  → 32
            conv_block(ndf*4, ndf*8, stride=1),                # 32  → 31
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),  # 31 → 30
        )

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        cond_map = condition.view(B, -1, 1, 1).expand(B, -1, H, W)
        return self.net(torch.cat([image, cond_map], dim=1))


class RegressionHead(nn.Module):
    """
    MLP regression head for numerical outputs.

    Input : (B, in_features)
    Output: (B, n_outputs)
    """

    def __init__(self, in_features: int, n_outputs: int, hidden_dims: List[int] = (256, 128)):
        super().__init__()
        dims = [in_features] + list(hidden_dims) + [n_outputs]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
