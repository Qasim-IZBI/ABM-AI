import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CGANConfig:
    n_inputs:   int               # number of conditioning input features
    n_outputs:  int = 0           # unused; kept for build_model interface parity
    noise_dim:  int = 64          # latent noise vector dimension
    image_size: int = 256         # output image H = W (must be a power of 2, ≥ 64)
    ngf:        int = 32          # base channel count in generator
    ndf:        int = 32          # base channel count in discriminator
    lambda_l1:  float = 10.0     # weight of L1 pixel loss relative to adversarial loss


# ── Generator ────────────────────────────────────────────────────────────────


class Generator(nn.Module):
    """
    Condition + noise → RGB image.

    Input : (B, n_inputs + noise_dim)  — concatenated condition and noise
    Output: (B, 3, image_size, image_size)  — tanh-normalised to [-1, 1]

    Architecture: FC → (ngf×8, 4, 4) → 6× ConvTranspose2d up-blocks.
    For image_size=256 the spatial path is 4→8→16→32→64→128→256.
    """

    def __init__(self, cfg: CGANConfig):
        super().__init__()
        self.cfg = cfg
        n_up = int(math.log2(cfg.image_size // 4))  # e.g. 6 for 256

        self.fc = nn.Linear(cfg.n_inputs + cfg.noise_dim, cfg.ngf * 8 * 4 * 4)

        # Channel schedule: ngf×8 halving down to ngf
        ch = [cfg.ngf * 8]
        for _ in range(n_up - 1):
            ch.append(max(cfg.ngf, ch[-1] // 2))

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

    def forward(self, condition: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        B = condition.size(0)
        if noise is None:
            noise = torch.randn(B, self.cfg.noise_dim, device=condition.device)
        z = torch.cat([condition, noise], dim=1)
        x = self.fc(z).view(B, self.cfg.ngf * 8, 4, 4)
        return self.up_blocks(x)


# ── Discriminator ─────────────────────────────────────────────────────────────


class Discriminator(nn.Module):
    """
    70×70 PatchGAN discriminator conditioned on numerical inputs.

    The condition vector is broadcast spatially and concatenated to the image
    as extra channels so every patch sees the conditioning information.

    Input : image (B, 3, H, W) + condition (B, n_inputs)
    Output: (B, 1, ~30, ~30)  patch-level real/fake scores (not sigmoided)
    """

    def __init__(self, cfg: CGANConfig):
        super().__init__()
        in_c = 3 + cfg.n_inputs
        ndf  = cfg.ndf

        def conv_block(ic, oc, stride, norm=True):
            layers = [nn.utils.spectral_norm(
                nn.Conv2d(ic, oc, kernel_size=4, stride=stride, padding=1, bias=False)
            )]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv_block(in_c,  ndf,   stride=2, norm=False),  # 256 → 128
            conv_block(ndf,   ndf*2, stride=2),               # 128 → 64
            conv_block(ndf*2, ndf*4, stride=2),               # 64  → 32
            conv_block(ndf*4, ndf*8, stride=1),               # 32  → 31
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),  # 31 → 30
        )

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        B, _, H, W = image.shape
        cond_map = condition.view(B, -1, 1, 1).expand(B, -1, H, W)
        return self.net(torch.cat([image, cond_map], dim=1))


# ── CGAN ─────────────────────────────────────────────────────────────────────


class CGAN(nn.Module):
    """
    Conditional GAN that generates cell population images conditioned on
    numerical ABM simulation parameters (diffusion rate, cellcycle time).

    Losses
    ------
    Generator     : LSGAN adversarial  +  λ_L1 × pixel L1 (vs. real image)
    Discriminator : LSGAN (real → 1, fake → 0), averaged

    Training interface (consumed by BaseTrainer)
    --------------------------------------------
    generator_parameters()                  → Iterable[Parameter]
    discriminator_parameters()              → Iterable[Parameter]
    compute_generator_loss(batch)           → (loss_G, logs, visuals)
    compute_discriminator_loss(batch, vis)  → (loss_D, logs)

    Inference
    ---------
    model.generate(condition)              → image tensor in [-1, 1]
    """

    def __init__(self, cfg: CGANConfig):
        super().__init__()
        self.cfg = cfg
        self.G = Generator(cfg)
        self.D = Discriminator(cfg)

    # ── Training interface ───────────────────────────────────────────────────

    def generator_parameters(self):
        return self.G.parameters()

    def discriminator_parameters(self):
        return self.D.parameters()

    def compute_generator_loss(self, batch: dict):
        condition = batch["inputs"]
        real      = batch["image"]

        fake = self.G(condition)

        pred_fake = self.D(fake, condition)
        loss_adv  = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
        loss_l1   = F.l1_loss(fake, real) * self.cfg.lambda_l1

        loss_G = loss_adv + loss_l1
        logs   = {"loss_G": loss_G.item(), "adv_G": loss_adv.item(), "l1": loss_l1.item()}
        visuals = {"fake": fake.detach(), "real": real}
        return loss_G, logs, visuals

    def compute_discriminator_loss(self, batch: dict, visuals: dict):
        condition = batch["inputs"]
        real      = batch["image"]
        fake      = visuals["fake"]   # already detached — no G gradient here

        pred_real = self.D(real, condition)
        pred_fake = self.D(fake, condition)

        loss_D = 0.5 * (
            F.mse_loss(pred_real, torch.ones_like(pred_real)) +
            F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        )
        return loss_D, {"loss_D": loss_D.item()}

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(self, condition: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Generate an image from a condition vector. Output is in [-1, 1]."""
        return self.G(condition, noise)

    def config_dict(self) -> dict:
        return {**asdict(self.cfg), "model_name": "cgan"}
