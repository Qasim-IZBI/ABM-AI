from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import ConvUpGenerator, PatchDiscriminator, RegressionHead


@dataclass
class MultiModalCGANConfig:
    n_inputs:    int
    n_outputs:   int                                           # number of numerical outputs
    noise_dim:   int = 64
    image_size:  int = 1024
    ngf:         int = 32
    ndf:         int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    lambda_l1:   float = 10.0  # weight of image L1 loss
    lambda_reg:  float = 1.0   # weight of numerical MSE loss


class MultiModalCGAN(nn.Module):
    """
    Conditional GAN with dual outputs: a generated image and numerical predictions.

    The image generator is stochastic (samples noise at every forward pass); the
    regression head is deterministic (condition → numerical outputs, no noise).

    Input  : condition  (B, n_inputs)
    Output 1: image     (B, 3, image_size, image_size)  in [-1, 1]
    Output 2: numerical (B, n_outputs)

    Architecture
    ------------
    G (image)   : ConvUpGenerator(n_inputs + noise_dim, ...)  — noise is sampled internally
    R (numbers) : RegressionHead(n_inputs, n_outputs, hidden_dims)
    D (disc.)   : PatchDiscriminator(n_inputs, ndf)  — sees image + condition only

    The discriminator does not see the numerical output; it is trained only on the
    real/fake image distinction. The regression head is trained via MSE loss inside
    the generator step.

    Generator loss
    --------------
    loss_G = LSGAN_adv(fake_img)  +  λ_l1 · L1(fake_img, real_img)  +  λ_reg · MSE(ŷ, y)

    Discriminator loss
    ------------------
    loss_D = 0.5 · [LSGAN(real → 1)  +  LSGAN(fake → 0)]

    Training interface (consumed by BaseTrainer)
    --------------------------------------------
    generator_parameters()                   → G + R parameters
    discriminator_parameters()               → D parameters
    compute_generator_loss(batch)            → (loss_G, logs, visuals)
    compute_discriminator_loss(batch, vis)   → (loss_D, logs)
    """

    def __init__(self, cfg: MultiModalCGANConfig):
        super().__init__()
        self.cfg      = cfg
        # G takes cat(condition, noise) as a flat vector
        self.G        = ConvUpGenerator(cfg.n_inputs + cfg.noise_dim, cfg.image_size, cfg.ngf)
        self.D        = PatchDiscriminator(cfg.n_inputs, cfg.ndf)
        self.R        = RegressionHead(cfg.n_inputs, cfg.n_outputs, cfg.hidden_dims)

    # ── Inference ────────────────────────────────────────────────────────────

    def forward(self, condition: torch.Tensor,
                noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (image, numerical_predictions)."""
        return self._gen(condition, noise), self.R(condition)

    def generate(self, condition: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Image output only."""
        return self._gen(condition, noise)

    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        """Numerical output only (deterministic)."""
        return self.R(condition)

    def _gen(self, condition: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        B = condition.size(0)
        if noise is None:
            noise = torch.randn(B, self.cfg.noise_dim, device=condition.device)
        return self.G(torch.cat([condition, noise], dim=1))

    # ── Training interface ───────────────────────────────────────────────────

    def generator_parameters(self):
        return list(self.G.parameters()) + list(self.R.parameters())

    def discriminator_parameters(self):
        return self.D.parameters()

    def compute_generator_loss(self, batch: dict):
        condition = batch["inputs"]
        real_img  = batch["image"]
        real_num  = batch["labels"]

        fake_img = self._gen(condition)
        pred_num = self.R(condition)

        pred_fake = self.D(fake_img, condition)
        loss_adv  = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
        loss_l1   = F.l1_loss(fake_img, real_img)  * self.cfg.lambda_l1
        loss_reg  = F.mse_loss(pred_num, real_num)  * self.cfg.lambda_reg

        loss_G = loss_adv + loss_l1 + loss_reg
        logs   = {
            "loss_G":  loss_G.item(),
            "adv_G":   loss_adv.item(),
            "l1":      loss_l1.item(),
            "mse_reg": loss_reg.item(),
        }
        visuals = {"fake": fake_img.detach(), "real": real_img,
                   "pred_num": pred_num.detach()}
        return loss_G, logs, visuals

    def compute_discriminator_loss(self, batch: dict, visuals: dict):
        condition = batch["inputs"]
        real      = batch["image"]
        fake      = visuals["fake"]   # already detached

        pred_real = self.D(real, condition)
        pred_fake = self.D(fake, condition)
        loss_D    = 0.5 * (
            F.mse_loss(pred_real, torch.ones_like(pred_real)) +
            F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        )
        return loss_D, {"loss_D": loss_D.item()}

    def config_dict(self) -> dict:
        return {**asdict(self.cfg), "model_name": "mmcgan"}
