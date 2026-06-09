from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_models import ConvUpGenerator, RegressionHead


@dataclass
class MultiModalImRegConfig:
    n_inputs:    int
    n_outputs:   int                                           # number of numerical outputs
    image_size:  int = 1024
    ngf:         int = 32
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    lambda_l1:   float = 1.0   # weight of image L1 loss
    lambda_mse:  float = 1.0   # weight of image MSE loss
    lambda_reg:  float = 1.0   # weight of numerical MSE loss


class MultiModalImReg(nn.Module):
    """
    Deterministic multimodal generator.

    Both outputs are produced from the same condition vector with no stochasticity,
    making inference fully reproducible.

    Input  : condition  (B, n_inputs)
    Output 1: image     (B, 3, image_size, image_size)  in [-1, 1]
    Output 2: numerical (B, n_outputs)

    Architecture
    ------------
    Image branch  : ConvUpGenerator  — FC → (ngf×8, 4, 4) → ConvTranspose2d × n_up
    Numerical branch : RegressionHead — FC → hidden → output

    Both branches are trained simultaneously with a combined pixel + regression loss.

    Loss
    ----
    loss = λ_l1 · L1(fake, real)  +  λ_mse · MSE(fake, real)  +  λ_reg · MSE(ŷ, y)

    Training interface (generator-only — no discriminator)
    -------------------------------------------------------
    generator_parameters()        → all parameters
    compute_generator_loss(batch) → (loss, logs, visuals)
    """

    def __init__(self, cfg: MultiModalImRegConfig):
        super().__init__()
        self.cfg       = cfg
        self.image_gen = ConvUpGenerator(cfg.n_inputs, cfg.image_size, cfg.ngf)
        self.reg_head  = RegressionHead(cfg.n_inputs, cfg.n_outputs, cfg.hidden_dims)

    # ── Inference ────────────────────────────────────────────────────────────

    def forward(self, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (image, numerical_predictions)."""
        return self.image_gen(condition), self.reg_head(condition)

    def generate(self, condition: torch.Tensor) -> torch.Tensor:
        """Image output only."""
        return self.image_gen(condition)

    def predict(self, condition: torch.Tensor) -> torch.Tensor:
        """Numerical output only."""
        return self.reg_head(condition)

    # ── Training interface ───────────────────────────────────────────────────

    def generator_parameters(self):
        return self.parameters()

    def compute_generator_loss(self, batch: dict):
        condition = batch["inputs"]
        real_img  = batch["image"]
        real_num  = batch["labels"]

        fake_img, pred_num = self.forward(condition)

        loss_l1  = F.l1_loss(fake_img, real_img)  * self.cfg.lambda_l1
        loss_mse = F.mse_loss(fake_img, real_img) * self.cfg.lambda_mse
        loss_reg = F.mse_loss(pred_num, real_num)  * self.cfg.lambda_reg
        loss     = loss_l1 + loss_mse + loss_reg

        logs = {
            "loss":    loss.item(),
            "l1":      loss_l1.item(),
            "mse_img": loss_mse.item(),
            "mse_reg": loss_reg.item(),
        }
        visuals = {"fake": fake_img.detach(), "real": real_img,
                   "pred_num": pred_num.detach()}
        return loss, logs, visuals

    def config_dict(self) -> dict:
        return {**asdict(self.cfg), "model_name": "mmimreg"}
