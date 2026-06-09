import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImageRegressorConfig:
    n_inputs:   int               # number of conditioning input features
    n_outputs:  int = 0           # unused; kept for build_model interface parity
    image_size: int = 512         # output image H = W (must be a power of 2, ≥ 64)
    ngf:        int = 32          # base channel count
    lambda_l1:  float = 1.0      # weight of L1 pixel loss
    lambda_mse: float = 1.0      # weight of MSE pixel loss


class ImageRegressor(nn.Module):
    """
    Deterministic image generator conditioned on numerical ABM parameters.

    Identical generator structure to CGAN but with no latent noise and no
    adversarial training — the model is trained purely with pixel-level losses
    (L1 + MSE), making outputs deterministic at inference.

    Input : condition  (B, n_inputs)
    Output: RGB image  (B, 3, image_size, image_size)  in [-1, 1]

    Architecture: FC → (ngf×8, 4, 4) → 6× ConvTranspose2d up-blocks.
    For image_size=256 the spatial path is 4→8→16→32→64→128→256.

    Training interface (consumed by BaseTrainer — generator-only path)
    ------------------------------------------------------------------
    generator_parameters()        → Iterable[Parameter]
    compute_generator_loss(batch) → (loss, logs, visuals)
    """

    def __init__(self, cfg: ImageRegressorConfig):
        super().__init__()
        self.cfg = cfg
        n_up = int(math.log2(cfg.image_size // 4))  # e.g. 6 for 256

        # Project condition directly to spatial seed — no noise concatenation
        self.fc = nn.Linear(cfg.n_inputs, cfg.ngf * 8 * 4 * 4)

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

    # ── Core forward ─────────────────────────────────────────────────────────

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        x = self.fc(condition).view(condition.size(0), self.cfg.ngf * 8, 4, 4)
        return self.up_blocks(x)

    # ── Training interface (generator-only, no discriminator) ─────────────────

    def generator_parameters(self):
        return self.parameters()

    def compute_generator_loss(self, batch: dict):
        condition = batch["inputs"]
        real      = batch["image"]

        fake = self.forward(condition)

        loss_l1  = F.l1_loss(fake, real)  * self.cfg.lambda_l1
        loss_mse = F.mse_loss(fake, real) * self.cfg.lambda_mse
        loss     = loss_l1 + loss_mse

        logs    = {"loss": loss.item(), "l1": loss_l1.item(), "mse": loss_mse.item()}
        visuals = {"fake": fake.detach(), "real": real}
        return loss, logs, visuals

    # ── Inference ────────────────────────────────────────────────────────────

    def generate(self, condition: torch.Tensor) -> torch.Tensor:
        """Generate a deterministic image from a condition vector. Output in [-1, 1]."""
        return self.forward(condition)

    def config_dict(self) -> dict:
        return {**asdict(self.cfg), "model_name": "imreg"}
