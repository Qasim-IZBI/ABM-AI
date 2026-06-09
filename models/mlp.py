from dataclasses import dataclass, field, asdict
from typing import List

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    n_inputs:     int
    n_outputs:    int
    hidden_dims:  List[int] = field(default_factory=lambda: [128, 64])
    dropout:      float = 0.2
    activation:   str = "relu"   # "relu" | "gelu" | "silu"


class MLP(nn.Module):
    """
    Fully-connected regression network.

    Input:  (B, n_inputs)   — normalised numerical parameters
    Output: (B, n_outputs)  — predicted simulation outputs
    """

    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if cfg.activation not in act_map:
            raise ValueError(f"Unknown activation '{cfg.activation}'")
        Act = act_map[cfg.activation]

        dims = [cfg.n_inputs] + cfg.hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d), Act(), nn.Dropout(cfg.dropout)]
        layers += [nn.Linear(dims[-1], cfg.n_outputs)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
