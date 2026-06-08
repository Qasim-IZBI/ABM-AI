import os
import random
import numpy as np
import torch


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Returns MSE, MAE and R² averaged over output dimensions."""
    with torch.no_grad():
        mse = torch.mean((preds - targets) ** 2).item()
        mae = torch.mean(torch.abs(preds - targets)).item()
        ss_res = torch.sum((targets - preds) ** 2, dim=0)
        ss_tot = torch.sum((targets - targets.mean(dim=0)) ** 2, dim=0)
        r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).mean().item()
    return {"mse": mse, "mae": mae, "r2": r2}


# ── Checkpoints ──────────────────────────────────────────────────────────────

def save_checkpoint(path: str, model, optimizer, epoch: int, cfg: dict,
                    opt_d=None):
    """Save model + optimizer(s) state. Pass opt_d for GAN discriminator."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    cfg,
    }
    if opt_d is not None:
        ckpt["optimizer_d"] = opt_d.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, opt_d=None):
    """Load model + optimizer(s) state. Returns (epoch, config)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if opt_d is not None and "optimizer_d" in ckpt:
        opt_d.load_state_dict(ckpt["optimizer_d"])
    return ckpt.get("epoch", 0), ckpt.get("config", {})


def latest_checkpoint(ckpt_dir: str):
    """Return path of the highest-epoch checkpoint in ckpt_dir, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None
    files.sort()
    return os.path.join(ckpt_dir, files[-1])
