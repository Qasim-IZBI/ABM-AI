import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import compute_metrics, save_checkpoint, load_checkpoint, latest_checkpoint


class BaseTrainer:
    """
    Universal training loop for ABM models.

    Supports two modes, detected automatically from the model interface:

    Regression (MLP)
        model.forward(x) → predictions
        Single AdamW optimizer, MSE loss, logs loss + R².

    GAN (CGAN)
        model.compute_generator_loss(batch)           → (loss_G, logs, visuals)
        model.compute_discriminator_loss(batch, vis)  → (loss_D, logs)
        model.generator_parameters()                  → Iterable[Parameter]
        model.discriminator_parameters()              → Iterable[Parameter]
        Two Adam optimizers (betas=(0.5, 0.999)), logs G/D losses.

    Both modes support checkpoint auto-resume and elapsed-time tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        out_dir: str = "results",
        device: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = out_dir
        self.ckpt_dir = os.path.join(out_dir, "checkpoints")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.is_gan         = hasattr(model, "compute_generator_loss")
        self.has_discriminator = hasattr(model, "discriminator_parameters")

        if self.is_gan:
            self.opt_G = torch.optim.Adam(
                model.generator_parameters(), lr=lr, betas=(0.5, 0.999)
            )
            if self.has_discriminator:
                self.opt_D = torch.optim.Adam(
                    model.discriminator_parameters(), lr=lr, betas=(0.5, 0.999)
                )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.criterion = nn.MSELoss()

        self.start_epoch = 0
        self.accumulated_seconds = 0.0

        ckpt = latest_checkpoint(self.ckpt_dir)
        if ckpt:
            if self.is_gan:
                opt_d = self.opt_D if self.has_discriminator else None
                epoch, _ = load_checkpoint(ckpt, self.model, self.opt_G, opt_d=opt_d)
            else:
                epoch, _ = load_checkpoint(ckpt, self.model, self.optimizer)
            self.start_epoch = epoch + 1
            print(f"Resumed from checkpoint: {ckpt} (epoch {epoch})")

    # ── Public API ───────────────────────────────────────────────────────────────

    def train(self, epochs: int, save_every: int = 10, log_every: int = 1):
        n_train = len(self.train_loader.dataset)
        mode = "GAN" if self.is_gan else "regression"
        print(f"Training ({mode}) on {self.device} — {epochs} epochs, {n_train} samples")

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()

            if self.is_gan:
                logs = self._train_epoch_gan()
                log_str = "  ".join(f"{k}={v:.4f}" for k, v in logs.items())
            else:
                train_loss, train_metrics = self._train_epoch_regression()
                log_str = f"loss={train_loss:.4f}  r2={train_metrics['r2']:.4f}"
                if self.val_loader is not None:
                    val_loss, val_metrics = self._val_epoch()
                    log_str += f"  val_loss={val_loss:.4f}  val_r2={val_metrics['r2']:.4f}"

            elapsed = time.time() - t0
            self.accumulated_seconds += elapsed

            if (epoch + 1) % log_every == 0:
                print(f"[{epoch+1:04d}/{epochs}]  {log_str}  ({elapsed:.1f}s)")

            if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
                self._save_checkpoint(epoch)

        print(f"Done. Total training time: {self.accumulated_seconds / 60:.1f} min")

    # ── GAN training ─────────────────────────────────────────────────────────────

    def _train_epoch_gan(self) -> dict:
        self.model.train()
        sum_logs: dict = {}
        n = 0

        for batch in self.train_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            bs = batch["inputs"].size(0)

            # ── Generator step
            self.opt_G.zero_grad()
            loss_G, logs_G, visuals = self.model.compute_generator_loss(batch)
            loss_G.backward()
            self.opt_G.step()

            # ── Discriminator step (skipped for generator-only models like imreg)
            if self.has_discriminator:
                self.opt_D.zero_grad()
                loss_D, logs_D = self.model.compute_discriminator_loss(batch, visuals)
                loss_D.backward()
                self.opt_D.step()
                logs_G.update(logs_D)

            for k, v in logs_G.items():
                sum_logs[k] = sum_logs.get(k, 0.0) + v * bs
            n += bs

        return {k: v / n for k, v in sum_logs.items()}

    # ── Regression training ───────────────────────────────────────────────────────

    def _train_epoch_regression(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in self.train_loader:
            x = batch["inputs"].to(self.device)
            y = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x)
            loss  = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(x)
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

        n = len(self.train_loader.dataset)
        return total_loss / n, compute_metrics(torch.cat(all_preds), torch.cat(all_targets))

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []

        for batch in self.val_loader:
            x = batch["inputs"].to(self.device)
            y = batch["labels"].to(self.device)
            preds = self.model(x)
            total_loss += self.criterion(preds, y).item() * len(x)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

        n = len(self.val_loader.dataset)
        return total_loss / n, compute_metrics(torch.cat(all_preds), torch.cat(all_targets))

    # ── Checkpoint ───────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(self.ckpt_dir, f"epoch_{epoch+1:04d}.pt")
        if self.is_gan:
            opt_d = self.opt_D if self.has_discriminator else None
            save_checkpoint(path, self.model, self.opt_G, epoch,
                            self.model.config_dict(), opt_d=opt_d)
        else:
            save_checkpoint(path, self.model, self.optimizer, epoch,
                            self.model.config_dict())
        print(f"  -> saved checkpoint: {path}")
