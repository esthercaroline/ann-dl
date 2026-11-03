"""Variational Autoencoder for MNIST.

Architecture (Exercise 1 / 2):
  Encoder: 784 -> 512 -> 256 (ReLU), then two heads 256 -> d for mu and log_var.
  Decoder: d -> 256 -> 512 -> 784 (ReLU on hidden, sigmoid on output).

The reparameterization, BCE reconstruction (summed over pixels), and the
closed-form KL are written by hand — no library VAE / kl_divergence helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class VAE(nn.Module):
    """Fully-connected VAE with Bernoulli decoder (sigmoid + BCE)."""

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)  # predicts log sigma^2

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # z = mu + sigma * eps, with eps ~ N(0, I); log_var = log(sigma^2)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, reconstruction, KL), each averaged over the batch (nats / image).

    Reconstruction: BCE summed over the 784 pixels (Bernoulli ELBO term).
    KL: closed-form 1/2 * sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2).
    """
    batch = x.size(0)
    # reduction="sum" keeps the pixel sum; divide by batch so the KL is not
    # secretly scaled by beta ≈ 784 (which reduction="mean" would do).
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch
    # KL(q(z|x) || N(0,I)) = 1/2 * sum (mu^2 + sigma^2 - 1 - log sigma^2)
    kl = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - 1.0 - log_var) / batch
    total = recon + beta * kl
    return total, recon, kl


@dataclass
class History:
    train_recon: list[float] = field(default_factory=list)
    train_kl: list[float] = field(default_factory=list)
    val_recon: list[float] = field(default_factory=list)
    val_kl: list[float] = field(default_factory=list)


def _flatten_batch(batch) -> torch.Tensor:
    x, _ = batch
    return x.view(x.size(0), -1)


@torch.no_grad()
def evaluate(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    beta: float = 1.0,
) -> tuple[float, float]:
    """Average reconstruction and KL (nats / image) over ``loader``."""
    model.eval()
    recon_sum = 0.0
    kl_sum = 0.0
    n = 0
    for batch in loader:
        x = _flatten_batch(batch).to(device)
        x_hat, mu, log_var = model(x)
        _, recon, kl = vae_loss(x_hat, x, mu, log_var, beta=beta)
        recon_sum += recon.item() * x.size(0)
        kl_sum += kl.item() * x.size(0)
        n += x.size(0)
    return recon_sum / n, kl_sum / n


def train_vae(
    model: VAE,
    train_set: Dataset,
    val_set: Dataset,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    beta: float = 1.0,
    device: torch.device | None = None,
) -> History:
    """Adam training; logs reconstruction and KL on train (epoch avg) and val."""
    device = device or torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    hist = History()
    for epoch in range(1, epochs + 1):
        model.train()
        recon_sum = 0.0
        kl_sum = 0.0
        n = 0
        for batch in train_loader:
            x = _flatten_batch(batch).to(device)
            opt.zero_grad()
            x_hat, mu, log_var = model(x)
            total, recon, kl = vae_loss(x_hat, x, mu, log_var, beta=beta)
            total.backward()
            opt.step()
            recon_sum += recon.item() * x.size(0)
            kl_sum += kl.item() * x.size(0)
            n += x.size(0)

        train_recon = recon_sum / n
        train_kl = kl_sum / n
        val_recon, val_kl = evaluate(model, val_loader, device, beta=beta)

        hist.train_recon.append(train_recon)
        hist.train_kl.append(train_kl)
        hist.val_recon.append(val_recon)
        hist.val_kl.append(val_kl)

        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"train recon={train_recon:.2f} kl={train_kl:.2f}  "
            f"val recon={val_recon:.2f} kl={val_kl:.2f}"
        )

    return hist
