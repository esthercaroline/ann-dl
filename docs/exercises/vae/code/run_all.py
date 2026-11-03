"""Train both VAEs, save Figures 1–6 and a metrics JSON for the report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from vae import VAE, evaluate, train_vae

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"
DEVICE = torch.device("cpu")

def load_splits():
    train_full = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transforms.ToTensor()
    )
    test = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    train, val = random_split(
        train_full, [50_000, 10_000], generator=torch.Generator().manual_seed(42)
    )
    return train, val, test


def plot_curves(hist, path: Path):
    epochs = np.arange(1, len(hist.train_recon) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, hist.train_recon, label="train")
    axes[0].plot(epochs, hist.val_recon, label="validation")
    axes[0].set_title("Reconstruction (BCE) vs epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Nats per image")
    axes[0].legend()

    axes[1].plot(epochs, hist.train_kl, label="train")
    axes[1].plot(epochs, hist.val_kl, label="validation")
    axes[1].set_title("KL vs epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Nats per image")
    axes[1].legend()

    fig.suptitle("Figure 1 — Training curves ($d = 16$)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_reconstructions(model: VAE, test, path: Path):
    model.eval()
    # First test image of each digit 0–9
    found = {}
    for i in range(len(test)):
        x, y = test[i]
        y = int(y)
        if y not in found:
            found[y] = (i, x)
        if len(found) == 10:
            break
    originals = torch.stack([found[d][1].view(-1) for d in range(10)]).to(DEVICE)
    mu, _ = model.encode(originals)
    recons = model.decode(mu).cpu()

    fig, axes = plt.subplots(2, 10, figsize=(12, 2.6))
    for d in range(10):
        axes[0, d].imshow(found[d][1].view(28, 28), cmap="gray")
        axes[0, d].axis("off")
        axes[0, d].set_title(str(d), fontsize=9)
        axes[1, d].imshow(recons[d].view(28, 28), cmap="gray")
        axes[1, d].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel(r"decode $\mu$", fontsize=9)
    fig.suptitle(r"Figure 2 — First test image of each digit and reconstruction from $\mu$")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_samples(model: VAE, path: Path):
    model.eval()
    torch.manual_seed(0)
    z = torch.randn(64, model.latent_dim, device=DEVICE)
    imgs = model.decode(z).cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
    fig.suptitle(r"Figure 3 — $8\times 8$ samples from $z\sim\mathcal{N}(0,I)$ (seed 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_latent_scatter(model: VAE, test, path: Path):
    model.eval()
    loader = DataLoader(test, batch_size=512, shuffle=False)
    mus, labels = [], []
    for x, y in loader:
        x = x.view(x.size(0), -1).to(DEVICE)
        mu, _ = model.encode(x)
        mus.append(mu.cpu())
        labels.append(y)
    mu = torch.cat(mus).numpy()
    y = torch.cat(labels).numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    for d in range(10):
        m = y == d
        ax.scatter(mu[m, 0], mu[m, 1], s=6, alpha=0.45, label=str(d))
    ax.set_title(r"Figure 4 — Encoded test set ($\mu$, $d=2$), colored by digit")
    ax.set_xlabel(r"$\mu_1$")
    ax.set_ylabel(r"$\mu_2$")
    ax.legend(title="digit", markerscale=3, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_decoded_plane(model: VAE, path: Path):
    model.eval()
    # Equal prior mass per cell: z = Phi^{-1}(q) with q evenly spaced in [0.05, 0.95]
    qs = np.linspace(0.05, 0.95, 15)
    coords = norm.ppf(qs)
    grid = np.stack(np.meshgrid(coords, coords[::-1]), axis=-1).reshape(-1, 2)
    z = torch.tensor(grid, dtype=torch.float32, device=DEVICE)
    imgs = model.decode(z).cpu().view(-1, 28, 28)

    canvas = np.zeros((15 * 28, 15 * 28))
    for i in range(15):
        for j in range(15):
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = imgs[i * 15 + j]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(canvas, cmap="gray")
    ax.set_title(r"Figure 5 — Decoded $15\times 15$ plane ($z=\Phi^{-1}(q)$)")
    ax.set_xlabel(r"$z_1$ (quantile grid)")
    ax.set_ylabel(r"$z_2$ (quantile grid)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def first_index(test, digit: int) -> int:
    for i in range(len(test)):
        if int(test[i][1]) == digit:
            return i
    raise ValueError(f"digit {digit} not found")


@torch.no_grad()
def plot_interpolation(model16: VAE, test, idx_a: int, idx_b: int, path: Path):
    model16.eval()
    xa, ya = test[idx_a]
    xb, yb = test[idx_b]
    xa_flat = xa.view(1, -1).to(DEVICE)
    xb_flat = xb.view(1, -1).to(DEVICE)
    mu_a, _ = model16.encode(xa_flat)
    mu_b, _ = model16.encode(xb_flat)

    ts = np.linspace(0, 1, 10)
    fig, axes = plt.subplots(2, 10, figsize=(12, 2.8))
    for k, t in enumerate(ts):
        # Top: pixel-space linear blend
        blend = ((1 - t) * xa + t * xb).view(28, 28)
        axes[0, k].imshow(blend, cmap="gray")
        axes[0, k].axis("off")
        axes[0, k].set_title(f"$t={t:.1f}$", fontsize=8)
        # Bottom: latent-space blend with d=16 model
        z = (1 - t) * mu_a + t * mu_b
        img = model16.decode(z).cpu().view(28, 28)
        axes[1, k].imshow(img, cmap="gray")
        axes[1, k].axis("off")
    axes[0, 0].set_ylabel("pixel", fontsize=9)
    axes[1, 0].set_ylabel("latent", fontsize=9)
    fig.suptitle(
        f"Figure 6 — Interpolation (test indices {idx_a} digit {int(ya)}"
        f" → {idx_b} digit {int(yb)}; latent row uses $d=16$)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return int(ya), int(yb)


def main():
    train, val, test = load_splits()
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    # ---- Exercise 1: d = 16 ----
    torch.manual_seed(42)
    model16 = VAE(latent_dim=16)
    hist16 = train_vae(model16, train, val, device=DEVICE)
    test_recon16, test_kl16 = evaluate(model16, test_loader, DEVICE)
    print(f"\n[Ex1] test recon={test_recon16:.2f}  kl={test_kl16:.2f}  "
          f"negELBO={test_recon16 + test_kl16:.2f}")

    plot_curves(hist16, FIG_DIR / "fig1_curves.png")
    plot_reconstructions(model16, test, FIG_DIR / "fig2_reconstructions.png")
    plot_samples(model16, FIG_DIR / "fig3_samples.png")

    torch.save(model16.state_dict(), ROOT / "vae_d16.pt")

    # ---- Exercise 2: d = 2 ----
    torch.manual_seed(42)
    model2 = VAE(latent_dim=2)
    hist2 = train_vae(model2, train, val, device=DEVICE)
    test_recon2, test_kl2 = evaluate(model2, test_loader, DEVICE)
    print(f"\n[Ex2] test recon={test_recon2:.2f}  kl={test_kl2:.2f}  "
          f"negELBO={test_recon2 + test_kl2:.2f}")

    plot_latent_scatter(model2, test, FIG_DIR / "fig4_latent_scatter.png")
    plot_decoded_plane(model2, FIG_DIR / "fig5_decoded_plane.png")

    # First test "4" and first test "9" — different digits for Figure 6.
    idx_a, idx_b = first_index(test, 4), first_index(test, 9)
    ya, yb = plot_interpolation(
        model16, test, idx_a, idx_b, FIG_DIR / "fig6_interpolation.png"
    )

    torch.save(model2.state_dict(), ROOT / "vae_d2.pt")

    metrics = {
        "ex1_test_recon": round(test_recon16, 2),
        "ex1_test_kl": round(test_kl16, 2),
        "ex1_test_neg_elbo": round(test_recon16 + test_kl16, 2),
        "ex1_val_kl_first": round(hist16.val_kl[0], 2),
        "ex1_val_kl_last": round(hist16.val_kl[-1], 2),
        "ex2_test_recon": round(test_recon2, 2),
        "ex2_test_kl": round(test_kl2, 2),
        "interp_idx_a": idx_a,
        "interp_idx_b": idx_b,
        "interp_digit_a": ya,
        "interp_digit_b": yb,
        "hist16": {
            "train_recon": hist16.train_recon,
            "train_kl": hist16.train_kl,
            "val_recon": hist16.val_recon,
            "val_kl": hist16.val_kl,
        },
        "hist2": {
            "train_recon": hist2.train_recon,
            "train_kl": hist2.train_kl,
            "val_recon": hist2.val_recon,
            "val_kl": hist2.val_kl,
        },
    }
    with open(ROOT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved metrics.json and figures to", FIG_DIR)


if __name__ == "__main__":
    main()
