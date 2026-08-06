"""Matplotlib figures (README Section 10, CLAUDE.md Section 8): field overlays,
throat sweep, total-head error map, loss history. All PNG, 300 dpi, labeled axes.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src import analytical


def _model_fields_nondim(model, xt: torch.Tensor, dtt: torch.Tensor):
    with torch.no_grad():
        out = model(xt, dtt)
    return out[..., 0], out[..., 1]


def plot_fields_vs_analytical(model, dt_values_tilde, out_path: str, n_x: int = 300):
    """(a) V_tilde(x_tilde), p_tilde(x_tilde): PINN vs analytical, several Dt_tilde."""
    device = next(model.parameters()).device
    xt = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64, device=device).reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for dtt_val in dt_values_tilde:
        dtt = torch.full_like(xt, dtt_val)
        V_pred, p_pred = _model_fields_nondim(model, xt, dtt)
        V_exact = analytical.velocity_exact_nondim(xt, dtt).squeeze(-1)
        p_exact = analytical.pressure_exact_nondim(xt, dtt).squeeze(-1)

        xt_np = xt.cpu().numpy().flatten()
        ax1.plot(xt_np, V_pred.cpu().numpy(), "-", label=f"PINN, Dt~={dtt_val:.2f}")
        ax1.plot(xt_np, V_exact.cpu().numpy(), "k--", linewidth=0.8)
        ax2.plot(xt_np, p_pred.cpu().numpy(), "-", label=f"PINN, Dt~={dtt_val:.2f}")
        ax2.plot(xt_np, p_exact.cpu().numpy(), "k--", linewidth=0.8)

    ax1.set_xlabel("x_tilde [-]"); ax1.set_ylabel("V_tilde [-]"); ax1.set_title("Velocity (dashed = analytical)")
    ax2.set_xlabel("x_tilde [-]"); ax2.set_ylabel("p_tilde [-]"); ax2.set_title("Pressure (dashed = analytical)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_throat_sweep(model, dt_values_tilde, out_path: str):
    """(b) Throat V_tilde, p_tilde vs Dt_tilde: PINN vs analytical."""
    device = next(model.parameters()).device
    dtt = torch.tensor(dt_values_tilde, dtype=torch.float64, device=device).reshape(-1, 1)
    xt = torch.full_like(dtt, 0.5)  # throat at x_tilde = 0.5

    V_pred, p_pred = _model_fields_nondim(model, xt, dtt)
    V_exact = analytical.velocity_exact_nondim(xt, dtt).squeeze(-1)
    p_exact = analytical.pressure_exact_nondim(xt, dtt).squeeze(-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    dt_np = np.array(dt_values_tilde)
    ax1.plot(dt_np, V_pred.cpu().numpy(), "o-", label="PINN")
    ax1.plot(dt_np, V_exact.cpu().numpy(), "kx--", label="Analytical")
    ax2.plot(dt_np, p_pred.cpu().numpy(), "o-", label="PINN")
    ax2.plot(dt_np, p_exact.cpu().numpy(), "kx--", label="Analytical")

    ax1.set_xlabel("Dt_tilde [-]"); ax1.set_ylabel("Throat V_tilde [-]"); ax1.set_title("Throat velocity vs Dt")
    ax2.set_xlabel("Dt_tilde [-]"); ax2.set_ylabel("Throat p_tilde [-]"); ax2.set_title("Throat pressure vs Dt")
    for ax in (ax1, ax2):
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_total_head_error_map(model, out_path: str, n_x: int = 100, n_dt: int = 50,
                               dt_min_t: float = 0.4, dt_max_t: float = 0.9):
    """(c) |p_tilde + V_tilde^2 - 1| over the full (x_tilde, Dt_tilde) domain."""
    device = next(model.parameters()).device
    xt_1d = torch.linspace(0.0, 1.0, n_x, dtype=torch.float64)
    dtt_1d = torch.linspace(dt_min_t, dt_max_t, n_dt, dtype=torch.float64)
    XT, DTT = torch.meshgrid(xt_1d, dtt_1d, indexing="ij")
    xt = XT.reshape(-1, 1).to(device)
    dtt = DTT.reshape(-1, 1).to(device)

    V_pred, p_pred = _model_fields_nondim(model, xt, dtt)
    error = torch.abs(p_pred + V_pred ** 2 - 1.0).cpu().numpy().reshape(n_x, n_dt)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(xt_1d.numpy(), dtt_1d.numpy(), error.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="|p_tilde + V_tilde^2 - 1|")
    ax.set_xlabel("x_tilde [-]"); ax.set_ylabel("Dt_tilde [-]")
    ax.set_title("Bernoulli total-head invariant error map")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_loss_history(history: dict, out_path: str):
    """(d) Loss history (log scale), per term."""
    epochs = history["epoch"]
    term_names = sorted(history["terms"][0].keys())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["loss_total"], label="total", linewidth=2, color="black")
    for name in term_names:
        vals = [t[name] for t in history["terms"]]
        ax.plot(epochs, vals, label=name, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (log scale)")
    ax.set_title("Training loss history")
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
