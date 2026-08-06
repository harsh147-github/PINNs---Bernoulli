"""Figures — README Sections 10 and 11.

(a) V(x), p(x): PINN vs analytical for several throat diameters
(b) throat quantities vs throat diameter (the parametric design sweep)
(c) colored CFD-style 2D field in the actual nozzle shape
(d) total-head (Bernoulli invariant) error map over (x, Dt)
(e) training loss history, per term, log scale
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.export import _axisym_mesh, exact_si, predict_si
from src.geometry import area_nondim
import torch


def _fig_dir(out_dir: str) -> Path:
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_fields_vs_exact(model, cfg: dict, out_dir: str) -> Path:
    """(a) PINN vs closed-form solution along the nozzle."""
    g = cfg["geometry"]
    x = np.linspace(0.0, g["L"], cfg["export"]["n_stations"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for dt_si in cfg["export"]["plot_dt"]:
        xt, dtt = x / g["L"], np.full_like(x, dt_si / g["D_in"])
        V, P = predict_si(model, cfg, xt, dtt)
        Ve, Pe = exact_si(cfg, xt, dtt)
        (l1,) = axes[0].plot(x, V, label=f"PINN, $D_t$={dt_si:.3f} m")
        axes[0].plot(x, Ve, "--", color=l1.get_color(), alpha=0.6)
        (l2,) = axes[1].plot(x, P / 1000, label=f"PINN, $D_t$={dt_si:.3f} m")
        axes[1].plot(x, Pe / 1000, "--", color=l2.get_color(), alpha=0.6)
    axes[0].set(xlabel="x [m]", ylabel="V [m/s]", title="Velocity — PINN (solid) vs exact (dashed)")
    axes[1].set(xlabel="x [m]", ylabel="p [kPa]", title="Pressure — PINN (solid) vs exact (dashed)")
    for ax in axes:
        ax.axvline(g["L"] / 2, color="gray", ls=":", lw=1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_dir(out_dir) / "fields_vs_exact.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_throat_sweep(model, cfg: dict, out_dir: str) -> Path:
    """(b) What changes when the throat diameter changes (README Section 7)."""
    g = cfg["geometry"]
    dts = np.linspace(g["dt_min"], g["dt_max"], cfg["export"]["sweep_n"])
    xt = np.full_like(dts, 0.5)  # throat station
    dtt = dts / g["D_in"]
    Vt, Pt = predict_si(model, cfg, xt, dtt)
    Vte, Pte = exact_si(cfg, xt, dtt)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(dts, Vt, "o-", label="PINN surrogate")
    axes[0].plot(dts, Vte, "--", label="exact $(D_{in}/D_t)^2 V_{in}$")
    axes[0].set(xlabel="throat diameter $D_t$ [m]", ylabel="throat velocity [m/s]",
                title="Throat velocity vs $D_t$ (inlet held constant)")
    axes[1].plot(dts, Pt / 1000, "o-", label="PINN surrogate")
    axes[1].plot(dts, Pte / 1000, "--", label="exact Bernoulli")
    axes[1].set(xlabel="throat diameter $D_t$ [m]", ylabel="throat pressure [kPa]",
                title="Throat pressure vs $D_t$")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = _fig_dir(out_dir) / "throat_sweep.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_colored_field(model, cfg: dict, out_dir: str, dt_list=(0.20, 0.45)) -> Path:
    """(c) CFD-style colored pressure & velocity fields in the nozzle shape."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for col, dt_si in enumerate(dt_list):
        X, Y, d, XT, DTT = _axisym_mesh(cfg, dt_si)
        V, P = predict_si(model, cfg, XT, DTT)
        for row, (field, name, unit, cmap) in enumerate(
            [(P / 1000, "Pressure", "kPa", "RdBu_r"), (V, "Velocity", "m/s", "turbo")]
        ):
            ax = axes[row, col]
            cf = ax.contourf(X, Y, field, levels=100, cmap=cmap)
            ax.plot(X[:, 0], d / 2, "k-", lw=2)
            ax.plot(X[:, 0], -d / 2, "k-", lw=2)
            ax.set_title(f"{name} field — $D_t$ = {dt_si:.2f} m")
            ax.set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
            fig.colorbar(cf, ax=ax, label=unit)
    path = _fig_dir(out_dir) / "colored_cfd_fields.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_total_head_map(model, cfg: dict, out_dir: str) -> Path:
    """(d) Bernoulli invariant check: p~ + V~^2 must equal 1 everywhere."""
    from src.physics import total_head

    g = cfg["geometry"]
    xt = np.linspace(0, 1, 200)
    dtt = np.linspace(g["dt_min"] / g["D_in"], g["dt_max"] / g["D_in"], 25)
    XT, DTT = np.meshgrid(xt, dtt)
    xt_t = torch.as_tensor(XT.reshape(-1, 1), dtype=torch.float64)
    dtt_t = torch.as_tensor(DTT.reshape(-1, 1), dtype=torch.float64)
    H = total_head(model, xt_t, dtt_t).numpy().reshape(XT.shape)
    fig, ax = plt.subplots(figsize=(9, 6))
    cf = ax.contourf(XT * g["L"], DTT * g["D_in"], np.abs(H - 1.0) * 100, levels=50, cmap="viridis")
    ax.set(xlabel="x [m]", ylabel="$D_t$ [m]",
           title="Bernoulli invariant error |p~ + V~² − 1| [%] over the whole design space")
    fig.colorbar(cf, ax=ax, label="%")
    fig.tight_layout()
    path = _fig_dir(out_dir) / "total_head_error_map.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_loss_history(history: dict, out_dir: str) -> Path:
    """(e) Training curves: total + per-term losses (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(history["epoch"], history["loss"], label="total")
    ax.semilogy(history["epoch"], history["loss_cont"], label="$\\mathcal{L}_{cont}$ (continuity)")
    ax.semilogy(history["epoch"], history["loss_mom"], label="$\\mathcal{L}_{mom}$ (Bernoulli)")
    ax.set(xlabel="epoch / L-BFGS iteration", ylabel="loss (log scale)",
           title="Training history — physics losses driven down by Adam then L-BFGS")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = _fig_dir(out_dir) / "loss_history.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def make_all_figures(model, cfg: dict, history: dict, out_dir: str) -> list[Path]:
    """Generate the full figure set (a)-(e)."""
    return [
        plot_fields_vs_exact(model, cfg, out_dir),
        plot_throat_sweep(model, cfg, out_dir),
        plot_colored_field(model, cfg, out_dir),
        plot_total_head_map(model, cfg, out_dir),
        plot_loss_history(history, out_dir),
    ]
