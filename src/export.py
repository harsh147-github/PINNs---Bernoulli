"""Exports: CSV, ParaView (.vtu), MATLAB (.mat) — README Section 11.

The 1D surrogate field is reconstructed into a 2D axisymmetric field for
real CFD-style visualization: at each station x the duct spans
y in [-D(x)/2, +D(x)/2] with plug flow u(x,y) = V(x), p(x,y) = p(x).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from src.analytical import pressure_exact_nondim, velocity_exact_nondim
from src.geometry import to_si_pressure, to_si_velocity


@torch.no_grad()
def predict_si(
    model, cfg: dict, xt_np: np.ndarray, dtt_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the surrogate, return (V [m/s], p [Pa]) as numpy arrays."""
    device = next(model.parameters()).device
    xt = torch.as_tensor(xt_np.reshape(-1, 1), dtype=torch.float64, device=device)
    dtt = torch.as_tensor(dtt_np.reshape(-1, 1), dtype=torch.float64, device=device)
    v, p = model(xt, dtt)
    v, p = v.cpu(), p.cpu()
    ph = cfg["physics"]
    V = to_si_velocity(v, ph["V_in"]).numpy().reshape(xt_np.shape)
    P = to_si_pressure(p, ph["p_in"], ph["rho"], ph["V_in"]).numpy().reshape(xt_np.shape)
    return V, P


@torch.no_grad()
def exact_si(
    cfg: dict, xt_np: np.ndarray, dtt_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical ground truth in SI for the same points."""
    xt = torch.as_tensor(xt_np.reshape(-1, 1), dtype=torch.float64)
    dtt = torch.as_tensor(dtt_np.reshape(-1, 1), dtype=torch.float64)
    v = velocity_exact_nondim(xt, dtt)
    p = pressure_exact_nondim(xt, dtt)
    ph = cfg["physics"]
    V = to_si_velocity(v, ph["V_in"]).numpy().reshape(xt_np.shape)
    P = to_si_pressure(p, ph["p_in"], ph["rho"], ph["V_in"]).numpy().reshape(xt_np.shape)
    return V, P


def _axisym_mesh(cfg: dict, dt_si: float):
    """Structured (nx x ny) axisymmetric grid + local diameter, for one Dt."""
    g, e = cfg["geometry"], cfg["export"]
    x = np.linspace(0.0, g["L"], e["n_stations"])
    d = g["D_in"] + (dt_si - g["D_in"]) * np.sin(np.pi * x / g["L"]) ** 2
    X = np.repeat(x[:, None], e["n_radial"], axis=1)
    Y = np.stack([np.linspace(-di / 2, di / 2, e["n_radial"]) for di in d])
    XT = X / g["L"]
    DTT = np.full_like(X, dt_si / g["D_in"])
    return X, Y, d, XT, DTT


def export_csv(model, cfg: dict, out_dir: str) -> Path:
    """One CSV per plotted throat diameter: x, Dt, V, p, V_exact, p_exact (SI)."""
    out = Path(out_dir) / "csv"
    out.mkdir(parents=True, exist_ok=True)
    g = cfg["geometry"]
    x = np.linspace(0.0, g["L"], cfg["export"]["n_stations"])
    for dt_si in cfg["export"]["plot_dt"]:
        xt, dtt = x / g["L"], np.full_like(x, dt_si / g["D_in"])
        V, P = predict_si(model, cfg, xt, dtt)
        Ve, Pe = exact_si(cfg, xt, dtt)
        data = np.column_stack([x, np.full_like(x, dt_si), V, P, Ve, Pe])
        np.savetxt(
            out / f"nozzle_Dt_{dt_si:.4f}m.csv",
            data,
            delimiter=",",
            header="x_m,Dt_m,V_m_per_s,p_Pa,V_exact_m_per_s,p_exact_Pa",
            comments="",
        )
    return out


def export_vtu(model, cfg: dict, out_dir: str, dt_list_si: list[float]) -> Path:
    """Write one .vtu per throat diameter for ParaView (README Section 11)."""
    import pyvista as pv

    out = Path(out_dir) / "paraview"
    out.mkdir(parents=True, exist_ok=True)
    for i, dt_si in enumerate(dt_list_si):
        X, Y, _, XT, DTT = _axisym_mesh(cfg, dt_si)
        V, P = predict_si(model, cfg, XT, DTT)
        Z = np.zeros_like(X)
        grid = pv.StructuredGrid(X, Y, Z)
        grid.point_data["V"] = V.ravel(order="F")        # velocity [m/s]
        grid.point_data["p"] = P.ravel(order="F")        # pressure [Pa]
        grid.point_data["q_dyn"] = (0.5 * cfg["physics"]["rho"] * V**2).ravel(order="F")
        grid.cast_to_unstructured_grid().save(out / f"nozzle_Dt_{i:03d}.vtu")
    return out


def export_mat(model, cfg: dict, out_dir: str) -> Path:
    """MATLAB .mat with the full parametric sweep (README Section 11)."""
    out = Path(out_dir) / "matlab"
    out.mkdir(parents=True, exist_ok=True)
    g = cfg["geometry"]
    dts = np.linspace(g["dt_min"], g["dt_max"], cfg["export"]["sweep_n"])
    x = np.linspace(0.0, g["L"], cfg["export"]["n_stations"])
    X, DT = np.meshgrid(x, dts)
    V = np.zeros_like(X)
    P = np.zeros_like(X)
    Ve = np.zeros_like(X)
    Pe = np.zeros_like(X)
    D = np.zeros_like(X)
    for i, dt_si in enumerate(dts):
        xt, dtt = x / g["L"], np.full_like(x, dt_si / g["D_in"])
        V[i], P[i] = predict_si(model, cfg, xt, dtt)
        Ve[i], Pe[i] = exact_si(cfg, xt, dtt)
        D[i] = g["D_in"] + (dt_si - g["D_in"]) * np.sin(np.pi * x / g["L"]) ** 2
    savemat(
        out / "nozzle_surrogate.mat",
        {"X": X, "DT": DT, "V": V, "P": P, "V_exact": Ve, "P_exact": Pe, "D": D},
    )
    return out


# ---------------------------------------------------------------------------
# Stage 2 — CSV export (CLAUDE.md Section 6: "same ... export ... pipeline").
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_si_stage2(model, cfg: dict, xt_np: np.ndarray, dtt_np: np.ndarray):
    """Evaluate the Stage 2 surrogate, return (rho, V, p, T) SI numpy arrays."""
    device = next(model.parameters()).device
    xt = torch.as_tensor(xt_np.reshape(-1, 1), dtype=torch.float64, device=device)
    dtt = torch.as_tensor(dtt_np.reshape(-1, 1), dtype=torch.float64, device=device)
    rho, v, p, t = (o.cpu() for o in model(xt, dtt))
    ph = cfg["physics"]
    rho_in = ph["p_in"] / (ph["R"] * ph["T_in"])
    RHO = (rho * rho_in).numpy().reshape(xt_np.shape)
    V = (v * ph["V_in"]).numpy().reshape(xt_np.shape)
    P = (p * ph["p_in"]).numpy().reshape(xt_np.shape)
    T = (t * ph["T_in"]).numpy().reshape(xt_np.shape)
    return RHO, V, P, T


def export_csv_stage2(model, cfg: dict, out_dir: str) -> Path:
    """One CSV per plotted throat diameter: x, Dt, rho, V, p, T + exact (SI)."""
    from src.analytical import stage2_exact_state_si

    out = Path(out_dir) / "csv"
    out.mkdir(parents=True, exist_ok=True)
    g = cfg["geometry"]
    x = np.linspace(0.0, g["L"], cfg["export"]["n_stations"])
    for dt_si in cfg["export"]["plot_dt"]:
        xt, dtt = x / g["L"], np.full_like(x, dt_si / g["D_in"])
        RHO, V, P, T = predict_si_stage2(model, cfg, xt, dtt)
        rho_e, v_e, p_e, t_e = stage2_exact_state_si(x, dt_si, cfg)
        data = np.column_stack([x, np.full_like(x, dt_si), RHO, V, P, T, rho_e, v_e, p_e, t_e])
        np.savetxt(
            out / f"nozzle_stage2_Dt_{dt_si:.4f}m.csv",
            data,
            delimiter=",",
            header="x_m,Dt_m,rho,V_m_per_s,p_Pa,T_K,rho_exact,V_exact,p_exact,T_exact",
            comments="",
        )
    return out
