"""Export trained-surrogate results to CSV, ParaView (.vtu), and MATLAB (.mat)
(README Section 11, CLAUDE.md Section 8).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from scipy.io import savemat

from src import analytical, geometry as geo


def export_csv(model, cfg, dt_values_si: List[float], n_x: int = 200, path: str = None) -> str:
    """Write a long-format CSV: one row per (x, Dt) with PINN and analytical fields, SI units."""
    geom, phys = cfg["geometry"], cfg["physics"]
    L, D_in = geom["L"], geom["D_in"]
    V_in, p_in, rho = phys["V_in"], phys["p_in"], phys["rho"]

    rows = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for Dt_si in dt_values_si:
            x = torch.linspace(0.0, L, n_x, dtype=torch.float64, device=device)
            Dt = torch.full_like(x, Dt_si)
            xt = (x / L).reshape(-1, 1)
            dtt = (Dt / D_in).reshape(-1, 1)

            out = model(xt, dtt)
            V_pred = (out[..., 0] * V_in).cpu().numpy()
            p_pred = (p_in + out[..., 1] * 0.5 * rho * V_in ** 2).cpu().numpy()

            V_exact = analytical.velocity_exact_si(x, Dt, D_in=D_in, L=L, V_in=V_in).cpu().numpy()
            p_exact = analytical.pressure_exact_si(x, Dt, D_in=D_in, L=L, V_in=V_in, p_in=p_in, rho=rho).cpu().numpy()
            D_arr = geo.diameter(x, Dt, D_in=D_in, L=L).cpu().numpy()

            for i in range(n_x):
                rows.append({
                    "x": x[i].item(), "Dt": Dt_si, "D": D_arr[i],
                    "V_pinn": V_pred[i], "p_pinn": p_pred[i],
                    "V_exact": V_exact[i], "p_exact": p_exact[i],
                })

    df = pd.DataFrame(rows)
    if path is None:
        path = os.path.join(cfg["export"]["results_dir"], "csv", "nozzle_surrogate.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_vtu(model, cfg, dt_values_si: List[float], n_x: int = 100) -> List[str]:
    """Reconstruct each 1D field as a 2D axisymmetric slab and write one .vtu per Dt.

    For each x-station, the duct spans y in [-D(x)/2, D(x)/2] with `n_radial_points`
    points; point data carries V, p in SI units (README Section 11).
    """
    geom, phys, exp = cfg["geometry"], cfg["physics"], cfg["export"]
    L, D_in = geom["L"], geom["D_in"]
    V_in, p_in, rho = phys["V_in"], phys["p_in"], phys["rho"]
    n_r = exp["n_radial_points"]

    out_dir = os.path.join(exp["results_dir"], "paraview")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, Dt_si in enumerate(dt_values_si):
            x = torch.linspace(0.0, L, n_x, dtype=torch.float64, device=device)
            Dt = torch.full_like(x, Dt_si)
            xt = (x / L).reshape(-1, 1)
            dtt = (Dt / D_in).reshape(-1, 1)
            out = model(xt, dtt)
            V_line = (out[..., 0] * V_in).cpu().numpy()
            p_line = (p_in + out[..., 1] * 0.5 * rho * V_in ** 2).cpu().numpy()
            D_line = geo.diameter(x, Dt, D_in=D_in, L=L).cpu().numpy()
            x_np = x.cpu().numpy()

            points = np.zeros((n_x * n_r, 3))
            V_field = np.zeros(n_x * n_r)
            p_field = np.zeros(n_x * n_r)
            for i in range(n_x):
                y = np.linspace(-D_line[i] / 2.0, D_line[i] / 2.0, n_r)
                sl = slice(i * n_r, (i + 1) * n_r)
                points[sl, 0] = x_np[i]
                points[sl, 1] = y
                V_field[sl] = V_line[i]
                p_field[sl] = p_line[i]

            faces = []
            for i in range(n_x - 1):
                for j in range(n_r - 1):
                    p0 = i * n_r + j
                    p1 = i * n_r + j + 1
                    p2 = (i + 1) * n_r + j + 1
                    p3 = (i + 1) * n_r + j
                    faces.append([4, p0, p1, p2, p3])
            faces = np.hstack(faces).astype(np.int64)
            cell_types = np.full(n_x - 1, pv.CellType.QUAD.value if hasattr(pv, "CellType") else 9)
            # cell types built into an UnstructuredGrid via faces + cell type list
            n_cells = (n_x - 1) * (n_r - 1)
            cell_types = np.full(n_cells, 9, dtype=np.uint8)  # VTK_QUAD = 9
            grid = pv.UnstructuredGrid(faces, cell_types, points)
            grid.point_data["V"] = V_field
            grid.point_data["p"] = p_field

            path = os.path.join(out_dir, f"nozzle_Dt_{idx:03d}.vtu")
            grid.save(path)
            paths.append(path)
    return paths


def export_mat(model, cfg, dt_values_si: List[float], n_x: int = 100, path: str = None) -> str:
    """Write a single .mat with grids X, DT and fields V, P, D, V_exact, P_exact (SI)."""
    geom, phys = cfg["geometry"], cfg["physics"]
    L, D_in = geom["L"], geom["D_in"]
    V_in, p_in, rho = phys["V_in"], phys["p_in"], phys["rho"]

    device = next(model.parameters()).device
    n_dt = len(dt_values_si)
    X = np.zeros((n_dt, n_x))
    DT = np.zeros((n_dt, n_x))
    V = np.zeros((n_dt, n_x))
    P = np.zeros((n_dt, n_x))
    D = np.zeros((n_dt, n_x))
    V_exact = np.zeros((n_dt, n_x))
    P_exact = np.zeros((n_dt, n_x))

    with torch.no_grad():
        for r, Dt_si in enumerate(dt_values_si):
            x = torch.linspace(0.0, L, n_x, dtype=torch.float64, device=device)
            Dt = torch.full_like(x, Dt_si)
            xt = (x / L).reshape(-1, 1)
            dtt = (Dt / D_in).reshape(-1, 1)
            out = model(xt, dtt)

            X[r] = x.cpu().numpy()
            DT[r] = Dt_si
            V[r] = (out[..., 0] * V_in).cpu().numpy()
            P[r] = (p_in + out[..., 1] * 0.5 * rho * V_in ** 2).cpu().numpy()
            D[r] = geo.diameter(x, Dt, D_in=D_in, L=L).cpu().numpy()
            V_exact[r] = analytical.velocity_exact_si(x, Dt, D_in=D_in, L=L, V_in=V_in).cpu().numpy()
            P_exact[r] = analytical.pressure_exact_si(x, Dt, D_in=D_in, L=L, V_in=V_in, p_in=p_in, rho=rho).cpu().numpy()

    if path is None:
        path = os.path.join(cfg["export"]["results_dir"], "matlab", "nozzle_surrogate.mat")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    savemat(path, {"X": X, "DT": DT, "V": V, "P": P, "D": D, "V_exact": V_exact, "P_exact": P_exact})
    return path
