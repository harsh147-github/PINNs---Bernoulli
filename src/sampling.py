"""Collocation, boundary, and validation point samplers (CLAUDE.md Section 6).

Collocation points are drawn with Latin Hypercube Sampling (`scipy.stats.qmc`)
rather than a uniform grid, and are periodically re-drawn during training
(`resample_every`) so the network never overfits to one fixed point set
(README Section 9, "resampling beats fixed collocation" per the stiff-PDE
PINN benchmark literature).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.stats import qmc


def _lhs_1d(n: int, low: float, high: float, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n)[:, 0]
    return low + u * (high - low)


def lhs_collocation(n_x: int, n_dt: int, seed: int, dt_min: float = 0.4, dt_max: float = 0.9,
                     dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """N_f = n_x * n_dt collocation points over (x_tilde, Dt_tilde) in [0,1] x [dt_min, dt_max].

    `n_dt` throat-diameter values are drawn by 1D LHS; for each, `n_x` axial
    positions are drawn by an independent 1D LHS draw, then combined as a
    Cartesian product (README Section 9: "512 in x_tilde x 8 LHS values of
    Dt_tilde"). Returns column tensors of shape (n_x*n_dt, 1).
    """
    dt_values = _lhs_1d(n_dt, dt_min, dt_max, seed=seed)
    xt_all, dtt_all = [], []
    for k, dt in enumerate(dt_values):
        xt_k = _lhs_1d(n_x, 0.0, 1.0, seed=seed + 1 + k)
        xt_all.append(xt_k)
        dtt_all.append(np.full(n_x, dt))
    xt = np.concatenate(xt_all)
    dtt = np.concatenate(dtt_all)
    xt_t = torch.tensor(xt, dtype=dtype).reshape(-1, 1)
    dtt_t = torch.tensor(dtt, dtype=dtype).reshape(-1, 1)
    return xt_t, dtt_t


def boundary_points(n: int, seed: int, dt_min: float = 0.4, dt_max: float = 0.9,
                     dtype: torch.dtype = torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """n boundary points at x_tilde=0, with Dt_tilde drawn by 1D LHS (soft-BC mode only)."""
    dt_values = _lhs_1d(n, dt_min, dt_max, seed=seed)
    xt_t = torch.zeros(n, 1, dtype=dtype)
    dtt_t = torch.tensor(dt_values, dtype=dtype).reshape(-1, 1)
    return xt_t, dtt_t


def validation_grid(n_total: int, dt_values_si: list, D_in: float, dtype: torch.dtype = torch.float64
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed validation set: evenly spaced x_tilde for each held-out Dt (SI, meters).

    `n_total` points are split evenly across `len(dt_values_si)` throat
    diameters (README Section 10, "generalization" check on held-out Dt).
    """
    n_dt = len(dt_values_si)
    n_per_dt = n_total // n_dt
    xt_all, dtt_all = [], []
    for dt_si in dt_values_si:
        dtt = dt_si / D_in
        xt_k = np.linspace(0.0, 1.0, n_per_dt)
        xt_all.append(xt_k)
        dtt_all.append(np.full(n_per_dt, dtt))
    xt = np.concatenate(xt_all)
    dtt = np.concatenate(dtt_all)
    xt_t = torch.tensor(xt, dtype=dtype).reshape(-1, 1)
    dtt_t = torch.tensor(dtt, dtype=dtype).reshape(-1, 1)
    return xt_t, dtt_t


def seed_everything(seed: int) -> None:
    """Seed torch, numpy, and python's random module (CLAUDE.md Section 6.5)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
