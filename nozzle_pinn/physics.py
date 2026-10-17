"""Governing equations: steady, incompressible, axisymmetric Navier–Stokes.

Non-dimensional form (lengths / D_in, velocity / U_mean_in, p / rho U^2),
cylindrical coordinates (x axial, r radial), no swirl  — README §3, §7:

    continuity :  du/dx + dv/dr + v/r                                 = 0
    x-momentum :  u du/dx + v du/dr + dp/dx - (1/Re) (u_xx + u_rr + u_r/r)       = 0
    r-momentum :  u dv/dx + v dv/dr + dp/dr - (1/Re) (v_xx + v_rr + v_r/r - v/r^2) = 0

Every derivative is computed by automatic differentiation of the network
(``torch.autograd.grad`` with ``create_graph=True``), so the residuals are
exact for the function the network represents — no mesh, no finite
differences.

Residuals are divided by local scales (U_m / R for continuity and
U_m^2 / R + U_m / (Re R^2) for momentum) so that a nozzle with a 6x faster
throat does not dominate the loss: every design contributes equally.
"""
from __future__ import annotations

from typing import Dict

import torch

from .model import NozzlePINN


def _grad(f: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """d f / d X for a batch, keeping the graph for higher derivatives."""
    return torch.autograd.grad(f, X, torch.ones_like(f), create_graph=True)[0]


def ns_residuals(model: NozzlePINN, X: torch.Tensor, dt: torch.Tensor, Re: float,
                 scale: str = "local") -> Dict[str, torch.Tensor]:
    """Scaled continuity / momentum residuals at collocation points.

    Parameters
    ----------
    model : NozzlePINN
    X : (N, 2) tensor of (x, r) with ``requires_grad=True``
    dt : (N,) throat-diameter ratio of each point's nozzle
    Re : Reynolds number
    scale : "local" divides by the local section scales (U_m, R); "design"
        divides every point of a nozzle by the same throat scales
        (U_t = dt^-2, R_t = dt/2), so momentum imbalance is weighted
        uniformly along the pipe — the pressure drop is the integral of it.
    """
    out = model(X[:, 0], X[:, 1], dt)
    u, v, p, R, Um = out["u"], out["v"], out["p"], out["R"], out["Um"]
    r = X[:, 1]
    gu, gv, gp = _grad(u, X), _grad(v, X), _grad(p, X)
    u_x, u_r = gu[:, 0], gu[:, 1]
    v_x, v_r = gv[:, 0], gv[:, 1]
    p_x, p_r = gp[:, 0], gp[:, 1]
    u_xx = _grad(u_x, X)[:, 0]
    u_rr = _grad(u_r, X)[:, 1]
    v_xx = _grad(v_x, X)[:, 0]
    v_rr = _grad(v_r, X)[:, 1]
    nu = 1.0 / Re

    cont = u_x + v_r + v / r
    mom_x = u * u_x + v * u_r + p_x - nu * (u_xx + u_rr + u_r / r)
    mom_r = u * v_x + v * v_r + p_r - nu * (v_xx + v_rr + v_r / r - v / (r * r))

    if scale == "design":
        Um, R = dt ** (-2), 0.5 * dt
    sc = Um / R
    sm = Um * Um / R + nu * Um / (R * R)
    return {"cont": cont / sc, "mom_x": mom_x / sm, "mom_r": mom_r / sm}


def outlet_residuals(model: NozzlePINN, X: torch.Tensor, dt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Soft outflow condition at x = L: fully developed, du/dx = 0 and v = 0."""
    out = model(X[:, 0], X[:, 1], dt)
    u_x = _grad(out["u"], X)[:, 0]
    return {"dudx": u_x, "v": out["v"]}
