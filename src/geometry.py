"""Nozzle geometry — README Section 3.2.

Implements the converging-diverging diameter law
    D(x; Dt) = D_in + (Dt - D_in) * sin^2(pi x / L),   x in [0, L]
and its non-dimensional counterpart. All functions accept and return
torch tensors so they are fully differentiable inside the PINN graph.
"""

from __future__ import annotations

import math

import torch


def diameter(x: torch.Tensor, dt: torch.Tensor, L: float, D_in: float) -> torch.Tensor:
    """Local diameter D(x; Dt) in SI units [m]. README Eq. geometry law."""
    return D_in + (dt - D_in) * torch.sin(math.pi * x / L) ** 2


def area(x: torch.Tensor, dt: torch.Tensor, L: float, D_in: float) -> torch.Tensor:
    """Local cross-sectional area A(x; Dt) [m^2] = pi/4 * D^2."""
    d = diameter(x, dt, L, D_in)
    return math.pi / 4.0 * d**2


def area_nondim(xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
    """Non-dimensional area A~/A_in. README Section 3.8.

    D/D_in = 1 + (dtt - 1) * sin^2(pi * xt),  A~ = (D/D_in)^2
    with xt = x/L in [0,1], dtt = Dt/D_in.
    """
    d_ratio = 1.0 + (dtt - 1.0) * torch.sin(math.pi * xt) ** 2
    return d_ratio**2


def to_nondim_x(x: torch.Tensor, L: float) -> torch.Tensor:
    """x [m] -> xt = x/L."""
    return x / L


def to_nondim_dt(dt: torch.Tensor, D_in: float) -> torch.Tensor:
    """Dt [m] -> dtt = Dt/D_in."""
    return dt / D_in


def to_si_velocity(vt: torch.Tensor, V_in: float) -> torch.Tensor:
    """V~ -> V [m/s]."""
    return vt * V_in


def to_si_pressure(pt: torch.Tensor, p_in: float, rho: float, V_in: float) -> torch.Tensor:
    """p~ -> p [Pa]. p~ = (p - p_in) / (0.5 * rho * V_in^2)."""
    return p_in + 0.5 * rho * V_in**2 * pt
