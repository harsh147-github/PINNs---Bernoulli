"""Closed-form analytical solution — README Eq. (4), Section 3.4/3.5.

This is the ground truth the PINN is validated against. No CFD data anywhere.

Non-dimensional:  V~ = 1/A~,   p~ = 1 - V~^2
SI:               V  = V_in * A_in/A,   p = p_in + 0.5*rho*(V_in^2 - V^2)
"""

from __future__ import annotations

import torch

from src.geometry import area_nondim


def velocity_exact_nondim(xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
    """V~(xt, dtt) = 1 / A~ — continuity + inlet BC V~(0)=1."""
    return 1.0 / area_nondim(xt, dtt)


def pressure_exact_nondim(xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
    """p~(xt, dtt) = 1 - V~^2 — Bernoulli + inlet BC p~(0)=0."""
    v = velocity_exact_nondim(xt, dtt)
    return 1.0 - v**2


def velocity_exact_si(x: torch.Tensor, dt: torch.Tensor, cfg: dict) -> torch.Tensor:
    """V(x; Dt) [m/s] = V_in * A_in / A(x)."""
    from src.geometry import area

    a_in = torch.as_tensor(cfg["geometry"]["D_in"]) ** 2  # proportional; ratio is what matters
    a = area(x, dt, cfg["geometry"]["L"], cfg["geometry"]["D_in"])
    return cfg["physics"]["V_in"] * (a_in / (a / (torch.pi / 4.0)))


def pressure_exact_si(x: torch.Tensor, dt: torch.Tensor, cfg: dict) -> torch.Tensor:
    """p(x; Dt) [Pa] = p_in + 0.5*rho*(V_in^2 - V^2)."""
    v = velocity_exact_si(x, dt, cfg)
    return cfg["physics"]["p_in"] + 0.5 * cfg["physics"]["rho"] * (cfg["physics"]["V_in"] ** 2 - v**2)
