"""FILE 2 OF 9 — The exact answer. Depends only on file 1 (geometry.py).

OBJECTIVE OF THIS FILE
-----------------------
Write down the correct velocity and pressure at every point in the duct,
in closed form — no network, no training, just algebra. This file is
never used *during* training (the network never gets to see it). It
exists purely so file 8 (evaluate.py) can grade the trained network
honestly afterward: run the network, run this file, compare.

WHY WE CAN SOLVE IT BY HAND AT ALL (full derivation: README Section 3.4-3.7)
--------------------------------------------------------------------------------
Two physical laws pin down the flow completely for water in this duct:
continuity (mass in = mass out, at every cross-section) and Bernoulli's
theorem (pressure trades off against velocity along the flow, derived from
F=ma on a slug of fluid). Solving them together by hand:

    continuity:  A(x) V(x) = A_in V_in           =>  V(x) = V_in * A_in / A(x)
    Bernoulli:   p + 0.5 rho V^2 = p_in + 0.5 rho V_in^2
                                                   =>  p(x) = p_in + 0.5*rho*(V_in^2 - V(x)^2)

In the non-dimensional variables every other file actually trains and
tests in (x_tilde = x/L, V_tilde = V/V_in, p_tilde = (p-p_in)/(0.5 rho
V_in^2) — README Section 3.8), these collapse to almost nothing:

    V_tilde = 1 / A_tilde(x_tilde; Dt_tilde)
    p_tilde = 1 - V_tilde^2

`velocity_exact_nondim` / `pressure_exact_nondim` below are exactly these
two lines. The `_si` versions below are the same idea in SI units.
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
