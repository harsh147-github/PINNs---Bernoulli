"""FILE 1 OF 9 — Nozzle geometry. Read this one first; it has no physics and
no neural network in it, purely the shape of the duct.

OBJECTIVE OF THIS FILE
-----------------------
Define D(x; Dt), the duct's diameter at position x for a given throat size
Dt, and its derived quantity A(x; Dt), the cross-sectional area — the one
piece of geometry every other file in this repo needs. Nothing here is
trained, guessed, or learned; it's the fixed "container" the fluid moves
through.

THE THEORY, SELF-CONTAINED (full derivation: README Section 3.2)
-------------------------------------------------------------------
The duct is a venturi: same diameter at both ends (D_in), pinched to a
narrower throat (Dt) exactly halfway along its length L. One formula
draws that whole shape with a single minimum and zero slope at the
throat (so nothing kinks, which matters later — physics.py differentiates
straight through this function):

    D(x; Dt) = D_in + (Dt - D_in) * sin^2(pi * x / L),   x in [0, L]
    A(x; Dt) = (pi/4) * D(x; Dt)^2

Check the two endpoints: sin(0) = sin(pi) = 0, so D(0) = D(L) = D_in (the
inlet and outlet match). At the midpoint x = L/2, sin(pi/2) = 1, so
D(L/2) = Dt (the throat, exactly where we want it).

WHY THIS FILE ONLY USES torch, NOT plain math/numpy
------------------------------------------------------
Every function below is called later with x wrapped in
`x.requires_grad_(True)` (see physics.py). PyTorch can only differentiate
through operations it recorded — so this shape law has to be built from
`torch.sin`, `torch.pow`, etc., not Python floats or numpy. That's the
only thing that makes this file different from a plain-Python geometry
script: it's written so it can later become part of a differentiable
chain (README Section 4.6, "automatic differentiation").

NON-DIMENSIONALIZATION (full reasoning: README Section 3.8)
----------------------------------------------------------------
The network never sees x in metres or D in metres — those get rescaled
first (x_tilde = x/L, in [0,1]; Dt_tilde = Dt/D_in, in [0.4, 0.9] for our
throat range) so every number the network handles is order-1, which is
what makes training well-behaved. `area_nondim` below is the
non-dimensional A_tilde = A/A_in that every other file actually calls.
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
