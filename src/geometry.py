"""Nozzle geometry: D(x; Dt), A(x; Dt), and SI <-> non-dimensional converters.

Implements README.md Eq. (nozzle law, Section 3.2):

    D(x; Dt) = D_in + (Dt - D_in) * sin^2(pi * x / L),   x in [0, L]
    A(x; Dt) = (pi / 4) * D(x; Dt)^2

and the non-dimensional form used for training (README.md Section 3.5):

    x_tilde  = x / L
    Dt_tilde = Dt / D_in
    D_tilde(x_tilde; Dt_tilde) = D(x; Dt) / D_in = 1 + (Dt_tilde - 1) * sin^2(pi * x_tilde)
    A_tilde(x_tilde; Dt_tilde) = A(x; Dt) / A_in = D_tilde(x_tilde; Dt_tilde) ** 2

Every function accepts and returns ``torch.Tensor`` and uses only differentiable
torch ops (``torch.sin``, ``torch.pow``, ...), so ``torch.autograd.grad`` can
differentiate straight through geometry when it is composed with the network
output inside a PDE residual (see physics.py).
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Dimensional (SI) geometry
# ---------------------------------------------------------------------------

def diameter(x: torch.Tensor, Dt: torch.Tensor, D_in: float, L: float) -> torch.Tensor:
    """Local nozzle diameter D(x; Dt) in metres.

    Parameters
    ----------
    x : torch.Tensor
        Axial position [m], any shape.
    Dt : torch.Tensor
        Throat diameter [m], broadcastable to ``x``'s shape.
    D_in : float
        Inlet/outlet diameter [m].
    L : float
        Nozzle length [m].
    """
    return D_in + (Dt - D_in) * torch.sin(math.pi * x / L) ** 2


def area(x: torch.Tensor, Dt: torch.Tensor, D_in: float, L: float) -> torch.Tensor:
    """Cross-sectional area A(x; Dt) = (pi/4) D(x; Dt)^2, in m^2."""
    D = diameter(x, Dt, D_in=D_in, L=L)
    return (math.pi / 4.0) * D ** 2


# ---------------------------------------------------------------------------
# Non-dimensional geometry (what the network actually trains on)
# ---------------------------------------------------------------------------

def diameter_nondim(xt: torch.Tensor, Dtt: torch.Tensor) -> torch.Tensor:
    """D_tilde(x_tilde; Dt_tilde) = 1 + (Dt_tilde - 1) * sin^2(pi * x_tilde)."""
    return 1.0 + (Dtt - 1.0) * torch.sin(math.pi * xt) ** 2


def area_nondim(xt: torch.Tensor, Dtt: torch.Tensor) -> torch.Tensor:
    """A_tilde(x_tilde; Dt_tilde) = D_tilde(x_tilde; Dt_tilde) ** 2."""
    return diameter_nondim(xt, Dtt) ** 2


# ---------------------------------------------------------------------------
# SI <-> non-dimensional converters (README.md Section 3.5)
# ---------------------------------------------------------------------------

def x_to_nondim(x: torch.Tensor, L: float) -> torch.Tensor:
    return x / L


def x_from_nondim(xt: torch.Tensor, L: float) -> torch.Tensor:
    return xt * L


def Dt_to_nondim(Dt: torch.Tensor, D_in: float) -> torch.Tensor:
    return Dt / D_in


def Dt_from_nondim(Dtt: torch.Tensor, D_in: float) -> torch.Tensor:
    return Dtt * D_in


def V_to_nondim(V: torch.Tensor, V_in: float) -> torch.Tensor:
    return V / V_in


def V_from_nondim(Vt: torch.Tensor, V_in: float) -> torch.Tensor:
    return Vt * V_in


def p_to_nondim(p: torch.Tensor, p_in: float, rho: float, V_in: float) -> torch.Tensor:
    return (p - p_in) / (0.5 * rho * V_in ** 2)


def p_from_nondim(pt: torch.Tensor, p_in: float, rho: float, V_in: float) -> torch.Tensor:
    return p_in + pt * (0.5 * rho * V_in ** 2)
