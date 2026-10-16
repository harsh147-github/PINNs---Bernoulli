"""Neural network and hard-constrained field ansatz (README §4).

The network itself is a plain MLP  N(x_hat, s_hat, d_hat) -> (N_u, N_v, N_p).
Its raw outputs are never used directly; they are wrapped in an *ansatz* that
satisfies every Dirichlet condition EXACTLY for any network weights:

    eta  = r / R(x; dt)                (0 on the axis, 1 on the wall)
    U_m  = (R_in / R)^2                (quasi-1D mean velocity, mass balance)
    xi   = x / L                       (0 at the inlet, 1 at the outlet)

    u = (1 - eta^2) U_m [ 2           + xi N_u ]
    v = eta (1 - eta^2) U_m [ 2 R'    + xi N_v ]
    p = (1 - U_m^2)   + (1 - xi) dt^-4 N_p

* wall  (eta = 1)  : u = v = 0                    (no-slip)
* inlet (xi = 0)   : u = 2 (1 - eta^2), v = 0     (Poiseuille, mean 1)
* axis  (eta = 0)  : v = 0 and u, p even in r     (symmetry)
* outlet (xi = 1)  : p = 0                        (reference pressure)

The ``2 U_m (1 - eta^2)`` / ``2 U_m R' eta (1 - eta^2)`` part is a locally
fully-developed flow that already conserves mass exactly; the network only
learns the *correction* to it, which is O(1) for every throat diameter
because each term is scaled by the local velocity scale U_m (and the pressure
correction by the throat dynamic-pressure scale dt^-4).

The network sees s = eta^2 (not eta), which makes u, v/eta and p smooth even
functions of r — the axis regularity condition comes for free.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .geometry import Nozzle


class MLP(nn.Module):
    """Fully connected tanh network with Xavier initialisation."""

    def __init__(self, n_in: int, n_out: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = width
        out = nn.Linear(d, n_out)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NozzlePINN(nn.Module):
    """Parametric surrogate (x, r, dt) -> (u, v, p) with hard boundary conditions."""

    def __init__(self, noz: Nozzle, width: int = 64, depth: int = 6) -> None:
        super().__init__()
        self.noz = noz
        self.mlp = MLP(3, 3, width, depth)

    # ------------------------------------------------------------------
    def normalise(self, x: torch.Tensor, s: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Map (x, eta^2, dt) to [-1, 1]^3, the range tanh networks like."""
        n = self.noz
        xh = 2.0 * x / n.length - 1.0
        sh = 2.0 * s - 1.0
        dh = 2.0 * (dt - n.dt_min) / (n.dt_max - n.dt_min) - 1.0
        return torch.stack([xh, sh, dh], dim=-1)

    def forward(self, x: torch.Tensor, r: torch.Tensor, dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the constrained fields at points (x, r) of nozzle ``dt``.

        ``x``, ``r`` and ``dt`` are 1-D tensors of equal length. Autograd
        flows through the geometry, so derivatives w.r.t. x and r of the
        returned fields are exact derivatives in physical coordinates.
        """
        n = self.noz
        R = n.radius(x, dt)
        dR = n.dradius_dx(x, dt)
        eta = r / R
        s = eta * eta
        Um = (0.5 / R) ** 2
        xi = x / n.length
        N = self.mlp(self.normalise(x, s, dt))
        Nu, Nv, Np = N[:, 0], N[:, 1], N[:, 2]
        wall = 1.0 - s
        u = wall * Um * (2.0 + xi * Nu)
        v = eta * wall * Um * (2.0 * dR + xi * Nv)
        p = (1.0 - Um * Um) + (1.0 - xi) * dt ** (-4) * Np
        return {"u": u, "v": v, "p": p, "R": R, "Um": Um}
