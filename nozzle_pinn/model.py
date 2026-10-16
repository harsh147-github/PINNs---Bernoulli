"""Neural network and hard-constrained field ansatz (README §4).

The network itself is a plain MLP  N(x_hat, s_hat, d_hat) -> (N_u, N_v, N_p).
Its raw outputs are never used directly; they are wrapped in an *ansatz* that
satisfies every Dirichlet condition EXACTLY for any network weights:

    eta  = r / R(x; dt)                (0 on the axis, 1 on the wall)
    U_m  = (R_in / R)^2                (quasi-1D mean velocity, mass balance)
    xi   = x / L                       (0 at the inlet, 1 at the outlet)

    u = (1 - eta^2) U_m [ 2           + xi N_u ]
    v = eta (1 - eta^2) U_m [ 2 R'    + xi N_v ]
    p = p_1D(x; dt)   + (1 - xi) dt^-4 N_p

    p_1D = (1 - U_m^2) + (32/Re) int_x^L U_m / D^2 dx'     (quasi-1D prior)

* wall  (eta = 1)  : u = v = 0                    (no-slip)
* inlet (xi = 0)   : u = 2 (1 - eta^2), v = 0     (Poiseuille, mean 1)
* axis  (eta = 0)  : v = 0 and u, p even in r     (symmetry)
* outlet (xi = 1)  : p = 0                        (reference pressure)

The ``2 U_m (1 - eta^2)`` / ``2 U_m R' eta (1 - eta^2)`` part is a locally
fully-developed flow that already conserves mass exactly, and ``p_1D`` is the
classic pipe-hydraulics estimate (1-D Bernoulli + Hagen–Poiseuille friction,
zero at the outlet; exact for the straight pipe). The network only learns the
*correction* to this engineering prior, which is O(1) for every throat
diameter because each term is scaled by the local velocity scale U_m (and the
pressure correction by the throat dynamic-pressure scale dt^-4).

The network sees s = eta^2 (not eta), which makes u, v/eta and p smooth even
functions of r — the axis regularity condition comes for free.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
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


class ModifiedMLP(nn.Module):
    """'Modified MLP' of Wang, Teng & Perdikaris (2021, SIAM J. Sci. Comput.).

    Two encoders U = tanh(W_u z), V = tanh(W_v z) are mixed into every hidden
    layer:  H <- (1 - Z) * U + Z * V  with  Z = tanh(W_k H).  The multiplicative
    skip connections mitigate the unbalanced back-propagated gradients that make
    plain MLP PINNs stall on Navier–Stokes problems.
    """

    def __init__(self, n_in: int, n_out: int, width: int, depth: int) -> None:
        super().__init__()
        self.enc_u = nn.Linear(n_in, width)
        self.enc_v = nn.Linear(n_in, width)
        self.inp = nn.Linear(n_in, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.out = nn.Linear(width, n_out)
        for lin in [self.enc_u, self.enc_v, self.inp, *self.hidden, self.out]:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        U = torch.tanh(self.enc_u(z))
        V = torch.tanh(self.enc_v(z))
        H = torch.tanh(self.inp(z))
        H = (1.0 - H) * U + H * V
        for lin in self.hidden:
            Z = torch.tanh(lin(H))
            H = (1.0 - Z) * U + Z * V
        return self.out(H)


class NozzlePINN(nn.Module):
    """Parametric surrogate (x, r, dt) -> (u, v, p) with hard boundary conditions."""

    def __init__(self, noz: Nozzle, Re: float, width: int = 64, depth: int = 6,
                 n_quad: int = 64, architecture: str = "modified_mlp",
                 correction_scale: str = "local") -> None:
        super().__init__()
        self.correction_scale = correction_scale
        self.noz = noz
        self.Re = float(Re)
        net_cls = {"mlp": MLP, "modified_mlp": ModifiedMLP}[architecture]
        self.mlp = net_cls(3, 3, width, depth)
        # Zero output layer: training starts EXACTLY from the engineering prior
        # (locally fully developed flow + quasi-1D pressure) instead of noise.
        last = self.mlp.out if architecture == "modified_mlp" else self.mlp.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.last_layer = last
        t, w = np.polynomial.legendre.leggauss(n_quad)
        self.register_buffer("gl_t", torch.as_tensor(t, dtype=torch.get_default_dtype()))
        self.register_buffer("gl_w", torch.as_tensor(w, dtype=torch.get_default_dtype()))

    def p_prior(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Quasi-1D pressure: Bernoulli on the mean velocity + Poiseuille friction.

        The friction integral int_x^L 32 U_m / (Re D^2) dx' is evaluated with
        Gauss–Legendre quadrature on [x, L] so it stays differentiable in x.
        """
        n = self.noz
        L = n.length
        xq = x[:, None] + (L - x)[:, None] * 0.5 * (1.0 + self.gl_t[None, :])
        D = 2.0 * n.radius(xq, dt[:, None])
        f = 32.0 / self.Re * n.mean_velocity(xq, dt[:, None]) / (D * D)
        friction = 0.5 * (L - x) * (f * self.gl_w[None, :]).sum(dim=1)
        Um = n.mean_velocity(x, dt)
        return (1.0 - Um * Um) + friction

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
        # scale of the learned correction: local mean velocity, or the jet
        # velocity (stays at the throat value downstream of the throat)
        Us = Um if self.correction_scale == "local" else n.jet_velocity(x, dt)
        u = wall * (2.0 * Um + xi * Us * Nu)
        v = eta * wall * (2.0 * Um * dR + xi * Us * Nv)
        p = self.p_prior(x, dt) + (1.0 - xi) * dt ** (-4) * Np
        return {"u": u, "v": v, "p": p, "R": R, "Um": Um}
