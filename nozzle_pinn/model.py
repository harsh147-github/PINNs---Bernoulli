"""Neural network and hard-constrained field ansatz (README §5, §6, §8).

The network itself is an MLP  N(x_hat, s_hat, d_hat) -> (N_psi, N_p).
Its raw outputs are never used directly; they are wrapped in an *ansatz* that
satisfies mass conservation and every Dirichlet condition EXACTLY, for any
network weights.

Velocity comes from the Stokes stream function psi (axisymmetric flow):

    u = (1/r) d psi/dr,     v = -(1/r) d psi/dx      =>  du/dx + dv/dr + v/r = 0

so continuity holds identically. With s = eta^2 = (r/R(x;dt))^2 and
xi = x / L we write psi = s * H and

    H(x, s; dt) = 1/8 [ (2 - s)  +  xi^2 (1 - s)^2 N_psi ]
    p           = p_1D(x; dt)  +  (1 - xi) dt^-4 N_p
    p_1D        = (1 - U_m^2) + (32/Re) int_x^L U_m / D^2 dx'   (quasi-1D prior)

* axis  (s = 0)  : psi = 0, psi ~ r^2 (regular), v = 0
* wall  (s = 1)  : psi = 1/8 (no flow through the wall) and d psi/ds = 0
                   (no-slip): u = v = 0
* every section  : volume flux = 2 pi psi_wall = pi/4 = inlet flux, EXACTLY
* inlet (xi = 0) : psi = (2s - s^2)/8 -> u = 2 (1 - eta^2) Poiseuille, v = 0
                   (xi^2 also kills d/dx of the correction there)
* outlet (xi = 1): p = 0

With N = 0 the field is a locally fully developed pipe flow
u = 2 U_m (1 - eta^2), v = eta R' u, plus pipe-hydraulics pressure p_1D
(1-D Bernoulli + Hagen–Poiseuille friction). The network learns the
correction to this engineering prior.

To stay regular on the axis without dividing by r:
    u = 2 F_s / R^2,   v = -(r / R^2) (H_x - 2 R' F_s / R),   F_s = H + s H_s
where H_x, H_s are partial derivatives (autograd).
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
    """Parametric surrogate (x, r, dt) -> (u, v, p) with hard constraints.

    Velocity comes from a Stokes stream function, so mass is conserved
    exactly (see the module docstring).
    """

    def __init__(self, noz: Nozzle, Re: float, width: int = 64, depth: int = 6,
                 n_quad: int = 64, architecture: str = "modified_mlp") -> None:
        super().__init__()
        self.noz = noz
        self.Re = float(Re)
        net_cls = {"mlp": MLP, "modified_mlp": ModifiedMLP}[architecture]
        self.mlp = net_cls(3, 2, width, depth)          # outputs (N_psi, N_p)
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

    def stream_H(self, xh: torch.Tensor, s: torch.Tensor, dt: torch.Tensor):
        """H(x, s; dt) with psi = s H, and the pressure network output N_p."""
        xi = xh / self.noz.length
        N = self.mlp(self.normalise(xh, s, dt))
        H = 0.125 * ((2.0 - s) + xi * xi * (1.0 - s) ** 2 * N[:, 0])
        return H, N[:, 1]

    def forward(self, x: torch.Tensor, r: torch.Tensor, dt: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate the constrained fields at points (x, r) of nozzle ``dt``.

        ``x``, ``r`` and ``dt`` are 1-D tensors of equal length. Autograd
        flows through the geometry and the stream function, so derivatives of
        the returned u, v, p w.r.t. x and r are exact derivatives in physical
        coordinates. Works with or without an outer ``torch.no_grad``.
        """
        n = self.noz
        with torch.enable_grad():
            if not (x.requires_grad or r.requires_grad):
                x = x.detach().requires_grad_(True)
            R = n.radius(x, dt)
            dR = n.dradius_dx(x, dt)
            s = (r / R) ** 2
            # separate handles so autograd can give PARTIAL derivatives of H
            # w.r.t. x at fixed s and w.r.t. s at fixed x
            xh = x * 1.0
            sh = s * 1.0
            H, Np = self.stream_H(xh, sh, dt)
            H_x, H_s = torch.autograd.grad(H, (xh, sh), torch.ones_like(H), create_graph=True)
        F_s = H + sh * H_s                       # d psi / d s
        u = 2.0 * F_s / (R * R)                  # = (1/r) d psi/dr
        v = -(r / (R * R)) * (H_x - 2.0 * dR * F_s / R)   # = -(1/r) d psi/dx
        p = self.p_prior(x, dt) + (1.0 - x / n.length) * dt ** (-4) * Np
        return {"u": u, "v": v, "p": p, "psi": s * H, "R": R, "Um": (0.5 / R) ** 2}
