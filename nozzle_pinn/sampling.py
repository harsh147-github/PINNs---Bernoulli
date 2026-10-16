"""Collocation-point sampling over the whole design space (README §5.1).

A point is a triple (x, r, dt): an axial station, a radius inside THAT
nozzle's wall, and the throat ratio of the nozzle it belongs to. Sampling dt
together with (x, r) is what makes the network *parametric*: one training run
sees thousands of different nozzles.

Densities:
* dt      : uniform on [dt_min, dt_max]
* x       : mixture — uniform on the pipe plus a share concentrated in the
            converge/throat/diffuser region where gradients live
* eta^2   : uniform (= uniform in cross-section area) plus a share clustered in
            the wall layer 0.75 < eta < 1 where shear is largest
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from .geometry import Nozzle

ETA2_MIN = 1.0e-6  # keep r > 0: the axisymmetric operator has 1/r terms


def sample_interior(noz: Nozzle, n: int, wall_frac: float, throat_frac: float,
                    gen: torch.Generator, dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return X = (x, r) of shape (n, 2) and dt of shape (n,)."""
    rnd = lambda m: torch.rand(m, generator=gen, dtype=dtype)  # noqa: E731
    dt = noz.dt_min + (noz.dt_max - noz.dt_min) * rnd(n)

    n_th = int(throat_frac * n)
    x = torch.cat([noz.length * rnd(n - n_th),
                   noz.x1 + (noz.x4 - noz.x1) * rnd(n_th)])
    x = x[torch.randperm(n, generator=gen)]

    n_w = int(wall_frac * n)
    eta2 = torch.cat([ETA2_MIN + (1.0 - ETA2_MIN) * rnd(n - n_w),
                      (0.75 + 0.25 * rnd(n_w)) ** 2])
    eta2 = eta2[torch.randperm(n, generator=gen)]
    eta2 = eta2.clamp(max=1.0 - 1.0e-9)

    r = noz.radius(x, dt) * torch.sqrt(eta2)
    return torch.stack([x, r], dim=1), dt


def sample_outlet(noz: Nozzle, n: int, gen: torch.Generator,
                  dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Points on the outlet plane x = L."""
    dt = noz.dt_min + (noz.dt_max - noz.dt_min) * torch.rand(n, generator=gen, dtype=dtype)
    eta2 = ETA2_MIN + (1.0 - ETA2_MIN) * torch.rand(n, generator=gen, dtype=dtype)
    x = torch.full((n,), noz.length, dtype=dtype)
    r = noz.radius(x, dt) * torch.sqrt(eta2)
    return torch.stack([x, r], dim=1), dt


def draw(noz: Nozzle, cfg: Dict[str, Any], gen: torch.Generator, n_interior: int | None = None):
    """One full collocation set (interior + outlet) from the config."""
    s = cfg["sampling"]
    Xi, di = sample_interior(noz, n_interior or int(s["n_interior"]), float(s["wall_fraction"]),
                             float(s["throat_fraction"]), gen)
    Xo, do = sample_outlet(noz, int(s["n_outlet"]), gen)
    return Xi.requires_grad_(True), di, Xo.requires_grad_(True), do
