"""Collocation-point sampling — README Sections 7 and 9.

Latin Hypercube Sampling over the (xt, dtt) design space
    [0, 1] x [dtt_min, dtt_max]
so the network sees positions AND throat diameters simultaneously — this is
what makes the model a parametric surrogate instead of a single-case solver.
Resampling periodically during training prevents overfitting to any fixed
point set.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc


def lhs_collocation(
    n_points: int, dtt_min: float, dtt_max: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """LHS pairs (xt, dtt) as float64 column tensors."""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    pts = sampler.random(n_points)
    xt = pts[:, 0:1]  # [0, 1]
    dtt = dtt_min + pts[:, 1:2] * (dtt_max - dtt_min)
    return (
        torch.as_tensor(xt, dtype=torch.float64),
        torch.as_tensor(dtt, dtype=torch.float64),
    )


def boundary_points(n: int, dtt_min: float, dtt_max: float, seed: int) -> torch.Tensor:
    """Throat-diameter samples for the soft-BC term at xt = 0."""
    rng = np.random.default_rng(seed)
    dtt = rng.uniform(dtt_min, dtt_max, size=(n, 1))
    return torch.as_tensor(dtt, dtype=torch.float64)


def validation_grid(
    held_out_dtt: list[float], n_x: int = 400
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed validation grid over x at held-out throat diameters only."""
    xt_1d = np.linspace(0.0, 1.0, n_x)
    xt_all = np.concatenate([xt_1d for _ in held_out_dtt])[:, None]
    dtt_all = np.concatenate([[d] * n_x for d in held_out_dtt])[:, None]
    return (
        torch.as_tensor(xt_all, dtype=torch.float64),
        torch.as_tensor(dtt_all, dtype=torch.float64),
    )
