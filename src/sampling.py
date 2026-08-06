"""FILE 6 OF 9 — Deciding WHERE to check the physics.

OBJECTIVE OF THIS FILE
-----------------------
physics.py's residuals are only ever computed at specific points — they
can't literally be checked at "every point" in a continuous duct. This
file decides which points. That choice matters more than it sounds: check
only a fixed grid and the network can quietly overfit to exactly those
points while still being wrong everywhere else; check too few and you
waste the physics equations' free supervision (README Section 4.4).

WHY LATIN HYPERCUBE SAMPLING, NOT A REGULAR GRID
------------------------------------------------------
`lhs_collocation` draws random points over the 2D box
    [0, 1] (x_tilde) x [dtt_min, dtt_max] (Dt_tilde)
using Latin Hypercube Sampling (LHS) rather than either a regular grid or
plain uniform-random points. LHS guarantees the samples are spread evenly
across both axes (no accidental clumping the way pure-random sampling can
produce, and no rigid repeating pattern the way a grid does) while still
being randomized every time it's called with a new seed. Both position
AND throat diameter are sampled together, every draw — this is *the*
mechanical reason one trained network becomes a parametric surrogate
instead of a solver for one fixed geometry: it never sees just one Dt,
it sees the whole design space simultaneously, every single training step.

WHY RESAMPLE PERIODICALLY (`resample_every` in train.py)
---------------------------------------------------------------
If the network only ever gets graded at the same 4,096 fixed points, nothing
stops it from becoming very good at exactly those points while still
being wrong in between them. `train.py` calls `lhs_collocation` again
every `resample_every` epochs with a new seed, so over the course of
training the network has been checked at many different random point
sets — good performance can no longer hide between a fixed grid's cracks.

WHY 5 THROAT DIAMETERS ARE HELD OUT ENTIRELY (`validation_grid`)
-----------------------------------------------------------------------
`held_out_dt` in the config (0.225, 0.275, 0.325, 0.375, 0.425 m) are
excluded from every `lhs_collocation` draw for the whole run — the
network is never trained on them. `evaluate.py` grades exclusively on
these. A network that merely memorized the training geometries would
fail there; one that actually learned the underlying equations doesn't
care that it's a "new" Dt, because the equations hold for every Dt.
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
