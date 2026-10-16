"""Parametric converging–diverging nozzle geometry (README §3.1).

The wall radius is a C2-smooth function of the axial position ``x`` and the
throat diameter ratio ``dt = D_throat / D_in``::

    R(x; dt) = 0.5 * [1 + (dt - 1) * Phi(x)]
    Phi(x)   = S((x - x1) / Lc) - S((x - x3) / Ld)
    S(t)     = 6 t^5 - 15 t^4 + 10 t^3   (t clipped to [0, 1])

``Phi`` rises 0 -> 1 through the converging section, stays 1 along the throat
and falls back to 0 through the diffuser. ``S`` ("smootherstep") has zero
first AND second derivative at both ends, so R, R' and R'' are continuous —
required because the PINN residual differentiates the wall shape twice.

All functions work on NumPy arrays and on torch tensors (autograd-friendly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


def _clip01(t):
    """Clip to [0, 1] for either numpy or torch input."""
    if isinstance(t, torch.Tensor):
        return torch.clamp(t, 0.0, 1.0)
    return np.clip(t, 0.0, 1.0)


def smootherstep_dt(t):
    """Derivative dS/dt = 30 t^2 (1 - t)^2 (zero outside [0, 1])."""
    t = _clip01(t)
    return 30.0 * t * t * (1.0 - t) * (1.0 - t)


def smootherstep(t):
    """C2 step 0 -> 1 on [0, 1]: S = 6t^5 - 15t^4 + 10t^3."""
    t = _clip01(t)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


@dataclass(frozen=True)
class Nozzle:
    """Axial layout of the nozzle (lengths in inlet diameters)."""

    L_inlet: float
    L_converge: float
    L_throat: float
    L_diverge: float
    L_outlet: float
    dt_min: float
    dt_max: float
    dt_baseline: float

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Nozzle":
        g = cfg["geometry"]
        return cls(**{k: float(g[k]) for k in cls.__dataclass_fields__})

    # --- key axial stations ------------------------------------------------
    @property
    def x1(self) -> float:  # start of converging section
        return self.L_inlet

    @property
    def x2(self) -> float:  # start of throat
        return self.x1 + self.L_converge

    @property
    def x3(self) -> float:  # end of throat / start of diffuser
        return self.x2 + self.L_throat

    @property
    def x4(self) -> float:  # end of diffuser
        return self.x3 + self.L_diverge

    @property
    def length(self) -> float:  # total pipe length
        return self.x4 + self.L_outlet

    @property
    def x_throat_mid(self) -> float:
        return 0.5 * (self.x2 + self.x3)

    # --- shape ---------------------------------------------------------------
    def phi(self, x):
        """Shape function: 0 in inlet/outlet pipes, 1 in the throat."""
        return smootherstep((x - self.x1) / self.L_converge) - smootherstep(
            (x - self.x3) / self.L_diverge
        )

    def dphi_dx(self, x):
        """Analytic derivative of the shape function."""
        return (smootherstep_dt((x - self.x1) / self.L_converge) / self.L_converge
                - smootherstep_dt((x - self.x3) / self.L_diverge) / self.L_diverge)

    def radius(self, x, dt):
        """Wall radius R(x; dt) (inlet radius = 0.5)."""
        return 0.5 * (1.0 + (dt - 1.0) * self.phi(x))

    def dradius_dx(self, x, dt):
        """Wall slope R'(x; dt)."""
        return 0.5 * (dt - 1.0) * self.dphi_dx(x)

    def jet_velocity(self, x, dt):
        """Velocity scale of the core flow, U_s = (R_in / R_c)^2.

        R_c follows the wall through the converging section but stays at the
        throat radius downstream: the jet leaving the throat keeps (roughly)
        its throat velocity through the diffuser, while the *mean* velocity
        U_m drops back to 1. Used to scale the network's velocity correction.
        """
        Rc = 0.5 * (1.0 + (dt - 1.0) * smootherstep((x - self.x1) / self.L_converge))
        return (0.5 / Rc) ** 2

    def mean_velocity(self, x, dt):
        """Quasi-1D mean velocity U_m = (R_in / R)^2 from mass conservation."""
        return (0.5 / self.radius(x, dt)) ** 2
