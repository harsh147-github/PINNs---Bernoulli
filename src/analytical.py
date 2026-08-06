"""Closed-form analytical solution — README Eq. (4), Section 3.4/3.5.

This is the ground truth the PINN is validated against. No CFD data anywhere.

Non-dimensional:  V~ = 1/A~,   p~ = 1 - V~^2
SI:               V  = V_in * A_in/A,   p = p_in + 0.5*rho*(V_in^2 - V^2)
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


# ---------------------------------------------------------------------------
# Stage 2 — exact quasi-1D isentropic subsonic solution (CLAUDE.md Section 6).
#
# The inlet is subsonic and slow (V_in=10 m/s, T_in=300K => M_in ~ 0.029), so
# the whole nozzle stays subsonic even at the narrowest throat in the design
# space. We fix the isentropic stagnation state from the inlet static state
# and M_in, derive the (constant) reference sonic area A* from the inlet
# area-Mach relation, then solve the area-Mach relation at every station for
# the local Mach number M(x) via a bracketed subsonic root (brentq).
# Every quantity is then non-dimensionalized as a RATIO TO ITS INLET VALUE
# (rho/rho_in, V/V_in, p/p_in, T/T_in) — see physics.residuals_euler for why.
# ---------------------------------------------------------------------------

import numpy as np
from scipy.optimize import brentq


def area_mach_ratio(M: float, gamma: float) -> float:
    """(A/A*)(M): the isentropic area-Mach relation (CLAUDE.md Section 6)."""
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M**2)
    return (1.0 / M) * term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def mach_from_area_ratio(area_ratio: float, gamma: float = 1.4) -> float:
    """Subsonic Mach number M solving A/A* = area_ratio (area_ratio >= 1).

    Bracketed subsonic root [1e-6, 1-1e-6] via `scipy.optimize.brentq`
    (CLAUDE.md Section 6: "implement the subsonic root with
    scipy.optimize.brentq, bracket [1e-6, 1-1e-6]").
    """
    if area_ratio < 1.0 - 1e-9:
        raise ValueError(f"area_ratio must be >= 1 (got {area_ratio}); A* is the sonic area.")
    if abs(area_ratio - 1.0) < 1e-12:
        return 1.0 - 1e-6
    return brentq(lambda M: area_mach_ratio(M, gamma) - area_ratio, 1e-6, 1.0 - 1e-6, xtol=1e-12)


def mach_from_area_ratio_array(area_ratio: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Element-wise `mach_from_area_ratio` (no closed form for the inverse)."""
    area_ratio = np.atleast_1d(np.asarray(area_ratio, dtype=np.float64))
    return np.array([mach_from_area_ratio(float(ar), gamma=gamma) for ar in area_ratio])


def stage2_exact_state_si(x_si: np.ndarray, dt_si: float, cfg: dict):
    """Exact isentropic subsonic (rho, V, p, T) at stations x_si [m] for throat dt_si [m].

    Returns SI arrays: rho [kg/m^3], V [m/s], p [Pa], T [K].
    """
    g, ph = cfg["geometry"], cfg["physics"]
    L, D_in = g["L"], g["D_in"]
    V_in, p_in, T_in = ph["V_in"], ph["p_in"], ph["T_in"]
    gamma, R = ph["gamma"], ph["R"]

    x_si = np.atleast_1d(np.asarray(x_si, dtype=np.float64))
    D_x = D_in + (dt_si - D_in) * np.sin(np.pi * x_si / L) ** 2
    A_x = (np.pi / 4.0) * D_x**2
    A_in = (np.pi / 4.0) * D_in**2

    rho_in = p_in / (R * T_in)
    a_in = np.sqrt(gamma * R * T_in)
    M_in = V_in / a_in

    A_star = A_in / area_mach_ratio(M_in, gamma)
    area_ratio = np.maximum(A_x / A_star, 1.0)  # clip tiny numerical dips below 1 at the throat
    M_x = mach_from_area_ratio_array(area_ratio, gamma=gamma)

    T0 = T_in * (1.0 + 0.5 * (gamma - 1.0) * M_in**2)
    p0 = p_in * (T0 / T_in) ** (gamma / (gamma - 1.0))
    rho0 = rho_in * (T0 / T_in) ** (1.0 / (gamma - 1.0))

    T_x = T0 / (1.0 + 0.5 * (gamma - 1.0) * M_x**2)
    p_x = p0 * (T_x / T0) ** (gamma / (gamma - 1.0))
    rho_x = rho0 * (T_x / T0) ** (1.0 / (gamma - 1.0))
    V_x = M_x * np.sqrt(gamma * R * T_x)
    return rho_x, V_x, p_x, T_x


def stage2_exact_state_nondim(x_si: np.ndarray, dt_si: float, cfg: dict):
    """Same as `stage2_exact_state_si` but as ratios to the inlet state
    (rho/rho_in, V/V_in, p/p_in, T/T_in) — matches physics.residuals_euler."""
    rho_x, V_x, p_x, T_x = stage2_exact_state_si(x_si, dt_si, cfg)
    ph = cfg["physics"]
    rho_in = ph["p_in"] / (ph["R"] * ph["T_in"])
    return rho_x / rho_in, V_x / ph["V_in"], p_x / ph["p_in"], T_x / ph["T_in"]
