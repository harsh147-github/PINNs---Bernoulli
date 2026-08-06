"""Closed-form ground truth used to validate the PINN (README.md Sections 3.4, 3.6, 10).

Stage 1 (incompressible Bernoulli, Eq. 4 of README.md):

    V(x) = V_in * A_in / A(x)          V_tilde = 1 / A_tilde
    p(x) = p_in + 0.5*rho*(V_in^2 - V(x)^2)     p_tilde = 1 - V_tilde^2

Stage 2 (compressible, subsonic isentropic area-Mach relation):

    A / A*  =  (1/M) * [ (2/(gamma+1)) * (1 + (gamma-1)/2 * M^2) ] ** ((gamma+1)/(2*(gamma-1)))

solved for the subsonic root M in (0, 1) via ``scipy.optimize.brentq``.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import brentq

from src import geometry as geo


# ---------------------------------------------------------------------------
# Stage 1: incompressible Bernoulli, non-dimensional and SI
# ---------------------------------------------------------------------------

def velocity_exact_nondim(xt: torch.Tensor, Dtt: torch.Tensor) -> torch.Tensor:
    """V_tilde(x_tilde; Dt_tilde) = 1 / A_tilde(x_tilde; Dt_tilde)  (README Eq. 4, nondim)."""
    return 1.0 / geo.area_nondim(xt, Dtt)


def pressure_exact_nondim(xt: torch.Tensor, Dtt: torch.Tensor) -> torch.Tensor:
    """p_tilde(x_tilde; Dt_tilde) = 1 - V_tilde^2  (README Eq. 4, nondim)."""
    Vt = velocity_exact_nondim(xt, Dtt)
    return 1.0 - Vt ** 2


def velocity_exact_si(x: torch.Tensor, Dt: torch.Tensor, D_in: float, L: float,
                       V_in: float) -> torch.Tensor:
    """V(x; Dt) = V_in * A_in / A(x; Dt), in m/s (README Eq. 4, SI)."""
    A_in = (np.pi / 4.0) * D_in ** 2
    A = geo.area(x, Dt, D_in=D_in, L=L)
    return V_in * A_in / A


def pressure_exact_si(x: torch.Tensor, Dt: torch.Tensor, D_in: float, L: float,
                       V_in: float, p_in: float, rho: float) -> torch.Tensor:
    """p(x; Dt) = p_in + 0.5*rho*(V_in^2 - V(x)^2), in Pa (README Eq. 4, SI)."""
    V = velocity_exact_si(x, Dt, D_in=D_in, L=L, V_in=V_in)
    return p_in + 0.5 * rho * (V_in ** 2 - V ** 2)


# ---------------------------------------------------------------------------
# Stage 2: isentropic area-Mach relation, subsonic branch
# ---------------------------------------------------------------------------

def area_mach_ratio(M: float, gamma: float) -> float:
    """(A/A*)(M), the isentropic area-Mach relation (README Section 3.6)."""
    term = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * M ** 2)
    return (1.0 / M) * term ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def _area_mach_relation(M: float, area_ratio: float, gamma: float) -> float:
    """f(M) = (A/A*)(M) - area_ratio; root f(M) = 0 gives the physical Mach number."""
    return area_mach_ratio(M, gamma) - area_ratio


def mach_from_area_ratio(area_ratio: float, gamma: float = 1.4) -> float:
    """Subsonic Mach number M in (0, 1) solving A/A* = area_ratio (area_ratio >= 1).

    Uses ``scipy.optimize.brentq`` bracketed on the subsonic branch [1e-6, 1-1e-6],
    per CLAUDE.md Section 3 ("subsonic branch, bracket [1e-6, 1-1e-6]").
    """
    if area_ratio < 1.0 - 1e-9:
        raise ValueError(f"area_ratio must be >= 1 (got {area_ratio}); A* is the throat area.")
    if abs(area_ratio - 1.0) < 1e-12:
        return 1.0 - 1e-6
    return brentq(_area_mach_relation, 1e-6, 1.0 - 1e-6, args=(area_ratio, gamma), xtol=1e-12)


def mach_from_area_ratio_array(area_ratio: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Vectorized subsonic Mach solve (element-wise brentq; A/A* has no closed form)."""
    area_ratio = np.atleast_1d(np.asarray(area_ratio, dtype=np.float64))
    out = np.empty_like(area_ratio)
    for i, ar in enumerate(area_ratio):
        out[i] = mach_from_area_ratio(float(ar), gamma=gamma)
    return out


# ---------------------------------------------------------------------------
# Stage 2: exact quasi-1D isentropic subsonic state, ratio-to-inlet nondim
# (same convention as physics.residuals_euler: rho/rho_in, V/V_in, p/p_in, T/T_in)
# ---------------------------------------------------------------------------

def stage2_exact_state_si(x_si: np.ndarray, Dt_si: float, D_in: float, L: float,
                           V_in: float, p_in: float, T_in: float, gamma: float, R: float):
    """Exact isentropic subsonic (rho, V, p, T) at stations x_si for throat Dt_si.

    Uses the inlet static state + inlet Mach number to fix the isentropic
    stagnation state and the reference sonic area A*, then solves the area-Mach
    relation (README Eq. area-Mach, Section 3.6) at every station. Returns SI
    arrays (rho [kg/m^3], V [m/s], p [Pa], T [K]).
    """
    x_si = np.atleast_1d(np.asarray(x_si, dtype=np.float64))
    A_in = (np.pi / 4.0) * D_in ** 2
    D_x = D_in + (Dt_si - D_in) * np.sin(np.pi * x_si / L) ** 2
    A_x = (np.pi / 4.0) * D_x ** 2

    rho_in = p_in / (R * T_in)
    a_in = np.sqrt(gamma * R * T_in)
    M_in = V_in / a_in

    A_star = A_in / area_mach_ratio(M_in, gamma)
    area_ratio = np.maximum(A_x / A_star, 1.0)  # clip tiny numerical dips below 1 at the throat
    M_x = mach_from_area_ratio_array(area_ratio, gamma=gamma)

    T0 = T_in * (1.0 + 0.5 * (gamma - 1.0) * M_in ** 2)
    p0 = p_in * (T0 / T_in) ** (gamma / (gamma - 1.0))
    rho0 = rho_in * (T0 / T_in) ** (1.0 / (gamma - 1.0))

    T_x = T0 / (1.0 + 0.5 * (gamma - 1.0) * M_x ** 2)
    p_x = p0 * (T_x / T0) ** (gamma / (gamma - 1.0))
    rho_x = rho0 * (T_x / T0) ** (1.0 / (gamma - 1.0))
    V_x = M_x * np.sqrt(gamma * R * T_x)

    return rho_x, V_x, p_x, T_x


def stage2_exact_state_nondim(x_si: np.ndarray, Dt_si: float, D_in: float, L: float,
                               V_in: float, p_in: float, T_in: float, gamma: float, R: float):
    """Same as `stage2_exact_state_si` but returned as ratios to the inlet state
    (rho/rho_in, V/V_in, p/p_in, T/T_in) — matches physics.residuals_euler's convention."""
    rho_x, V_x, p_x, T_x = stage2_exact_state_si(x_si, Dt_si, D_in, L, V_in, p_in, T_in, gamma, R)
    rho_in = p_in / (R * T_in)
    return rho_x / rho_in, V_x / V_in, p_x / p_in, T_x / T_in
