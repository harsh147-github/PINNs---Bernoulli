"""The equations-to-code file: PDE residuals via ``torch.autograd.grad``.

This is the heart of the "physics replaces data" idea (README Section 4.4):
any candidate field that drives these residuals to zero everywhere, together
with the boundary conditions, *is* the flow solution. We never use finite
differences for these derivatives (CLAUDE.md Section 11) — `x_tilde` is given
`requires_grad_(True)` and PyTorch differentiates the *exact* computational
graph of geometry + network composed together.

Stage 1 (README Eq. 5, non-dimensional continuity + Bernoulli/momentum):

    r_cont = d/dx_tilde [ A_tilde(x_tilde; Dt_tilde) * V_tilde_hat ]
    r_mom  = d/dx_tilde [ p_tilde_hat + V_tilde_hat^2 ]

Stage 2 (compressible quasi-1D Euler, subsonic branch). We non-dimensionalize
every primitive variable as a ratio to its inlet reference value:

    rho_tilde = rho/rho_in,  V_tilde = V/V_in,  p_tilde = p/p_in,  T_tilde = T/T_in

(chosen so rho_tilde, p_tilde, T_tilde are automatically the natural targets
for the positivity guard in networks.py, and so all four equal 1 at the
inlet). Substituting into the SI equations of CLAUDE.md Section 3 / README
Eq. 6 and using p_in = rho_in * R * T_in (ideal gas at the inlet) gives:

    r_cont = d/dx_tilde [ rho_tilde * A_tilde * V_tilde ]
    r_mom  = rho_tilde * V_tilde * dV_tilde/dx_tilde + kappa * dp_tilde/dx_tilde,
             kappa = p_in / (rho_in * V_in^2)
    r_energy = T_tilde + beta * V_tilde^2 - (1 + beta),
               beta = V_in^2 / (2 * c_p * T_in)     (stagnation enthalpy, algebraic)
    r_eos    = p_tilde - rho_tilde * T_tilde         (ideal gas law, algebraic)
"""
from __future__ import annotations

from typing import Dict

import torch

from src import geometry as geo


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """d y / d x via autograd, keeping the graph alive for backprop through the loss."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def residual_continuity(model, xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
    """r_cont = d/dx_tilde [ A_tilde * V_tilde_hat ]  (README Eq. 5, Stage 1)."""
    xt = xt.clone().requires_grad_(True)
    out = model(xt, dtt)
    V_hat = out[..., 0:1]
    A_tilde = geo.area_nondim(xt, dtt)
    flux = A_tilde * V_hat
    return _grad(flux, xt)


def residual_momentum(model, xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
    """r_mom = d/dx_tilde [ p_tilde_hat + V_tilde_hat^2 ]  (README Eq. 5, Stage 1)."""
    xt = xt.clone().requires_grad_(True)
    out = model(xt, dtt)
    V_hat, p_hat = out[..., 0:1], out[..., 1:2]
    total_head = p_hat + V_hat ** 2
    return _grad(total_head, xt)


def residuals_stage1(model, xt: torch.Tensor, dtt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Both Stage 1 residuals, sharing one autograd-enabled forward pass."""
    xt = xt.clone().requires_grad_(True)
    out = model(xt, dtt)
    V_hat, p_hat = out[..., 0:1], out[..., 1:2]
    A_tilde = geo.area_nondim(xt, dtt)
    flux = A_tilde * V_hat
    total_head = p_hat + V_hat ** 2
    r_cont = _grad(flux, xt)
    r_mom = _grad(total_head, xt)
    return {"continuity": r_cont, "momentum": r_mom}


def residuals_euler(model, xt: torch.Tensor, dtt: torch.Tensor, gamma: float, R: float,
                     V_in: float, p_in: float, T_in: float) -> Dict[str, torch.Tensor]:
    """Stage 2 quasi-1D compressible Euler residuals (subsonic branch).

    `rho_in` is *not* a free parameter: the inlet reference density must be
    the ideal-gas-consistent value `p_in / (R * T_in)` (NOT the Stage 1
    incompressible `physics.rho` from the config, which is an independently
    prescribed constant for the inviscid-incompressible model and is in
    general inconsistent with `p = rho*R*T` at `T_in`). Deriving it here
    guarantees `r_eos` and `r_energy` are satisfiable simultaneously with
    `r_mom` by the exact isentropic solution (see analytical.py).
    """
    rho_in = p_in / (R * T_in)

    xt = xt.clone().requires_grad_(True)
    out = model(xt, dtt)
    rho_hat, V_hat, p_hat, T_hat = (out[..., i:i + 1] for i in range(4))

    A_tilde = geo.area_nondim(xt, dtt)
    mass_flux = rho_hat * A_tilde * V_hat
    r_cont = _grad(mass_flux, xt)

    dV_dxt = _grad(V_hat, xt)
    dp_dxt = _grad(p_hat, xt)
    kappa = p_in / (rho_in * V_in ** 2)
    r_mom = rho_hat * V_hat * dV_dxt + kappa * dp_dxt

    cp = gamma * R / (gamma - 1.0)
    beta = V_in ** 2 / (2.0 * cp * T_in)
    r_energy = T_hat + beta * V_hat ** 2 - (1.0 + beta)

    r_eos = p_hat - rho_hat * T_hat

    return {"continuity": r_cont, "momentum": r_mom, "energy": r_energy, "eos": r_eos}
