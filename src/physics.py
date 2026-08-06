"""THE equations-to-code file — README Section 5.

Every residual below is one governing equation written as `torch` code.
Derivatives are exact: torch.autograd.grad differentiates the network's
output through its own computational graph (machine precision, no mesh,
no finite differences). create_graph=True keeps the derivative itself
differentiable so backpropagation can reach the weights.

Stage 1 (README Eq. 5):
    continuity : r_cont = d/dxt [ A~(xt,dtt) * V~ ]            = 0
    momentum   : r_mom  = d/dxt [ p~ + V~^2 ]                  = 0
                 (differential form of Bernoulli's theorem)
"""

from __future__ import annotations

import torch

from src.geometry import area_nondim


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Exact dy/dx through the network graph. Requires x.requires_grad."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


def residual_continuity(
    model: torch.nn.Module, xt: torch.Tensor, dtt: torch.Tensor
) -> torch.Tensor:
    """r_cont = d(A~ * V~)/dxt. Zero everywhere for a perfect solution."""
    xt = xt.requires_grad_(True)
    v, _ = model(xt, dtt)
    flux = area_nondim(xt, dtt) * v
    return _grad(flux, xt)


def residual_momentum(
    model: torch.nn.Module, xt: torch.Tensor, dtt: torch.Tensor
) -> torch.Tensor:
    """r_mom = d(p~ + V~^2)/dxt — Bernoulli says total head is constant."""
    xt = xt.requires_grad_(True)
    v, p = model(xt, dtt)
    total_head = p + v**2
    return _grad(total_head, xt)


def total_head(model: torch.nn.Module, xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
    """p~ + V~^2 — should equal 1 everywhere (invariant check, README Sec. 10).

    Always returns a CPU tensor: callers build xt/dtt with plain numpy/torch
    (CPU) tensors, and moving the result back here (rather than in every
    caller) keeps this safe to call on a CUDA-resident model.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        v, p = model(xt.to(device), dtt.to(device))
    return (p + v**2).cpu()


# ---------------------------------------------------------------------------
# Stage 2 — compressible quasi-1D Euler, subsonic branch (CLAUDE.md Section 6).
#
# Model interface: model(xt, dtt) -> (rho~, V~, p~, T~), each a ratio to its
# inlet reference value (rho/rho_in, V/V_in, p/p_in, T/T_in) — chosen so every
# field equals 1 at the inlet (clean hard-BC target) and rho~, p~, T~ are the
# natural quantities for the softplus positivity guard (networks.PINNStage2).
#
# Substituting into the SI equations of CLAUDE.md Section 6:
#   d(rho A V)/dx = 0
#   rho V dV/dx + dp/dx = 0
#   cp T + V^2/2 = h0
#   p = rho R T
# and using p_in = rho_in * R * T_in (ideal gas at the inlet) gives the
# non-dimensional residuals below, all in terms of x~ = x/L:
#
#   r_cont   = d/dx~ [ rho~ * A~ * V~ ]
#   r_mom    = rho~ * V~ * dV~/dx~ + kappa * dp~/dx~,   kappa = p_in/(rho_in*V_in^2)
#   r_energy = T~ + beta * V~^2 - (1 + beta),            beta  = V_in^2/(2*cp*T_in)
#   r_eos    = p~ - rho~ * T~
# ---------------------------------------------------------------------------


def residuals_euler(
    model: torch.nn.Module, xt: torch.Tensor, dtt: torch.Tensor, phys_cfg: dict
) -> dict[str, torch.Tensor]:
    """Stage 2 quasi-1D compressible Euler residuals (subsonic branch).

    Parameters
    ----------
    phys_cfg : the `cfg["physics"]` dict (needs gamma, R, rho_in derived from
        p_in/(R*T_in), V_in, p_in, T_in).
    """
    gamma, R = phys_cfg["gamma"], phys_cfg["R"]
    V_in, p_in, T_in = phys_cfg["V_in"], phys_cfg["p_in"], phys_cfg["T_in"]
    rho_in = p_in / (R * T_in)
    cp = gamma * R / (gamma - 1.0)

    xt = xt.requires_grad_(True)
    rho, v, p, t = model(xt, dtt)

    a_tilde = area_nondim(xt, dtt)
    mass_flux = rho * a_tilde * v
    r_cont = _grad(mass_flux, xt)

    dv_dxt = _grad(v, xt)
    dp_dxt = _grad(p, xt)
    kappa = p_in / (rho_in * V_in**2)
    r_mom = rho * v * dv_dxt + kappa * dp_dxt

    beta = V_in**2 / (2.0 * cp * T_in)
    r_energy = t + beta * v**2 - (1.0 + beta)

    r_eos = p - rho * t

    return {"cont": r_cont, "mom": r_mom, "energy": r_energy, "eos": r_eos}


def total_state_stage2(
    model: torch.nn.Module, xt: torch.Tensor, dtt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(rho~, V~, p~, T~) with no grad tracking — for validation/plotting."""
    with torch.no_grad():
        return model(xt, dtt)
