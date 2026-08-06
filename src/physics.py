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
    """p~ + V~^2 — should equal 1 everywhere (invariant check, README Sec. 10)."""
    with torch.no_grad():
        v, p = model(xt, dtt)
    return p + v**2
