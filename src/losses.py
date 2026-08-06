"""Loss terms and weighting strategies (README.md Section 5).

Each physics statement becomes one mean-squared-residual term:

    L_cont = mean(r_cont^2)      L_mom = mean(r_mom^2)      L_bc = ... (soft BC only)
    L = lambda_cont * L_cont + lambda_mom * L_mom + lambda_bc * L_bc + lambda_data * L_data

`lambda_bc` and `L_bc` are dropped entirely when `hard_bc: true` (the BC is
then satisfied exactly by the network's output transform, see networks.py).

Two weighting strategies (`loss_weighting` in config):

- `fixed`: use the lambdas from the config, unchanged.
- `annealing` (default): Wang, Teng & Perdikaris (2021) learning-rate
  annealing, README Section 5 / CLAUDE.md Section 5, recomputed every
  `annealing_every` epochs:

      lambda_hat_i = max_theta |grad_theta L_r| / mean(|grad_theta L_i|)
      lambda_i <- (1 - alpha) * lambda_i + alpha * lambda_hat_i,   alpha = 0.9

  where `L_r` is the reference term (we use the momentum residual, the
  stiffest term in this problem) and `L_i` ranges over the other terms.
"""
from __future__ import annotations

from typing import Dict, List

import torch

from src import physics


def mse(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(residual ** 2)


def stage1_loss_terms(model, xt_f: torch.Tensor, dtt_f: torch.Tensor,
                       xt_b: torch.Tensor = None, dtt_b: torch.Tensor = None,
                       hard_bc: bool = True) -> Dict[str, torch.Tensor]:
    """Compute each unweighted Stage 1 loss term (README Section 5 ①②③).

    `xt_f, dtt_f` are collocation points (PDE residuals). `xt_b, dtt_b` are
    boundary points at x_tilde=0, only used (and only required) when
    `hard_bc=False`.
    """
    res = physics.residuals_stage1(model, xt_f, dtt_f)
    terms = {
        "continuity": mse(res["continuity"]),
        "momentum": mse(res["momentum"]),
    }
    if not hard_bc:
        assert xt_b is not None and dtt_b is not None, "boundary points required when hard_bc=False"
        out_b = model(xt_b, dtt_b)
        V_b, p_b = out_b[..., 0:1], out_b[..., 1:2]
        terms["bc"] = torch.mean((V_b - 1.0) ** 2) + torch.mean(p_b ** 2)
    return terms


def stage2_loss_terms(model, xt_f: torch.Tensor, dtt_f: torch.Tensor, gamma: float, R: float,
                       V_in: float, p_in: float, T_in: float,
                       xt_b: torch.Tensor = None, dtt_b: torch.Tensor = None,
                       hard_bc: bool = True) -> Dict[str, torch.Tensor]:
    """Compute each unweighted Stage 2 loss term (physics.residuals_euler)."""
    res = physics.residuals_euler(model, xt_f, dtt_f, gamma=gamma, R=R,
                                   V_in=V_in, p_in=p_in, T_in=T_in)
    terms = {
        "continuity": mse(res["continuity"]),
        "momentum": mse(res["momentum"]),
        "energy": mse(res["energy"]),
        "eos": mse(res["eos"]),
    }
    if not hard_bc:
        assert xt_b is not None and dtt_b is not None, "boundary points required when hard_bc=False"
        out_b = model(xt_b, dtt_b)
        terms["bc"] = torch.mean((out_b - 1.0) ** 2)
    return terms


def weighted_total(terms: Dict[str, torch.Tensor], lambdas: Dict[str, float]) -> torch.Tensor:
    total = 0.0
    for name, value in terms.items():
        total = total + lambdas.get(name, 1.0) * value
    return total


def update_lambdas_annealing(model, terms: Dict[str, torch.Tensor], lambdas: Dict[str, float],
                              reference: str = "momentum", alpha: float = 0.9,
                              lambda_min: float = 0.1, lambda_max: float = 100.0) -> Dict[str, float]:
    """One step of Wang et al. (2021) learning-rate annealing (Algorithm 1).

    lambda_hat_i = max_theta |grad_theta L_r| / mean_theta |grad_theta L_i|
    lambda_i <- (1 - alpha) * lambda_i + alpha * lambda_hat_i
    `reference` is never reweighted (its own lambda stays fixed at 1, matching
    the algorithm where L_r is the anchor term).

    The raw ratio `lambda_hat` is clipped to [lambda_min, lambda_max] before the
    EMA update. This is not in the paper's pseudocode, but is standard practice
    in every real implementation of this scheme: late in training one residual
    term's mean gradient can shrink much faster than the reference term's max
    gradient, so the raw ratio is unbounded and, without a ceiling, compounds
    through the EMA every 500 epochs into a runaway feedback loop (observed
    directly in this repo: lambda_continuity reached 5e5 within 1500 epochs and
    diverged the training loss before this guard was added).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    def max_abs_grad(loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
        return flat.abs().max()

    def mean_abs_grad(loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat = torch.cat([g.reshape(-1) for g in grads if g is not None])
        return flat.abs().mean()

    max_grad_ref = max_abs_grad(terms[reference])

    new_lambdas = dict(lambdas)
    for name, loss_i in terms.items():
        if name == reference:
            continue
        mean_grad_i = mean_abs_grad(loss_i)
        if mean_grad_i.item() == 0.0:
            continue
        lambda_hat = (max_grad_ref / mean_grad_i).item()
        lambda_hat = min(max(lambda_hat, lambda_min), lambda_max)
        new_lambdas[name] = (1.0 - alpha) * lambdas.get(name, 1.0) + alpha * lambda_hat
    return new_lambdas
