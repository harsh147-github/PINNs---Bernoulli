"""Loss functions and weighting strategy — README Section 5.

Each physics statement becomes ONE mean-squared-error term:

    L_cont = mean( r_cont^2 )   continuity residual  (Eq. 1)
    L_mom  = mean( r_mom^2  )   Bernoulli/momentum residual (Eq. 2)
    L_bc   = mean( (V~(0)-1)^2 + p~(0)^2 )   inlet BCs (Eq. 3, soft mode only)

Total:  L = lambda_cont * L_cont + lambda_mom * L_mom + lambda_bc * L_bc

Weighting strategies (README Section 5, item 4):
  - 'fixed'     : user-set lambdas from config.
  - 'annealing' : Wang, Teng & Perdikaris (2021) learning-rate annealing,
                  Algorithm 1. Every `anneal_every` epochs:
                      lambda_hat_i = max|grad L_ref| / mean|grad L_i|
                      lambda_i <- (1 - alpha) * lambda_i + alpha * lambda_hat_i
                  with alpha = 0.9. This stops one equation's gradients from
                  drowning out the other's ("gradient pathologies").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.physics import residual_continuity, residual_momentum


def loss_continuity(model, xt, dtt) -> torch.Tensor:
    """L_cont — mean squared continuity residual (Eq. 1)."""
    return torch.mean(residual_continuity(model, xt, dtt) ** 2)


def loss_momentum(model, xt, dtt) -> torch.Tensor:
    """L_mom — mean squared Bernoulli/momentum residual (Eq. 2)."""
    return torch.mean(residual_momentum(model, xt, dtt) ** 2)


def loss_bc_soft(model, dtt_b) -> torch.Tensor:
    """L_bc — soft inlet-condition penalty (only when hard_bc: false)."""
    xt0 = torch.zeros_like(dtt_b)
    v, p = model(xt0, dtt_b)
    return torch.mean((v - 1.0) ** 2 + p**2)


@dataclass
class LossWeights:
    """Holds lambdas and implements the two weighting strategies."""

    weighting: str = "annealing"
    anneal_alpha: float = 0.9
    anneal_every: int = 500
    anneal_warmup: int = 500
    values: dict = field(
        default_factory=lambda: {"cont": 1.0, "mom": 1.0, "bc": 1.0}
    )

    def total(
        self,
        model,
        xt_f,
        dtt_f,
        dtt_b=None,
        hard_bc: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """Assemble the weighted total loss. Returns (total, parts)."""
        l_cont = loss_continuity(model, xt_f, dtt_f)
        l_mom = loss_momentum(model, xt_f, dtt_f)
        parts = {"cont": l_cont, "mom": l_mom}
        total = self.values["cont"] * l_cont + self.values["mom"] * l_mom
        if not hard_bc and dtt_b is not None:
            l_bc = loss_bc_soft(model, dtt_b)
            parts["bc"] = l_bc
            total = total + self.values["bc"] * l_bc
        return total, parts

    def maybe_anneal(self, epoch: int, model, parts: dict) -> bool:
        """Apply Wang et al. Algorithm 1 when due. Returns True if updated."""
        if self.weighting != "annealing":
            return False
        if epoch < self.anneal_warmup or epoch % self.anneal_every != 0:
            return False
        params = [p for p in model.parameters() if p.requires_grad]
        ref_max, means = 0.0, {}
        for name, loss_i in parts.items():
            g = torch.autograd.grad(loss_i, params, retain_graph=True, allow_unused=True)
            flat = torch.cat([gi.abs().flatten() for gi in g if gi is not None])
            means[name] = flat.mean().item()
            ref_max = max(ref_max, flat.max().item())
        for name in self.values:
            if name in means and means[name] > 0:
                lam_hat = ref_max / means[name]
                lam_hat = min(max(lam_hat, 1e-3), 1e4)  # stability clip
                a = self.anneal_alpha
                self.values[name] = (1 - a) * self.values[name] + a * lam_hat
        return True
