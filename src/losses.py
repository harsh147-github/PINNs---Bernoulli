"""FILE 5 OF 9 — Turning "how wrong is the physics" into the one number
the optimizer actually descends.

OBJECTIVE OF THIS FILE
-----------------------
physics.py hands back a RESIDUAL — a raw number (well, thousands of them,
one per collocation point) measuring how far the network's guess is from
obeying one equation, at one point. A residual isn't a loss yet: it can be
positive or negative, and it's evaluated at hundreds of scattered points,
not one number. This file turns residuals into a loss:

    L_cont = mean( r_cont^2 )   -- continuity residual, squared and averaged
    L_mom  = mean( r_mom^2  )   -- Bernoulli/momentum residual, squared and averaged
    L_bc   = mean( (V~(0)-1)^2 + p~(0)^2 )   -- inlet BCs (soft mode only)

Squaring does two jobs at once: it makes a residual of -0.3 count exactly
as bad as +0.3 (direction of violation doesn't matter, only size), and it
punishes large violations much harder than small ones (0.3^2=0.09 vs
0.03^2=0.0009 -- a 10x bigger error costs 100x more loss), which pushes
training to stamp out the worst offenders first. Averaging over all the
collocation points (README Section 5's N_f) turns "thousands of
per-point numbers" into the single scalar `total.backward()` needs.

Total loss the optimizer actually sees:

    L = lambda_cont * L_cont + lambda_mom * L_mom + lambda_bc * L_bc

WHY THE lambda WEIGHTS EXIST AT ALL
---------------------------------------
Continuity and momentum don't naturally produce gradients of the same
size. Left with lambda=1 for both, the optimizer will happily drive
whichever term has the louder gradient toward zero while quietly
neglecting the other (Wang, Teng & Perdikaris 2021 call this a "gradient
pathology"). Two fixes are implemented:

  - 'fixed'     : you set the lambdas yourself in config and they never change.
  - 'annealing' (default): every `anneal_every` epochs, re-measure each
                  term's actual gradient magnitude and rebalance:
                      lambda_hat_i = max|grad L_ref| / mean|grad L_i|
                      lambda_i <- (1 - alpha) * lambda_i + alpha * lambda_hat_i
                  with alpha=0.9 -- the update leans 90% toward the fresh
                  estimate each time it fires. `maybe_anneal()` below is
                  exactly this formula in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.physics import residual_continuity, residual_momentum, residuals_euler


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


# ---------------------------------------------------------------------------
# Stage 2 — compressible quasi-1D Euler loss (CLAUDE.md Section 6).
#
#   L_cont, L_mom, L_energy, L_eos = mean squared residual, one per equation
#   L_bc (soft mode only)          = mean squared deviation from the inlet
#                                     state (rho~=V~=p~=T~=1 at x~=0)
#   L = lambda_cont*L_cont + lambda_mom*L_mom + lambda_energy*L_energy
#       + lambda_eos*L_eos (+ lambda_bc*L_bc if hard_bc: false)
#
# `lambda_mom_boost` (config switch, Hong et al. 2023): when true, the
# momentum weight is pinned at `lambda_mom_boost_value` (typically 20)
# instead of being annealed, for cases where residuals stall.
# ---------------------------------------------------------------------------


def loss_terms_stage2(model, xt, dtt, phys_cfg: dict) -> dict:
    """Each unweighted Stage 2 residual loss term (mean squared residual)."""
    res = residuals_euler(model, xt, dtt, phys_cfg)
    return {name: torch.mean(r**2) for name, r in res.items()}


def loss_bc_soft_stage2(model, dtt_b) -> torch.Tensor:
    """L_bc — soft inlet-condition penalty: rho~=V~=p~=T~=1 at x~=0."""
    xt0 = torch.zeros_like(dtt_b)
    rho, v, p, t = model(xt0, dtt_b)
    return torch.mean((rho - 1.0) ** 2 + (v - 1.0) ** 2 + (p - 1.0) ** 2 + (t - 1.0) ** 2)


@dataclass
class Stage2LossWeights:
    """Holds lambdas and implements the two weighting strategies for Stage 2."""

    weighting: str = "annealing"
    anneal_alpha: float = 0.9
    anneal_every: int = 500
    anneal_warmup: int = 500
    lambda_mom_boost: bool = False
    lambda_mom_boost_value: float = 20.0
    values: dict = field(
        default_factory=lambda: {"cont": 1.0, "mom": 1.0, "energy": 1.0, "eos": 1.0, "bc": 1.0}
    )

    def __post_init__(self) -> None:
        if self.lambda_mom_boost:
            self.values["mom"] = self.lambda_mom_boost_value

    def total(
        self, model, xt_f, dtt_f, phys_cfg: dict, dtt_b=None, hard_bc: bool = True
    ) -> tuple[torch.Tensor, dict]:
        """Assemble the weighted total loss. Returns (total, parts)."""
        parts = loss_terms_stage2(model, xt_f, dtt_f, phys_cfg)
        total = sum(self.values[name] * parts[name] for name in parts)
        if not hard_bc and dtt_b is not None:
            l_bc = loss_bc_soft_stage2(model, dtt_b)
            parts["bc"] = l_bc
            total = total + self.values["bc"] * l_bc
        return total, parts

    def maybe_anneal(self, epoch: int, model, parts: dict) -> bool:
        """Wang et al. Algorithm 1, skipping 'mom' when lambda_mom_boost is pinned."""
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
            if self.lambda_mom_boost and name == "mom":
                continue  # pinned per Hong et al. 2023 config switch
            if name in means and means[name] > 0:
                lam_hat = ref_max / means[name]
                lam_hat = min(max(lam_hat, 1e-3), 1e4)
                a = self.anneal_alpha
                self.values[name] = (1 - a) * self.values[name] + a * lam_hat
        return True
