"""Network architecture — README Section 6.

PINN(xt, dtt) -> (V~, p~)

Fully connected MLP: 2 inputs -> K hidden layers of N neurons (tanh) -> 2 raw
outputs. Boundary conditions are HARD-WIRED into the output transform
(README Section 5, item 3; Lagaris 1998 / Sun et al. 2020):

    V~(xt) = 1 + xt * N_V(xt, dtt)      =>  V~(0) = 1  identically
    p~(xt) =     xt * N_p(xt, dtt)      =>  p~(0) = 0  identically

The optimizer literally cannot violate the inlet conditions.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class Sine(nn.Module):
    """sin activation (SIREN-style), optional alternative to tanh."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.sin(z)


_ACTS = {"tanh": nn.Tanh, "sin": Sine}


class PINN(nn.Module):
    """Parametric PINN for the nozzle surrogate.

    Parameters
    ----------
    n_hidden_layers, n_neurons : architecture (README Section 6 table).
    activation : 'tanh' (default) or 'sin'. Must be C^inf-smooth because we
        differentiate outputs to build PDE residuals.
    hard_bc : if True, apply the exact-BC output transform above; if False,
        return raw network outputs (BCs then enter as a soft loss term).
    """

    def __init__(
        self,
        n_hidden_layers: int = 6,
        n_neurons: int = 64,
        activation: str = "tanh",
        hard_bc: bool = True,
    ) -> None:
        super().__init__()
        self.hard_bc = hard_bc
        act = _ACTS[activation]

        layers: list[nn.Module] = []
        n_in = 2
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_in, n_neurons), act()]
            n_in = n_neurons
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(n_in, 2)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/Glorot-uniform init; biases zero (README Section 6)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def raw(self, xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
        """Forward propagation through the MLP: z -> W z + b -> tanh -> ..."""
        z = torch.cat([xt, dtt], dim=1)
        for layer in self.hidden:
            z = layer(z)
        return self.output(z)

    def forward(
        self, xt: torch.Tensor, dtt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (V~, p~) with boundary conditions enforced if hard_bc."""
        out = self.raw(xt, dtt)
        n_v, n_p = out[:, 0:1], out[:, 1:2]
        if self.hard_bc:
            v = 1.0 + xt * n_v   # V~(0) = 1 exactly
            p = xt * n_p         # p~(0) = 0 exactly
        else:
            v, p = n_v, n_p
        return v, p

    def count_parameters(self) -> int:
        """Total number of trainable weights + biases."""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Stage 2 — compressible quasi-1D Euler network (CLAUDE.md Section 6).
#
# PINNStage2(xt, dtt) -> (rho~, V~, p~, T~), each a ratio to its inlet value,
# so all four equal 1 at the inlet (see physics.residuals_euler docstring).
#
# CLAUDE.md Section 6 requires "softplus positivity on rho~, p~, T~". A plain
# `1 + xt*raw` hard-BC ansatz (as used for Stage 1) cannot guarantee
# positivity away from the inlet, so for the three positive-definite fields
# we shift the softplus so it lands exactly on 1 at xt=0:
#
#     c0 = softplus^-1(1) = ln(e - 1)
#     rho~ = softplus(c0 + xt * raw_rho)      # = softplus(c0) = 1 at xt=0, always > 0
#     p~   = softplus(c0 + xt * raw_p)
#     T~   = softplus(c0 + xt * raw_T)
#     V~   = 1 + xt * raw_V                    # velocity has no sign constraint
#
# which satisfies both requirements (exact inlet BC + positivity everywhere)
# using literally the softplus CLAUDE.md asks for. When `hard_bc: false`,
# softplus is applied directly to the raw outputs (no xt-shift), and the
# inlet condition is left to the soft BC loss (losses.loss_bc_soft_stage2).
# `hard_pressure_bc` (only consulted when hard_bc: false) forces just the
# pressure field through the exact shifted-softplus ansatz, per Hong et al.
# (2023) — CLAUDE.md Section 6: "if residuals stall: ... hard pressure BCs".
# ---------------------------------------------------------------------------

_SOFTPLUS_INV_1 = math.log(math.e - 1.0)  # softplus(c0) = 1


class PINNStage2(nn.Module):
    """Parametric PINN for the Stage 2 compressible nozzle surrogate.

    Parameters mirror `PINN` (README/CLAUDE Section 6 architecture table);
    the only difference is 4 outputs instead of 2 and the positivity-aware
    output transform described above.
    """

    def __init__(
        self,
        n_hidden_layers: int = 6,
        n_neurons: int = 64,
        activation: str = "tanh",
        hard_bc: bool = True,
        hard_pressure_bc: bool = True,
    ) -> None:
        super().__init__()
        self.hard_bc = hard_bc
        self.hard_pressure_bc = hard_pressure_bc
        act = _ACTS[activation]

        layers: list[nn.Module] = []
        n_in = 2
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(n_in, n_neurons), act()]
            n_in = n_neurons
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(n_in, 4)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def raw(self, xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
        z = torch.cat([xt, dtt], dim=1)
        for layer in self.hidden:
            z = layer(z)
        return self.output(z)

    def _positive_field(self, xt: torch.Tensor, raw_i: torch.Tensor, hard: bool) -> torch.Tensor:
        if hard:
            return nn.functional.softplus(_SOFTPLUS_INV_1 + xt * raw_i)
        return nn.functional.softplus(raw_i)

    def forward(
        self, xt: torch.Tensor, dtt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (rho~, V~, p~, T~) with positivity + BCs enforced per config."""
        out = self.raw(xt, dtt)
        n_rho, n_v, n_p, n_t = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

        rho = self._positive_field(xt, n_rho, hard=self.hard_bc)
        t = self._positive_field(xt, n_t, hard=self.hard_bc)
        p = self._positive_field(xt, n_p, hard=self.hard_bc or self.hard_pressure_bc)
        v = 1.0 + xt * n_v if self.hard_bc else n_v
        return rho, v, p, t

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
