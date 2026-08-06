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
