"""The PINN itself: MLP + activation + init + hard/soft boundary-condition output transform.

Implements README.md Section 4.1 (network as nested function), Section 5 point ③
(hard-BC ansatz), and Section 6 (architecture table).

    z^(0) = (x_tilde, Dt_tilde)                              # network input
    z^(k) = phi(W^(k) z^(k-1) + b^(k)),  k = 1..K             # hidden layers
    raw   = W^(K+1) z^(K) + b^(K+1)                           # linear output head

`phi` is tanh (or optionally sin/SIREN) — both C^infinity, required because we
differentiate the network's output twice through autograd to build PDE
residuals; a ReLU network's kinked second derivative would poison the physics
loss (README Section 4.1).

Stage 1 hard-BC ansatz (README Section 5 ③, exact at x_tilde = 0):

    V_hat = 1 + x_tilde * raw_V
    p_hat = x_tilde * raw_p

Stage 2 hard-BC ansatz: all four primitive variables equal 1 at the inlet in
our ratio-to-inlet non-dimensionalization (see physics.py docstring), and
rho_hat, p_hat, T_hat must stay strictly positive (CLAUDE.md Section 3,
"Enforce positivity with softplus"). An additive ansatz `1 + x*raw` cannot
guarantee positivity away from the inlet, and softplus-after-affine would
break the exact BC at x_tilde=0 (softplus(1) != 1). We instead use an
exponential hard-BC ansatz for the positive-definite fields:

    rho_hat = exp(x_tilde * raw_rho)     # = 1 at x_tilde=0, always > 0
    p_hat   = exp(x_tilde * raw_p)
    T_hat   = exp(x_tilde * raw_T)
    V_hat   = 1 + x_tilde * raw_V         # velocity has no sign constraint

which satisfies the same two requirements (exact inlet BC + positivity
everywhere) as softplus would, and is what is used when `hard_bc: true`. When
`hard_bc: false`, softplus is applied directly to the raw outputs for the
positive fields (per CLAUDE.md), and the inlet condition is left to the soft
BC loss penalty in losses.py.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Optional random Fourier input embedding (off by default, CLAUDE.md Section 4).

    z -> [sin(2*pi*B*z), cos(2*pi*B*z)] with fixed (non-trainable) B ~ N(0, scale^2).
    """

    def __init__(self, n_inputs: int, n_frequencies: int, scale: float, dtype=torch.float64):
        super().__init__()
        B = torch.randn(n_inputs, n_frequencies, dtype=dtype) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return 2 * self.B.shape[1]

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * (z @ self.B)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class PINN(nn.Module):
    """Fully connected PINN: (x_tilde, Dt_tilde) -> primitive flow variables."""

    def __init__(
        self,
        stage: int,
        hidden_layers: int = 6,
        hidden_neurons: int = 64,
        activation: str = "tanh",
        hard_bc: bool = True,
        fourier_features: bool = False,
        fourier_scale: float = 2.0,
        fourier_n_frequencies: int = 16,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        self.stage = stage
        self.hard_bc = hard_bc
        self.n_outputs = 2 if stage == 1 else 4
        self.dtype = dtype

        n_inputs_raw = 2
        self.fourier: Optional[FourierFeatures] = None
        if fourier_features:
            self.fourier = FourierFeatures(n_inputs_raw, fourier_n_frequencies, fourier_scale, dtype=dtype)
            n_in = self.fourier.out_dim
        else:
            n_in = n_inputs_raw

        if activation == "tanh":
            self._activation = torch.tanh
        elif activation == "sin":
            self._activation = torch.sin
        else:
            raise ValueError(f"Unknown activation '{activation}' (use 'tanh' or 'sin')")

        dims = [n_in] + [hidden_neurons] * hidden_layers + [self.n_outputs]
        layers = [nn.Linear(dims[i], dims[i + 1], dtype=dtype) for i in range(len(dims) - 1)]
        self.layers = nn.ModuleList(layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _mlp(self, xt: torch.Tensor, Dtt: torch.Tensor) -> torch.Tensor:
        z = torch.cat([xt, Dtt], dim=-1)
        if self.fourier is not None:
            z = self.fourier(z)
        for layer in self.layers[:-1]:
            z = self._activation(layer(z))
        return self.layers[-1](z)

    def forward(self, xt: torch.Tensor, Dtt: torch.Tensor) -> torch.Tensor:
        """Evaluate the network.

        Parameters
        ----------
        xt, Dtt : torch.Tensor
            Non-dimensional axial position and throat diameter, shape (N, 1).

        Returns
        -------
        torch.Tensor, shape (N, 2) for Stage 1 [(V_tilde_hat, p_tilde_hat)] or
        (N, 4) for Stage 2 [(rho_tilde_hat, V_tilde_hat, p_tilde_hat, T_tilde_hat)].
        """
        raw = self._mlp(xt, Dtt)

        if self.stage == 1:
            raw_V, raw_p = raw[..., 0:1], raw[..., 1:2]
            if self.hard_bc:
                V_hat = 1.0 + xt * raw_V
                p_hat = xt * raw_p
            else:
                V_hat, p_hat = raw_V, raw_p
            return torch.cat([V_hat, p_hat], dim=-1)

        raw_rho, raw_V, raw_p, raw_T = (raw[..., i:i + 1] for i in range(4))
        if self.hard_bc:
            rho_hat = torch.exp(xt * raw_rho)
            V_hat = 1.0 + xt * raw_V
            p_hat = torch.exp(xt * raw_p)
            T_hat = torch.exp(xt * raw_T)
        else:
            rho_hat = torch.nn.functional.softplus(raw_rho)
            V_hat = raw_V
            p_hat = torch.nn.functional.softplus(raw_p)
            T_hat = torch.nn.functional.softplus(raw_T)
        return torch.cat([rho_hat, V_hat, p_hat, T_hat], dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_network(cfg: dict, stage: int) -> PINN:
    """Construct a PINN from a loaded YAML config dict (see configs/default.yaml)."""
    net_cfg = cfg["network"]
    return PINN(
        stage=stage,
        hidden_layers=net_cfg["hidden_layers"],
        hidden_neurons=net_cfg["hidden_neurons"],
        activation=net_cfg["activation"],
        hard_bc=net_cfg["hard_bc"],
        fourier_features=net_cfg["fourier_features"],
        fourier_scale=net_cfg["fourier_scale"],
        fourier_n_frequencies=net_cfg["fourier_n_frequencies"],
    )
