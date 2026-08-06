"""Gate 3: every loss term is exactly zero when fed the analytical solution."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytical import pressure_exact_nondim, velocity_exact_nondim
from src.losses import loss_continuity, loss_momentum


class ExactSolution(torch.nn.Module):
    def forward(self, xt, dtt):
        return velocity_exact_nondim(xt, dtt), pressure_exact_nondim(xt, dtt)


def test_losses_vanish_on_exact_solution():
    xt = torch.rand(1000, 1, dtype=torch.float64)
    dtt = torch.rand(1000, 1, dtype=torch.float64) * 0.5 + 0.4
    m = ExactSolution()
    assert loss_continuity(m, xt, dtt).item() < 1e-20
    assert loss_momentum(m, xt, dtt).item() < 1e-20
