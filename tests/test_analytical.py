"""Gate 1: the closed-form solution satisfies the residuals to machine precision."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytical import pressure_exact_nondim, velocity_exact_nondim
from src.geometry import area_nondim


class ExactSolution(torch.nn.Module):
    """Wraps the analytical solution in the model interface (xt, dtt) -> (V~, p~)."""

    def forward(self, xt, dtt):
        return velocity_exact_nondim(xt, dtt), pressure_exact_nondim(xt, dtt)


def test_exact_solution_satisfies_continuity():
    from src.physics import residual_continuity

    xt = torch.linspace(1e-6, 1 - 1e-6, 500, dtype=torch.float64).unsqueeze(1)
    for dtt_val in (0.4, 0.55, 0.7, 0.9):
        dtt = torch.full_like(xt, dtt_val)
        r = residual_continuity(ExactSolution(), xt, dtt)
        assert r.abs().max().item() < 1e-10, f"continuity residual {r.abs().max()}"


def test_exact_solution_satisfies_momentum():
    from src.physics import residual_momentum

    xt = torch.linspace(1e-6, 1 - 1e-6, 500, dtype=torch.float64).unsqueeze(1)
    for dtt_val in (0.4, 0.55, 0.7, 0.9):
        dtt = torch.full_like(xt, dtt_val)
        r = residual_momentum(ExactSolution(), xt, dtt)
        assert r.abs().max().item() < 1e-10, f"momentum residual {r.abs().max()}"


def test_area_law_endpoints():
    xt = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float64)
    dtt = torch.tensor([[0.4], [0.4], [0.4]], dtype=torch.float64)
    a = area_nondim(xt, dtt)
    assert torch.isclose(a[0, 0], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(a[1, 0], torch.tensor(0.16, dtype=torch.float64))  # 0.4^2
    assert torch.isclose(a[2, 0], torch.tensor(1.0, dtype=torch.float64))
