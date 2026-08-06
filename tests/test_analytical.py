"""CLAUDE.md Section 7: the analytical solution must drive every residual to ~0
(float64, < 1e-12) when plugged directly into the residual functions."""
import torch

from src import analytical, physics


class ExactStage1Model:
    """Wraps the closed-form solution (README Eq. 4) as a `model(xt, dtt)` callable,
    exactly the interface `physics.py` expects from a trained PINN."""

    def __call__(self, xt: torch.Tensor, dtt: torch.Tensor) -> torch.Tensor:
        V = analytical.velocity_exact_nondim(xt, dtt)
        p = analytical.pressure_exact_nondim(xt, dtt)
        return torch.cat([V, p], dim=-1)


def _sample_points(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    xt = torch.rand(n, 1, generator=g, dtype=torch.float64)
    dtt = 0.4 + 0.5 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    return xt, dtt


def test_continuity_residual_zero_for_exact_solution():
    model = ExactStage1Model()
    xt, dtt = _sample_points()
    r = physics.residual_continuity(model, xt, dtt)
    assert torch.max(torch.abs(r)).item() < 1e-12


def test_momentum_residual_zero_for_exact_solution():
    model = ExactStage1Model()
    xt, dtt = _sample_points()
    r = physics.residual_momentum(model, xt, dtt)
    assert torch.max(torch.abs(r)).item() < 1e-12


def test_boundary_conditions_satisfied_exactly():
    model = ExactStage1Model()
    n = 50
    xt = torch.zeros(n, 1, dtype=torch.float64)
    dtt = 0.4 + 0.5 * torch.rand(n, 1, dtype=torch.float64)
    out = model(xt, dtt)
    V, p = out[..., 0], out[..., 1]
    assert torch.max(torch.abs(V - 1.0)).item() < 1e-12
    assert torch.max(torch.abs(p)).item() < 1e-12


def test_total_head_invariant():
    """p_tilde + V_tilde^2 == 1 everywhere for the exact solution (Bernoulli constant)."""
    model = ExactStage1Model()
    xt, dtt = _sample_points()
    out = model(xt, dtt)
    V, p = out[..., 0], out[..., 1]
    total_head = p + V ** 2
    assert torch.max(torch.abs(total_head - 1.0)).item() < 1e-12
