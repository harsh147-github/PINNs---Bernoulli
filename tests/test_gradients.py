"""CLAUDE.md Section 7: autodiff d/dx_tilde vs central finite difference must
agree to < 1e-6 relative, on the actual network (random-initialized PINN)."""
import torch

from src import geometry as geo
from src.networks import PINN


def _central_diff(f, xt: torch.Tensor, dtt: torch.Tensor, h: float = 1e-6) -> torch.Tensor:
    return (f(xt + h, dtt) - f(xt - h, dtt)) / (2 * h)


def test_network_output_gradient_matches_finite_difference():
    torch.manual_seed(0)
    model = PINN(stage=1, hidden_layers=3, hidden_neurons=16, hard_bc=True)

    n = 30
    xt = 0.1 + 0.8 * torch.rand(n, 1, dtype=torch.float64)  # avoid boundary for FD stencil
    dtt = 0.4 + 0.5 * torch.rand(n, 1, dtype=torch.float64)

    xt_ad = xt.clone().requires_grad_(True)
    out = model(xt_ad, dtt)
    V_hat = out[..., 0:1]
    dV_dx_autodiff = torch.autograd.grad(V_hat, xt_ad, grad_outputs=torch.ones_like(V_hat),
                                          create_graph=False)[0]

    with torch.no_grad():
        dV_dx_fd = _central_diff(lambda x, d: model(x, d)[..., 0:1], xt, dtt)

    rel_err = torch.abs(dV_dx_autodiff - dV_dx_fd) / (torch.abs(dV_dx_fd) + 1e-12)
    assert torch.max(rel_err).item() < 1e-6


def test_geometry_area_gradient_matches_finite_difference():
    """Same check on geometry.area_nondim, since physics.py differentiates through it."""
    n = 30
    xt = 0.1 + 0.8 * torch.rand(n, 1, dtype=torch.float64)
    dtt = 0.4 + 0.5 * torch.rand(n, 1, dtype=torch.float64)

    xt_ad = xt.clone().requires_grad_(True)
    A = geo.area_nondim(xt_ad, dtt)
    dA_dx_autodiff = torch.autograd.grad(A, xt_ad, grad_outputs=torch.ones_like(A))[0]

    with torch.no_grad():
        dA_dx_fd = _central_diff(geo.area_nondim, xt, dtt)

    rel_err = torch.abs(dA_dx_autodiff - dA_dx_fd) / (torch.abs(dA_dx_fd) + 1e-12)
    assert torch.max(rel_err).item() < 1e-6
