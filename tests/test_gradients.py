"""Gate 2: torch.autograd derivatives agree with finite differences (verification
only — training itself never uses finite differences)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.networks import PINN


def test_autodiff_vs_finite_difference():
    torch.manual_seed(0)
    model = PINN(2, 16).to(dtype=torch.float64)
    xt = torch.linspace(0.05, 0.95, 50, dtype=torch.float64).unsqueeze(1)
    dtt = torch.full_like(xt, 0.6)

    xt_req = xt.clone().requires_grad_(True)
    v, _ = model(xt_req, dtt)
    dv_auto = torch.autograd.grad(v, xt_req, torch.ones_like(v), create_graph=True)[0]

    eps = 1e-5
    v_plus, _ = model(xt + eps, dtt)
    v_minus, _ = model(xt - eps, dtt)
    dv_fd = (v_plus - v_minus) / (2 * eps)

    rel = (dv_auto - dv_fd).abs() / dv_fd.abs().clamp_min(1e-8)
    assert rel.median().item() < 1e-6, f"median rel err {rel.median()}"


def test_hard_bc_exact_at_inlet():
    torch.manual_seed(0)
    model = PINN(2, 16, hard_bc=True).to(dtype=torch.float64)
    dtt = torch.rand(32, 1, dtype=torch.float64) * 0.5 + 0.4
    v, p = model(torch.zeros_like(dtt), dtt)
    assert (v - 1.0).abs().max().item() == 0.0
    assert p.abs().max().item() == 0.0
