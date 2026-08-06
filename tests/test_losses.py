"""CLAUDE.md Section 7: every loss term must be 0 (to 1e-12) when the network
is replaced by the exact solution."""
import torch

from src import losses
from src.sampling import lhs_collocation, boundary_points
from tests.test_analytical import ExactStage1Model


def test_stage1_loss_terms_zero_hard_bc():
    model = ExactStage1Model()
    xt_f, dtt_f = lhs_collocation(n_x=64, n_dt=4, seed=1)
    terms = losses.stage1_loss_terms(model, xt_f, dtt_f, hard_bc=True)
    for name, value in terms.items():
        assert value.item() < 1e-12, f"{name} loss = {value.item()}"


def test_stage1_loss_terms_zero_soft_bc():
    model = ExactStage1Model()
    xt_f, dtt_f = lhs_collocation(n_x=64, n_dt=4, seed=1)
    xt_b, dtt_b = boundary_points(n=20, seed=2)
    terms = losses.stage1_loss_terms(model, xt_f, dtt_f, xt_b, dtt_b, hard_bc=False)
    for name, value in terms.items():
        assert value.item() < 1e-12, f"{name} loss = {value.item()}"


def test_weighted_total_matches_manual_sum():
    model = ExactStage1Model()
    xt_f, dtt_f = lhs_collocation(n_x=32, n_dt=4, seed=3)
    terms = losses.stage1_loss_terms(model, xt_f, dtt_f, hard_bc=True)
    lambdas = {"continuity": 2.0, "momentum": 3.0}
    total = losses.weighted_total(terms, lambdas)
    expected = 2.0 * terms["continuity"] + 3.0 * terms["momentum"]
    assert torch.isclose(total, expected)
