"""Validation metrics vs. the analytical ground truth (README Section 10).

- `relative_l2`: standard relative L2 error, used for both training-time logging
  and the final acceptance-gate check (CLAUDE.md Section 7).
- `evaluate_held_out`: runs the full "generalization" check — Dt values never
  seen during training (README Section 10, "Generalization").
"""
from __future__ import annotations

import json
from typing import Dict

import torch

from src import analytical, sampling


def relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """||pred - ref||_2 / ||ref||_2, a scalar float."""
    num = torch.linalg.vector_norm(pred - ref)
    den = torch.linalg.vector_norm(ref) + 1e-14
    return (num / den).item()


def evaluate_held_out(model, cfg: Dict, device: torch.device = torch.device("cpu")) -> Dict:
    """Evaluate the Stage 1 surrogate on the held-out Dt validation grid.

    Returns a metrics dict: rel_l2 for V and p, max |Bernoulli invariant - 1|,
    and per-Dt breakdowns. See README Section 10 for the three validation checks
    and CLAUDE.md Section 7 for the acceptance thresholds.
    """
    samp_cfg, geom_cfg = cfg["sampling"], cfg["geometry"]
    xt, dtt = sampling.validation_grid(
        n_total=samp_cfg["n_validation"],
        dt_values_si=samp_cfg["validation_dt_holdout"],
        D_in=geom_cfg["D_in"],
    )
    xt, dtt = xt.to(device), dtt.to(device)

    with torch.no_grad():
        out = model(xt, dtt)
        V_pred, p_pred = out[..., 0], out[..., 1]
        V_exact = analytical.velocity_exact_nondim(xt, dtt).squeeze(-1)
        p_exact = analytical.pressure_exact_nondim(xt, dtt).squeeze(-1)

        rel_l2_V = relative_l2(V_pred, V_exact)
        rel_l2_p = relative_l2(p_pred, p_exact)
        total_head = p_pred + V_pred ** 2
        max_head_err_pct = torch.max(torch.abs(total_head - 1.0)).item() * 100.0

    metrics = {
        "rel_l2_V": rel_l2_V,
        "rel_l2_p": rel_l2_p,
        "max_bernoulli_invariant_error_pct": max_head_err_pct,
        "n_points": xt.shape[0],
        "dt_values_si": samp_cfg["validation_dt_holdout"],
    }
    return metrics


def save_metrics_json(metrics: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
