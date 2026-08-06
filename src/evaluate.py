"""Validation against the analytical solution — README Section 10.

No CFD data anywhere: ground truth is the closed-form Bernoulli solution.
The generalization check evaluates ONLY on throat diameters the network
never trained on (held_out_dt in config) — a memorizer fails here, a
physics-trained surrogate passes.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analytical import pressure_exact_nondim, velocity_exact_nondim
from src.physics import total_head
from src.sampling import validation_grid


def relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """||pred - ref||_2 / ||ref||_2."""
    return (torch.norm(pred - ref) / torch.norm(ref)).item()


def validate_held_out(model, cfg: dict) -> dict:
    """Rel-L2 of V~ and p~ on held-out throat diameters."""
    dtt_held = [d / cfg["geometry"]["D_in"] for d in cfg["training"]["held_out_dt"]]
    xt, dtt = validation_grid(dtt_held)
    with torch.no_grad():
        v, p = model(xt, dtt)
    v_ref = velocity_exact_nondim(xt, dtt)
    p_ref = pressure_exact_nondim(xt, dtt)
    return {
        "rel_l2_V": relative_l2(v, v_ref),
        "rel_l2_p": relative_l2(p, p_ref),
    }


def full_report(model, cfg: dict) -> dict:
    """All acceptance-gate metrics (README Section 9, item 6)."""
    dtt_held = [d / cfg["geometry"]["D_in"] for d in cfg["training"]["held_out_dt"]]
    xt, dtt = validation_grid(dtt_held, n_x=400)
    with torch.no_grad():
        v, p = model(xt, dtt)
    m = validate_held_out(model, cfg)
    h = total_head(model, xt, dtt)
    m["total_head_max_err"] = (h - 1.0).abs().max().item()
    a = cfg["acceptance"]
    m["gates"] = {
        "rel_l2_V": m["rel_l2_V"] < a["rel_l2_V"],
        "rel_l2_p": m["rel_l2_p"] < a["rel_l2_p"],
        "total_head": m["total_head_max_err"] < a["total_head_tol"],
    }
    m["all_passed"] = all(m["gates"].values())
    return m
