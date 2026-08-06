"""Validation against the analytical solution — README Section 10.

No CFD data anywhere: ground truth is the closed-form Bernoulli solution.
The generalization check evaluates ONLY on throat diameters the network
never trained on (held_out_dt in config) — a memorizer fails here, a
physics-trained surrogate passes.
"""

from __future__ import annotations

import numpy as np
import torch

from src.analytical import pressure_exact_nondim, stage2_exact_state_nondim, velocity_exact_nondim
from src.physics import total_head
from src.sampling import validation_grid


def relative_l2(pred: torch.Tensor, ref: torch.Tensor) -> float:
    """||pred - ref||_2 / ||ref||_2."""
    return (torch.norm(pred - ref) / torch.norm(ref)).item()


def _forward_cpu(model, xt: torch.Tensor, dtt: torch.Tensor):
    """Evaluate `model` on its own device, return outputs moved back to CPU.

    Bug fix (found during CLAUDE.md Section 2 VERIFY on a CUDA model):
    `validation_grid`/numpy-built tensors are always CPU; calling a CUDA
    model on them raises "Expected all tensors to be on the same device".
    This keeps every caller's tensors CPU-only while still using the GPU.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(xt.to(device), dtt.to(device))
    return tuple(o.cpu() for o in out)


def validate_held_out(model, cfg: dict) -> dict:
    """Rel-L2 of V~ and p~ on held-out throat diameters."""
    dtt_held = [d / cfg["geometry"]["D_in"] for d in cfg["training"]["held_out_dt"]]
    xt, dtt = validation_grid(dtt_held)
    v, p = _forward_cpu(model, xt, dtt)
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


# ---------------------------------------------------------------------------
# Stage 2 — validation against the isentropic subsonic solution (CLAUDE.md
# Section 6: "Validation vs isentropic area-Mach relation ... Gate: rel-L2 < 1e-2").
# ---------------------------------------------------------------------------


def validate_held_out_stage2(model, cfg: dict) -> dict:
    """Rel-L2 of (rho~, V~, p~, T~) vs the exact isentropic solution, on
    held-out throat diameters (never trained on)."""
    g = cfg["geometry"]
    held_dt_si = cfg["stage2"]["training"]["held_out_dt"]
    n_x = 400

    rho_p, v_p, p_p, t_p = [], [], [], []
    rho_e, v_e, p_e, t_e = [], [], [], []
    for dt_si in held_dt_si:
        x = np.linspace(0.0, g["L"], n_x)
        xt = torch.as_tensor((x / g["L"]).reshape(-1, 1), dtype=torch.float64)
        dtt = torch.full_like(xt, dt_si / g["D_in"])
        rho, v, p, t = _forward_cpu(model, xt, dtt)
        rho_p.append(rho); v_p.append(v); p_p.append(p); t_p.append(t)

        rho_ex, v_ex, p_ex, t_ex = stage2_exact_state_nondim(x, dt_si, cfg)
        rho_e.append(torch.as_tensor(rho_ex).reshape(-1, 1))
        v_e.append(torch.as_tensor(v_ex).reshape(-1, 1))
        p_e.append(torch.as_tensor(p_ex).reshape(-1, 1))
        t_e.append(torch.as_tensor(t_ex).reshape(-1, 1))

    rho_p, v_p, p_p, t_p = (torch.cat(z) for z in (rho_p, v_p, p_p, t_p))
    rho_e, v_e, p_e, t_e = (torch.cat(z) for z in (rho_e, v_e, p_e, t_e))
    return {
        "rel_l2_rho": relative_l2(rho_p, rho_e),
        "rel_l2_V": relative_l2(v_p, v_e),
        "rel_l2_p": relative_l2(p_p, p_e),
        "rel_l2_T": relative_l2(t_p, t_e),
    }


def full_report_stage2(model, cfg: dict) -> dict:
    """All Stage 2 acceptance-gate metrics."""
    m = validate_held_out_stage2(model, cfg)
    gate = cfg["stage2"]["acceptance"]["rel_l2"]
    m["gates"] = {k: v < gate for k, v in m.items()}
    m["all_passed"] = all(m["gates"].values())
    return m
