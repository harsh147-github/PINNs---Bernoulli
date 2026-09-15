"""FILE 8 OF 9 — Grading the trained network, honestly.

OBJECTIVE OF THIS FILE
-----------------------
Everything up to this point never once told the network the correct
answer -- it only ever saw whether it violated the equations (files 3-5).
This file is the first (and only) place file 2's exact answer
(analytical.py) actually gets compared against the network's output.
It's deliberately kept separate from training: if grading logic leaked
into the loss, "the network learned the physics" and "the network
memorized the grading points" would become impossible to tell apart.

WHY GRADE ONLY ON `held_out_dt` (sampling.py's held-out throat diameters)
-------------------------------------------------------------------------------
This is the actual test of whether training worked. A network graded on
throat diameters it trained on could be passing simply by memorizing
those specific curves -- not by having learned continuity and Bernoulli
in general. Grading exclusively on Dt values that never once appeared in
any `lhs_collocation` draw (sampling.py) closes that loophole: the only
way to score well here is for the governing equations to actually hold in
the network's output, since that's the only signal it was ever given.

`relative_l2` -- THE METRIC, IN PLAIN TERMS
-----------------------------------------------
    rel_l2 = ||prediction - exact|| / ||exact||

The numerator is "how far off is the whole prediction, as one number"
(the Euclidean length of the error vector across every validation point
at once). Dividing by the exact solution's own length turns that into a
*relative* error -- 1e-4 means "off by about 0.01% of the true answer's
overall scale," independent of whatever units or magnitude V_tilde/p_tilde
happen to be in. That's what the acceptance gates (README Section 9,
CLAUDE.md Section 7) actually threshold against.

WHAT THE THREE GATES ARE CHECKING, ONE EACH
-------------------------------------------------
    rel_l2_V     < 1e-3   -- does the velocity field actually match?
    rel_l2_p     < 1e-2   -- does pressure match? (looser: p depends on V^2,
                              so it amplifies whatever error V already has)
    total_head   < 1%     -- is Bernoulli's constant ACTUALLY constant in
                              the trained output, or does it drift? This is
                              the one gate that isn't just "close to the
                              exact solution" -- it's a direct physics
                              self-consistency check on the network alone.
"""

from __future__ import annotations

import torch

from src.analytical import pressure_exact_nondim, velocity_exact_nondim
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
