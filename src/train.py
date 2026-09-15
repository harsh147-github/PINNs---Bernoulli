"""FILE 7 OF 9 — The training loop. Everything in files 1-6 was building
blocks; this file is where they actually run, thousands of times, and the
network's random-garbage weights slowly turn into a working surrogate.

OBJECTIVE OF THIS FILE
-----------------------
Repeat this cycle until the loss is small (README Section 4.7 explains
each step conceptually; this is that same cycle as real code):

    1. forward pass    -- model(xt, dtt) -> current guess (networks.py)
    2. residuals        -- how wrong is that guess, physically? (physics.py)
    3. loss              -- one scalar summarizing all the wrongness (losses.py)
    4. backward pass    -- loss.backward() -> gradient for all ~21,000 weights
    5. optimizer step  -- nudge every weight slightly downhill
    6. repeat

That six-line cycle, unmodified, run 20,000 times, IS Stage 1 training.
Everything else in this file (resampling, annealing, logging,
checkpointing) is instrumentation wrapped around that one loop so you can
watch it happen and recover a trained model afterward.

TWO OPTIMIZERS, TWO JOBS (full reasoning: README Section 4.7)
--------------------------------------------------------------------
  Phase 1 -- Adam.  Adapts its own step size per weight and remembers a
  running average of recent gradients, which makes it robust to the
  wildly uneven, initially chaotic loss landscape a freshly-initialized
  network starts on. Runs for `adam_epochs` (20,000) full-batch steps,
  learning rate decaying x0.95 every `lr_decay_every` (2,000) epochs. It
  does the bulk of the work and gets the loss most of the way down.

  Phase 2 -- L-BFGS.  A quasi-Newton method: once Adam has found the
  right neighbourhood, L-BFGS uses *curvature* (how the gradient itself
  is changing, not just its current value) to take far more precise
  steps, squeezing the loss down another 1-2 orders of magnitude where
  Adam stalls. More expensive per step (it needs several forward/backward
  evaluations per iteration for its internal line search), which is
  exactly why it only runs for the last `lbfgs_max_iter` (5,000) iterations
  instead of the whole run.

WHAT `weights.maybe_anneal(...)` IS DOING RIGHT BEFORE `.backward()`
-------------------------------------------------------------------------
Look at the order in the Adam loop below: `maybe_anneal` runs BEFORE
`total.backward()`, not after. This is deliberate, not arbitrary --
`maybe_anneal` (losses.py) needs to call `torch.autograd.grad` on each
individual loss term to measure its gradient size, and that only works
while the computation graph is still alive. `backward()` frees that graph
once it runs. Swap the order and annealing silently breaks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.evaluate import relative_l2, validate_held_out
from src.losses import LossWeights
from src.sampling import boundary_points, lhs_collocation


def _sample_batch(cfg: dict, seed: int):
    g = cfg["geometry"]
    dtt_min = g["dt_min"] / g["D_in"]
    dtt_max = g["dt_max"] / g["D_in"]
    n = cfg["training"]["n_collocation_x"] * cfg["training"]["n_dt_values"]
    xt, dtt = lhs_collocation(n, dtt_min, dtt_max, seed)
    dtt_b = boundary_points(cfg["training"]["n_boundary"], dtt_min, dtt_max, seed + 1)
    return xt, dtt, dtt_b


def train(model, cfg: dict, out_dir: str, device: str = "cpu") -> dict:
    """Full Adam -> L-BFGS protocol. Returns the training history dict."""
    t = cfg["training"]
    hard_bc = cfg["network"]["hard_bc"]
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    weights = LossWeights(
        weighting=cfg["loss"]["weighting"],
        anneal_alpha=cfg["loss"]["anneal_alpha"],
        anneal_every=cfg["loss"]["anneal_every"],
        anneal_warmup=cfg["loss"]["anneal_warmup"],
        values={
            "cont": cfg["loss"]["lambda_cont"],
            "mom": cfg["loss"]["lambda_mom"],
            "bc": cfg["loss"]["lambda_bc"],
        },
    )

    xt, dtt, dtt_b = _sample_batch(cfg, cfg["seed"])
    xt, dtt, dtt_b = xt.to(device), dtt.to(device), dtt_b.to(device)

    dash = None
    if t.get("live_dashboard", False):
        from src.liveviz import LiveDashboard

        dash = LiveDashboard(
            cfg, out_dir,
            frame_every=t.get("frame_every", 200),
            live_display=t.get("live_display", False),
        )

    netviz = None
    if t.get("live_network_viz", False):
        from src.netviz import NetworkViz

        netviz = NetworkViz(
            model, cfg, out_dir,
            frame_every=t.get("netviz_frame_every", 25),
            live_display=t.get("live_display", False),
        )

    opt = torch.optim.Adam(model.parameters(), lr=t["adam_lr"])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=t["lr_decay_every"], gamma=t["lr_decay_gamma"]
    )

    history: dict[str, list] = {
        k: []
        for k in ("epoch", "loss", "loss_cont", "loss_mom", "lam_cont", "lam_mom",
                  "rel_l2_V", "rel_l2_p", "phase")
    }

    # ---------------- Phase 1: Adam ----------------
    pbar = tqdm(range(1, t["adam_epochs"] + 1), desc="Adam", unit="ep")
    for epoch in pbar:
        if epoch % t["resample_every"] == 0:
            xt, dtt, dtt_b = _sample_batch(cfg, cfg["seed"] + epoch)
            xt, dtt, dtt_b = xt.to(device), dtt.to(device), dtt_b.to(device)

        opt.zero_grad()
        total, parts = weights.total(model, xt, dtt, dtt_b, hard_bc)
        weights.maybe_anneal(epoch, model, parts)  # before backward: parts' graph is still alive
        if dash is not None:
            dash.maybe_update(epoch, model, parts, total, tag="adam")
        if netviz is not None:
            netviz.maybe_update(epoch, t["adam_epochs"], model, total.item(), mode="ADAM")
        total.backward()
        opt.step()
        sched.step()

        if epoch % t["log_every"] == 0:
            history["epoch"].append(epoch)
            history["loss"].append(total.item())
            history["loss_cont"].append(parts["cont"].item())
            history["loss_mom"].append(parts["mom"].item())
            history["lam_cont"].append(weights.values["cont"])
            history["lam_mom"].append(weights.values["mom"])
            history["phase"].append("adam")
            if epoch % t["validate_every"] == 0:
                m = validate_held_out(model, cfg)
                history["rel_l2_V"].append(m["rel_l2_V"])
                history["rel_l2_p"].append(m["rel_l2_p"])
                pbar.set_postfix(
                    loss=f"{total.item():.2e}",
                    L2_V=f"{m['rel_l2_V']:.2e}",
                    L2_p=f"{m['rel_l2_p']:.2e}",
                )
        if epoch % t["checkpoint_every"] == 0:
            _save_ckpt(model, cfg, history, out / "checkpoints" / f"stage1_ep{epoch}.pt")

    # ---------------- Phase 2: L-BFGS polish ----------------
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=t["lbfgs_max_iter"],
        history_size=t["lbfgs_history_size"],
        tolerance_grad=1e-9,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    pbar2 = tqdm(total=t["lbfgs_max_iter"], desc="L-BFGS", unit="it")
    state = {"it": 0}

    def closure():
        lbfgs.zero_grad()
        total, parts = weights.total(model, xt, dtt, dtt_b, hard_bc)
        total.backward()
        state["it"] += 1
        if state["it"] % 10 == 0:
            pbar2.update(10)
            pbar2.set_postfix(loss=f"{total.item():.2e}")
            history["epoch"].append(t["adam_epochs"] + state["it"])
            history["loss"].append(total.item())
            history["loss_cont"].append(parts["cont"].item())
            history["loss_mom"].append(parts["mom"].item())
            history["lam_cont"].append(weights.values["cont"])
            history["lam_mom"].append(weights.values["mom"])
            history["phase"].append("lbfgs")
            if netviz is not None:
                netviz.maybe_update(state["it"], t["lbfgs_max_iter"], model, total.item(), mode="L-BFGS")
        return total

    lbfgs.step(closure)
    pbar2.close()
    if netviz is not None:
        netviz.close()

    _save_ckpt(model, cfg, history, out / "checkpoints" / "stage1_final.pt")
    (out / "logs" / "train_stage1_history.json").write_text(json.dumps(history))
    return history


def _save_ckpt(model, cfg: dict, history: dict, path: Path) -> None:
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg, "history": history}, path
    )
