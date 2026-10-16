"""Physics-only training loop: Adam then L-BFGS (README §5).

No CFD data enters the loss. The loss is

    L = w_c  <cont^2>  + w_m (<mom_x^2> + <mom_r^2>)  + w_o (<(du/dx)^2> + <v^2>)_outlet

averaged over collocation points that span every nozzle in the design space.
OpenFOAM solutions, if present, are used ONLY to *monitor* accuracy during
training (``monitor``), never to compute gradients.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .geometry import Nozzle
from .model import NozzlePINN
from .physics import ns_residuals, outlet_residuals
from .sampling import draw


def build_model(cfg: Dict[str, Any]) -> NozzlePINN:
    """Construct the network described in the config (float64)."""
    torch.set_default_dtype(torch.float64)
    noz = Nozzle.from_config(cfg)
    net = cfg["network"]
    return NozzlePINN(noz, float(cfg["physics"]["Re"]), int(net["hidden_width"]),
                      int(net["hidden_layers"]), int(net["quad_nodes"]), str(net["architecture"]))


def loss_terms(model: NozzlePINN, Xi, di, Xo, do, Re: float) -> Dict[str, torch.Tensor]:
    """Mean-squared residual of every equation."""
    res = ns_residuals(model, Xi, di, Re)
    out = outlet_residuals(model, Xo, do)
    return {
        "cont": (res["cont"] ** 2).mean(),
        "mom_x": (res["mom_x"] ** 2).mean(),
        "mom_r": (res["mom_r"] ** 2).mean(),
        "outlet": (out["dudx"] ** 2).mean() + (out["v"] ** 2).mean(),
    }


def total_loss(terms: Dict[str, torch.Tensor], tc: Dict[str, Any]) -> torch.Tensor:
    return (float(tc["w_continuity"]) * terms["cont"]
            + float(tc["w_momentum"]) * (terms["mom_x"] + terms["mom_r"])
            + float(tc["w_outlet"]) * terms["outlet"])


def train(cfg: Dict[str, Any], out_dir: Path,
          monitor: Optional[Callable[[NozzlePINN], Dict[str, float]]] = None,
          monitor_every: int = 500) -> Dict[str, Any]:
    """Train the parametric PINN; return the history dictionary.

    Writes ``checkpoints/*.pt``, ``history.json`` and ``model_final.pt`` to
    ``out_dir``. Wall-clock time for each stage is recorded.
    """
    tc = cfg["training"]
    Re = float(cfg["physics"]["Re"])
    torch.manual_seed(int(tc["seed"]))
    np.random.seed(int(tc["seed"]))
    torch.set_num_threads(int(tc["threads"]))
    gen = torch.Generator().manual_seed(int(tc["seed"]))

    model = build_model(cfg)
    noz = model.noz
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    hist: Dict[str, Any] = {"epoch": [], "time_s": [], "loss": [], "cont": [], "mom_x": [],
                            "mom_r": [], "outlet": [], "lr": [], "stage": [], "monitor": []}

    def log(ep: int, stage: str, terms: Dict[str, torch.Tensor], loss: torch.Tensor, lr: float, t: float):
        hist["epoch"].append(ep)
        hist["time_s"].append(t)
        hist["loss"].append(float(loss.detach()))
        for k in ("cont", "mom_x", "mom_r", "outlet"):
            hist[k].append(float(terms[k].detach()))
        hist["lr"].append(lr)
        hist["stage"].append(stage)

    def do_monitor(ep: int, t: float):
        if monitor is None:
            return
        m = monitor(model)
        m.update({"epoch": ep, "time_s": t})
        hist["monitor"].append(m)
        print("   monitor " + "  ".join(f"{k}={v:.3e}" for k, v in m.items()
                                        if k not in ("epoch", "time_s")), flush=True)

    # ------------------------------ Adam -----------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["adam_lr"]))
    sched = torch.optim.lr_scheduler.StepLR(opt, int(tc["lr_decay_every"]), float(tc["lr_decay_factor"]))
    n_adam = int(tc["adam_epochs"])
    resample = int(cfg["sampling"]["resample_every"])
    t0 = time.perf_counter()
    Xi = di = Xo = do = None
    for ep in range(n_adam):
        if ep % resample == 0:
            # first draw is space-filling; later draws are residual-adaptive
            Xi, di, Xo, do = draw(noz, cfg, gen, model=model if ep > 0 else None, Re=Re)
        terms = loss_terms(model, Xi, di, Xo, do, Re)
        loss = total_loss(terms, tc)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if ep % int(tc["log_every"]) == 0 or ep == n_adam - 1:
            t = time.perf_counter() - t0
            log(ep, "adam", terms, loss, sched.get_last_lr()[0], t)
            print(f"[adam {ep:6d}] loss={hist['loss'][-1]:.3e} cont={hist['cont'][-1]:.2e} "
                  f"mx={hist['mom_x'][-1]:.2e} mr={hist['mom_r'][-1]:.2e} "
                  f"out={hist['outlet'][-1]:.2e} lr={hist['lr'][-1]:.1e} t={t:.0f}s", flush=True)
        if monitor is not None and (ep % monitor_every == 0 or ep == n_adam - 1):
            do_monitor(ep, time.perf_counter() - t0)
        if (ep + 1) % int(tc["checkpoint_every"]) == 0:
            torch.save(model.state_dict(), out_dir / "checkpoints" / f"adam_{ep + 1:06d}.pt")
    t_adam = time.perf_counter() - t0

    # ------------------------------ L-BFGS ---------------------------------
    n_lbfgs = int(tc["lbfgs_iters"])
    t1 = time.perf_counter()
    if n_lbfgs > 0:
        Xi, di, Xo, do = draw(noz, cfg, gen, n_interior=int(tc["lbfgs_points"]), model=model, Re=Re)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=n_lbfgs,
                                  history_size=int(tc["lbfgs_history"]),
                                  tolerance_grad=1e-12, tolerance_change=1e-15,
                                  line_search_fn="strong_wolfe")
        state = {"n": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            terms = loss_terms(model, Xi, di, Xo, do, Re)
            loss = total_loss(terms, tc)
            loss.backward()
            k = state["n"]
            state["n"] += 1
            if k % int(tc["log_every"]) == 0:
                t = time.perf_counter() - t0
                log(n_adam + k, "lbfgs", terms, loss, 1.0, t)
                print(f"[lbfgs {k:5d}] loss={hist['loss'][-1]:.3e} cont={hist['cont'][-1]:.2e} "
                      f"mx={hist['mom_x'][-1]:.2e} mr={hist['mom_r'][-1]:.2e} "
                      f"out={hist['outlet'][-1]:.2e} t={t:.0f}s", flush=True)
                if monitor is not None and k % monitor_every == 0:
                    do_monitor(n_adam + k, t)
            return loss

        lbfgs.step(closure)
        hist["lbfgs_evaluations"] = state["n"]
    t_lbfgs = time.perf_counter() - t1

    terms = loss_terms(model, Xi, di, Xo, do, Re)
    final = float(total_loss(terms, tc).detach())
    hist["final_loss"] = final
    hist["final_terms"] = {k: float(v.detach()) for k, v in terms.items()}
    hist["time_adam_s"] = t_adam
    hist["time_lbfgs_s"] = t_lbfgs
    hist["time_total_s"] = t_adam + t_lbfgs
    hist["n_parameters"] = sum(p.numel() for p in model.parameters())
    do_monitor(n_adam + hist.get("lbfgs_evaluations", 0), t_adam + t_lbfgs)

    torch.save({"state_dict": model.state_dict(), "config": cfg}, out_dir / "model_final.pt")
    (out_dir / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"[train] done: Adam {t_adam:.0f}s + L-BFGS {t_lbfgs:.0f}s = {t_adam + t_lbfgs:.0f}s, "
          f"final loss {final:.3e}", flush=True)
    return hist


def load_model(path: Path) -> NozzlePINN:
    """Rebuild a trained model from ``model_final.pt`` (config embedded)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
