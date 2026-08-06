"""Training loop: Adam (bulk) -> L-BFGS (polish), loss-weight annealing,
checkpointing, logging (CLAUDE.md Section 6, README Sections 4.3 & 9).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict

import torch
from tqdm import tqdm

from src import losses, sampling
from src.evaluate import evaluate_held_out, relative_l2
from src import analytical


def get_device(cfg: Dict) -> torch.device:
    choice = cfg.get("device", "auto")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_logger(stage: int, results_dir: str) -> logging.Logger:
    os.makedirs(os.path.join(results_dir, "logs"), exist_ok=True)
    log_path = os.path.join(results_dir, "logs", f"train_stage{stage}.log")
    logger = logging.getLogger(f"train_stage{stage}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


def _compute_terms(model, stage: int, xt_f, dtt_f, xt_b, dtt_b, hard_bc: bool, phys: Dict):
    if stage == 1:
        return losses.stage1_loss_terms(model, xt_f, dtt_f, xt_b, dtt_b, hard_bc=hard_bc)
    return losses.stage2_loss_terms(
        model, xt_f, dtt_f,
        gamma=phys["gamma"], R=phys["R"],
        V_in=phys["V_in"], p_in=phys["p_in"], T_in=phys["T_in"],
        xt_b=xt_b, dtt_b=dtt_b, hard_bc=hard_bc,
    )


def train(model: torch.nn.Module, cfg: Dict, stage: int, epochs_override: int = None,
          lbfgs_max_iter_override: int = None) -> Dict:
    """Train `model` (a `networks.PINN`) under `cfg` for the given `stage` (1 or 2).

    Returns a history dict (also embedded in the saved checkpoint) with, per
    logged epoch: loss terms, lambdas, and held-out relative-L2 metrics.
    """
    device = get_device(cfg)
    model.to(device)
    sampling.seed_everything(cfg["seed"])

    geom, samp, loss_cfg, train_cfg, phys = (
        cfg["geometry"], cfg["sampling"], cfg["loss"], cfg["training"], cfg["physics"]
    )
    results_dir = cfg["export"]["results_dir"]
    logger = _make_logger(stage, results_dir)

    hard_bc = cfg["network"]["hard_bc"]
    dt_min_t = geom["Dt_nondim_min"]
    dt_max_t = geom["Dt_nondim_max"]

    def resample():
        xt_f, dtt_f = sampling.lhs_collocation(
            samp["n_collocation_x"], samp["n_dt_values"], seed=cfg["seed"] + epoch,
            dt_min=dt_min_t, dt_max=dt_max_t,
        )
        return xt_f.to(device), dtt_f.to(device)

    epoch = 0
    xt_f, dtt_f = resample()
    if not hard_bc:
        xt_b, dtt_b = sampling.boundary_points(samp["n_boundary"], seed=cfg["seed"] + 1,
                                                dt_min=dt_min_t, dt_max=dt_max_t)
        xt_b, dtt_b = xt_b.to(device), dtt_b.to(device)
    else:
        xt_b = dtt_b = None

    lambdas = {"continuity": loss_cfg["lambda_cont"], "momentum": loss_cfg["lambda_mom"]}
    if not hard_bc:
        lambdas["bc"] = loss_cfg["lambda_bc"]
    if stage == 2:
        lambdas["energy"] = 1.0
        lambdas["eos"] = 1.0
        if loss_cfg.get("stage2_lambda_mom_boost"):
            lambdas["momentum"] = loss_cfg["stage2_lambda_mom_boost"]

    adam_cfg = train_cfg["optimizer_adam"]
    epochs = epochs_override if epochs_override is not None else adam_cfg["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=adam_cfg["lr_decay_every"], gamma=adam_cfg["lr_decay_gamma"]
    )

    history = {"epoch": [], "loss_total": [], "terms": [], "lambdas": [], "rel_l2_V": [], "rel_l2_p": []}
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"=== Stage {stage} training: {epochs} Adam epochs, device={device} ===")
    logger.info(f"Network parameters: {model.count_parameters()}")

    t0 = time.time()
    pbar = tqdm(range(1, epochs + 1), desc=f"Stage{stage} Adam")
    for epoch in pbar:
        if epoch > 1 and (epoch - 1) % samp["resample_every"] == 0:
            xt_f, dtt_f = resample()

        optimizer.zero_grad()
        terms = _compute_terms(model, stage, xt_f, dtt_f, xt_b, dtt_b, hard_bc, phys)
        loss = losses.weighted_total(terms, lambdas)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if loss_cfg["loss_weighting"] == "annealing" and epoch % loss_cfg["annealing_every"] == 0:
            terms_fresh = _compute_terms(model, stage, xt_f, dtt_f, xt_b, dtt_b, hard_bc, phys)
            lambdas = losses.update_lambdas_annealing(
                model, terms_fresh, lambdas, reference="momentum", alpha=loss_cfg["annealing_alpha"],
                lambda_min=loss_cfg["annealing_lambda_min"], lambda_max=loss_cfg["annealing_lambda_max"],
            )

        if epoch % train_cfg["log_every"] == 0 or epoch == epochs:
            with torch.no_grad():
                metrics = evaluate_held_out(model, cfg, device=device) if stage == 1 else {}
            term_vals = {k: v.item() for k, v in terms.items()}
            history["epoch"].append(epoch)
            history["loss_total"].append(loss.item())
            history["terms"].append(term_vals)
            history["lambdas"].append(dict(lambdas))
            history["rel_l2_V"].append(metrics.get("rel_l2_V"))
            history["rel_l2_p"].append(metrics.get("rel_l2_p"))
            pbar.set_postfix(loss=f"{loss.item():.3e}")
            logger.info(
                f"epoch {epoch:6d}  loss={loss.item():.6e}  terms={term_vals}  "
                f"lambdas={ {k: round(v, 3) for k, v in lambdas.items()} }  "
                f"relL2_V={metrics.get('rel_l2_V')}  relL2_p={metrics.get('rel_l2_p')}"
            )

        if epoch % train_cfg["checkpoint_every"] == 0:
            _save_checkpoint(model, cfg, history, stage, os.path.join(ckpt_dir, f"stage{stage}_epoch{epoch}.pt"))

    logger.info(f"Adam phase done in {time.time() - t0:.1f}s. Starting L-BFGS polish.")

    lbfgs_cfg = train_cfg["optimizer_lbfgs"]
    lbfgs_max_iter = lbfgs_max_iter_override if lbfgs_max_iter_override is not None else lbfgs_cfg["max_iter"]
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_cfg["history_size"],
        line_search_fn=lbfgs_cfg["line_search_fn"],
        tolerance_grad=lbfgs_cfg["tolerance_grad"],
    )

    def closure():
        lbfgs.zero_grad()
        terms = _compute_terms(model, stage, xt_f, dtt_f, xt_b, dtt_b, hard_bc, phys)
        loss = losses.weighted_total(terms, lambdas)
        loss.backward()
        return loss

    lbfgs.step(closure)

    final_terms = _compute_terms(model, stage, xt_f, dtt_f, xt_b, dtt_b, hard_bc, phys)
    final_loss = losses.weighted_total(final_terms, lambdas)
    final_metrics = evaluate_held_out(model, cfg, device=device) if stage == 1 else {}
    history["epoch"].append(epochs + lbfgs_max_iter)
    history["loss_total"].append(final_loss.item())
    history["terms"].append({k: v.item() for k, v in final_terms.items()})
    history["lambdas"].append(dict(lambdas))
    history["rel_l2_V"].append(final_metrics.get("rel_l2_V"))
    history["rel_l2_p"].append(final_metrics.get("rel_l2_p"))
    logger.info(f"L-BFGS final loss={final_loss.item():.6e}  metrics={final_metrics}")

    final_path = os.path.join(ckpt_dir, f"stage{stage}_final.pt")
    _save_checkpoint(model, cfg, history, stage, final_path)
    logger.info(f"Saved final checkpoint to {final_path}")

    return history


def _save_checkpoint(model, cfg, history, stage, path):
    torch.save({"state_dict": model.state_dict(), "config": cfg, "history": history, "stage": stage}, path)
