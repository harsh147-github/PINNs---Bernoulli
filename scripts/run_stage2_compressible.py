"""Train the Stage 2 (compressible quasi-1D Euler, subsonic branch) PINN surrogate.

Usage:
    python scripts/run_stage2_compressible.py --config configs/default.yaml
    python scripts/run_stage2_compressible.py --config configs/default.yaml --epochs 200   # smoke test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml

from src import analytical
from src.evaluate import relative_l2
from src.networks import build_network
from src.train import get_device, train


def evaluate_stage2_held_out(model, cfg, device) -> dict:
    """Rel-L2 of the PINN's Mach & pressure vs. the isentropic subsonic relations
    (CLAUDE.md Section 7: "Mach/pressure vs isentropic relations, rel-L2 < 1e-2")."""
    geom, phys, samp = cfg["geometry"], cfg["physics"], cfg["sampling"]
    L, D_in = geom["L"], geom["D_in"]
    V_in, p_in, T_in = phys["V_in"], phys["p_in"], phys["T_in"]
    gamma, R = phys["gamma"], phys["R"]

    n_x = 200
    all_p_pred, all_p_exact, all_M_pred, all_M_exact = [], [], [], []
    with torch.no_grad():
        for Dt_si in samp["validation_dt_holdout"]:
            x = np.linspace(0.0, L, n_x)
            rho_t, V_t, p_t, T_t = analytical.stage2_exact_state_nondim(
                x, Dt_si, D_in, L, V_in, p_in, T_in, gamma, R
            )
            T_si = T_t * T_in
            V_si = V_t * V_in
            M_exact = V_si / np.sqrt(gamma * R * T_si)

            xt = torch.tensor(x / L, dtype=torch.float64, device=device).reshape(-1, 1)
            dtt = torch.full_like(xt, Dt_si / D_in)
            out = model(xt, dtt)
            rho_p, V_p, p_p, T_p = (out[..., i].cpu().numpy() for i in range(4))
            M_pred = (V_p * V_in) / np.sqrt(gamma * R * (T_p * T_in))

            all_p_pred.append(p_p); all_p_exact.append(p_t)
            all_M_pred.append(M_pred); all_M_exact.append(M_exact)

    p_pred = torch.tensor(np.concatenate(all_p_pred))
    p_exact = torch.tensor(np.concatenate(all_p_exact))
    M_pred = torch.tensor(np.concatenate(all_M_pred))
    M_exact = torch.tensor(np.concatenate(all_M_exact))
    return {
        "rel_l2_pressure": relative_l2(p_pred, p_exact),
        "rel_l2_mach": relative_l2(M_pred, M_exact),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lbfgs-max-iter", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["stage"] = 2

    device = get_device(cfg)
    print(f"Device: {device}")

    model = build_network(cfg, stage=2)
    print(f"Network parameters: {model.count_parameters()}")

    train(model, cfg, stage=2, epochs_override=args.epochs, lbfgs_max_iter_override=args.lbfgs_max_iter)

    metrics = evaluate_stage2_held_out(model, cfg, device)
    print("\n=== Stage 2 acceptance gate (CLAUDE.md Section 7) ===")
    gate_p = metrics["rel_l2_pressure"] < 1e-2
    gate_M = metrics["rel_l2_mach"] < 1e-2
    print(f"rel_l2(pressure) = {metrics['rel_l2_pressure']:.3e}  (< 1e-2 required) -> {'PASS' if gate_p else 'FAIL'}")
    print(f"rel_l2(Mach)     = {metrics['rel_l2_mach']:.3e}  (< 1e-2 required) -> {'PASS' if gate_M else 'FAIL'}")
    if not (gate_p and gate_M):
        print("WARNING: one or more Stage 2 acceptance gates FAILED. See CLAUDE.md Section 7 / Section 11.")


if __name__ == "__main__":
    main()
