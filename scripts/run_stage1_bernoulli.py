"""Train the Stage 1 (incompressible Bernoulli) parametric PINN surrogate end-to-end.

Usage:
    python scripts/run_stage1_bernoulli.py --config configs/default.yaml
    python scripts/run_stage1_bernoulli.py --config configs/default.yaml --epochs 200   # smoke test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.evaluate import evaluate_held_out, save_metrics_json
from src.networks import build_network
from src.train import get_device, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override Adam epoch count (for smoke tests)")
    parser.add_argument("--lbfgs-max-iter", type=int, default=None,
                         help="override L-BFGS max_iter (for smoke tests)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["stage"] = 1

    device = get_device(cfg)
    print(f"Device: {device}")

    model = build_network(cfg, stage=1)
    print(f"Network parameters: {model.count_parameters()}")

    history = train(model, cfg, stage=1, epochs_override=args.epochs,
                     lbfgs_max_iter_override=args.lbfgs_max_iter)

    metrics = evaluate_held_out(model, cfg, device=device)
    os.makedirs(os.path.join(cfg["export"]["results_dir"], "logs"), exist_ok=True)
    save_metrics_json(metrics, os.path.join(cfg["export"]["results_dir"], "logs", "stage1_final_metrics.json"))

    print("\n=== Stage 1 acceptance gates (CLAUDE.md Section 7) ===")
    gate_V = metrics["rel_l2_V"] < 1e-3
    gate_p = metrics["rel_l2_p"] < 1e-2
    gate_head = metrics["max_bernoulli_invariant_error_pct"] < 1.0
    print(f"rel_l2(V_tilde) = {metrics['rel_l2_V']:.3e}  (< 1e-3 required) -> {'PASS' if gate_V else 'FAIL'}")
    print(f"rel_l2(p_tilde) = {metrics['rel_l2_p']:.3e}  (< 1e-2 required) -> {'PASS' if gate_p else 'FAIL'}")
    print(f"max |p_tilde + V_tilde^2 - 1| = {metrics['max_bernoulli_invariant_error_pct']:.3f}%  "
          f"(< 1% required) -> {'PASS' if gate_head else 'FAIL'}")
    if not (gate_V and gate_p and gate_head):
        print("WARNING: one or more acceptance gates FAILED. See CLAUDE.md Section 7 / Section 11.")


if __name__ == "__main__":
    main()
