#!/usr/bin/env python3
"""Train the Stage 1 parametric Bernoulli surrogate end-to-end.

    python scripts/run_stage1_bernoulli.py --config configs/default.yaml

Pipeline: seed -> build PINN -> Adam -> L-BFGS -> validate against the
analytical solution (held-out throat diameters) -> export CSV/.vtu/.mat ->
figures. Prints the acceptance-gate report at the end.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import full_report
from src.export import export_csv, export_mat, export_vtu
from src.networks import PINN
from src.train import train
from src.visualize import make_all_figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--adam-epochs", type=int, default=None, help="override config")
    ap.add_argument("--lbfgs-iter", type=int, default=None, help="override config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.adam_epochs is not None:
        cfg["training"]["adam_epochs"] = args.adam_epochs
    if args.lbfgs_iter is not None:
        cfg["training"]["lbfgs_max_iter"] = args.lbfgs_iter

    # deterministic seeds (README Section 9)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    n = cfg["network"]
    model = PINN(
        n_hidden_layers=n["n_hidden_layers"],
        n_neurons=n["n_neurons"],
        activation=n["activation"],
        hard_bc=n["hard_bc"],
    ).to(device=device, dtype=torch.float64)
    print(f"PINN: {n['n_hidden_layers']}x{n['n_neurons']} {n['activation']}, "
          f"hard_bc={n['hard_bc']}, parameters={model.count_parameters()}")

    history = train(model, cfg, args.out, device)

    report = full_report(model, cfg)
    Path(args.out, "logs").mkdir(parents=True, exist_ok=True)
    Path(args.out, "logs", "acceptance_report.json").write_text(json.dumps(report, indent=2))

    export_csv(model, cfg, args.out)
    dts = np.linspace(cfg["geometry"]["dt_min"], cfg["geometry"]["dt_max"],
                      cfg["export"]["sweep_n"]).tolist()
    export_vtu(model, cfg, args.out, dts)
    export_mat(model, cfg, args.out)
    figs = make_all_figures(model, cfg, history, args.out)

    print("\n===== ACCEPTANCE GATES (README Section 9) =====")
    print(f"rel-L2(V~) on held-out Dt : {report['rel_l2_V']:.3e}  "
          f"(gate < {cfg['acceptance']['rel_l2_V']:.0e})  -> {'PASS' if report['gates']['rel_l2_V'] else 'FAIL'}")
    print(f"rel-L2(p~) on held-out Dt : {report['rel_l2_p']:.3e}  "
          f"(gate < {cfg['acceptance']['rel_l2_p']:.0e})  -> {'PASS' if report['gates']['rel_l2_p'] else 'FAIL'}")
    print(f"max |p~+V~^2-1|           : {report['total_head_max_err']:.3e}  "
          f"(gate < {cfg['acceptance']['total_head_tol']:.0e})  -> {'PASS' if report['gates']['total_head'] else 'FAIL'}")
    print(f"ALL GATES: {'PASSED' if report['all_passed'] else 'FAILED'}")
    print("\nfigures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
