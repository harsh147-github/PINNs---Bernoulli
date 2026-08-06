#!/usr/bin/env python3
"""Train the Stage 2 compressible quasi-1D Euler surrogate end-to-end.

    python scripts/run_stage2_compressible.py --config configs/default.yaml

Pipeline: seed -> build PINNStage2 -> Adam -> L-BFGS -> validate against the
isentropic subsonic area-Mach solution (held-out throat diameters) -> export
CSV -> figures. Prints the acceptance-gate report at the end (CLAUDE.md
Section 6: "same train/evaluate/export/visualize pipeline, same
acceptance-gate printout" as Stage 1).
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

from src.evaluate import full_report_stage2
from src.export import export_csv_stage2
from src.networks import PINNStage2
from src.train import train_stage2
from src.visualize import make_all_figures_stage2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--adam-epochs", type=int, default=None, help="override config")
    ap.add_argument("--lbfgs-iter", type=int, default=None, help="override config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["stage"] = 2
    # Stage 2 (air) needs its own inlet velocity, independent of Stage 1's
    # (water) V_in in cfg["physics"] -- see configs/default.yaml's stage2.physics
    # comment. Every downstream call (train_stage2, evaluate, export, visualize)
    # receives this same cfg object, so overriding it once here is sufficient.
    cfg["physics"]["V_in"] = cfg["stage2"]["physics"]["V_in"]
    if args.adam_epochs is not None:
        cfg["stage2"]["training"]["adam_epochs"] = args.adam_epochs
    if args.lbfgs_iter is not None:
        cfg["stage2"]["training"]["lbfgs_max_iter"] = args.lbfgs_iter

    # deterministic seeds (README Section 9 / CLAUDE.md Section 5)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    n = cfg["stage2"]["network"]
    model = PINNStage2(
        n_hidden_layers=n["n_hidden_layers"],
        n_neurons=n["n_neurons"],
        activation=n["activation"],
        hard_bc=n["hard_bc"],
        hard_pressure_bc=n["hard_pressure_bc"],
    ).to(device=device, dtype=torch.float64)
    print(f"PINNStage2: {n['n_hidden_layers']}x{n['n_neurons']} {n['activation']}, "
          f"hard_bc={n['hard_bc']}, hard_pressure_bc={n['hard_pressure_bc']}, "
          f"parameters={model.count_parameters()}")

    history = train_stage2(model, cfg, args.out, device)

    report = full_report_stage2(model, cfg)
    Path(args.out, "logs").mkdir(parents=True, exist_ok=True)
    Path(args.out, "logs", "stage2_acceptance_report.json").write_text(json.dumps(report, indent=2))

    export_csv_stage2(model, cfg, args.out)
    figs = make_all_figures_stage2(model, cfg, history, args.out)

    gate = cfg["stage2"]["acceptance"]["rel_l2"]
    print("\n===== STAGE 2 ACCEPTANCE GATES (CLAUDE.md Section 6) =====")
    for name in ("rel_l2_rho", "rel_l2_V", "rel_l2_p", "rel_l2_T"):
        status = "PASS" if report["gates"][name] else "FAIL"
        print(f"{name:12s}: {report[name]:.3e}  (gate < {gate:.0e})  -> {status}")
    print(f"ALL GATES: {'PASSED' if report['all_passed'] else 'FAILED'}")
    print("\nfigures:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
