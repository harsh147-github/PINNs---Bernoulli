"""Load a trained Stage 1 surrogate, sweep the throat diameter, and export
ParaView (.vtu), MATLAB (.mat), CSV, and figures for the whole design space.

Usage:
    python scripts/run_parametric_sweep.py --checkpoint results/checkpoints/stage1_final.pt \\
        --dt-min 0.20 --dt-max 0.45 --n 25
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.export import export_csv, export_mat, export_vtu
from src.networks import build_network
from src.visualize import plot_fields_vs_analytical, plot_throat_sweep, plot_total_head_error_map, plot_loss_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/stage1_final.pt")
    parser.add_argument("--dt-min", type=float, default=0.20)
    parser.add_argument("--dt-max", type=float, default=0.45)
    parser.add_argument("--n", type=int, default=25)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    stage = ckpt.get("stage", 1)

    model = build_network(cfg, stage=stage)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dt_values_si = list(np.linspace(args.dt_min, args.dt_max, args.n))
    D_in = cfg["geometry"]["D_in"]
    dt_values_tilde = [d / D_in for d in dt_values_si]

    print(f"Sweeping {args.n} throat diameters from {args.dt_min} to {args.dt_max} m ...")

    vtu_paths = export_vtu(model, cfg, dt_values_si)
    print(f"Wrote {len(vtu_paths)} .vtu files to results/paraview/")

    mat_path = export_mat(model, cfg, dt_values_si)
    print(f"Wrote {mat_path}")

    csv_path = export_csv(model, cfg, dt_values_si)
    print(f"Wrote {csv_path}")

    fig_dir = os.path.join(cfg["export"]["results_dir"], "figures")
    plot_fields_vs_analytical(model, dt_values_tilde[:4] or dt_values_tilde,
                               os.path.join(fig_dir, "fields_vs_analytical.png"))
    plot_throat_sweep(model, dt_values_tilde, os.path.join(fig_dir, "throat_sweep.png"))
    plot_total_head_error_map(model, os.path.join(fig_dir, "total_head_error_map.png"))
    if "history" in ckpt and ckpt["history"]["epoch"]:
        plot_loss_history(ckpt["history"], os.path.join(fig_dir, "loss_history.png"))
    print(f"Wrote figures to {fig_dir}/")

    assert len(vtu_paths) == args.n, f"expected {args.n} .vtu files, got {len(vtu_paths)}"
    print("Parametric sweep complete.")


if __name__ == "__main__":
    main()
