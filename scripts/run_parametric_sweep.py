#!/usr/bin/env python3
"""Load a trained surrogate and sweep the throat diameter, exporting all
post-processing files (ParaView .vtu series + MATLAB .mat + sweep figure).

    python scripts/run_parametric_sweep.py \
        --checkpoint results/checkpoints/stage1_final.pt \
        --dt-min 0.20 --dt-max 0.45 --n 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export import export_mat, export_vtu
from src.networks import PINN
from src.visualize import plot_throat_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dt-min", type=float, default=0.20)
    ap.add_argument("--dt-max", type=float, default=0.45)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    n = cfg["network"]
    model = PINN(n["n_hidden_layers"], n["n_neurons"], n["activation"], n["hard_bc"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(dtype=torch.float64).eval()

    dts = np.linspace(args.dt_min, args.dt_max, args.n).tolist()
    vtu_dir = export_vtu(model, cfg, args.out, dts)
    mat_dir = export_mat(model, cfg, args.out)
    fig = plot_throat_sweep(model, cfg, args.out)
    print(f"ParaView series ({args.n} files): {vtu_dir}")
    print(f"MATLAB file: {mat_dir}/nozzle_surrogate.mat")
    print(f"sweep figure: {fig}")


if __name__ == "__main__":
    main()
