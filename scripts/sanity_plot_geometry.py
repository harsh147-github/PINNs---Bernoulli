"""Quick sanity plot for src/geometry.py: D(x; Dt) and A(x; Dt) for several throat diameters.

Run: python scripts/sanity_plot_geometry.py
Writes results/figures/geometry_sanity.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import yaml

from src import geometry as geo

if __name__ == "__main__":
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    L = cfg["geometry"]["L"]
    D_in = cfg["geometry"]["D_in"]
    Dt_min, Dt_max = cfg["geometry"]["Dt_min"], cfg["geometry"]["Dt_max"]

    x = torch.linspace(0.0, L, 300, dtype=torch.float64)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for Dt_val in [Dt_min, 0.30, 0.35, Dt_max]:
        Dt = torch.full_like(x, Dt_val)
        D = geo.diameter(x, Dt, D_in=D_in, L=L)
        A = geo.area(x, Dt, D_in=D_in, L=L)
        ax1.plot(x.numpy(), D.numpy(), label=f"Dt={Dt_val:.2f} m")
        ax2.plot(x.numpy(), A.numpy(), label=f"Dt={Dt_val:.2f} m")

    ax1.set_xlabel("x [m]"); ax1.set_ylabel("D(x) [m]"); ax1.set_title("Nozzle diameter")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("A(x) [m^2]"); ax2.set_title("Cross-sectional area")
    for ax in (ax1, ax2):
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs("results/figures", exist_ok=True)
    out_path = "results/figures/geometry_sanity.png"
    plt.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")
