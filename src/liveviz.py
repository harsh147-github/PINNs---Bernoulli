"""Live learning dashboard — watch the PINN learn in real time.

Every `frame_every` epochs during training this saves a 3-panel frame:
  1. loss curves (log scale) — how much the equations are still violated
  2. V~(x) the network currently "guesses" vs the exact Bernoulli solution
  3. p~(x) current guess vs exact

On a machine with a display, set `live_display: true` in the config and the
dashboard updates in a window epoch by epoch. Headless (servers, Colab):
frames land in results/frames/ and can be stitched into a GIF/MP4 replay:

    python scripts/make_learning_replay.py --frames results/frames --out results/learning_replay.gif
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analytical import pressure_exact_nondim, velocity_exact_nondim


class LiveDashboard:
    """Periodic training dashboard. Save-only by default; interactive on demand."""

    def __init__(
        self,
        cfg: dict,
        out_dir: str,
        frame_every: int = 200,
        live_display: bool = False,
    ) -> None:
        self.cfg = cfg
        self.frame_every = frame_every
        self.live_display = live_display
        self.dir = Path(out_dir) / "frames"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hist: dict[str, list] = {"epoch": [], "cont": [], "mom": [], "total": []}
        self.dtt_probe = [0.4, 0.65, 0.9]  # nondim throat diameters to watch
        if live_display:
            plt.ion()

    def maybe_update(
        self, epoch: int, model, parts: dict, total: torch.Tensor, tag: str = "adam"
    ) -> None:
        if epoch % self.frame_every != 0:
            return
        self.hist["epoch"].append(epoch)
        self.hist["cont"].append(parts["cont"].item())
        self.hist["mom"].append(parts["mom"].item())
        self.hist["total"].append(total.item())

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        fig.suptitle(f"PINN learning live — {tag} epoch {epoch}", fontsize=13)

        ax = axes[0]
        ax.semilogy(self.hist["epoch"], self.hist["total"], "k-", label="total")
        ax.semilogy(self.hist["epoch"], self.hist["cont"], label="continuity")
        ax.semilogy(self.hist["epoch"], self.hist["mom"], label="Bernoulli/momentum")
        ax.set(xlabel="epoch", ylabel="loss", title="Equation violation (log scale)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        xt = torch.linspace(0, 1, 300, dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            for dtt_v in self.dtt_probe:
                dtt = torch.full_like(xt, dtt_v)
                v, p = model(xt, dtt)
                ve = velocity_exact_nondim(xt, dtt)
                pe = pressure_exact_nondim(xt, dtt)
                (l,) = axes[1].plot(xt, v, label=f"$\\tilde D_t$={dtt_v}")
                axes[1].plot(xt, ve, "--", color=l.get_color(), alpha=0.5)
                (l,) = axes[2].plot(xt, p, label=f"$\\tilde D_t$={dtt_v}")
                axes[2].plot(xt, pe, "--", color=l.get_color(), alpha=0.5)
        axes[1].set(xlabel="$\\tilde x$", ylabel="$\\tilde V$",
                    title="Velocity guess (solid) vs exact (dashed)")
        axes[2].set(xlabel="$\\tilde x$", ylabel="$\\tilde p$",
                    title="Pressure guess (solid) vs exact (dashed)")
        for ax in axes[1:]:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.dir / f"frame_{tag}_{epoch:06d}.png", dpi=110)
        if self.live_display:
            plt.pause(0.01)
            plt.close(fig)
        else:
            plt.close(fig)
