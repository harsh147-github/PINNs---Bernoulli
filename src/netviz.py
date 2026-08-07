"""Live network-architecture visualizer — watch the actual weights during training.

This is a different view from `liveviz.py` (which plots loss curves and
V(x)/p(x) guesses). This one draws the network itself: every neuron as a
dot, every weight as a line between two dots, colored and thickened by
that weight's current value, redrawn periodically as training runs. Early
in training every connection looks similar (small random weights, all
faint); watching it over a few thousand epochs you can see specific
connections darken and strengthen while others fade — that's training,
made visible, not a metaphor for it.

WHY LineCollection AND NOT ONE `ax.plot()` PER CONNECTION
------------------------------------------------------------
This network has ~21,000 weights, i.e. ~21,000 lines to draw. Calling
`ax.plot()` once per line would create 21,000 separate matplotlib
artists -- far too slow to redraw live. `matplotlib.collections.
LineCollection` batches many line segments into a single artist with one
array of colors/widths, so updating "all the weights in one layer" is one
numpy-vectorized call, not a Python loop over thousands of plot objects.

HOW A WEIGHT NUMBER BECOMES A LINE'S COLOR
----------------------------------------------
Each linear layer's weight matrix W has shape (n_out, n_in) -- W[j, i] is
the weight on the connection from input neuron i to output neuron j
(`networks.py`'s `raw()` computes exactly `W @ z + b` for each layer).
Positive weights are drawn red, negative blue (a standard diverging
colormap, "coolwarm"), and BOTH the opacity and the line width scale with
|W[j,i]| relative to that layer's largest weight -- a connection near
zero is nearly invisible, the layer's strongest connection is fully
opaque and thick. That's the same design the reference visualizers (e.g.
Tensorflow Playground) use, and it's what makes "training" visually mean
something: you're watching this scaling factor change, live.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

# matplotlib's automatic backend detection sometimes falls back to the
# headless Agg backend even on a machine with a real display (e.g. when
# stdout/stderr are piped, as they are when this training script runs
# under a wrapping tool). Force an interactive backend explicitly so
# `live_display=True` actually opens a window instead of silently no-op'ing
# (matplotlib only warns "FigureCanvasAgg is non-interactive" and moves on).
try:
    matplotlib.use("TkAgg")
except Exception:
    pass  # fall back to whatever matplotlib already picked (headless-safe)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from torch import nn


def _linear_layers(model: nn.Module) -> list[nn.Linear]:
    """Every nn.Linear in the model, in forward-pass order (hidden layers, then output head)."""
    linears = [m for m in model.hidden if isinstance(m, nn.Linear)]
    linears.append(model.output)
    return linears


class NetworkViz:
    """Live (or frame-saved) node-link diagram of the PINN's actual weights.

    Parameters
    ----------
    model : the PINN (or PINNStage2) instance being trained.
    out_dir : where to save frames (results/netviz_frames/), used whether
        or not `live_display` is also on.
    frame_every : redraw every this many epochs. ~20,000 weights per
        redraw is cheap, but redrawing every single epoch (tens of
        thousands of times over a run) is wasted work -- default matches
        `liveviz.py`'s `frame_every`.
    live_display : pop an interactive window that updates in place
        (needs a real display -- your laptop, not a headless server).
    probe : one (x_tilde, Dt_tilde) point whose live prediction is shown
        in the text overlay, the same role as "Prediction: 6" in a
        classifier visualizer -- here it's a continuous regression
        prediction instead of a class, so it's a number, not a label.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        out_dir: str,
        frame_every: int = 25,
        live_display: bool = True,
        probe: tuple[float, float] = (0.5, 0.65),
    ) -> None:
        self.frame_every = frame_every
        self.live_display = live_display
        self.probe = probe
        self.stage = cfg.get("stage", 1)
        self.dir = Path(out_dir) / "netviz_frames"
        self.dir.mkdir(parents=True, exist_ok=True)

        linears = _linear_layers(model)
        layer_sizes = [linears[0].in_features] + [lin.out_features for lin in linears]
        n_layers = len(layer_sizes)

        # one x-column per layer, neurons spread vertically within a column
        self.xs = np.arange(n_layers, dtype=float)
        self.ys = [
            np.linspace(-1, 1, n) if n > 1 else np.array([0.0]) for n in layer_sizes
        ]

        if live_display:
            plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.set_xlim(-0.5, n_layers - 0.5)
        self.ax.set_ylim(-1.15, 1.15)
        self.ax.axis("off")

        # one LineCollection per weight matrix (one per gap between layers)
        self.collections: list[LineCollection] = []
        self._n_pairs: list[tuple[int, int]] = []  # (n_out, n_in) per gap, for flatten order
        for gap, lin in enumerate(linears):
            n_out, n_in = lin.out_features, lin.in_features
            segs = [
                [(self.xs[gap], self.ys[gap][i]), (self.xs[gap + 1], self.ys[gap + 1][j])]
                for j in range(n_out)
                for i in range(n_in)
            ]
            lc = LineCollection(segs, linewidths=0.3, colors=(0.3, 0.3, 0.3, 0.15))
            self.ax.add_collection(lc)
            self.collections.append(lc)
            self._n_pairs.append((n_out, n_in))

        # neurons as scatter points, one per layer
        node_colors = ["#2f9d94"] + ["#c9ccd1"] * (n_layers - 2) + ["#c17a4a"]
        for x, y, c in zip(self.xs, self.ys, node_colors):
            self.ax.scatter(np.full_like(y, x), y, s=18, color=c, zorder=5,
                             edgecolors="black", linewidths=0.4)

        self.text = self.ax.text(
            0.01, 0.99, "", transform=self.ax.transAxes, va="top", ha="left",
            fontsize=11, color="#e8e6df", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#12181f", edgecolor="#a8683a", alpha=0.9),
        )
        self.fig.tight_layout()

    def _refresh_weights(self, model: nn.Module) -> None:
        for lc, lin in zip(self.collections, _linear_layers(model)):
            w = lin.weight.detach().cpu().numpy()  # (n_out, n_in), matches segment build order
            flat = w.flatten()
            vmax = float(np.abs(flat).max()) + 1e-12
            norm = Normalize(vmin=-vmax, vmax=vmax)
            rgba = plt.get_cmap("coolwarm")(norm(flat))
            rgba[:, 3] = 0.06 + 0.9 * (np.abs(flat) / vmax)  # opacity ~ |weight|
            lc.set_color(rgba)
            lc.set_linewidths(0.15 + 1.6 * (np.abs(flat) / vmax))

    def _probe_text(self, model: nn.Module, epoch: int, total_epochs: int,
                     loss_val: float, mode: str, metrics: dict | None) -> str:
        from src.analytical import pressure_exact_nondim, velocity_exact_nondim

        xt = torch.tensor([[self.probe[0]]], dtype=torch.float64)
        dtt = torch.tensor([[self.probe[1]]], dtype=torch.float64)
        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(xt.to(device), dtt.to(device))
        if self.stage == 1:
            v, p = (o.cpu().item() for o in out)
            ve = velocity_exact_nondim(xt, dtt).item()
            pe = pressure_exact_nondim(xt, dtt).item()
            probe_line = f"probe x~={self.probe[0]} Dt~={self.probe[1]}: V~={v:+.4f} (exact {ve:+.4f})  p~={p:+.4f} (exact {pe:+.4f})"
        else:
            rho, v, p, t = (o.cpu().item() for o in out)
            probe_line = f"probe x~={self.probe[0]} Dt~={self.probe[1]}: rho~={rho:.4f} V~={v:.4f} p~={p:.4f} T~={t:.4f}"

        lines = [
            f"MODE: {mode}   Stage {self.stage}",
            f"epoch: {epoch}/{total_epochs}",
            f"loss:  {loss_val:.4e}",
        ]
        if metrics:
            lines.append("  ".join(f"{k}={v:.3e}" for k, v in metrics.items()))
        lines.append(probe_line)
        return "\n".join(lines)

    def maybe_update(
        self,
        epoch: int,
        total_epochs: int,
        model: nn.Module,
        loss_val: float,
        mode: str = "Adam",
        metrics: dict | None = None,
    ) -> None:
        if epoch % self.frame_every != 0:
            return
        self._refresh_weights(model)
        self.text.set_text(self._probe_text(model, epoch, total_epochs, loss_val, mode, metrics))
        self.fig.canvas.draw_idle()
        if self.live_display:
            plt.pause(0.001)
        self.fig.savefig(self.dir / f"frame_{epoch:06d}.png", dpi=90,
                          facecolor=self.fig.get_facecolor())

    def close(self) -> None:
        plt.close(self.fig)
