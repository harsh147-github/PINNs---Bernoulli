"""Figures, animation and data export (README §8).

Every plot shows the FULL pipe: the solution lives on the half plane r >= 0
and is mirrored to r < 0 (the flow is axisymmetric). Streamlines are iso-lines
of the Stokes stream function psi — in axisymmetric incompressible flow these
are exactly the streamlines, and equal spacing in psi means equal volume flux
between neighbouring lines (crowded lines = fast flow).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from .inference import field_grid  # noqa: E402
from .model import NozzlePINN  # noqa: E402

CMAP = "turbo"          # CFD-style rainbow for scalar fields
ERR_CMAP = "magma"
PSI_LEVELS = np.linspace(0.05, 0.95, 10)   # 10 stream tubes of equal flux


def _mirror(a: np.ndarray, sign: float = 1.0) -> np.ndarray:
    """Stack the half-plane field with its mirror image (row 0 = axis)."""
    return np.vstack([sign * a[::-1], a[1:]])


def _full(g: Dict[str, np.ndarray], key: str) -> np.ndarray:
    return _mirror(g[key])


def _draw_walls(ax, g: Dict[str, np.ndarray]) -> None:
    ax.plot(g["x"][0], g["R"][0], "k", lw=1.6)
    ax.plot(g["x"][0], -g["R"][0], "k", lw=1.6)
    ax.set_aspect("equal")
    ax.set_xlim(g["x"][0, 0], g["x"][0, -1])
    ax.set_ylabel("r / D_in")


def _streamlines(ax, g: Dict[str, np.ndarray], color: str = "k") -> None:
    X, Y, P = _full(g, "x"), _mirror(g["r"], -1.0), _full(g, "psi")
    ax.contour(X, Y, P, levels=PSI_LEVELS, colors=color, linewidths=0.7)
    # recirculating fluid has psi > 1 (between the wall streamline and the wall)
    if np.nanmax(g["psi"]) > 1.0 + 1e-3:
        ax.contour(X, Y, P, levels=np.linspace(1.0005, np.nanmax(g["psi"]), 4)[:-1],
                   colors="w", linewidths=0.7, linestyles="--")


def plot_design_fields(model: NozzlePINN, dt: float, path: Path, nx: int, neta: int,
                       title_extra: str = "") -> None:
    """Static pressure, velocity magnitude and dynamic pressure + streamlines."""
    g = field_grid(model, dt, nx, neta)
    X, Y = _full(g, "x"), _mirror(g["r"], -1.0)
    panels = [("p", "static pressure  p / (rho U^2)"),
              ("speed", "velocity magnitude  |U| / U_in"),
              ("q", "dynamic pressure  q = |U|^2/2  (/ rho U^2)")]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), constrained_layout=True)
    for ax, (key, lab) in zip(axes, panels):
        cf = ax.contourf(X, Y, _full(g, key), levels=60, cmap=CMAP)
        _streamlines(ax, g)
        _draw_walls(ax, g)
        fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.01).set_label(lab, fontsize=8)
    axes[-1].set_xlabel("x / D_in")
    axes[0].set_title(f"PINN surrogate — throat ratio d_t = D_t/D_in = {dt:.3f}{title_extra}"
                      "   (black lines: streamlines = iso-psi)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _of_grid(ref: Dict[str, np.ndarray], nx: int) -> Dict[str, np.ndarray]:
    """Reshape OpenFOAM cell data (blockMesh order: x fastest) to (nr, nx)."""
    n = ref["x"].size
    return {k: np.asarray(ref[k]).reshape(n // nx, nx) for k in ("x", "r", "u", "v", "p")}


def plot_pinn_vs_openfoam(model: NozzlePINN, ref: Dict[str, np.ndarray], of_nx: int, path: Path) -> None:
    """Upper half = OpenFOAM, lower half = PINN, same colour scale; plus |error|."""
    from .inference import predict_points
    dt = float(ref["dt"])
    o = _of_grid(ref, of_nx)
    pr = predict_points(model, o["x"], o["r"], dt)
    fields = [("p", o["p"], pr["p"], "p / (rho U^2)"),
              ("|U|", np.hypot(o["u"], o["v"]), np.hypot(pr["u"], pr["v"]), "|U| / U_in")]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9.5), constrained_layout=True)
    x_axis = o["x"][0]
    R = model.noz.radius(x_axis, dt)
    for i, (name, f_of, f_pi, lab) in enumerate(fields):
        ax = axes[2 * i]
        lo, hi = min(f_of.min(), f_pi.min()), max(f_of.max(), f_pi.max())
        lev = np.linspace(lo, hi, 60)
        ax.contourf(o["x"], o["r"], f_of, levels=lev, cmap=CMAP)
        cf = ax.contourf(o["x"], -o["r"], f_pi, levels=lev, cmap=CMAP)
        ax.plot(x_axis, R, "k", x_axis, -R, "k", lw=1.4)
        ax.axhline(0, color="w", lw=0.8, ls="--")
        ax.text(0.05, 0.3, "OpenFOAM", color="w", fontsize=9, weight="bold")
        ax.text(0.05, -0.38, "PINN", color="w", fontsize=9, weight="bold")
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.01).set_label(lab, fontsize=8)
        ax.set_title(f"{name}: OpenFOAM (top) vs PINN (bottom), d_t = {dt:.2f}", fontsize=10)
        ax2 = axes[2 * i + 1]
        err = np.abs(f_pi - f_of)
        cf2 = ax2.contourf(o["x"], o["r"], err, levels=40, cmap=ERR_CMAP)
        ax2.contourf(o["x"], -o["r"], err, levels=cf2.levels, cmap=ERR_CMAP)
        ax2.plot(x_axis, R, "k", x_axis, -R, "k", lw=1.0)
        ax2.set_aspect("equal")
        fig.colorbar(cf2, ax=ax2, shrink=0.9, pad=0.01).set_label(f"|error| in {name}", fontsize=8)
        rel = np.linalg.norm(f_pi - f_of) / np.linalg.norm(f_of)
        ax2.set_title(f"|PINN - OpenFOAM| in {name}   (relative L2 = {rel:.2e})", fontsize=10)
    axes[-1].set_xlabel("x / D_in")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_axial_profiles(model: NozzlePINN, dts: Sequence[float], refs: List[Dict[str, np.ndarray]],
                        of_nx: int, path: Path) -> None:
    """Centre-line velocity, static pressure and dynamic pressure along the pipe."""
    from .inference import axial_profiles
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for i, dt in enumerate(dts):
        c = cmap(i / max(1, len(dts) - 1))
        pr = axial_profiles(model, dt)
        axes[0].plot(pr["x"], pr["u_axis"], color=c, lw=2, label=f"PINN d_t={dt:.2f}")
        axes[1].plot(pr["x"], pr["p_axis"], color=c, lw=2)
        axes[2].plot(pr["x"], pr["q_axis"], color=c, lw=2)
        ref = next((r for r in refs if abs(float(r["dt"]) - dt) < 1e-6), None)
        if ref is not None:
            o = _of_grid(ref, of_nx)
            k = slice(None, None, 12)
            axes[0].plot(o["x"][0][k], o["u"][0][k], "o", color=c, mfc="none", ms=5)
            axes[1].plot(o["x"][0][k], o["p"][0][k], "o", color=c, mfc="none", ms=5)
            axes[2].plot(o["x"][0][k], 0.5 * (o["u"][0] ** 2 + o["v"][0] ** 2)[k], "o", color=c,
                         mfc="none", ms=5)
    axes[0].plot([], [], "ko", mfc="none", label="OpenFOAM (first cell row)")
    axes[0].set_ylabel("centre-line u / U_in")
    axes[1].set_ylabel("centre-line p / (rho U^2)")
    axes[2].set_ylabel("centre-line q / (rho U^2)")
    axes[2].set_xlabel("x / D_in")
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Axial profiles: lines = PINN, circles = OpenFOAM")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sweep(sweep: Dict[str, np.ndarray], of_rows: List[Dict[str, float]], dt_base: float, path: Path) -> None:
    """Design-space sweep of throat quantities, PINN line vs OpenFOAM markers."""
    keys = [("q_throat_axis", "throat centre-line dynamic pressure q / (rho U^2)"),
            ("q_throat_mean", "throat area-mean dynamic pressure"),
            ("u_throat_axis", "throat centre-line velocity u / U_in"),
            ("p_throat_axis", "throat centre-line static pressure"),
            ("dp_total", "total static pressure drop  p_in - p_out"),
            ("recirc_length", "recirculation bubble length / D_in")]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, (k, lab) in zip(axes.ravel(), keys):
        ax.plot(sweep["dt"], sweep[k], "-", color="C0", lw=2, label=f"PINN ({len(sweep['dt'])} designs)")
        if of_rows:
            ax.plot([r["dt"] for r in of_rows], [r[k] for r in of_rows], "s", color="C3", ms=7,
                    mfc="none", mew=2, label="OpenFOAM")
        ax.axvline(dt_base, color="k", ls=":", lw=1)
        ax.text(dt_base, ax.get_ylim()[1], " d1 (baseline)", va="top", fontsize=8)
        ax.set_xlabel("throat ratio d_t = D_t / D_in")
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].annotate("throat narrower than d1\n-> dynamic pressure UP", xy=(0.45, 0.6),
                        xycoords="axes fraction", fontsize=8, color="C3")
    axes[0, 0].annotate("throat wider than d1\n-> dynamic pressure DOWN", xy=(0.62, 0.25),
                        xycoords="axes fraction", fontsize=8, color="C2")
    fig.suptitle("Amortised design sweep over the throat diameter")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_history(hist: Dict[str, Any], path: Path) -> None:
    """Loss components and (monitor-only) accuracy vs OpenFOAM during training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ep = np.array(hist["epoch"])
    for k, lab in (("loss", "total"), ("cont", "continuity"), ("mom_x", "x-momentum"),
                   ("mom_r", "r-momentum"), ("outlet", "outlet")):
        axes[0].semilogy(ep, hist[k], label=lab, lw=1.2 if k != "loss" else 2)
    n_adam = sum(1 for s in hist["stage"] if s == "adam")
    if n_adam < len(ep):
        axes[0].axvline(ep[n_adam - 1], color="k", ls=":", lw=1)
        axes[0].text(ep[n_adam - 1], axes[0].get_ylim()[1], " L-BFGS", va="top", fontsize=8)
    axes[0].set_xlabel("iteration (Adam epochs, then L-BFGS evaluations)")
    axes[0].set_ylabel("mean squared scaled residual")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Physics loss (the ONLY training signal)")
    if hist.get("monitor"):
        m = hist["monitor"]
        e = [r["epoch"] for r in m]
        axes[1].semilogy(e, [r["mean_relL2_u"] for r in m], "o-", label="mean rel-L2 u")
        axes[1].semilogy(e, [r["mean_relL2_p"] for r in m], "s-", label="mean rel-L2 p")
        axes[1].semilogy(e, [r["max_relL2_u"] for r in m], "o--", label="worst-design rel-L2 u", alpha=0.6)
        axes[1].semilogy(e, [r["max_relL2_p"] for r in m], "s--", label="worst-design rel-L2 p", alpha=0.6)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].set_xlabel("iteration")
        axes[1].set_title("Error vs OpenFOAM (monitoring only, never used in the loss)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_timing(timing: Dict[str, Any], path: Path) -> None:
    """Cumulative cost of exploring N designs: OpenFOAM vs PINN."""
    n = np.arange(1, 1001)
    of = timing["openfoam_s_per_design"] * n
    pinn = timing["pinn_train_s"] + timing["pinn_infer_s_per_design"] * n
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.loglog(n, of, lw=2, label=f"OpenFOAM: {timing['openfoam_s_per_design']:.1f} s / design")
    ax.loglog(n, pinn, lw=2, label=f"PINN: {timing['pinn_train_s'] / 60:.0f} min training once + "
                                   f"{timing['pinn_infer_s_per_design'] * 1e3:.1f} ms / design")
    if timing.get("break_even_designs"):
        be = timing["break_even_designs"]
        ax.axvline(be, color="k", ls=":")
        ax.text(be, of[0], f"  break-even ~{be:.0f} designs", fontsize=9)
    ax.set_xlabel("number of throat diameters explored")
    ax.set_ylabel("cumulative wall-clock time [s] (same 4-core CPU)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title("Amortisation: pay for training once, query designs for ~free")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def make_animation(model: NozzlePINN, dts: Sequence[float], path: Path, nx: int, neta: int,
                   fps: int = 12) -> None:
    """GIF of the pressure field + streamlines as the throat opens and closes."""
    grids = [field_grid(model, float(d), nx, neta) for d in dts]
    pmin = min(g["p"].min() for g in grids)
    pmax = max(g["p"].max() for g in grids)
    qmax = max(g["q"].max() for g in grids)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), constrained_layout=True)
    lev_p = np.linspace(pmin, pmax, 60)
    lev_q = np.linspace(0.0, qmax, 60)
    sm_p = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(pmin, pmax))
    sm_q = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, qmax))
    fig.colorbar(sm_p, ax=axes[0], shrink=0.9, pad=0.01).set_label("p / (rho U^2)", fontsize=8)
    fig.colorbar(sm_q, ax=axes[1], shrink=0.9, pad=0.01).set_label("q / (rho U^2)", fontsize=8)

    def frame(i: int):
        g = grids[i]
        X, Y = _full(g, "x"), _mirror(g["r"], -1.0)
        for ax, key, lev in ((axes[0], "p", lev_p), (axes[1], "q", lev_q)):
            ax.clear()
            ax.contourf(X, Y, np.clip(_full(g, key), lev[0], lev[-1]), levels=lev, cmap=CMAP)
            _streamlines(ax, g)
            _draw_walls(ax, g)
        axes[0].set_title(f"static pressure + streamlines   d_t = {dts[i]:.3f}")
        axes[1].set_title("dynamic pressure q = |U|^2 / 2 + streamlines")
        axes[1].set_xlabel("x / D_in")
        return []

    anim = FuncAnimation(fig, frame, frames=len(grids), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Data export
# -----------------------------------------------------------------------------
def write_vts(path: Path, g: Dict[str, np.ndarray]) -> None:
    """ParaView structured grid (.vts, ASCII) of one design, half plane z = 0."""
    ny, nx = g["x"].shape
    pts = "\n".join(f"{x:.6g} {r:.6g} 0" for x, r in zip(g["x"].ravel(), g["r"].ravel()))
    arrays = []
    for k in ("u", "v", "p", "speed", "q", "p0", "psi"):
        vals = " ".join(f"{v:.6g}" for v in g[k].ravel())
        arrays.append(f'<DataArray type="Float32" Name="{k}" format="ascii">{vals}</DataArray>')
    vel = " ".join(f"{u:.6g} {v:.6g} 0" for u, v in zip(g["u"].ravel(), g["v"].ravel()))
    arrays.append(f'<DataArray type="Float32" Name="U" NumberOfComponents="3" format="ascii">{vel}</DataArray>')
    ext = f"0 {nx - 1} 0 {ny - 1} 0 0"
    path.write_text(f"""<?xml version="1.0"?>
<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">
<StructuredGrid WholeExtent="{ext}"><Piece Extent="{ext}">
<PointData Scalars="p" Vectors="U">{''.join(arrays)}</PointData>
<Points><DataArray type="Float32" NumberOfComponents="3" format="ascii">{pts}</DataArray></Points>
</Piece></StructuredGrid></VTKFile>
""")


def write_pvd(path: Path, files: List[str], dts: Sequence[float]) -> None:
    """ParaView collection: the design sweep as a 'time' series (time = d_t)."""
    rows = "\n".join(f'<DataSet timestep="{d:.6f}" file="{f}"/>' for f, d in zip(files, dts))
    path.write_text(f'<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
                    f"<Collection>\n{rows}\n</Collection>\n</VTKFile>\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))
