"""Amortised inference sweep + every figure, table and export (README §11, §13).

Usage
-----
    python scripts/infer_and_postprocess.py [--model results/pinn/model_final.pt]

Steps
-----
1. Load the trained surrogate.
2. Sweep ``inference.n_designs`` throat diameters across the design space,
   timing the inference of the full flow field of every design.
3. Compare against every OpenFOAM case in ``results/openfoam``.
4. Write figures (``results/figures``), tables (``results/tables``), field
   data (``results/fields``), ParaView files (``results/paraview``) and the
   design-sweep animation.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nozzle_pinn import postprocess as P  # noqa: E402
from nozzle_pinn.compare import compare_case, load_openfoam  # noqa: E402
from nozzle_pinn.config import REPO_ROOT  # noqa: E402
from nozzle_pinn.inference import field_grid  # noqa: E402
from nozzle_pinn.train import load_model  # noqa: E402


def section_metrics(x2d, r2d, u, v, p, R, x_throat: float) -> dict:
    """Throat / global quantities from any structured (nr, nx) field.

    Works for both the PINN grid and the OpenFOAM cell grid: values at the
    axis (r = 0) and wall (r = R, u = v = 0) are appended before integrating
    so both use the same area-weighted mean over the throat cross-section.
    """
    j = int(np.argmin(np.abs(x2d[0] - x_throat)))
    # interior only: drop a wall row (eta = 1, where u = 0 exactly) if present
    interior = slice(None, -1) if np.allclose(u[-1], 0.0) else slice(None)
    back = (u[interior] < 0).any(axis=0)
    xb = x2d[0][back]
    r = r2d[:, j]
    q = 0.5 * (u[:, j] ** 2 + v[:, j] ** 2)
    if r[0] > 0:  # cell data: add axis and wall points
        r = np.concatenate([[0.0], r, [R[j]]])
        q = np.concatenate([[q[0]], q, [0.0]])
    q_mean = np.trapezoid(q * r, r) / np.trapezoid(r, r)
    return {
        "u_throat_axis": float(u[0, j]),
        "p_throat_axis": float(p[0, j]),
        "q_throat_axis": float(0.5 * (u[0, j] ** 2 + v[0, j] ** 2)),
        "q_throat_mean": float(q_mean),
        "dp_total": float(p[0, 0]),
        "p_min": float(p.min()),
        "u_min": float(u[interior].min()),
        "recirc_length": float(xb.max() - xb.min()) if back.any() else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(REPO_ROOT / "results" / "pinn" / "model_final.pt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results"))
    ap.add_argument("--no-animation", action="store_true")
    ap.add_argument("--no-paraview", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    figs, tabs, flds, pv = (out / d for d in ("figures", "tables", "fields", "paraview"))
    for d in (figs, tabs, flds, pv):
        d.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.model))
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    torch.set_num_threads(int(cfg["training"]["threads"]))
    noz = model.noz
    inf = cfg["inference"]
    nx, neta = int(inf["nx"]), int(inf["neta"])
    of_nx = int(cfg["openfoam"]["nx"])
    Re = float(cfg["physics"]["Re"])

    # ------------------------------------------------------------------ 1. sweep
    dts = np.linspace(noz.dt_min, noz.dt_max, int(inf["n_designs"]))
    field_grid(model, float(dts[0]), nx, neta)  # warm-up (thread pool, allocator)
    rows, t_inf = [], []
    for dt in dts:
        t0 = time.perf_counter()
        g = field_grid(model, float(dt), nx, neta)
        t_inf.append(time.perf_counter() - t0)
        m = section_metrics(g["x"], g["r"], g["u"], g["v"], g["p"], g["R"][0], noz.x_throat_mid)
        rows.append({"dt": float(dt), **m})
    sweep = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    infer_s = float(np.mean(t_inf))
    print(f"[infer] {len(dts)} designs x {nx * neta} points: mean {infer_s * 1e3:.2f} ms / design "
          f"(total {sum(t_inf):.2f} s)")

    rho, nu, D = (float(cfg["physics"][k]) for k in ("rho_SI", "nu_SI", "D_in_m"))
    U_SI = Re * nu / D
    pa = rho * U_SI ** 2  # Pa per unit non-dimensional pressure
    with open(tabs / "design_sweep_pinn.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(rows[0])
        w.writerow(keys + ["D_throat_mm", "q_throat_axis_Pa", "dp_total_Pa"])
        for r in rows:
            w.writerow([f"{r[k]:.6g}" for k in keys]
                       + [f"{r['dt'] * D * 1e3:.4g}", f"{r['q_throat_axis'] * pa:.6g}",
                          f"{r['dp_total'] * pa:.6g}"])

    # ------------------------------------------------------------- 2. OpenFOAM
    refs = load_openfoam(out / "openfoam")
    of_rows, cmp_rows = [], []
    for ref in refs:
        o = P._of_grid(ref, of_nx)
        dt = float(ref["dt"])
        Rcol = noz.radius(o["x"][0], dt)
        of_rows.append({"dt": dt, **section_metrics(o["x"], o["r"], o["u"], o["v"], o["p"], Rcol,
                                                    noz.x_throat_mid)})
        c = compare_case(model, ref)
        pinn_m = rows[int(np.argmin(np.abs(dts - dt)))]
        c["q_throat_axis_err_pct"] = 100 * abs(field_grid(model, dt, nx, neta)["q"][0][
            int(np.argmin(np.abs(np.linspace(0, noz.length, nx) - noz.x_throat_mid)))]
            - of_rows[-1]["q_throat_axis"]) / of_rows[-1]["q_throat_axis"]
        c["dp_total_err_pct"] = 100 * abs(pinn_m["dp_total"] - of_rows[-1]["dp_total"]) / of_rows[-1]["dp_total"]
        cmp_rows.append(c)
        P.plot_pinn_vs_openfoam(model, ref, of_nx, figs / f"compare_openfoam_dt{int(round(dt * 100)):03d}.png")
        print(f"[compare] d_t={dt:.2f}  relL2 u={c['relL2_u']:.2e}  v={c['relL2_v']:.2e}  "
              f"p={c['relL2_p']:.2e}  |U|={c['relL2_speed']:.2e}")
    if cmp_rows:
        with open(tabs / "pinn_vs_openfoam.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(cmp_rows[0]))
            for r in cmp_rows:
                w.writerow([f"{v:.6g}" for v in r.values()])
        with open(tabs / "design_sweep_openfoam.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(of_rows[0]))
            for r in of_rows:
                w.writerow([f"{v:.6g}" for v in r.values()])

    # --------------------------------------------------------------- 3. timing
    hist = json.loads((Path(args.model).parent / "history.json").read_text())
    tfile = out / "openfoam" / "timings.json"
    of_t = json.loads(tfile.read_text()) if tfile.exists() else {}
    base = {k: v for k, v in of_t.items() if k.startswith("dt_") and k[3:].isdigit()}
    of_per = float(np.mean([v["mesh_s"] + v["solve_s"] for v in base.values()])) if base else float("nan")
    timing = {
        "pinn_train_s": hist["time_total_s"],
        "pinn_train_adam_s": hist["time_adam_s"],
        "pinn_train_lbfgs_s": hist["time_lbfgs_s"],
        "pinn_parameters": hist["n_parameters"],
        "pinn_infer_s_per_design": infer_s,
        "pinn_infer_points_per_design": nx * neta,
        "pinn_infer_total_s": float(sum(t_inf)),
        "n_designs_swept": len(dts),
        "openfoam_s_per_design": of_per,
        "openfoam_cells": next(iter(base.values()))["cells"] if base else None,
        "openfoam_designs_run": len(base),
        "openfoam_equivalent_sweep_s": of_per * len(dts),
        "break_even_designs": hist["time_total_s"] / max(of_per - infer_s, 1e-12) if base else None,
    }
    pfile = out / "openfoam" / "timings_parallel.json"
    if pfile.exists():
        par = json.loads(pfile.read_text())
        of_par = float(par["s_per_design_throughput"])
        timing.update({
            "openfoam_parallel_jobs": par["jobs"],
            "openfoam_parallel_s_per_design": of_par,
            "break_even_designs_vs_parallel": hist["time_total_s"] / max(of_par - infer_s, 1e-12),
        })
    for n in (121, 1000, 10000):
        timing[f"pinn_total_s_{n}_designs"] = hist["time_total_s"] + infer_s * n
        timing[f"openfoam_serial_s_{n}_designs"] = of_per * n
        if "openfoam_parallel_s_per_design" in timing:
            timing[f"openfoam_parallel_s_{n}_designs"] = timing["openfoam_parallel_s_per_design"] * n
    P.write_json(tabs / "timing.json", timing)
    if base:
        P.plot_timing(timing, figs / "timing_amortisation.png")
    print("[timing] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in timing.items()}))

    # -------------------------------------------------------------- 4. figures
    P.plot_history(hist, figs / "training_history.png")
    P.plot_sweep(sweep, of_rows, noz.dt_baseline, figs / "design_sweep.png")
    plot_dts = [float(d) for d in inf["plot_designs"]]
    for dt in plot_dts:
        tag = " (baseline d1)" if abs(dt - noz.dt_baseline) < 1e-9 else ""
        P.plot_design_fields(model, dt, figs / f"fields_dt{int(round(dt * 100)):03d}.png", nx, neta, tag)
        g = field_grid(model, dt, nx, neta)
        np.savez_compressed(flds / f"pinn_fields_dt{int(round(dt * 100)):03d}.npz", dt=dt, **g)
    P.plot_axial_profiles(model, [float(r["dt"]) for r in refs] or plot_dts, refs, of_nx,
                          figs / "axial_profiles.png")

    if not args.no_animation:
        up = np.linspace(noz.dt_baseline, noz.dt_max, 16)
        down = np.linspace(noz.dt_max, noz.dt_min, 31)[1:]
        back = np.linspace(noz.dt_min, noz.dt_baseline, 16)[1:]
        seq = np.concatenate([up, down, back])
        P.make_animation(model, seq, figs / "throat_sweep_animation.gif", 241, 41)

    if not args.no_paraview:
        pv_dts = np.linspace(noz.dt_min, noz.dt_max, 25)
        names = []
        for i, dt in enumerate(pv_dts):
            name = f"nozzle_{i:03d}.vts"
            P.write_vts(pv / name, field_grid(model, float(dt), nx, neta))
            names.append(name)
        P.write_pvd(pv / "nozzle_sweep.pvd", names, pv_dts)
    print(f"[done] outputs in {out}")


if __name__ == "__main__":
    main()
