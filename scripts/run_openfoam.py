"""Run the OpenFOAM (simpleFoam) reference cases for the configured designs.

Usage
-----
    python scripts/run_openfoam.py [--config configs/nozzle.yaml] [--designs 0.4 0.7]

For every throat ratio d_t a case is written to ``openfoam/cases/dt_XXX``,
meshed with blockMesh, solved with simpleFoam, and the cell-centre solution is
saved to ``results/openfoam/dt_XXX.npz``. Wall-clock times go to
``results/openfoam/timings.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nozzle_pinn import openfoam as F  # noqa: E402
from nozzle_pinn.config import REPO_ROOT, load_config  # noqa: E402
from nozzle_pinn.geometry import Nozzle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--designs", type=float, nargs="*", default=None)
    ap.add_argument("--nx", type=int, default=None, help="override axial cells (mesh study)")
    ap.add_argument("--nr", type=int, default=None, help="override radial cells (mesh study)")
    ap.add_argument("--tag", default="", help="suffix for output names (mesh study)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.nx:
        cfg["openfoam"]["nx"] = args.nx
    if args.nr:
        cfg["openfoam"]["nr"] = args.nr
    noz = Nozzle.from_config(cfg)
    designs = args.designs or cfg["openfoam"]["designs"]
    out_dir = REPO_ROOT / "results" / "openfoam"
    out_dir.mkdir(parents=True, exist_ok=True)
    tfile = out_dir / f"timings{args.tag}.json"
    timings = json.loads(tfile.read_text()) if tfile.exists() else {}

    for dt in designs:
        name = f"dt_{int(round(dt * 1000)):04d}{args.tag}"
        case = REPO_ROOT / "openfoam" / "cases" / name
        F.write_case(case, noz, float(dt), cfg)
        t = F.run_case(case)
        t.update(F.final_residuals(case))
        t["cells"] = int(cfg["openfoam"]["nx"]) * int(cfg["openfoam"]["nr"])
        sol = F.read_solution(case)
        np.savez_compressed(out_dir / f"{name}.npz", dt=dt, **sol)
        timings[name] = t
        tfile.write_text(json.dumps(timings, indent=2))
        print(f"[openfoam] d_t={dt:.3f}  cells={t['cells']}  iters={t['iterations']:.0f}  "
              f"solve={t['solve_s']:.1f}s  res(Ux,p)=({t['Ux']:.1e},{t['p']:.1e})", flush=True)


if __name__ == "__main__":
    main()
