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
    ap.add_argument("--jobs", type=int, default=1,
                    help="run this many cases concurrently (one core each) and record the "
                         "throughput in timings_parallel.json — the fair multi-core baseline")
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

    if args.jobs > 1:
        run_parallel(cfg, noz, designs, args.jobs, out_dir)
        return

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


def _one(job):
    cfg, noz, dt, case = job
    F.write_case(case, noz, float(dt), cfg)
    return dt, F.run_case(case)


def run_parallel(cfg, noz, designs, jobs: int, out_dir: Path) -> None:
    """Throughput test: ``jobs`` independent cases at a time on ``jobs`` cores."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    work = [(cfg, noz, dt, REPO_ROOT / "openfoam" / "cases" / f"par_{int(round(dt * 1000)):04d}")
            for dt in designs]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=jobs) as ex:  # each case is its own OpenFOAM process
        res = list(ex.map(_one, work))
    wall = time.perf_counter() - t0
    out = {"jobs": jobs, "n_designs": len(designs), "wall_s": wall,
           "s_per_design_throughput": wall / len(designs),
           "cases": {f"{dt:.3f}": t for dt, t in res}}
    (out_dir / "timings_parallel.json").write_text(json.dumps(out, indent=2))
    print(f"[openfoam] {len(designs)} cases, {jobs} concurrent: wall {wall:.1f} s "
          f"-> {wall / len(designs):.1f} s per design (throughput)")


if __name__ == "__main__":
    main()
