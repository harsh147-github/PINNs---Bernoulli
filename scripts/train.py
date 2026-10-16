"""Train the parametric nozzle PINN (physics only, no data).

Usage
-----
    python scripts/train.py                                  # full run from config
    python scripts/train.py --adam-epochs 500 --lbfgs-iters 0 --out results/smoke

If OpenFOAM solutions exist in ``results/openfoam`` they are used to PRINT
accuracy during training (monitoring only — they never enter the loss).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nozzle_pinn.compare import load_openfoam, make_monitor  # noqa: E402
from nozzle_pinn.config import REPO_ROOT, load_config  # noqa: E402
from nozzle_pinn.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "pinn"))
    ap.add_argument("--adam-epochs", type=int, default=None)
    ap.add_argument("--lbfgs-iters", type=int, default=None)
    ap.add_argument("--monitor-every", type=int, default=500)
    ap.add_argument("--no-monitor", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.adam_epochs is not None:
        cfg["training"]["adam_epochs"] = args.adam_epochs
    if args.lbfgs_iters is not None:
        cfg["training"]["lbfgs_iters"] = args.lbfgs_iters
    refs = [] if args.no_monitor else load_openfoam(REPO_ROOT / "results" / "openfoam")
    monitor = make_monitor(refs) if refs else None
    if refs:
        print(f"[train] monitoring against {len(refs)} OpenFOAM cases "
              f"(d_t = {', '.join(f'{float(r['dt']):.2f}' for r in refs)})")
    train(cfg, Path(args.out), monitor=monitor, monitor_every=args.monitor_every)


if __name__ == "__main__":
    main()
