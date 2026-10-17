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
import re
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
    ap.add_argument("--resume-from", default=None, help="state_dict checkpoint to continue from")
    ap.add_argument("--start-epoch", type=int, default=0, help="Adam epoch of that checkpoint")
    ap.add_argument("--prev-log", default=None,
                    help="log of the interrupted run: restores loss history and elapsed time")
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
    resume = None
    if args.resume_from:
        hist, t_off = parse_log(Path(args.prev_log), args.start_epoch) if args.prev_log else ({}, 0.0)
        resume = {"state_dict": args.resume_from, "start_epoch": args.start_epoch,
                  "time_offset": t_off, "history": hist}
        print(f"[train] resuming at Adam epoch {args.start_epoch} (elapsed {t_off:.0f} s)")
    train(cfg, Path(args.out), monitor=monitor, monitor_every=args.monitor_every, resume=resume)


def parse_log(path: Path, upto: int):
    """Rebuild the history dict (epochs < ``upto``) and elapsed time from a log."""
    num = r"([-+0-9.eE]+)"
    pat = re.compile(rf"\[adam\s+(\d+)\] loss={num} cont={num} mx={num} mr={num} out={num} lr={num} t={num}s")
    mon = re.compile(r"monitor mean_relL2_u=([-+0-9.eE]+)\s+mean_relL2_p=([-+0-9.eE]+)\s+"
                     r"max_relL2_u=([-+0-9.eE]+)\s+max_relL2_p=([-+0-9.eE]+)")
    h = {k: [] for k in ("epoch", "time_s", "loss", "cont", "mom_x", "mom_r", "outlet", "lr", "stage", "monitor")}
    last_ep, last_t, t_at = -1, 0.0, 0.0
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            ep = int(m.group(1))
            last_ep, last_t = ep, float(m.group(8))
            if ep >= upto:
                break
            t_at = last_t
            for k, g in zip(("loss", "cont", "mom_x", "mom_r", "outlet", "lr"), range(2, 8)):
                h[k].append(float(m.group(g)))
            h["epoch"].append(ep)
            h["time_s"].append(last_t)
            h["stage"].append("adam")
            continue
        m = mon.search(line)
        if m and 0 <= last_ep < upto:
            h["monitor"].append({"mean_relL2_u": float(m.group(1)), "mean_relL2_p": float(m.group(2)),
                                 "max_relL2_u": float(m.group(3)), "max_relL2_p": float(m.group(4)),
                                 "epoch": last_ep, "time_s": last_t})
    # time at the checkpoint epoch: interpolate from the last logged epoch
    rate = t_at / max(h["epoch"][-1], 1) if h["epoch"] else 0.0
    return h, t_at + rate * (upto - (h["epoch"][-1] if h["epoch"] else 0))


if __name__ == "__main__":
    main()
