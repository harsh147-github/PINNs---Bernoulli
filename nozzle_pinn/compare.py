"""PINN vs OpenFOAM comparison metrics (README §7.3)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .inference import predict_points
from .model import NozzlePINN


def load_openfoam(results_dir: Path, tag: str = "") -> List[Dict[str, np.ndarray]]:
    """All ``dt_XXXX{tag}.npz`` OpenFOAM solutions, sorted by d_t."""
    files = sorted(results_dir.glob(f"dt_[0-9][0-9][0-9][0-9]{tag}.npz"))
    return [dict(np.load(f)) for f in files]


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b||_2 / ||b||_2."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare_case(model: NozzlePINN, ref: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Error metrics of the PINN at the OpenFOAM cell centres of one design.

    Cells are equally sized in the axial direction, so plain (unweighted)
    norms over cells are an adequate discrete L2 norm; the pressure norm is
    taken after removing nothing — both solvers fix p = 0 at the outlet.
    """
    dt = float(ref["dt"])
    pr = predict_points(model, ref["x"], ref["r"], dt)
    speed_ref = np.hypot(ref["u"], ref["v"])
    speed = np.hypot(pr["u"], pr["v"])
    return {
        "dt": dt,
        "relL2_u": rel_l2(pr["u"], ref["u"]),
        # v is ~0 in a straight pipe, so normalise its error by |U| instead of |v|
        "relL2_v": float(np.linalg.norm(pr["v"] - ref["v"]) / np.linalg.norm(speed_ref)),
        "relL2_speed": rel_l2(speed, speed_ref),
        "relL2_p": rel_l2(pr["p"], ref["p"]),
        "maxabs_u": float(np.max(np.abs(pr["u"] - ref["u"]))),
        "maxabs_p": float(np.max(np.abs(pr["p"] - ref["p"]))),
        "p_inlet_pinn": float(predict_points(model, np.array([0.0]), np.array([0.0]), dt)["p"][0]),
    }


def make_monitor(refs: List[Dict[str, np.ndarray]]):
    """Callable used during training to print accuracy vs OpenFOAM (no gradients)."""
    def monitor(model: NozzlePINN) -> Dict[str, float]:
        rows = [compare_case(model, r) for r in refs]
        return {"mean_relL2_u": float(np.mean([r["relL2_u"] for r in rows])),
                "mean_relL2_p": float(np.mean([r["relL2_p"] for r in rows])),
                "max_relL2_u": float(np.max([r["relL2_u"] for r in rows])),
                "max_relL2_p": float(np.max([r["relL2_p"] for r in rows]))}
    return monitor
