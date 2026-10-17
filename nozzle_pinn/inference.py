"""Inference: evaluate the trained surrogate for ANY throat diameter (README §11).

After training, a new design costs one forward pass of a small MLP — this is
the "amortisation": the expensive part (training) is paid once, every further
geometry is (almost) free.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .model import NozzlePINN


@torch.no_grad()
def predict_points(model: NozzlePINN, x: np.ndarray, r: np.ndarray, dt: float | np.ndarray) -> Dict[str, np.ndarray]:
    """Fields (u, v, p) at arbitrary points of nozzle(s) ``dt``."""
    xt = torch.as_tensor(np.asarray(x, dtype=np.float64).ravel())
    rt = torch.as_tensor(np.asarray(r, dtype=np.float64).ravel())
    dtt = torch.as_tensor(np.broadcast_to(np.asarray(dt, dtype=np.float64), xt.shape).copy())
    out = model(xt, rt, dtt)
    shape = np.shape(x)
    return {k: out[k].detach().numpy().reshape(shape) for k in ("u", "v", "p")}


def field_grid(model: NozzlePINN, dt: float, nx: int, neta: int) -> Dict[str, np.ndarray]:
    """Structured (x, eta) grid of one nozzle with every derived quantity.

    Returns x, r (2-D, shape (neta, nx)), u, v, p, speed |U|, dynamic pressure
    q = |U|^2 / 2, total pressure p0 = p + q, and the Stokes stream function
    psi normalised to 0 on the axis and 1 on the wall.
    """
    noz = model.noz
    xs = np.linspace(0.0, noz.length, nx)
    etas = np.linspace(0.0, 1.0, neta)
    X, E = np.meshgrid(xs, etas)
    Rw = noz.radius(X, dt)
    Rr = E * Rw
    f = predict_points(model, X, Rr, dt)
    speed = np.hypot(f["u"], f["v"])
    q = 0.5 * speed ** 2
    # Stokes stream function psi = int_0^r u r' dr' = R^2 int_0^eta u eta' deta'
    integrand = f["u"] * E
    psi = np.zeros_like(integrand)
    psi[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(etas)[:, None], axis=0)
    psi *= Rw ** 2
    psi /= 0.125  # Q / (2 pi) with Q = pi R_in^2 * 1 = pi / 4
    return {"x": X, "r": Rr, "eta": E, "R": Rw, "u": f["u"], "v": f["v"], "p": f["p"],
            "speed": speed, "q": q, "p0": f["p"] + q, "psi": psi}


def axial_profiles(model: NozzlePINN, dt: float, nx: int = 401, neta: int = 201) -> Dict[str, np.ndarray]:
    """Centre-line and cross-section-averaged quantities along x."""
    g = field_grid(model, dt, nx, neta)
    eta = g["eta"][:, 0]
    w = eta  # area weight in eta (dA ∝ eta deta)
    area = np.trapezoid(w, eta)
    p_mean = np.trapezoid(g["p"] * w[:, None], eta, axis=0) / area
    q_mean = np.trapezoid(g["q"] * w[:, None], eta, axis=0) / area
    flux = 2.0 * np.trapezoid(g["u"] * w[:, None], eta, axis=0) * g["R"][0] ** 2 / 0.25
    return {"x": g["x"][0], "R": g["R"][0], "u_axis": g["u"][0], "p_axis": g["p"][0],
            "p_wall": g["p"][-1], "q_axis": g["q"][0], "p_mean": p_mean, "q_mean": q_mean,
            "mass_flux": flux}
