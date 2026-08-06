"""Stage 2 sanity check: the exact isentropic subsonic solution
(analytical.stage2_exact_state_nondim) must satisfy physics.residuals_euler's
governing equations to ~0.

The exact solution goes through a brentq root solve (no closed form for the
inverse area-Mach relation), so it is not torch-differentiable and can't be
plugged into `physics.residuals_euler` (which differentiates the model via
autograd, for use with an actual PINN). Instead this test recomputes the same
residuals with a central finite difference directly against
`stage2_exact_state_nondim`, independently checking that the exact solution
satisfies the PDEs CLAUDE.md Section 6 specifies."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytical import stage2_exact_state_nondim
from src.geometry import area_nondim
import torch


def test_stage2_exact_solution_satisfies_euler_residuals():
    cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
    g, ph = cfg["geometry"], cfg["physics"]
    gamma, R = ph["gamma"], ph["R"]
    V_in, p_in, T_in = ph["V_in"], ph["p_in"], ph["T_in"]
    rho_in = p_in / (R * T_in)
    cp = gamma * R / (gamma - 1.0)
    kappa = p_in / (rho_in * V_in**2)
    beta = V_in**2 / (2.0 * cp * T_in)

    dt_si = 0.30
    h = 1e-6  # step in x_tilde
    xt_vals = np.linspace(0.05, 0.95, 15)

    def state(xt):
        x_si = xt * g["L"]
        return stage2_exact_state_nondim(np.array([x_si]), dt_si, cfg)

    def area_t(xt):
        return area_nondim(torch.tensor(xt), torch.tensor(dt_si / g["D_in"])).item()

    for xt in xt_vals:
        rho_m, v_m, p_m, t_m = (z[0] for z in state(xt - h))
        rho_0, v_0, p_0, t_0 = (z[0] for z in state(xt))
        rho_p, v_p, p_p, t_p = (z[0] for z in state(xt + h))

        flux_m = rho_m * area_t(xt - h) * v_m
        flux_p = rho_p * area_t(xt + h) * v_p
        r_cont = (flux_p - flux_m) / (2 * h)
        assert abs(r_cont) < 1e-3, f"r_cont={r_cont} at xt={xt}"

        dv_dxt = (v_p - v_m) / (2 * h)
        dp_dxt = (p_p - p_m) / (2 * h)
        r_mom = rho_0 * v_0 * dv_dxt + kappa * dp_dxt
        assert abs(r_mom) < 1e-3, f"r_mom={r_mom} at xt={xt}"

        r_energy = t_0 + beta * v_0**2 - (1.0 + beta)
        assert abs(r_energy) < 1e-9, f"r_energy={r_energy} at xt={xt}"

        r_eos = p_0 - rho_0 * t_0
        assert abs(r_eos) < 1e-9, f"r_eos={r_eos} at xt={xt}"
