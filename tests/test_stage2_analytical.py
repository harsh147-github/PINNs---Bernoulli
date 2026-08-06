"""Stage 2 physics sanity check: the exact isentropic subsonic solution
(src.analytical.stage2_exact_state_nondim) must satisfy physics.residuals_euler's
four equations. The exact solution is built from a brentq root solve (not a
torch-differentiable graph), so we can't route it through physics.py's
autograd-based `_grad` helper directly; instead we check the two algebraic
residuals (energy, eos) directly, and check continuity/momentum with a
central finite difference of the exact fields (independent of autograd,
consistent with the PDEs in physics.py's docstring). Looser tolerance than
Stage 1 (CLAUDE.md Section 7: Stage 2 uses rel-L2 vs. isentropic relations,
not machine-precision residuals) because of the brentq root solve.
"""
import numpy as np
import yaml

from src import analytical


def _load_cfg():
    with open("configs/default.yaml") as f:
        return yaml.safe_load(f)


def test_stage2_algebraic_residuals_near_zero():
    """Energy (stagnation enthalpy) and ideal-gas EOS residuals, no derivatives needed."""
    cfg = _load_cfg()
    geom, phys = cfg["geometry"], cfg["physics"]
    L, D_in = geom["L"], geom["D_in"]
    V_in, p_in, T_in = phys["V_in"], phys["p_in"], phys["T_in"]
    gamma, R = phys["gamma"], phys["R"]
    cp = gamma * R / (gamma - 1.0)
    beta = V_in ** 2 / (2.0 * cp * T_in)

    x = np.linspace(0.05, 2.95, 20)
    for Dt in [0.22, 0.30, 0.40]:
        rho_t, V_t, p_t, T_t = analytical.stage2_exact_state_nondim(
            x, Dt, D_in, L, V_in, p_in, T_in, gamma, R
        )
        r_energy = T_t + beta * V_t ** 2 - (1.0 + beta)
        r_eos = p_t - rho_t * T_t
        assert np.max(np.abs(r_energy)) < 1e-8
        assert np.max(np.abs(r_eos)) < 1e-8


def test_stage2_continuity_and_momentum_residuals_near_zero_fd():
    """Central finite difference of the exact fields against physics.py's PDEs
    (mass flux constant, momentum balance), matching physics.residuals_euler's
    non-dimensional form in its docstring."""
    cfg = _load_cfg()
    geom, phys = cfg["geometry"], cfg["physics"]
    L, D_in = geom["L"], geom["D_in"]
    V_in, p_in, T_in = phys["V_in"], phys["p_in"], phys["T_in"]
    gamma, R = phys["gamma"], phys["R"]
    rho_in = p_in / (R * T_in)  # ideal-gas-consistent inlet density (see physics.residuals_euler)
    kappa = p_in / (rho_in * V_in ** 2)

    Dt = 0.30
    h = 1e-5
    x0 = np.linspace(0.2, 2.8, 15)

    def fields_at(x):
        rho_t, V_t, p_t, _ = analytical.stage2_exact_state_nondim(
            x, Dt, D_in, L, V_in, p_in, T_in, gamma, R
        )
        return rho_t, V_t, p_t

    xt_plus = (x0 + h) / L
    xt_minus = (x0 - h) / L
    dxt = 2 * h / L

    rho_p, V_p, p_p = fields_at(x0 + h)
    rho_m, V_m, p_m = fields_at(x0 - h)
    rho_0, V_0, p_0 = fields_at(x0)

    D_p = D_in + (Dt - D_in) * np.sin(np.pi * xt_plus) ** 2
    A_tilde_p = (D_p / D_in) ** 2
    D_m = D_in + (Dt - D_in) * np.sin(np.pi * xt_minus) ** 2
    A_tilde_m = (D_m / D_in) ** 2

    r_cont = (rho_p * A_tilde_p * V_p - rho_m * A_tilde_m * V_m) / dxt
    dV_dxt = (V_p - V_m) / dxt
    dp_dxt = (p_p - p_m) / dxt
    r_mom = rho_0 * V_0 * dV_dxt + kappa * dp_dxt

    assert np.max(np.abs(r_cont)) < 1e-4, r_cont
    assert np.max(np.abs(r_mom)) < 1e-4, r_mom
