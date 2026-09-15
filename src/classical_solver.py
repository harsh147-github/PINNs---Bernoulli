"""AN INDEPENDENT BASELINE — a classical numerical method that never touches
PyTorch, autodiff, or a neural network of any kind.

OBJECTIVE OF THIS FILE
-----------------------
Everything else in `src/` either trains a network against the physics
(train.py/losses.py/physics.py) or writes down the physics' closed-form
answer directly (analytical.py). This file exists to answer a different
question: if we hand the exact same governing equations to a *completely
different, well-established numerical technique* — explicit adaptive
Runge-Kutta ODE marching, the same family of method classical CFD texts
(e.g. Anderson, "Computational Fluid Dynamics: The Basics with
Applications", Ch. 7) use for quasi-1D nozzle flow — does it agree with
both the PINN and the closed form? Two independent methods landing on the
same answer is real evidence the physics was implemented correctly
everywhere; agreement with the closed form alone only proves the algebra
in `analytical.py`, not that a genuinely different numerical technique
sees the same flow field.

WHY THIS SUBSTITUTES FOR A FULL MESH-BASED CFD CASE (e.g. OpenFOAM) — READ
THIS BEFORE TRUSTING THE COMPARISON
-------------------------------------------------------------------------------
Be precise about what this file is and is not:

- It IS an independent numerical solve of the same governing ODEs
  (continuity + the differential form of Bernoulli/momentum), integrated
  by explicit adaptive-step Runge-Kutta-Fehlberg 4(5) (`scipy.integrate.
  solve_ivp(method="RK45")`) rather than by a trained neural network.
  Cross-checking the PINN against it is a legitimate "do two different
  numerical methods on the same equations agree" test.
- It is NOT a meshed, multi-dimensional CFD solve. A real OpenFOAM case for
  this nozzle would discretize the full 2D/3D Navier-Stokes equations
  (viscous terms, wall boundary layers, possible flow separation) over a
  volume mesh, and could in principle disagree with the quasi-1D inviscid
  answer wherever those effects matter. This repo's environment has no
  OpenFOAM installation and no meshing pipeline, so an actual mesh-based
  run was not performed here.
- Both this file AND analytical.py AND the PINN all start from the SAME
  quasi-1D, inviscid, incompressible model (README Section 3). Agreement
  between all three proves that model was solved correctly three
  different ways. It does NOT prove the quasi-1D inviscid model itself is
  a good description of the real, physical, viscous, 3D flow in an actual
  machined nozzle -- that would require the mesh-based comparison this
  repo does not have the tooling to run. Treat the numbers below as
  "three ways of solving the same equations agree," not as "validated
  against ground-truth experimental CFD."

THE ODE-MARCHING METHOD, DERIVED (not just asserted)
------------------------------------------------------------
`analytical.py` gets V(x) and p(x) by solving continuity and Bernoulli's
theorem algebraically, all at once, in their INTEGRATED (algebraic) form:

    A(x) V(x) = A_in V_in                     (continuity, integrated)
    p(x) + 0.5 rho V(x)^2 = p_in + 0.5 rho V_in^2   (Bernoulli, integrated)

This file instead marches their DIFFERENTIAL form step by step from the
inlet (x=0) to the outlet (x=L), exactly the way a real transient or
spatial-marching CFD code advances a solution -- it never "solves for V(x)"
in one algebraic step; it takes a lot of small, adaptive steps and
accumulates the answer as it goes. Differentiate continuity's integrated
form with respect to x (product rule) to get the ODE it actually integrates:

    d/dx[A(x) V(x)] = 0
    =>  A(x) dV/dx + V(x) dA/dx = 0
    =>  dV/dx = -(V(x) / A(x)) * dA/dx(x)                      (Eq. C1)

and differentiate Bernoulli's integrated form the same way:

    d/dx[p(x) + 0.5 rho V(x)^2] = 0
    =>  dp/dx + rho V(x) dV/dx = 0
    =>  dp/dx = -rho * V(x) * dV/dx(x)                          (Eq. C2)

Eq. C1 and C2 are a coupled first-order ODE system in the state
y(x) = [V(x), p(x)], with initial condition y(0) = [V_in, p_in] (the
SAME inlet condition the PINN hard-wires into its output transform, and
the SAME inlet condition analytical.py starts its algebra from). Handing
this y'(x) = f(x, y) system plus its initial condition to
`scipy.integrate.solve_ivp(method="RK45")` is precisely what "classical
ODE marching" means: take a step in x, evaluate f, adapt the step size to
hold a target error tolerance, repeat until x=L.

DELIBERATELY INDEPENDENT OF THE REST OF THIS REPO
-----------------------------------------------------
`diameter`/`area`/`d_area_dx` below are a **second, separate, pure-NumPy**
implementation of the same duct-shape law already in `geometry.py` --
duplicated on purpose, not imported. If this file re-used `geometry.py`,
a bug in that shared function could pass a "cross-check" that was
secretly checking the PINN against itself. Independence is the entire
point of a baseline.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp


def diameter(x: np.ndarray, Dt: float, L: float, D_in: float) -> np.ndarray:
    """D(x; Dt) [m], the same geometry law as geometry.py -- reimplemented
    independently in plain NumPy (see module docstring: no shared code with
    the PINN path)."""
    return D_in + (Dt - D_in) * np.sin(math.pi * x / L) ** 2


def area(x: np.ndarray, Dt: float, L: float, D_in: float) -> np.ndarray:
    """A(x; Dt) [m^2] = pi/4 * D(x)^2."""
    return (math.pi / 4.0) * diameter(x, Dt, L, D_in) ** 2


def d_area_dx(x: np.ndarray, Dt: float, L: float, D_in: float) -> np.ndarray:
    """dA/dx [m], analytically differentiated by hand (not autodiff -- this
    file uses ordinary calculus done on paper, which is the whole point of
    being an independent check on the autodiff-based PINN path).

    D(x) = D_in + (Dt-D_in) sin^2(pi x/L)
    dD/dx = (Dt-D_in) * 2 sin(pi x/L) cos(pi x/L) * (pi/L)
          = (Dt-D_in) * (pi/L) * sin(2 pi x/L)          [double-angle identity]
    A(x) = (pi/4) D(x)^2  =>  dA/dx = (pi/2) D(x) dD/dx
    """
    D = diameter(x, Dt, L, D_in)
    dD_dx = (Dt - D_in) * (math.pi / L) * np.sin(2.0 * math.pi * x / L)
    return (math.pi / 2.0) * D * dD_dx


def _rhs(x: float, y: np.ndarray, Dt: float, L: float, D_in: float, rho: float) -> np.ndarray:
    """The coupled ODE system y' = f(x, y), y = [V, p]. Eq. C1 and C2."""
    V, p = y
    A = area(np.array([x]), Dt, L, D_in)[0]
    dA_dx = d_area_dx(np.array([x]), Dt, L, D_in)[0]
    dV_dx = -(V / A) * dA_dx          # Eq. C1
    dp_dx = -rho * V * dV_dx          # Eq. C2
    return np.array([dV_dx, dp_dx])


def solve_nozzle_rk45(
    Dt: float,
    V_in: float,
    cfg: dict,
    n_report: int = 201,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> dict:
    """March continuity + Bernoulli's differential form from x=0 to x=L with
    explicit adaptive Runge-Kutta-Fehlberg 4(5) (`solve_ivp(method="RK45")`).

    Parameters
    ----------
    Dt : throat diameter [m] for this query.
    V_in : inlet velocity [m/s] for this query (an independent input, exactly
        like the PINN's own (x, Dt, V_in) query interface).
    cfg : the same `configs/default.yaml` dict every other file reads, for
        L, D_in, rho, p_in -- no new magic numbers (CLAUDE.md Section 6).
    n_report : number of x-stations to report the solution at (the ODE
        integrator internally takes as many adaptive steps as it needs;
        this only controls the output grid density).
    rtol, atol : `solve_ivp` error tolerances -- tight, since this run's
        entire purpose is to be a trustworthy independent reference.

    Returns
    -------
    dict with 'x' [m], 'V' [m/s], 'p' [Pa], plus solve_ivp's own diagnostics
    ('n_steps', 'success') so a failed integration is never silently
    treated as a valid baseline.
    """
    g = cfg["geometry"]
    L, D_in = g["L"], g["D_in"]
    p_in = cfg["physics"]["p_in"]

    x_eval = np.linspace(0.0, L, n_report)
    sol = solve_ivp(
        _rhs,
        t_span=(0.0, L),
        y0=[V_in, p_in],
        args=(Dt, L, D_in, cfg["physics"]["rho"]),
        method="RK45",
        t_eval=x_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"RK45 marching failed for Dt={Dt}, V_in={V_in}: {sol.message}")

    return {
        "x": sol.t,
        "V": sol.y[0],
        "p": sol.y[1],
        "n_steps": sol.t.size if sol.t_events is None else len(sol.t),
        "n_func_evals": sol.nfev,
        "success": sol.success,
    }
