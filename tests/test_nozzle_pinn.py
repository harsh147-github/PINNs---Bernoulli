"""Unit tests: geometry, hard constraints, PDE operator, stream function, I/O."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nozzle_pinn.config import load_config  # noqa: E402
from nozzle_pinn.geometry import Nozzle  # noqa: E402
from nozzle_pinn.inference import field_grid  # noqa: E402
from nozzle_pinn.model import NozzlePINN  # noqa: E402
from nozzle_pinn.openfoam import _parse_list  # noqa: E402
from nozzle_pinn.physics import ns_residuals  # noqa: E402

torch.set_default_dtype(torch.float64)
CFG = load_config()
NOZ = Nozzle.from_config(CFG)
RE = float(CFG["physics"]["Re"])


def _model(seed: int = 0) -> NozzlePINN:
    torch.manual_seed(seed)
    return NozzlePINN(NOZ, 32, 3)


def _points(n: int, dt: float | None = None, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    d = (torch.full((n,), dt) if dt is not None
         else NOZ.dt_min + (NOZ.dt_max - NOZ.dt_min) * torch.rand(n, generator=g))
    x = NOZ.length * torch.rand(n, generator=g)
    eta = 0.02 + 0.96 * torch.rand(n, generator=g)
    X = torch.stack([x, NOZ.radius(x, d) * eta], 1).requires_grad_(True)
    return X, d


def test_geometry_endpoints_and_smoothness():
    for dt in (0.4, 0.7, 1.0):
        assert NOZ.radius(0.0, dt) == pytest.approx(0.5)
        assert NOZ.radius(NOZ.length, dt) == pytest.approx(0.5)
        assert NOZ.radius(NOZ.x_throat_mid, dt) == pytest.approx(dt / 2)
    # analytic slope agrees with a central difference everywhere
    x = np.linspace(0.01, NOZ.length - 0.01, 2001)
    h = 1e-6
    fd = (NOZ.radius(x + h, 0.4) - NOZ.radius(x - h, 0.4)) / (2 * h)
    assert np.max(np.abs(fd - NOZ.dradius_dx(x, 0.4))) < 1e-7


def test_hard_boundary_conditions_hold_for_random_weights():
    m = _model()
    n = 200
    g = torch.Generator().manual_seed(3)
    d = NOZ.dt_min + (NOZ.dt_max - NOZ.dt_min) * torch.rand(n, generator=g)
    x = NOZ.length * torch.rand(n, generator=g)
    wall = m(x, NOZ.radius(x, d), d)                       # no-slip
    assert torch.max(torch.abs(wall["u"])) < 1e-12 and torch.max(torch.abs(wall["v"])) < 1e-12
    r = 0.5 * torch.rand(n, generator=g)
    inlet = m(torch.zeros(n), r, d)                        # Poiseuille inlet
    assert torch.allclose(inlet["u"], 2 * (1 - (2 * r) ** 2), atol=1e-12)
    assert torch.max(torch.abs(inlet["v"])) < 1e-12
    out = m(torch.full((n,), NOZ.length), r, d)            # p = 0 at outlet
    assert torch.max(torch.abs(out["p"])) < 1e-12
    axis = m(x, torch.zeros(n), d)                         # v = 0 on axis
    assert torch.max(torch.abs(axis["v"])) < 1e-12


def test_base_ansatz_conserves_mass_exactly():
    """With N = 0 the ansatz is a mass-conserving flow for every nozzle."""
    m = _model()
    for p in m.mlp.net[-1].parameters():
        torch.nn.init.zeros_(p)
    X, d = _points(500)
    res = ns_residuals(m, X, d, RE)
    assert torch.max(torch.abs(res["cont"])) < 1e-9


def test_poiseuille_is_exact_solution_of_residual_operator():
    """dt = 1 (straight pipe): u = 2(1-4r^2), p = 32 (L - x)/Re zeroes all residuals."""
    m = _model()
    last = m.mlp.net[-1]
    torch.nn.init.zeros_(last.weight)
    with torch.no_grad():
        last.bias.copy_(torch.tensor([0.0, 0.0, 32.0 * NOZ.length / RE]))
    X, d = _points(500, dt=1.0)
    res = ns_residuals(m, X, d, RE)
    for k, v in res.items():
        assert torch.max(torch.abs(v)) < 1e-9, k


def test_stream_function_reaches_one_on_wall():
    m = _model()
    for p in m.mlp.net[-1].parameters():
        torch.nn.init.zeros_(p)
    g = field_grid(m, 0.55, 51, 401)
    assert np.allclose(g["psi"][0], 0.0)
    assert np.allclose(g["psi"][-1], 1.0, atol=1e-4)


def test_openfoam_list_parser():
    s = "internalField nonuniform List<vector> 2\n(\n(1 2 3)\n(4 5 6)\n);"
    arr, _ = _parse_list(s, 0, vector=True)
    assert arr.shape == (2, 3) and arr[1, 2] == 6.0
    arr, _ = _parse_list("3 ( 1.5 2.5 -3e-2 )", 0, vector=False)
    assert np.allclose(arr, [1.5, 2.5, -0.03])
