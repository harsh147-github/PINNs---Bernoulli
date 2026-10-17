"""Generate, run and read OpenFOAM reference cases (README §12).

Each throat diameter becomes an axisymmetric *wedge* case solved with
``simpleFoam`` (steady, incompressible, laminar SIMPLE algorithm) using
EXACTLY the same non-dimensional problem the PINN solves:

* inlet  : fully developed Poiseuille profile u = 2 (1 - (r/R_in)^2), mean 1
* wall   : no-slip
* outlet : p = 0, zero-gradient velocity
* nu     : 1 / Re   (D_in = 1, U_mean = 1  ->  p is p/(rho U^2))

The mesh is one blockMesh hex block whose axis edge is collapsed (a standard
OpenFOAM wedge) and whose wall edge follows R(x; dt) through a ``polyLine``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .geometry import Nozzle

HEADER = """FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _write(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER.format(cls=cls, obj=obj) + body, encoding="utf-8")


def foam_env() -> Dict[str, str]:
    """Environment for the Ubuntu ``openfoam`` package (binaries in /usr/bin)."""
    env = dict(os.environ)
    env.setdefault("WM_PROJECT_DIR", "/usr/share/openfoam")
    return env


# -----------------------------------------------------------------------------
# Case writing
# -----------------------------------------------------------------------------
def write_block_mesh(case: Path, noz: Nozzle, dt: float, of: Dict[str, Any]) -> None:
    """blockMeshDict: single collapsed-axis wedge block with a polyLine wall."""
    half = np.deg2rad(0.5 * float(of["wedge_angle_deg"]))
    c, s = np.cos(half), np.sin(half)
    L = noz.length
    r_in = float(noz.radius(0.0, dt))
    r_out = float(noz.radius(L, dt))
    xs = np.linspace(0.0, L, 241)[1:-1]
    rs = noz.radius(xs, dt)
    front = "\n".join(f"        ({x:.8f} {r * c:.8f} {-r * s:.8f})" for x, r in zip(xs, rs))
    back = "\n".join(f"        ({x:.8f} {r * c:.8f} {r * s:.8f})" for x, r in zip(xs, rs))
    body = f"""
scale 1;

vertices
(
    (0 0 0)                                         // 0 axis, inlet
    ({L:.8f} 0 0)                                   // 1 axis, outlet
    ({L:.8f} {r_out * c:.8f} {-r_out * s:.8f})      // 2 wall, outlet, front
    (0 {r_in * c:.8f} {-r_in * s:.8f})              // 3 wall, inlet,  front
    (0 {r_in * c:.8f} {r_in * s:.8f})               // 4 wall, inlet,  back
    ({L:.8f} {r_out * c:.8f} {r_out * s:.8f})       // 5 wall, outlet, back
);

blocks
(
    hex (0 1 2 3 0 1 5 4) ({int(of["nx"])} {int(of["nr"])} 1)
    simpleGrading (1 {float(of["wall_grading"])} 1)
);

edges
(
    polyLine 3 2
    (
{front}
    )
    polyLine 4 5
    (
{back}
    )
);

boundary
(
    inlet  {{ type patch;  faces ((0 3 4 0)); }}
    outlet {{ type patch;  faces ((1 5 2 1)); }}
    wall   {{ type wall;   faces ((3 2 5 4)); }}
    front  {{ type wedge;  faces ((0 1 2 3)); }}
    back   {{ type wedge;  faces ((0 4 5 1)); }}
    axis   {{ type empty;  faces ((0 1 1 0)); }}
);
"""
    _write(case / "system" / "blockMeshDict", "dictionary", "blockMeshDict", body)


def write_case(case: Path, noz: Nozzle, dt: float, cfg: Dict[str, Any]) -> None:
    """Write a complete simpleFoam case for throat ratio ``dt``."""
    of = cfg["openfoam"]
    nu = 1.0 / float(cfg["physics"]["Re"])
    if case.exists():
        shutil.rmtree(case)
    write_block_mesh(case, noz, dt, of)

    _write(case / "system" / "controlDict", "dictionary", "controlDict", f"""
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {int(of["max_iters"])};
deltaT          1;
writeControl    timeStep;
writeInterval   {int(of["max_iters"])};
purgeWrite      1;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
""")
    _write(case / "system" / "fvSchemes", "dictionary", "fvSchemes", """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
""")
    tol = float(of["residual_tol"])
    _write(case / "system" / "fvSolution", "dictionary", "fvSolution", f"""
solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-10; relTol 0.01; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.05; }}
}}
SIMPLE
{{
    consistent      yes;
    nNonOrthogonalCorrectors 0;
    // Uz on a wedge is round-off noise with a meaningless normalised
    // residual, so convergence is judged on p (and Ux in final_residuals).
    residualControl {{ p {tol}; }}
}}
relaxationFactors
{{
    equations {{ U 0.9; }}
    fields    {{ p 1.0; }}
}}
""")
    _write(case / "constant" / "transportProperties", "dictionary", "transportProperties",
           f"\ntransportModel Newtonian;\nnu {nu:.10g};\n")
    _write(case / "constant" / "turbulenceProperties", "dictionary", "turbulenceProperties",
           "\nsimulationType laminar;\n")
    _write(case / "0" / "p", "volScalarField", "p", """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    wall   { type zeroGradient; }
    front  { type wedge; }
    back   { type wedge; }
    axis   { type empty; }
}
""")
    # U is written after meshing, once inlet face centres are known.


def _write_velocity(case: Path, r_inlet_faces: np.ndarray) -> None:
    """0/U with the Poiseuille inlet profile u = 2 (1 - (2r)^2), D_in = 1."""
    u = 2.0 * (1.0 - (2.0 * r_inlet_faces) ** 2)
    vals = "\n".join(f"({ui:.10g} 0 0)" for ui in u)
    _write(case / "0" / "U", "volVectorField", "U", f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (1 0 0);
boundaryField
{{
    inlet  {{ type fixedValue; value nonuniform List<vector> {len(u)}
(
{vals}
);
    }}
    outlet {{ type zeroGradient; }}
    wall   {{ type noSlip; }}
    front  {{ type wedge; }}
    back   {{ type wedge; }}
    axis   {{ type empty; }}
}}
""")


# -----------------------------------------------------------------------------
# Running
# -----------------------------------------------------------------------------
def _run(cmd: list[str], case: Path, log: str) -> float:
    t0 = time.perf_counter()
    with open(case / log, "w", encoding="utf-8") as fh:
        subprocess.run(cmd + ["-case", str(case)], stdout=fh, stderr=subprocess.STDOUT,
                       env=foam_env(), check=True)
    return time.perf_counter() - t0


def run_case(case: Path) -> Dict[str, float]:
    """Mesh, set the inlet profile, solve one case; return timings.

    ``mesh_s`` is blockMesh wall-clock time, ``solve_s`` is simpleFoam
    wall-clock time and ``iterations`` the number of SIMPLE iterations used.
    """
    times = {"mesh_s": _run(["blockMesh"], case, "log.blockMesh")}
    mesh = read_mesh(case)
    fc = mesh["face_centres"][mesh["patches"]["inlet"]]
    _write_velocity(case, np.hypot(fc[:, 1], fc[:, 2]))
    times["solve_s"] = _run(["simpleFoam"], case, "log.simpleFoam")
    times["iterations"] = float(latest_time(case))
    return times


# -----------------------------------------------------------------------------
# Reading ASCII fields
# -----------------------------------------------------------------------------
def latest_time(case: Path) -> str:
    """Name of the newest numeric time directory (excluding 0)."""
    times = [d.name for d in case.iterdir() if d.is_dir() and re.fullmatch(r"[0-9.eE+-]+", d.name)]
    times = [t for t in times if float(t) > 0]
    return max(times, key=float)


def _parse_list(txt: str, start: int, vector: bool) -> Tuple[np.ndarray, int]:
    """Parse an OpenFOAM ``N ( ... )`` list starting at/after ``start``."""
    m = re.compile(r"(\d+)\s*\(").search(txt, start)
    n = int(m.group(1))
    i = m.end()
    depth, j = 1, i
    while depth:
        ch = txt[j]
        depth += ch == "("
        depth -= ch == ")"
        j += 1
    inner = txt[i:j - 1]
    if vector:
        inner = inner.replace("(", " ").replace(")", " ")
    arr = np.array(inner.split(), dtype=float)
    return (arr.reshape(n, 3) if vector else arr.reshape(n)), j


def read_internal(path: Path, vector: bool) -> np.ndarray:
    """Read ``internalField`` of an ASCII volField file."""
    txt = path.read_text()
    k = txt.index("internalField")
    if txt[k:k + 40].split()[1] == "uniform":
        raise ValueError(f"{path} has a uniform internalField")
    return _parse_list(txt, k, vector)[0]


def _strip_header(txt: str) -> str:
    """Drop the FoamFile header and C/C++ comments."""
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//[^\n]*", "", txt)
    return txt[txt.index("}", txt.index("FoamFile")) + 1:]


def read_mesh(case: Path) -> Dict[str, Any]:
    """Read an ASCII ``constant/polyMesh`` and compute face and cell centres.

    Centres are vertex averages, which coincide with OpenFOAM's centroids to
    second order on this smooth, structured mesh.
    """
    pm = case / "constant" / "polyMesh"
    pts = _parse_list(_strip_header((pm / "points").read_text()), 0, vector=True)[0]
    ftxt = _strip_header((pm / "faces").read_text())
    m = re.search(r"(\d+)\s*\(", ftxt)
    faces = [list(map(int, f.split()))
             for f in re.findall(r"\d+\(([\d\s]+)\)", ftxt[m.end():])]
    owner = _parse_list(_strip_header((pm / "owner").read_text()), 0, vector=False)[0].astype(int)
    neigh = _parse_list(_strip_header((pm / "neighbour").read_text()), 0, vector=False)[0].astype(int)
    fc = np.array([pts[f].mean(axis=0) for f in faces])
    n_cells = int(owner.max()) + 1
    # unique vertices of each cell -> vertex-average cell centre
    cell_pts: list[set] = [set() for _ in range(n_cells)]
    for fi, f in enumerate(faces):
        cell_pts[owner[fi]].update(f)
        if fi < len(neigh):
            cell_pts[neigh[fi]].update(f)
    cc = np.array([pts[list(c)].mean(axis=0) for c in cell_pts])
    btxt = _strip_header((pm / "boundary").read_text())
    patches = {}
    for name, body in re.findall(r"(\w+)\s*\{([^}]*)\}", btxt):
        n = int(re.search(r"nFaces\s+(\d+)", body).group(1))
        s0 = int(re.search(r"startFace\s+(\d+)", body).group(1))
        patches[name] = np.arange(s0, s0 + n)
    return {"points": pts, "face_centres": fc, "cell_centres": cc, "patches": patches}


def read_solution(case: Path) -> Dict[str, np.ndarray]:
    """Cell-centre coordinates (x, r) and fields (u, v, p) of the final iterate."""
    t = latest_time(case)
    C = read_mesh(case)["cell_centres"]
    U = read_internal(case / t / "U", vector=True)
    p = read_internal(case / t / "p", vector=False)
    r = np.hypot(C[:, 1], C[:, 2])
    # radial velocity: project (Uy, Uz) on the radial unit vector
    v = (U[:, 1] * C[:, 1] + U[:, 2] * C[:, 2]) / np.maximum(r, 1e-30)
    return {"x": C[:, 0], "r": r, "u": U[:, 0], "v": v, "p": p}


def final_residuals(case: Path) -> Dict[str, float]:
    """Last reported initial residuals for Ux and p from the solver log."""
    txt = (case / "log.simpleFoam").read_text()
    out = {}
    for name in ("Ux", "p"):
        hits = re.findall(rf"Solving for {name}, Initial residual = ([0-9.eE+-]+)", txt)
        out[name] = float(hits[-1]) if hits else float("nan")
    return out
