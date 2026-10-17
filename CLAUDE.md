# CLAUDE.md — instructions for Claude Code in this repository

## What this repo is

A **parametric, data-free PINN solver** for steady, incompressible, laminar,
axisymmetric flow (water, Re = 50) through a converging–diverging nozzle
whose **throat diameter** is a network input. One training run produces a
surrogate for every throat ratio d_t = D_t/D_in in [0.40, 1.00]; inference of a
new design is a single forward pass. OpenFOAM (`simpleFoam`, same
non-dimensional problem) is the reference solver.

The previous contents of the repo (a quasi-1D Bernoulli PINN) were removed at
the user's explicit request; do not resurrect them.

Current result (committed): mean rel-L2 error vs OpenFOAM 0.40 % (u) / 1.1 % (p),
worst 1.3 % / 3.9 % at d_t = 0.4; training 3.6 h on 4 CPU cores; inference
0.26 s per design vs 34.6 s per OpenFOAM run. Velocity comes from a Stokes
stream function (exact mass conservation) — do not revert to primitive
(u, v, p) outputs with soft continuity; README §7.3 explains why.

## Layout

- `configs/nozzle.yaml` — every number (geometry, Re, network, training, OpenFOAM mesh).
- `nozzle_pinn/` — `geometry`, `model` (hard-BC ansatz), `physics` (NS residuals),
  `sampling`, `train`, `inference`, `compare`, `openfoam` (case writer/runner/reader),
  `postprocess` (figures, GIF, VTK).
- `scripts/run_openfoam.py` → `scripts/train.py` → `scripts/infer_and_postprocess.py`
  (`scripts/run_all.sh` runs all three).
- `tests/` — `pytest tests -q` must pass.
- `results/` — committed figures, tables, trained model; `openfoam/cases/` and
  `results/paraview/` are generated and git-ignored.

## Rules

- The PINN loss must stay **physics-only**. OpenFOAM data may be used for
  monitoring/validation, never inside the loss.
- Keep all derivatives in the loss as autograd derivatives (no finite differences).
- Keep numbers in `configs/nozzle.yaml`, not in code.
- After changing physics/model code: run `pytest tests -q`, retrain, rerun
  post-processing, and report the PINN-vs-OpenFOAM errors explicitly.
- OpenFOAM here is the Ubuntu `openfoam` package (v1912); binaries need
  `WM_PROJECT_DIR=/usr/share/openfoam` (set automatically by `nozzle_pinn/openfoam.py`).
