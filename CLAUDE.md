# CLAUDE.md — Build Instructions for Claude Code

You are building **PINNs-Bernoulli**: a complete, runnable, parametric Physics-Informed Neural Network (PINN) surrogate solver for flow through a converging–diverging nozzle. The README.md in this repo contains the full theory; treat it as the specification's narrative. **This file is the binding build contract.** Follow it literally. Where this file and your instincts disagree, this file wins.

## 0. Mission (one paragraph)

Build a PyTorch codebase in which a fully connected neural network, with inputs $(\tilde{x}, \tilde{D}_t)$ (dimensionless axial position and dimensionless throat diameter) and outputs $(\hat{\tilde V}, \hat{\tilde p})$ (dimensionless velocity and pressure), is trained **with no CFD data** by minimizing residuals of the quasi-1D continuity and Bernoulli/momentum equations plus boundary conditions, across the whole throat-diameter design space. Then extend to compressible quasi-1D Euler (Stage 2). Export results to ParaView (`.vtu`), MATLAB (`.mat`), CSV, and figures. Everything must run end-to-end from a fresh clone with `pip install -r requirements.txt` and three commands.

## 1. Tech stack (fixed — do not substitute)

- Python ≥ 3.10, **PyTorch ≥ 2.0** (autodiff + optimizers), NumPy, SciPy (`qmc` Latin Hypercube, `io.savemat`), Matplotlib (figures), **pyvista** (`.vtu` export), PyYAML (config), pandas (CSV), tqdm (progress), pytest (tests).
- All physics-path tensors: `torch.float64`. Device-agnostic: `DEVICE = cuda if available else cpu`.
- No PINN frameworks (no DeepXDE/Modulus): the educational point is building everything explicitly.

## 2. Repository tree to create (exact)

```
configs/default.yaml
src/__init__.py  src/geometry.py  src/physics.py  src/analytical.py  src/networks.py
src/losses.py  src/sampling.py  src/train.py  src/evaluate.py  src/export.py  src/visualize.py
scripts/run_stage1_bernoulli.py  scripts/run_stage2_compressible.py  scripts/run_parametric_sweep.py
postprocessing/paraview/README.md  postprocessing/matlab/load_and_plot.m
tests/test_analytical.py  tests/test_gradients.py  tests/test_losses.py
requirements.txt  .gitignore   # ignore results/, .venv/, __pycache__
```
Create `results/` at runtime with subfolders `checkpoints/ figures/ paraview/ matlab/ csv/ logs/`.

## 3. Physics specification (exact — copy these numbers)

**Geometry:** $L=3.0\,\text{m}$, $x\in[0,L]$, throat at $x_t=L/2$.
$D(x;D_t)=D_{in}+(D_t-D_{in})\sin^2(\pi x/L)$, $A(x;D_t)=\frac{\pi}{4}D^2$.
Defaults: $D_{in}=0.5\,\text{m}$, $D_t\in[0.20,0.45]\,\text{m}$, $\rho=1.225\,\text{kg/m}^3$, $V_{in}=10\,\text{m/s}$, $p_{in}=101325\,\text{Pa}$.

**Non-dimensionalization (train entirely in these variables):**
$\tilde x=x/L,\ \tilde A=A/A_{in},\ \tilde D_t=D_t/D_{in}\in[0.4,0.9],\ \tilde V=V/V_{in},\ \tilde p=(p-p_{in})/(\tfrac12\rho V_{in}^2)$.

**Stage 1 residuals (the loss physics):**
- continuity: $r_{cont}=\partial_{\tilde x}\!\left[\tilde A(\tilde x;\tilde D_t)\,\hat{\tilde V}\right]=0$
- momentum/Bernoulli: $r_{mom}=\partial_{\tilde x}\!\left(\hat{\tilde p}+\hat{\tilde V}^2\right)=0$
- BCs: $\hat{\tilde V}(0,\tilde D_t)=1,\ \hat{\tilde p}(0,\tilde D_t)=0$ (enforce as **hard constraint by default**, soft penalty as config option).

**Analytical ground truth:** $\tilde V=1/\tilde A,\ \tilde p=1-\tilde V^2$. Implement in SI and nondim forms.

**Stage 2 (compressible, subsonic branch only):** outputs $(\tilde\rho,\tilde V,\tilde p,\tilde T)$; residuals: $\partial_x(\rho A V)=0$; $\rho V\,\partial_x V+\partial_x p=0$; $c_p T+V^2/2=h_0$; $p=\rho R T$. Air: $\gamma=1.4$, $R=287\ \text{J/(kg·K)}$, $c_p=\gamma R/(\gamma-1)$. Reference state $(\rho_{in},V_{in},p_{in},T_{in})=(1.225,10,101325,300)$. Enforce positivity with softplus on $\tilde\rho,\tilde p,\tilde T$. Validate against isentropic area–Mach relation (implement its subsonic root solve with `scipy.optimize.brentq`). If residuals stall or solutions look oscillatory, apply $\lambda_{mom}=20$ and hard pressure BCs (per Hong et al. 2023) — expose both as config switches; do NOT attempt shock capturing in v1.

## 4. Network specification (exact)

- MLP: 2 inputs → **6 hidden layers × 64 neurons, tanh** → 2 outputs (Stage 1) / 4 (Stage 2). Xavier-uniform init on all linear layers; biases zero.
- Hard-BC output transform (default): $\hat{\tilde V}=1+\tilde x\,\mathcal N_V(\tilde x,\tilde D_t)$, $\hat{\tilde p}=\tilde x\,\mathcal N_p(\tilde x,\tilde D_t)$. Config flag `hard_bc: true|false`; when false, raw outputs + soft BC loss.
- Optional Fourier-feature input embedding (`fourier_features: false` default, scale 2.0, 16 frequencies) — flag exists, off for Stage 1.
- Implement `networks.py` as a single `PINN(nn.Module)` class with `forward(x, dt)` returning a tensor of outputs, plus a `count_parameters()` helper.

## 5. Loss specification (exact)

$$\mathcal L=\lambda_{cont}\mathcal L_{cont}+\lambda_{mom}\mathcal L_{mom}+\lambda_{bc}\mathcal L_{bc}+\lambda_{data}\mathcal L_{data}\ (\text{stub}=0)$$
Each term = mean of squared residuals over its point set. Residuals computed with `torch.autograd.grad(..., create_graph=True)` — **never finite differences inside training**.
Weighting (`loss_weighting: annealing|fixed`): default `annealing` = Wang et al. learning-rate annealing (Algorithm 1 of the paper), recomputed every 500 epochs: $\hat\lambda_i=\max_\theta|\nabla_\theta\mathcal L_{r}|/\overline{|\nabla_\theta\mathcal L_i|}$, then $\lambda_i\leftarrow(1-\alpha)\lambda_i+\alpha\hat\lambda_i$ with $\alpha=0.9$ (i.e. the update leans 90% toward the fresh estimate — implement exactly as written, with $\alpha=0.9$). When `hard_bc: true`, drop $\mathcal L_{bc}$.

## 6. Training specification (exact)

1. Collocation: LHS over $(\tilde x,\tilde D_t)\in[0,1]\times[0.4,0.9]$: `n_collocation_x=512` × `n_dt_values=8`; resample every 1000 epochs. Also sample 64 boundary points at $\tilde x=0$ across $\tilde D_t$ (used only when `hard_bc: false`).
2. **Adam** `lr=1e-3`, exponential decay gamma 0.95 every 2000 epochs, **20,000 epochs**, full batch. Log every 100 epochs: each loss term, each $\lambda$, rel-$L^2$ vs analytical (on 2000 fixed validation points covering 5 held-out $D_t$: 0.225/0.275/0.325/0.375/0.425 m).
3. **L-BFGS** polish: `max_iter=5000`, `history_size=50`, strong-Wolfe, tolerance_grad 1e-9.
4. Checkpoints: save every 5000 epochs + final: `results/checkpoints/stage{1,2}_final.pt` (state_dict + config + training history).
5. Seed everything (`seed: 1234`) — torch, numpy, python random.

## 7. Acceptance gates (the build is NOT done until all pass)

- `pytest tests/ -q` green:
  - `test_analytical`: analytical solution plugged into residual functions gives < 1e-12 (float64).
  - `test_gradients`: autodiff $\partial_{\tilde x}$ vs central finite difference agree to < 1e-6 relative.
  - `test_losses`: every loss term = 0 (to 1e-12) when the network is replaced by the exact solution.
- After Stage 1 training: rel-$L^2(\tilde V)<1\mathrm e{-3}$ and rel-$L^2(\tilde p)<1\mathrm e{-2}$ on **held-out** $D_t$ values; $|\tilde p+\tilde V^2-1|<1\%$ everywhere.
- After Stage 2: Mach/pressure vs isentropic relations, rel-$L^2<1\mathrm e{-2}$ (subsonic branch).
- `scripts/run_parametric_sweep.py` produces 25 `.vtu` files + 1 `.mat` + figures without error.

## 8. File-by-file implementation notes

- `geometry.py`: `diameter(x, dt)`, `area(x, dt)`, `area_nondim(xt, dtt)`, plus SI↔nondim converters; must accept torch tensors and be differentiable.
- `physics.py`: `residual_continuity(model, xt, dtt)`, `residual_momentum(...)`, `residuals_euler(...)`; enable `requires_grad_(True)` on inputs internally; return residual tensors.
- `analytical.py`: `velocity_exact`, `pressure_exact` (nondim + SI), `mach_from_area_ratio(...)` via brentq (subsonic branch, bracket [1e-6, 1-1e-6]).
- `sampling.py`: `lhs_collocation(n_x, n_dt, seed)`, `boundary_points(n, seed)`, `validation_grid(...)`.
- `train.py`: `train(model, cfg, stage)` → history dict; implements Adam phase, annealing, L-BFGS phase, checkpointing, logging to `results/logs/train_stage{1,2}.log`.
- `evaluate.py`: `relative_l2(pred, ref)`, `evaluate_held_out(model, cfg)` returning a metrics dict; save JSON.
- `export.py`: `export_csv`, `export_vtu(Dt_list)` — axisymmetric reconstruction: for each $x$-station, rectangle mesh $y\in[-D/2,D/2]$ (64 radial pts), point data `V`, `p` (SI units), via `pyvista.UnstructuredGrid.save()`; `export_mat(...)` via `scipy.io.savemat` with keys `X, DT, V, P, D, V_exact, P_exact` (SI).
- `visualize.py`: (a) $\tilde V(\tilde x),\tilde p(\tilde x)$ PINN vs analytical for 4 $D_t$; (b) throat values vs $D_t$ sweep; (c) total-head error map over $(\tilde x,\tilde D_t)$; (d) loss history (log scale, per term). PNG, 300 dpi, labeled axes with units.
- `configs/default.yaml`: every number in Sections 3–6 as a nested key (`physics:`, `geometry:`, `network:`, `training:`, `loss:`, `export:`). No magic numbers in code — everything reads from config.
- `postprocessing/paraview/README.md`: open `.vtu` → color by `p`; filters: Calculator (`V_mag`), Warp by Vector, Stream Tracer; animation over the 25-file series via ParaView's time-grouping (files named `nozzle_Dt_###.vtu` with sequential indices).
- `postprocessing/matlab/load_and_plot.m`: pure MATLAB script, no toolboxes: load `.mat`, reproduce figure set (a)–(c).

## 9. Coding standards

Type hints everywhere; docstrings (NumPy style) citing the README equation number each function implements; PEP8; no global state; CLI via `argparse` (`--config`, `--checkpoint`, `--dt-min/--dt-max/--n`); deterministic seeds; graceful CPU fallback; tqdm bars for epochs and exports.

## 10. Build order (strict)

1. `requirements.txt`, `configs/default.yaml`, `geometry.py` + its quick sanity plot.
2. `analytical.py`, `physics.py`, then the **three tests** — they must pass before any training code is written.
3. `networks.py`, `losses.py`, `sampling.py`, `train.py` → run Stage 1 short smoke test (200 epochs) → then full run → verify acceptance gates.
4. `evaluate.py`, `export.py`, `visualize.py`, `scripts/`, postprocessing helpers.
5. Stage 2 via the same skeleton; config `stage: 2`.
6. Final: run all three scripts fresh, run pytest, confirm Section 7 gates, update README "How to run" if any command differs.

## 11. Hard "do not" list

- Do not use DeepXDE/Modulus/SciANN; do not use finite differences for PDE residuals; do not train in dimensional variables; do not hardcode hyperparameters outside the config; do not clamp/gate the analytical validation to training $D_t$ values only; do not silently swallow failed acceptance gates — report them in the final summary; do not attempt shocks/turbulence in v1.
