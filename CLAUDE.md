# CLAUDE.md — Claude Code Instructions for PINNs-Bernoulli

## 0. STATUS — READ THIS FIRST

**This repository is a parametric incompressible Bernoulli (water) surrogate, and ONLY that. It is ALREADY IMPLEMENTED AND TESTED.** The code exists: `src/` (geometry, physics, analytical, networks, losses, sampling, train, evaluate, export, visualize, liveviz, netviz), `scripts/`, `tests/`, `configs/default.yaml`, `postprocessing/`. All 6 pytest tests pass, and a smoke training run produced all figures, ParaView `.vtu` files, and the MATLAB `.mat` export correctly. Everything here runs on CPU alone (no GPU required); a GPU speeds it up but nothing in the pipeline depends on one.

**There is no Stage 2 / compressible / air flow work in scope.** An earlier revision of this repo had a compressible quasi-1D Euler (air) extension; it has been deliberately removed — this project is scoped to incompressible water flow only. Do not reintroduce compressible/air-flow code, a `stage` config switch, or anything resembling it unless the user explicitly asks for it again.

**DO NOT rewrite this from scratch.** Do not restructure the repo, do not rename files, do not replace working modules. Your jobs, strictly in order:

1. **VERIFY** the existing build (Section 2): run the tests, run training, confirm the acceptance gates pass.
2. **ROADMAP** items (Section 7) — only when the user explicitly asks.

README.md contains the full theory narrative. This file is the binding build/run contract. Where they conflict, this file wins.

## 1. Mission (one paragraph)

The repo trains a PyTorch neural network $(\tilde{x}, \tilde{D}_t) \to (\hat{\tilde V}, \hat{\tilde p})$ **with no CFD data** by minimizing residuals of quasi-1D continuity and Bernoulli/momentum equations for incompressible water flow, across the whole throat-diameter design space — a parametric CFD surrogate. It exports colored CFD-style plots, a ParaView `.vtu` time series, and a MATLAB `.mat` file, all validated against the closed-form analytical solution.

## 2. VERIFY — run the existing build (do this first)

Windows (VS Code terminal, PowerShell):

```powershell
git pull origin main                      # get ALL code (src/, scripts/, tests/, configs/)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # CUDA is optional; CPU-only is fine and fully supported
pytest tests/ -q                          # 6 tests MUST pass (physics verified vs analytical solution)
python scripts/run_stage1_bernoulli.py --config configs/default.yaml
```

The training script trains (Adam 20,000 epochs → L-BFGS 5,000 iters), then automatically: evaluates acceptance gates, exports CSV + 25× `.vtu` + `.mat`, and writes 5 figures. It prints `device: cuda` when a GPU is available and used, `device: cpu` otherwise — both are correct. Final output must show **ALL GATES: PASSED**:
- rel-L2(Ṽ) < 1e-3 and rel-L2(p̃) < 1e-2 on held-out throat diameters (0.225–0.425 m, never trained on)
- max |p̃ + Ṽ² − 1| < 1% (Bernoulli invariant)

For a quick check before the full run: `python scripts/run_stage1_bernoulli.py --adam-epochs 300 --lbfgs-iter 0 --out results_smoke` (gates will NOT pass at 300 epochs — that is expected; it proves the pipeline end-to-end in ~1 minute).

## 3. WATCH THE LEARNING LIVE (the user wants dynamic visuals)

In `configs/default.yaml`:
- `training.live_dashboard: true` — saves a 3-panel dashboard frame (loss curves + velocity guess vs exact + pressure guess vs exact) every `frame_every` epochs to `results/frames/`.
- `training.live_display: true` — additionally pops a live matplotlib window updating during training (local machine with display).
- `training.live_network_viz: true` — a *different* live window (`src/netviz.py`): the actual network drawn as neurons + weighted connections (colored/thickened by each weight's live value), redrawn every `netviz_frame_every` epochs, saved to `results/netviz_frames/` either way. `live_display: true` is required for it to pop an interactive window rather than only save frames — note `netviz.py` forces the `TkAgg` backend explicitly, because matplotlib's auto-detection has been observed to silently fall back to a headless backend on this machine when launched from a wrapping tool/piped stdout, which sinks `live_display` without an error (only a swallowed `FigureCanvasAgg is non-interactive` warning) — do not remove that `matplotlib.use("TkAgg")` call without confirming the underlying auto-detection issue is actually fixed.

After training, stitch the replay:
```powershell
python scripts/make_learning_replay.py --frames results/frames --out results/learning_replay.gif
```

## 4. POST-PROCESSING (must work end-to-end, dynamically)

**ParaView:** `results/paraview/nozzle_Dt_000.vtu … nozzle_Dt_024.vtu` — open all 25 together (ParaView groups them as a time series) → color by `p` or `V` → press **Play** to animate the flow field as the throat diameter sweeps. Full instructions: `postprocessing/paraview/README.md`.

**MATLAB:** `results/matlab/nozzle_surrogate.mat` — run `postprocessing/matlab/load_and_plot.m`; it reproduces the colored pressure field, throat velocity/pressure vs Dₜ sweeps, and the error map. No toolboxes required.

**Parametric sweep on demand:**
```powershell
python scripts/run_parametric_sweep.py --checkpoint results/checkpoints/stage1_final.pt --dt-min 0.20 --dt-max 0.45 --n 25
```

## 5. PHYSICS / NETWORK / LOSS / TRAINING SPEC (unchanged contract — use for verification)

**Geometry:** L = 3.0 m, D(x;Dₜ) = D_in + (Dₜ − D_in)·sin²(πx/L), A = πD²/4, D_in = 0.5 m, Dₜ ∈ [0.20, 0.45] m, throat at x = L/2.
**Fluid:** water, ρ = 998 kg/m³, V_in = 2.0 m/s, p_in = 101325 Pa.
**Non-dimensional:** x̃ = x/L, Ã = A/A_in, D̃ₜ = Dₜ/D_in ∈ [0.4, 0.9], Ṽ = V/V_in, p̃ = (p − p_in)/(½ρV_in²).
**Residuals:** continuity ∂(ÃṼ)/∂x̃ = 0; Bernoulli/momentum ∂(p̃ + Ṽ²)/∂x̃ = 0; BCs Ṽ(0) = 1, p̃(0) = 0 (hard-wired via output transform Ṽ = 1 + x̃·N_V, p̃ = x̃·N_p).
**Analytical truth:** Ṽ = 1/Ã, p̃ = 1 − Ṽ².
**Network:** 2 → 6×64 tanh → 2, Xavier-uniform init, float64 physics paths.
**Loss:** L = λ_cont·L_cont + λ_mom·L_mom (+λ_bc·L_bc only if hard_bc: false); each term = mean squared residual via `torch.autograd.grad(..., create_graph=True)` — never finite differences in training. Weighting: Wang et al. (2021) LR-annealing Algorithm 1, α = 0.9, every 500 epochs. **Annealing must run BEFORE `total.backward()`** (the graph is freed by backward — this bug existed and was fixed; do not regress it).
**Training:** LHS collocation 512×8 over (x̃, D̃ₜ), resample every 1000 epochs; Adam lr 1e-3, ×0.95/2000 epochs, 20,000 epochs; L-BFGS max_iter 5000, history 50, strong-Wolfe; seed 1234 everywhere; checkpoints every 5000 epochs + final.

## 6. RULES (hard)

- Do not use DeepXDE/Modulus/SciANN. Do not rewrite or "improve" the existing modules. Do not train in dimensional variables. No magic numbers outside `configs/default.yaml`. Do not touch the fixed random seeds. Do not add compressible/air-flow (Euler) code or a `stage` config switch — this project is water/incompressible only. All new code: type hints, NumPy-style docstrings citing README equation numbers, PEP8. After any change: `pytest tests/ -q` must stay green and gates must still pass — report gate status explicitly, never silently.

## 7. ROADMAP (only when the user asks)

- [ ] Data assimilation — sparse pressure-tap measurements via λ_data (stub exists in config)
- [ ] Throat optimization using ∂p/∂Dₜ from the surrogate (free via autodiff)
