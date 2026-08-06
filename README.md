# PINNs-Bernoulli

**A Physics-Informed Neural Network (PINN) surrogate solver for computational fluid dynamics, built from scratch, taught from scratch — using water flowing through a converging–diverging venturi and Bernoulli's theorem as the complete worked example.**

This repository contains a fully runnable, parametric PINN-based CFD surrogate. It predicts velocity and pressure fields as water flows through a converging–diverging duct **without a mesh, without a CFD solver, and without any CFD training data** — the governing equations themselves are the training signal. The throat diameter is a network input, so a single trained model answers: *"what happens to the flow if I narrow the throat?"* in milliseconds. Stage 2 extends the same approach to compressible air flow (quasi-1D Euler) for when the incompressible assumption stops holding.

## Status (verified against the code in this repo, not aspirational)

- **Stage 1 (incompressible, water) — trained and passing all three acceptance gates**, on an RTX 3060, seed 1234:
  rel-L2(Ṽ) = **1.03×10⁻⁴** (gate < 1e-3) · rel-L2(p̃) = **2.33×10⁻⁴** (gate < 1e-2) · max Bernoulli-invariant error = **3.4×10⁻²%** (gate < 1%) — all on throat diameters held out of training.
- **Stage 2 (compressible Euler, air) — code complete and smoke-tested**, full 20k-epoch training run not yet executed.
- 7/7 tests pass (`pytest tests/ -q`): the 6 original Stage 1 tests plus one new Stage 2 physics sanity check.
- A CUDA device-mismatch bug (fresh CPU tensors passed to a GPU-resident model) was found and fixed in `evaluate.py`, `physics.py`, and `export.py` while verifying training actually ran on this machine's GPU — see §9.1 below.

---

## Table of Contents

1. [Why this project exists](#1-why-this-project-exists)
2. [Why a PINN surrogate instead of a CFD solver](#2-why-a-pinn-surrogate-instead-of-a-cfd-solver)
3. [The physics: from Navier–Stokes down to Bernoulli](#3-the-physics-from-navierstokes-down-to-bernoulli)
4. [PINN theory from absolute zero](#4-pinn-theory-from-absolute-zero)
5. [How the equations become loss functions](#5-how-the-equations-become-loss-functions)
6. [Network architecture design (and why)](#6-network-architecture-design-and-why)
7. [The parametric surrogate: throat diameter as an input](#7-the-parametric-surrogate-throat-diameter-as-an-input)
8. [Repository layout — every file explained](#8-repository-layout--every-file-explained)
9. [Training protocol](#9-training-protocol)
10. [Validation against the analytical solution](#10-validation-against-the-analytical-solution)
11. [Post-processing: ParaView and MATLAB](#11-post-processing-paraview-and-matlab)
12. [How to run everything](#12-how-to-run-everything)
13. [Literature survey: everything that has been done before us](#13-literature-survey-everything-that-has-been-done-before-us)
14. [Roadmap](#14-roadmap)

---

## 1. Why this project exists

The goal is **complete, from-scratch understanding**: given the physics of internal flow (Bernoulli's theorem, continuity, momentum, eventually Navier–Stokes), how do you turn differential equations into a neural network that *is* the solver? By the end of this repository you will be able to answer, concretely and in code:

- What the governing equations are, what each term means physically, and which assumptions reduce Navier–Stokes to Bernoulli.
- What a neural network actually is mathematically (weights, biases, activations), and what forward propagation and backpropagation compute.
- How automatic differentiation gives exact PDE residuals of a network's output — the single trick that makes PINNs possible.
- How each equation (continuity, Bernoulli/momentum, boundary conditions) is converted, term by term, into a PyTorch loss function.
- How the network is designed (inputs, outputs, layers, neurons, activations, initialization, output transforms), and *why* each choice is made.
- How boundary conditions are enforced (soft penalty vs. hard architectural constraint).
- How a geometry parameter (throat diameter) becomes just another network input, turning one trained model into a **surrogate** for infinitely many nozzle designs.
- How results are exported to ParaView (`.vtk/.vtu`) and MATLAB (`.mat`) for professional post-processing.

We deliberately start with **Bernoulli's theorem in a converging–diverging nozzle** because it is the simplest flow model that still contains the essential structure of every CFD problem: a conservation law (mass), a momentum/energy law (Bernoulli), boundary conditions, a geometry, and a closed-form analytical solution to validate against. Everything learned here transfers directly to the compressible Euler equations and the full Navier–Stokes equations — which are included as staged extensions.

---

## 2. Why a PINN surrogate instead of a CFD solver

A classical CFD solver (finite volume, finite difference, finite element) discretizes the domain into a mesh and marches equations cell by cell. This is powerful but has pain points that the literature has repeatedly identified:

1. **Meshing cost.** Every new geometry (e.g., a new throat diameter) needs a new mesh and a new solve. Design sweeps of 100 throat diameters = 100 meshes + 100 solves.
2. **Many-query problems.** Design optimization, uncertainty quantification, and real-time control need *thousands* of flow evaluations. Classical CFD is too slow for that loop.
3. **Data assimilation is awkward.** Classical solvers cannot naturally absorb sparse experimental measurements; PINNs can (Raissi's *Hidden Fluid Mechanics* reconstructed full velocity and pressure fields from sparse flow-visualization data alone [Raissi et al., Science 2020]).
4. **Differentiability.** A PINN is a differentiable function of everything — position, time, *and* design parameters. You can ask for ∂(pressure)/∂(throat diameter) by autodiff, for free, enabling gradient-based shape optimization.

A **PINN surrogate** is a neural network trained so that its output satisfies the governing PDEs (by penalizing their residuals in the loss), the boundary conditions, and optionally any available data. The landmark result for our use case is **Sun, Gao, Pan & Wang (CMAME 2020)**: a physics-constrained, *label-free* deep network that learns solutions of parametric Navier–Stokes flows over varying geometries with **zero CFD simulation data** — training is driven purely by PDE residuals. That is exactly the philosophy of this repository.

Once trained, the surrogate evaluates a full flow field in **milliseconds**, for **any** throat diameter in its training range. That is the payoff.

> **Honest caveat (from the literature):** PINNs are not universally better than CFD. They struggle with shocks/discontinuities (Hong et al. 2023 needed weighted losses + hard constraints to capture normal shocks in a CD nozzle), high Reynolds number turbulence, and stiff multiscale problems (Krishnapriyan et al., NeurIPS 2021, "Characterizing possible failure modes in PINNs"). Our staged design keeps us in regimes where PINNs are provably accurate, and Section 13 documents exactly where the known limits are.

---

## 3. The physics: from Navier–Stokes down to Bernoulli

### 3.1 The reduction ladder

The full, general statement of fluid motion is the Navier–Stokes equations. Every simpler model is Navier–Stokes plus assumptions. This ladder is the conceptual backbone of the repo — the code is organized into the same stages:

| Stage | Model | Assumptions dropped | Equations | Analytical check? |
|-------|-------|--------------------|-----------|-------------------|
| 0 | **Navier–Stokes (2D, incompressible)** | none (laminar) | continuity + 2 momentum PDEs | lid-driven cavity, cylinder benchmarks |
| 1 | **Quasi-1D incompressible inviscid** = *Bernoulli regime* | viscosity, 2D/3D effects, compressibility | continuity ODE + Bernoulli/momentum ODE | **exact closed form** ✅ |
| 2 | **Quasi-1D compressible isentropic (Euler)** | viscosity, 2D/3D effects | mass + momentum + energy ODEs + EOS | isentropic area–Mach relations ✅ |
| 3 | **Parametric surrogate** (Stage 1 or 2 physics + throat diameter as input) | — | same, now ∀D_t | analytical per D_t ✅ |

Stage 1 is the heart of this repository. Stage 2 is included. Stage 0 is the stretch goal with its own module.

### 3.2 Geometry: the converging–diverging nozzle

The nozzle is an axisymmetric duct of length $L = 3\,\text{m}$. Its local diameter is defined by one smooth law with a minimum (the **throat**) at $x_t = L/2$:

$$D(x; D_t) = D_{in} + (D_t - D_{in})\sin^2\!\left(\frac{\pi x}{L}\right), \qquad x \in [0, L]$$

- $D(0) = D(L) = D_{in}$ (inlet and outlet have the same diameter),
- $D(L/2) = D_t$ (the throat — our design parameter),
- smooth everywhere, zero slope at the throat → well-behaved pressure/velocity gradients.

Cross-sectional area: $A(x; D_t) = \dfrac{\pi}{4}D(x;D_t)^2$.

**Default values:** $D_{in} = 0.5\,\text{m}$, parametric range $D_t \in [0.2,\ 0.45]\,\text{m}$, fluid = **water** at $\rho = 998\,\text{kg/m}^3$ (20°C, incompressible), inlet velocity $V_{in} = 2\,\text{m/s}$, inlet pressure $p_{in} = 101{,}325\,\text{Pa}$ (atmospheric reference). Stage 2 (§3.6) instead models compressible **air**, since that's the regime where compressibility actually matters.

> **Any incompressible, inviscid fluid works here, not just water.** $\rho$, $V_{in}$, and $p_{in}$ never appear inside the residuals the network is trained against (§3.5) — they only rescale the network's dimensionless output back into SI units at export time. The trained network itself is fluid-agnostic; swapping the config from air to water (or any other density) requires no retraining, only re-running the SI conversion.

### 3.3 Stage 1 physics — the exact equations the PINN must satisfy

Steady, incompressible, inviscid, quasi-1D flow (area changes are gradual, so the flow is treated as one-dimensional across each cross-section):

**Conservation of mass (continuity)** — the same volumetric flow rate $Q$ passes every cross-section:

$$A(x)\,V(x) = Q = \text{const} \qquad\Longleftrightarrow\qquad \frac{d}{dx}\big[A(x)\,V(x)\big] = 0 \tag{1}$$

**Bernoulli's theorem (energy/momentum along a streamline)** — for steady incompressible inviscid flow, total head is constant:

$$p(x) + \tfrac{1}{2}\rho V(x)^2 = p_0 = \text{const} \qquad\Longleftrightarrow\qquad V\frac{dV}{dx} + \frac{1}{\rho}\frac{dp}{dx} = 0 \tag{2}$$

(The differential form on the right is what the PINN differentiates; it is the quasi-1D Euler momentum equation with constant $\rho$.)

**Boundary conditions:**

$$V(0) = V_{in}, \qquad p(0) = p_{in} \tag{3}$$

### 3.4 The closed-form analytical solution (our ground truth)

Equations (1)–(3) are exactly solvable — this is why Bernoulli is the perfect teaching case. Given $A(x)$:

$$\boxed{\;V(x) = V_{in}\,\frac{A_{in}}{A(x)}, \qquad p(x) = p_{in} + \tfrac{1}{2}\rho\left(V_{in}^2 - V(x)^2\right)\;} \tag{4}$$

Physical intuition the PINN must reproduce: where the duct **converges**, area drops → continuity forces velocity **up** → Bernoulli forces pressure **down**. The throat is the point of maximum velocity and minimum pressure. In the diverging section everything mirrors back. Every training run is validated against Eq. (4) with relative $L^2$ errors — no CFD data is ever needed.

### 3.5 Non-dimensionalization (do not skip this — it makes training work)

Neural networks train poorly when inputs/outputs span wildly different scales (here: $x \sim 1$, $p \sim 10^5$). We non-dimensionalize:

$$\tilde{x} = \frac{x}{L},\quad \tilde{A} = \frac{A}{A_{in}},\quad \tilde{D}_t = \frac{D_t}{D_{in}},\quad \tilde{V} = \frac{V}{V_{in}},\quad \tilde{p} = \frac{p - p_{in}}{\tfrac{1}{2}\rho V_{in}^2}$$

The physics becomes parameter-free and elegant:

$$\frac{d}{d\tilde{x}}\big[\tilde{A}\,\tilde{V}\big] = 0, \qquad \frac{d}{d\tilde{x}}\left(\tilde{p} + \tilde{V}^2\right) = 0, \qquad \tilde{V}(0)=1,\ \ \tilde{p}(0)=0 \tag{5}$$

and the analytical target is $\tilde{V} = 1/\tilde{A}$, $\tilde{p} = 1 - \tilde{V}^2$. Everything the network sees and produces is $O(1)$. The code converts back to SI units only at export time.

### 3.6 Stage 2 physics — compressible quasi-1D Euler (included extension)

When the throat velocity approaches the speed of sound, incompressibility breaks. The quasi-1D steady Euler system replaces Eq. (2):

$$\frac{d(\rho A V)}{dx} = 0, \qquad \rho V \frac{dV}{dx} + \frac{dp}{dx} = 0, \qquad h + \frac{V^2}{2} = h_0,\qquad p = \rho R T \tag{6}$$

with network outputs $(\tilde{\rho}, \tilde{V}, \tilde{p}, \tilde{T})$ and validation against the isentropic area–Mach relation $\frac{A}{A^*} = \frac{1}{M}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}$. We stay in the **subsonic branch** (no normal shocks): the literature (Hong et al. 2023; WHC-PINN 2025) shows vanilla PINNs converge to trivial/wrong solutions at shocks and need loss reweighting + hard pressure constraints — documented as future work, not hidden from you.

---

## 4. PINN theory from absolute zero

### 4.1 What the neural network is, mathematically

A fully connected network is a nested function. With input vector $\mathbf{z}^{(0)} = (\tilde{x}, \tilde{D}_t)$:

$$\mathbf{z}^{(k)} = \phi\!\left(W^{(k)} \mathbf{z}^{(k-1)} + \mathbf{b}^{(k)}\right),\quad k = 1..K; \qquad \hat{\mathbf{y}} = W^{(K+1)}\mathbf{z}^{(K)} + \mathbf{b}^{(K+1)}$$

- $W^{(k)}$ = **weight matrices**, $\mathbf{b}^{(k)}$ = **bias vectors** → together $\theta$, all trainable parameters.
- $\phi$ = **activation** (we use $\tanh$; smooth and infinitely differentiable — *required*, because we will differentiate the network output to build PDE residuals; ReLU's kinks would poison second derivatives).
- $\hat{\mathbf{y}} = (\hat{\tilde{V}}, \hat{\tilde{p}})$ — the predicted velocity and pressure at that point, for that throat diameter.

**Forward propagation** = evaluating this chain. By the universal approximation theorem, enough neurons can approximate any smooth function — including our nozzle solution.

### 4.2 The PINN trick: automatic differentiation of the network

Because the network is a closed-form composition of differentiable ops, we can ask PyTorch for **exact** derivatives of outputs w.r.t. inputs:

$$\frac{\partial \hat{\tilde{V}}}{\partial \tilde{x}},\quad \frac{\partial \hat{\tilde{p}}}{\partial \tilde{x}},\quad \frac{\partial \hat{\tilde{V}}}{\partial \tilde{D}_t},\ \dots$$

via `torch.autograd.grad` — machine-precision, no finite-difference error, no mesh. This is what separates a PINN from curve fitting: **the derivatives in the differential equations are computed analytically through the network's own graph.**

### 4.3 Backpropagation

The loss $\mathcal{L}(\theta)$ (Section 5) is a scalar. Backpropagation applies the chain rule through the same computational graph to get $\nabla_\theta \mathcal{L}$ — the gradient of the loss w.r.t. every weight and bias. The optimizer then steps downhill. Two optimizers, two roles (standard practice since Raissi et al. 2019):

- **Adam** (adaptive learning rates, robust to bad conditioning) does the bulk of training: `lr=1e-3`, ~20,000 epochs.
- **L-BFGS** (quasi-Newton, full-batch, uses curvature) polishes convergence to $10^{-5}$–$10^{-7}$ loss levels where Adam stalls.

### 4.4 Why the equations can replace training data

A data-driven network needs labeled examples $(x \to y)$. We have none — and need none. The governing equations are *a complete specification of the solution*: any function $(\hat V, \hat p)$ that satisfies continuity, momentum, and the boundary conditions everywhere **is** the flow. So we penalize the *violation* of the equations at thousands of random **collocation points**. The equations themselves manufacture the supervision — this is the core advantage you asked about: *physics supplies infinite, free, exact training signal with physical reality baked in.*

---

## 5. How the equations become loss functions

This is the heart of the build. Each physics statement maps to one mean-squared-error term. From Eq. (5):

**① Continuity residual** (Eq. 1, penalized at $N_f$ collocation points $\{\tilde{x}_i, \tilde{D}_{t,i}\}$):

$$r_{cont} = \frac{\partial}{\partial \tilde{x}}\big[\tilde{A}(\tilde{x};\tilde{D}_t)\,\hat{\tilde{V}}\big], \qquad \mathcal{L}_{cont} = \frac{1}{N_f}\sum_{i=1}^{N_f} r_{cont}(\tilde{x}_i, \tilde{D}_{t,i})^2$$

$\tilde{A}(\tilde x;\tilde D_t)$ is known analytically (geometry module) and differentiated together with $\hat{\tilde V}$ — the chain rule handles the product.

**② Momentum/Bernoulli residual** (Eq. 2):

$$r_{mom} = \frac{\partial}{\partial \tilde{x}}\left(\hat{\tilde{p}} + \hat{\tilde{V}}^2\right), \qquad \mathcal{L}_{mom} = \frac{1}{N_f}\sum_{i=1}^{N_f} r_{mom}(\tilde{x}_i, \tilde{D}_{t,i})^2$$

**③ Boundary conditions** (Eq. 3) — two options, both implemented:

- *Soft* (penalty): $\mathcal{L}_{bc} = \big(\hat{\tilde V}(0,\tilde D_t) - 1\big)^2 + \big(\hat{\tilde p}(0,\tilde D_t)\big)^2$ averaged over sampled $\tilde D_t$ values.
- *Hard* (architectural, Lagaris-style / Sun et al. 2020): bake the BC into the output transform so it is satisfied exactly, by construction:

$$\hat{\tilde V}(\tilde x) = 1 + \tilde{x}\,\mathcal{N}_V(\tilde x, \tilde D_t), \qquad \hat{\tilde p}(\tilde x) = \tilde{x}\,\mathcal{N}_p(\tilde x, \tilde D_t)$$

At $\tilde x = 0$ these give $\hat{\tilde V}=1$, $\hat{\tilde p}=0$ identically — the optimizer cannot violate them. Hard BCs are the default (the literature shows they train cleaner in data-free regimes).

**④ Total loss:**

$$\mathcal{L}(\theta) = \lambda_{cont}\mathcal{L}_{cont} + \lambda_{mom}\mathcal{L}_{mom} + \lambda_{bc}\mathcal{L}_{bc} \;(+\,\lambda_{data}\mathcal{L}_{data}\ \text{if any measurements exist})$$

**Loss weighting matters.** Different terms have different gradient magnitudes, and imbalanced gradients cause the optimizer to satisfy one equation while ignoring another (Wang, Teng & Perdikaris 2021, "gradient pathologies"). Two supported strategies:

- `fixed`: user-set weights (Stage 1 works with all $\lambda=1$; Hong et al. found $\lambda_{mom}\approx 20$ necessary in shock cases).
- `annealing` (default): Wang et al.'s learning-rate annealing — periodically set $\hat\lambda_i = \max|\nabla_\theta \mathcal{L}_{r}| \,/\, \overline{|\nabla_\theta \mathcal{L}_i|}$ with a 0.9 moving average. This self-balances the terms.

**⑤ Where each piece lives in code:** ① and ② → `src/losses.py` (calling `src/physics.py` for residuals, which calls `torch.autograd.grad`); ③ → `src/networks.py` (hard transform) and `src/losses.py` (soft penalty); ④ → `src/train.py`.

---

## 6. Network architecture design (and why)

| Choice | Value | Why |
|---|---|---|
| Inputs | $(\tilde{x}, \tilde{D}_t)$ — 2 | position + design parameter = parametric surrogate |
| Outputs | $(\hat{\tilde{V}}, \hat{\tilde{p}})$ — 2 (Stage 1); $(\hat{\tilde\rho}, \hat{\tilde V}, \hat{\tilde p}, \hat{\tilde T})$ — 4 (Stage 2) | primitive flow variables |
| Hidden layers | 6 × 64 neurons | comfortably fits smooth 1D solutions (literature uses 3×30 for nozzle Euler; we overprovision slightly for the parametric dimension) |
| Activation | `tanh` | $C^\infty$ smooth → clean high-order autodiff derivatives (HFM used `sin`; `sin` (SIREN) is available as a flag) |
| Initialization | Xavier/Glorot uniform | keeps activation variances stable at depth, avoids early saturation of tanh |
| Output transform | hard-BC ansatz (Section 5③) | exact BC satisfaction, faster convergence (Sun et al. 2020) |
| Positivity guards | $\tilde A > 0$ by construction; Stage 2: $\tilde\rho, \tilde p = \text{softplus}(\cdot)$ | prevents unphysical states during early training |
| Precision | float64 for physics paths | PDE residuals amplify round-off; float64 costs little at this size |

---

## 7. The parametric surrogate: throat diameter as an input

This is the *surrogate* part. Instead of training one model per nozzle, $\tilde D_t$ is simply concatenated to the network input. Training samples pairs $(\tilde{x}_i, \tilde{D}_{t,i})$ by **Latin Hypercube Sampling** over $[0,1] \times [0.4, 0.9]$ (i.e. $D_t \in [0.2, 0.45]$ m), so the network sees the whole design space at once.

**What physically changes with $D_t$ (and the surrogate must capture):** smaller throat → smaller $\tilde A$ at $x_t$ → by continuity higher throat velocity $\tilde V_t = 1/\tilde A_t = (D_{in}/D_t)^2$ → by Bernoulli deeper pressure drop $\tilde p_t = 1 - \tilde V_t^2$. At $D_t = 0.2$ m: $\tilde V_t = 6.25$, $\tilde p_t = -38$ (≈ 2.3 kPa drop in SI). The surrogate reproduces this entire family instantly, and gives $\partial \hat p / \partial D_t$ by autodiff for design studies.

**Why this matters:** this is precisely the Sun et al. (2020) paradigm — geometry as input, PDE residuals as the only teacher. One training run replaces a whole CFD campaign.

---

## 8. Repository layout — every file explained

```
PINNs---Bernoulli/
├── README.md                  ← you are here (theory + how-to)
├── CLAUDE.md                  ← complete build instructions for Claude Code
├── requirements.txt           ← all dependencies, pinned minimums
├── configs/
│   └── default.yaml           ← every hyperparameter in one place (physics, network, training, export)
├── src/
│   ├── geometry.py            ← D(x; Dt), A(x; Dt), dA/dx — Eq. geometry law, nondim + SI
│   ├── physics.py             ← residuals r_cont, r_mom (Stage 1) & quasi-1D Euler residuals (Stage 2)
│   │                            via torch.autograd.grad — THE equations-to-code file
│   ├── analytical.py          ← Eq. (4) + compressible isentropic reference — ground truth for validation
│   ├── networks.py            ← MLP builder, tanh/sin activations, Xavier init, hard-BC output transforms
│   ├── losses.py              ← L_cont, L_mom, L_bc(soft), weighting strategies (fixed / LR-annealing)
│   ├── sampling.py            ← Latin Hypercube collocation sampler over (x, Dt) space; boundary samplers
│   ├── train.py               ← training loop: Adam → L-BFGS, weight annealing, logging, checkpoints
│   ├── evaluate.py            ← relative L2 vs analytical, residual maps, error metrics
│   ├── export.py              ← CSV, ParaView (.vtk/.vtu via pyvista, 2D axisymmetric reconstruction),
│   │                            MATLAB (.mat via scipy.io.savemat)
│   └── visualize.py           ← matplotlib plots: V(x), p(x) vs analytical, parametric sweeps, loss curves
├── scripts/
│   ├── run_stage1_bernoulli.py    ← one command: train Stage 1 surrogate end-to-end
│   ├── run_stage2_compressible.py ← one command: train Stage 2 (quasi-1D Euler)
│   └── run_parametric_sweep.py    ← load trained surrogate, sweep Dt, export all post-processing files
├── postprocessing/
│   ├── paraview/README.md     ← how to open .vtu, apply filters (Warp, Calculator, StreamTracer)
│   └── matlab/load_and_plot.m ← loads .mat, reproduces all key figures in MATLAB
├── tests/
│   ├── test_analytical.py     ← sanity: analytical solution satisfies residuals to machine precision
│   ├── test_gradients.py      ← autodiff vs finite-difference agreement
│   └── test_losses.py         ← each loss term = 0 when fed the exact solution
└── results/                   ← created at runtime: checkpoints, figures, .vtk/.vtu, .mat, logs
```

---

## 9. Training protocol

1. **Sample** $N_f = 512 \times 8$ collocation points (512 in $\tilde x$ × 8 LHS values of $\tilde D_t$, resampled every 1000 epochs — resampling prevents overfitting to fixed points, per the stiff-PDE PINN benchmark study).
2. **Adam**, `lr = 1e-3` (exponential decay ×0.95 / 2000 epochs), 20,000 epochs, full batch.
3. **Loss-weight annealing** every 500 epochs (Wang et al. algorithm, moving average 0.9).
4. **L-BFGS** (strong-Wolfe, `max_iter = 5,000`) polish.
5. **Validate** every 500 epochs against the analytical solution on 10,000 fresh points; track relative $L^2$ for $\tilde V$ and $\tilde p$.
6. **Acceptance gates** (the run is only "done" when): rel-$L^2(\tilde V) < 10^{-3}$, rel-$L^2(\tilde p) < 10^{-2}$ (pressure is more sensitive — it scales with $\tilde V^2$), Bernoulli constant variation < 1% across the domain, for **all** sampled $\tilde D_t$, including values never seen in training (interpolation check).

Measured wall time on an RTX 3060: ~12 minutes for the 20,000-epoch Adam phase, ~5–6 minutes for the L-BFGS polish — call it 15–18 minutes end to end, including exports and figures. (The network is tiny — 21,122 parameters — so a GPU's advantage here is real but modest; expect low tens of minutes on CPU too.)

### 9.1 A bug found while verifying on GPU

`evaluate.py`'s `validate_held_out`/`full_report`, `physics.py`'s `total_head`, and `export.py`'s `predict_si` all built fresh input tensors with plain `torch.as_tensor(...)` (which defaults to CPU) and called the model directly. That's silent on a CPU-only machine but fatal — `RuntimeError: Expected all tensors to be on the same device` — the moment the model lives on a CUDA device, since PyTorch refuses to multiply a CPU tensor against CUDA weights. Fixed by moving inputs to `next(model.parameters()).device` right before each model call and moving results back to CPU right after, at each of the three call sites. Training/loss math is unchanged — this only touches how validation and export reach the model.

---

## 10. Validation against the analytical solution

Three independent checks, all automated:

- **Pointwise fields:** $\hat{\tilde V}(\tilde x)$ vs $1/\tilde A(\tilde x)$, $\hat{\tilde p}(\tilde x)$ vs $1 - \tilde V^2$ — overlay plots for several $D_t$.
- **Invariant check:** $\hat{\tilde p} + \hat{\tilde V}^2 \equiv 1$ (total head constancy) plotted as a map over $(\tilde x, \tilde D_t)$.
- **Generalization:** hold out $D_t = 0.225, 0.275, 0.325, 0.375, 0.425$ m from training; evaluate only on them. A surrogate that memorizes training geometries fails here; a physics-trained one passes, because the *equations* hold everywhere.

**Actual result of the reference training run** (seed 1234, RTX 3060, config as committed): rel-$L^2(\tilde V) = 1.03\times10^{-4}$, rel-$L^2(\tilde p) = 2.33\times10^{-4}$, max Bernoulli-invariant error $= 3.4\times10^{-4}$ (0.034%) — all three gates pass with roughly a 10× margin, on throat diameters never seen during training.

---

## 11. Post-processing: ParaView and MATLAB

### ParaView (`.vtk` / `.vtu`)

The 1D surrogate field is reconstructed into a **2D axisymmetric field** for real visualization: for each station $x$, the duct spans $y \in [-D(x)/2, D(x)/2]$, with plug-flow $u(x,y) = V(x)$, $p(x,y) = p(x)$. `src/export.py` builds a triangle mesh with `pyvista` and writes `results/paraview/nozzle_field_<Dt>.vtu`. Open in ParaView → *Apply* → color by `p` or `V`; suggested filters (documented in `postprocessing/paraview/README.md`): **Calculator** (Mach proxy, dynamic pressure), **Warp by Vector**, **Stream Tracer**, and a parametric animation over the exported $D_t$ series.

### MATLAB (`.mat`)

`src/export.py` writes `results/matlab/nozzle_surrogate.mat` via `scipy.io.savemat` containing grids `X`, `DT`, fields `V`, `P`, geometry `D`, and the analytical references. `postprocessing/matlab/load_and_plot.m` loads it and reproduces every key figure (field maps, throat sweep curves, error plots) — no Python needed at analysis time.

---

## 12. How to run everything

```bash
git clone https://github.com/harsh147-github/PINNs---Bernoulli.git
cd PINNs---Bernoulli
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt                        # torch, numpy, scipy, matplotlib, pyvista, pyyaml, pandas, tqdm, imageio

python scripts/run_stage1_bernoulli.py --config configs/default.yaml   # train the Bernoulli surrogate
python scripts/run_stage2_compressible.py --config configs/default.yaml # compressible extension
python scripts/run_parametric_sweep.py  --checkpoint results/checkpoints/stage1_final.pt \
        --dt-min 0.20 --dt-max 0.45 --n 25                              # sweep throat, export ParaView+MATLAB
pytest tests/ -q                                       # physics/unit sanity checks
```

Outputs land in `results/`: figures (`figures/`), ParaView files (`paraview/`), MATLAB file (`matlab/`), checkpoints (`checkpoints/`), logs (`logs/`).

---

## 13. Literature survey: everything that has been done before us

Annotated map of the field this repo stands on — every claim in this README traces here.

**Foundations**
- **Raissi, Perdikaris, Karniadakis (2019), *Physics-informed neural networks*, J. Comput. Phys. 378:686–707** — the founding paper: PDE residuals in the loss, autodiff, Adam→L-BFGS. Also the 2017 two-part arXiv preprints. Code: `github.com/maziarraissi/PINNs` (TensorFlow v1).
- **Lagaris, Likas, Fotiadis (1998), IEEE TNN** — 20 years earlier: ANNs as trial solutions for ODEs/PDEs with hard-coded BCs. The intellectual ancestor of our hard-BC ansatz.
- **Karniadakis et al. (2021), *Physics-informed machine learning*, Nature Reviews Physics 3:422–440** — the field-level review: why physics-informed learning beats black-box ML when data is scarce; capabilities and honest limitations.

**PINNs for fluid mechanics / Navier–Stokes**
- **Raissi, Yazdani, Karniadakis (2020), *Hidden Fluid Mechanics*, Science 367:1026–1030** — Navier–Stokes-informed network (6 outputs, 10×50/layer per output, `sin` activation) recovers velocity & pressure from passive-scalar data alone; geometry/BC-agnostic; differentiable surrogate used for wall shear stress & forces.
- **Jin, Cai, Li, Karniadakis (2021), *NSFnets*, J. Comput. Phys. 426:109951** — systematic PINN formulation of incompressible Navier–Stokes (velocity–pressure and vorticity–velocity forms), laminar→turbulent limits.
- **Sun, Gao, Pan, Wang (2020), *Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data*, CMAME 361:112732** — **the single most important reference for this repo**: parametric geometry as network input, hard BC enforcement, *zero* CFD data, vascular-flow surrogates + uncertainty propagation.

**Nozzle flows & compressible PINNs (our exact problem family)**
- **Hong et al. (2023), *Continuous and discontinuous compressible flows in a converging–diverging channel solved by PINNs without data*, arXiv:2306.11749 / Sci. Rep.** — quasi-1D Euler CD nozzle; vanilla PINNs collapse to trivial solutions at shocks; fixed by ~20× momentum-loss weight + hard pressure BCs. Their nozzle law and Pb regimes inform our Stage 2.
- **WHC-PINN (2025), *PINN with weighted loss and hard constraints for hyperbolic conservation laws*, Sci. Rep.** — gradient-based weighting + global conservation term + pressure-only hard constraints; captures Riemann problems and nozzle shocks with a unified solver.
- **Anderson, *Computational Fluid Dynamics: The Basics with Applications*, Ch. 7** — the classical quasi-1D nozzle CFD tutorial (MacCormack scheme) — what we replace with a PINN.

**Training dynamics & failure modes**
- **Wang, Teng, Perdikaris (2021), *Understanding and mitigating gradient pathologies in PINNs*, SIAM SISC** — loss terms compete; learning-rate annealing algorithm (implemented here) + improved architectures.
- **Wang, Yu, Perdikaris (2022), NTK-based weighting, JCP** — weights from Neural Tangent Kernel eigenvalues.
- **Krishnapriyan et al. (2021), *Characterizing possible failure modes in PINNs*, NeurIPS** — why PINNs fail on stiff/multiscale problems and how curriculum/regularization help.
- **Stiff-PDE benchmark review (2023), Springer Comput. Mech.** — DeepXDE vs NVIDIA Modulus baselines, Fourier/modified-Fourier/DGM architectures, SDF weights, importance sampling; resampling beats fixed collocation.

**Parametric PINNs for design**
- ***Investigating the Effects of Labeled Data on Parameterized PINNs for Surrogate Modeling* (2024), MDPI Fluids 9(12):296** — parametric PINN design optimization (drag reduction over a forward-facing step); small amounts of labeled data + low data-loss weight improve parametric surrogates; gradients w.r.t. design parameters from autodiff.

**Reference codebases worth reading**
- `maziarraissi/PINNs` — the original (TF1). `rezaakb/pinns-torch` — PyTorch re-implementation, up to 9× faster via CUDA graphs (NeurIPS DLDE 2023). `chen-yingfa/pinn-torch` — clean minimal PyTorch Navier–Stokes PINN. `pjuangph/PINN-Torch` — tutorial-style, includes Euler/shock-tube limitations demo. `parfenyev/2d-turb-PINN` — PINN inference from PIV data. **Frameworks:** DeepXDE (multi-backend, beginner-friendly), NVIDIA Modulus/PhysicsNeMo (industrial scale).

---

## 14. Roadmap

- [x] Stage 1 — parametric incompressible Bernoulli surrogate (trained, all acceptance gates pass — §9.1, §10)
- [x] Stage 2 — quasi-1D compressible Euler code (network, residuals, isentropic validation, training loop, run script) — smoke-tested, **full 20k-epoch training run + gate confirmation still pending**
- [ ] Stage 2 — ParaView `.vtu` / MATLAB `.mat` export parity with Stage 1 (currently CSV + figures only)
- [ ] Stage 2b — shock capturing (weighted loss + hard pressure constraints, per Hong et al. / WHC-PINN)
- [ ] Stage 0 — 2D incompressible Navier–Stokes nozzle (NSFnets VP formulation, `sin` activations)
- [ ] Data assimilation mode — inject sparse pressure-tap measurements (`λ_data` term already stubbed)
- [ ] Gradient-based throat optimization: maximize thrust/recovery using $\partial p/\partial D_t$ from the surrogate
- [ ] Uncertainty quantification over $\rho$, $V_{in}$ via the parametric surrogate
