# Parametric PINN solver for a converging–diverging nozzle

A **data-free physics-informed neural network (PINN)** that solves the
steady, incompressible, laminar, axisymmetric **Navier–Stokes equations** for
water flowing through a converging–diverging nozzle, for **every throat
diameter in a design range at once**. It is benchmarked against **OpenFOAM**
(`simpleFoam`) on the identical problem.

```
          inlet      converging   throat      diverging (diffuser)   outlet
        ________                                               ______________
                \___                                     ___/
  flow  ======>     \____________________________________/        ======>
                     <---------- D_t (design input) ----------->
        ________ ___/                                    \___  ______________
                                                              \
```

The throat diameter `D_t` is an **input to the network**. One training run
produces a surrogate for every nozzle in `d_t = D_t / D_in ∈ [0.40, 1.00]`,
including the sketch's baseline `d1` (`d_t = 0.70`) and every wider and
narrower throat. After training, a new geometry costs one forward pass of a
small network (milliseconds) instead of a fresh CFD run. This is the
**amortisation**: you pay for training once and query designs almost for free.

> **Results summary** ([§9](#9-results)):
>
> - **Accuracy.** One physics-only PINN covers every throat ratio
>   0.40–1.00. Against OpenFOAM it agrees to **0.40 % (velocity) and 1.1 %
>   (pressure)** mean relative L2 error, worst case 1.3 % / 3.9 % at the
>   narrowest, separated nozzle.
> - **Dynamic pressure.** Throat dynamic pressure rises **9×** when the
>   throat closes from d1 to 0.4, and falls to **0.27×** when it opens to
>   1.0.
> - **Cost.** Training took 3.6 h on a 4-core CPU. After that each new design
>   costs **0.26 s**, versus **34.6 s** for an OpenFOAM run (13.7 s with 4
>   in parallel).
> - **Amortisation.** Break-even is at ~376 designs (vs serial OpenFOAM) or
>   ~962 (vs 4-parallel OpenFOAM). At 10,000 designs the PINN is 22× / 8.8×
>   cheaper.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Governing equations — what we solve and why](#2-governing-equations--what-we-solve-and-why)
3. [Problem set-up: geometry, fluid, non-dimensionalisation, boundary conditions](#3-problem-set-up)
4. [The network and the hard-constraint ansatz](#4-the-network-and-the-hard-constraint-ansatz)
5. [Training (physics-only)](#5-training-physics-only)
6. [Inference and amortisation](#6-inference-and-amortisation)
7. [OpenFOAM reference solver](#7-openfoam-reference-solver)
8. [Post-processing: plots, animation, data](#8-post-processing)
9. [Results](#9-results)
10. [Code walkthrough, file by file](#10-code-walkthrough-file-by-file)
11. [Limitations and next steps](#11-limitations-and-next-steps)
12. [References](#12-references)

---

## 1. Quick start

```bash
pip install -r requirements.txt          # torch, numpy, scipy, matplotlib, pyyaml, pillow, pytest
sudo apt-get install openfoam            # Ubuntu package (OpenFOAM v1912) — reference solver only

pytest tests -q                          # 7 unit tests
python scripts/run_openfoam.py           # 7 OpenFOAM reference cases  (~4 min)
python scripts/train.py                  # train the parametric PINN    (~3.6 h on a 4-core CPU, minutes on a GPU)
python scripts/infer_and_postprocess.py  # 121-design sweep, comparisons, figures, GIF, VTK (~2 min)
# or everything at once:
bash scripts/run_all.sh
```

The trained model is committed as `results/pinn/model_final.pt`. To reuse it
without retraining, skip `train.py` and run `infer_and_postprocess.py`.

Quick smoke test of the training loop (about 1 minute, inaccurate on purpose):

```bash
python scripts/train.py --adam-epochs 100 --lbfgs-iters 0 --out results_smoke --no-monitor
```

Everything runs on a CPU. No GPU is needed; on a GPU-equipped machine the same
code is simply faster.

---

## 2. Governing equations — what we solve and why

### 2.1 Choice of model

The pipe is round and the inflow is straight, so the flow is
**axisymmetric**: it depends only on the axial coordinate `x` and the radius
`r`, with no swirl. That reduces the 3-D problem to a 2-D (x, r) half-plane
**without any approximation**.

Water at room temperature is incompressible at these speeds (Mach ≈ 10⁻⁵). The
equations that govern it are the **incompressible Navier–Stokes equations**.
Bernoulli's equation is the inviscid, irrotational special case of these and
is only accurate in the core flow. Navier–Stokes also captures what Bernoulli
cannot:
- wall friction (boundary layers);
- the pressure *loss* across the nozzle;
- the jet that leaves the throat;
- flow separation and recirculation in the diffuser.

Navier–Stokes is therefore the "best" model here. It is the same model
OpenFOAM's `simpleFoam` solves, so the comparison is like-for-like.

**Reynolds number.** We solve at **Re = U D_in / ν = 50**, i.e. laminar flow.
Physically this is water at 20 °C in a 10 mm pipe at 5 mm/s (see
`physics:` in `configs/nozzle.yaml`). The narrowest throat (`d_t = 0.4`)
runs at a throat Reynolds number of 125 and produces a strong jet with a
separated diffuser. §7.2 explains why Re = 50 rather than 100: the domain
length needed for a well-posed outlet.

Why not the 0.5 m / 2 m s⁻¹ pipe of a full-scale water main? That flow has
Re ≈ 10⁶ and is turbulent. Solving it means RANS turbulence modelling (k-ω SST
and similar). Vanilla PINNs are known to fail at such Reynolds numbers, and
a RANS-PINN is a research project of its own (see §11).

Laminar Re = 50 keeps every physical effect the study is about:
- acceleration in the throat;
- the static-to-dynamic pressure exchange;
- viscous losses;
- jet formation;
- separation in the diffuser at narrow throats.

It also stays inside what a PINN can solve accurately, and OpenFOAM solves the
same equations exactly with its laminar model.

### 2.2 The equations (non-dimensional, cylindrical, steady, no swirl)

With velocity `(u, v)` in the `(x, r)` directions and kinematic pressure `p`:

```
continuity   ∂u/∂x + ∂v/∂r + v/r                                                         = 0   (1)
x-momentum   u ∂u/∂x + v ∂u/∂r + ∂p/∂x − (1/Re)(∂²u/∂x² + ∂²u/∂r² + (1/r)∂u/∂r)        = 0   (2)
r-momentum   u ∂v/∂x + v ∂v/∂r + ∂p/∂r − (1/Re)(∂²v/∂x² + ∂²v/∂r² + (1/r)∂v/∂r − v/r²) = 0   (3)
```

The `v/r`, `(1/r)∂/∂r` and `v/r²` terms are what the cylindrical geometry adds
to the Cartesian equations. Equations (2)–(3) are implemented line-for-line in
`nozzle_pinn/physics.py`.

Equation (1) is satisfied **exactly by construction**. The velocity is
derived from a Stokes stream function `ψ`:

```
u = (1/r) ∂ψ/∂r,     v = −(1/r) ∂ψ/∂x
```

Substituting gives `∂u/∂x + ∂v/∂r + v/r ≡ 0` for any `ψ`. See §4.2 and §7.3
for why this matters here.

### 2.3 Derived quantities we report

| symbol | meaning | formula (non-dim.) |
|---|---|---|
| `p` | static pressure | network output |
| `q` | **dynamic pressure** | `½ (u² + v²)` |
| `p0` | total pressure | `p + q` |
| `ψ` | Stokes stream function | `ψ = ∫₀ʳ u r' dr'`, shown normalised: 0 on the axis, 1 on the wall |

In axisymmetric incompressible flow, **lines of constant ψ are exactly the
streamlines**. Every streamline plot here is drawn as iso-ψ contours, at 10
equal flux increments. Lines crowding together therefore means fast flow, and
any region with ψ > 1 is recirculating fluid.

---

## 3. Problem set-up

### 3.1 Geometry (`nozzle_pinn/geometry.py`)

Lengths are in units of the inlet diameter `D_in`. The pipe is built from five
segments, 12.5 inlet diameters in total:

| segment | length | radius |
|---|---|---|
| inlet pipe | 1.0 | 0.5 |
| converging | 1.5 | 0.5 → d_t/2 |
| throat | 1.5 | d_t/2 |
| diverging (diffuser) | 2.5 | d_t/2 → 0.5 |
| outlet pipe | 6.0 | 0.5 |

The wall is a single analytic formula:

```
R(x; d_t) = ½ [1 + (d_t − 1) Φ(x)],   Φ(x) = S((x − x₁)/L_c) − S((x − x₃)/L_d)
S(t) = 6t⁵ − 15t⁴ + 10t³   (t clipped to [0, 1])          ← "smootherstep"
```

`S` has zero first **and second** derivatives at both ends, so `R`, `R'` and
`R''` are continuous. This matters because the PINN residual differentiates
the wall shape twice (through `η = r/R(x)`). A wall with a slope kink would
inject spurious delta functions into the second derivatives. OpenFOAM meshes
the same curve (sampled at 240 points along a `polyLine`).

### 3.2 Non-dimensionalisation

| quantity | scale |
|---|---|
| length | `D_in` |
| velocity | mean inlet velocity `U` |
| pressure | `ρU²` (kinematic pressure, same as OpenFOAM's `p` when U = 1) |
| viscosity | enters only as `1/Re` |

Non-dimensional variables keep every input and output O(1). That is what
neural networks and optimisers need. It also makes the PINN and OpenFOAM
literally solve the same numbers: the OpenFOAM case uses `D = 1`, `U = 1`,
`ν = 1/Re`.

To convert to SI: `p[Pa] = p̂ · ρU²`, which is 0.0998 Pa per unit for the
water case in the config (ρ = 998.2 kg/m³, U = 5 mm/s → ρU² = 0.0250 Pa).
The sweep CSV contains SI columns.

### 3.3 Boundary conditions (identical in PINN and OpenFOAM)

| boundary | condition | PINN | OpenFOAM |
|---|---|---|---|
| inlet x = 0 | fully developed Poiseuille, `u = 2(1 − (2r)²)`, v = 0 (mean = 1) | exact (ansatz) | `fixedValue` (nonuniform list) |
| wall r = R(x) | no-slip u = v = 0 | exact (ansatz) | `noSlip` |
| axis r = 0 | symmetry: v = 0, ∂u/∂r = ∂p/∂r = 0 | exact (ansatz, input η²) | wedge axis |
| outlet x = L | p = 0, zero velocity gradient | p exact; ∂u/∂x = 0, v = 0 as soft loss | `fixedValue 0` / `zeroGradient` |
| mass flux | `∫ 2πr u dr = π/4` at every section | **exact** (stream function, ψ_wall = 1/8) | conservative finite volume |

---

## 4. The network and the hard-constraint ansatz

### 4.1 Inputs and outputs

The surrogate is a map `(x, r, d_t) ↦ (u, v, p)`. Internally the network
outputs two scalars `(N_ψ, N_p)`, from which the stream function and the
pressure are built (§4.2). It sees **normalised** inputs
(`NozzlePINN.normalise`):

```
x̂ = 2x/L − 1,    ŝ = 2η² − 1  with η = r / R(x; d_t),    d̂ = 2(d_t − 0.4)/0.6 − 1
```

Three choices matter here:

- **`η = r/R(x)` maps every nozzle onto the same unit strip.** Points at the
  same relative radius of different nozzles look alike to the network, which
  is what lets one set of weights generalise across geometries.
- **`η²` instead of `η` makes the outputs even functions of `r`.** That is
  exactly the smoothness an axisymmetric field must have on the axis, so the
  symmetry condition is satisfied for free.
- **`d̂` is the design parameter.** Feeding it as an input is what makes the
  network a *parametric* solver.

### 4.2 Hard constraints: exact mass conservation and boundary conditions

The raw network output `N = (N_ψ, N_p)` is never used directly. It is wrapped
in an **ansatz** (`NozzlePINN.forward`) built on the Stokes stream function.
With `s = η²` and `ξ = x/L`:

```
ψ(x, r)   = s · H(x, s)
H(x, s)   = ⅛ [ (2 − s)  +  ξ² (1 − s)² · N_ψ ]
u = (1/r) ∂ψ/∂r = 2 F_s / R²                      F_s = ∂ψ/∂s = H + s ∂H/∂s
v = −(1/r) ∂ψ/∂x = −(r/R²) (∂H/∂x − 2 R' F_s / R)
p = p_1D(x; d_t) + (1 − ξ) · d_t⁻⁴ · N_p
p_1D = (1 − U_m²) + (32/Re) ∫ₓᴸ U_m / D² dx'      U_m = (0.5/R)²  (quasi-1-D mean velocity)
```

Read it factor by factor:

- **Continuity holds identically.** `u` and `v` are derivatives of the same
  `ψ`, so equation (1) is satisfied for any weights. There is no continuity
  loss to balance.
- **The flow rate is exact.** `ψ = ⅛` on the wall (`s = 1`) for every `x` and
  every nozzle. The volume flux `2π ψ_wall = π/4` therefore equals the inlet
  flux at *every* cross-section. A unit test checks this to 1e-6 with random
  weights.
- **No-slip.** `∂ψ/∂s = 0` on the wall (the `(1 − s)²` factor) gives `u = 0`.
  `ψ = const` along the wall gives `v = 0`.
- **Axis.** `ψ ∝ s ∝ r²` near the axis, which is the regularity condition of
  axisymmetric flow. `v ∝ r` vanishes on the axis. The formulas above never
  divide by `r`, so they are well-defined on the centre-line itself.
- **Inlet.** The correction is multiplied by `ξ²`. Its value *and* its
  x-derivative vanish at the inlet, so `ψ = (2s − s²)/8`. That gives
  `u = 2(1 − η²)` (Poiseuille, mean 1) and `v = 0`.
- **Outlet.** `(1 − ξ)` gives `p = 0` there.
- **Engineering prior.** With `N = 0` the field is a locally fully-developed
  pipe flow, `u = 2U_m(1 − η²)` and `v = ηR'u` (streamlines follow
  `η = const`).
  - The pressure prior `p_1D` is textbook pipe hydraulics: 1-D Bernoulli on
    the mean velocity plus Hagen–Poiseuille friction.
  - The prior is the *exact* solution for the straight pipe (`d_t = 1`).
  - The friction integral uses 64-point Gauss–Legendre quadrature, so it stays
    differentiable in torch (`p_prior`).
  - The network learns only the correction to this prior: entrance effects,
    the jet, the separation and the extra losses.
- **`d_t⁻⁴` scaling.** The throat dynamic pressure scales as `d_t⁻⁴`, so this
  keeps the pressure correction O(1) for every nozzle.

`∂H/∂x` (at fixed `s`) and `∂H/∂s` (at fixed `x`) are obtained with one
autograd call. `forward` passes separate tensor handles `xh = x·1`,
`sh = s·1` into the network, so the gradients with respect to them are true
*partial* derivatives.

With these constraints the optimiser spends no effort on boundary conditions
or mass conservation, and no penalty weights need tuning.

### 4.3 Architecture: the "modified MLP"

We use a 6-hidden-layer, 64-unit tanh network. It is the **modified MLP** of
Wang, Teng & Perdikaris (2021) (`ModifiedMLP`): two input encoders
`U = tanh(W_u z)` and `V = tanh(W_v z)` are blended into every layer as
`H ← (1 − Z)·U + Z·V`.

This architecture was proposed to fix the stiff, unbalanced gradients of PINNs
on Navier–Stokes problems. In our single-design test at d_t = 0.7 it reached
**0.9 % velocity error after 750 steps**. A plain MLP still had **10–12 %
error after 3,000 steps**. The **output layer is zero-initialised**, so
training starts *exactly* at the engineering prior and improves from there.
On the final setup the prior is ≈11 % (u) / 10 % (p) from OpenFOAM, averaged
over the seven reference designs.

---

## 5. Training (physics-only)

### 5.1 Collocation points (`nozzle_pinn/sampling.py`)

Each training point is a triple `(x, r, d_t)`: a location inside a randomly
drawn nozzle. One batch therefore samples thousands of *different* nozzles.

- **Design parameter.** `d_t = 0.4 + 0.6·U^1.3`, so narrow throats (the hard
  cases) are slightly over-sampled.
- **Axial.** 30 % of the points are concentrated in the
  converge–throat–diffuser region.
- **Radial.** 30 % are clustered in the wall layer `0.75 < η < 1`, where shear
  is largest. The rest are uniform in `η²`, i.e. in cross-section area, with a
  small offset so `r > 0` (the momentum equations contain `1/r`).
- **Residual-based adaptive sampling (RAD, Wu et al. 2023).** Every 500
  epochs a fresh set of 3,000 interior + 400 outlet points is drawn.
  - Half of the interior points are chosen *from a 4× larger candidate pool*
    with probability `∝ e/mean(e) + 1`, where `e` is the current squared PDE
    residual.
  - Points therefore move to wherever the network currently violates the
    physics most, typically the throat jet and the separated diffuser of
    narrow nozzles.
  - The `+1` keeps every region covered.

### 5.2 Loss (`nozzle_pinn/train.py`, `nozzle_pinn/physics.py`)

```
L = ⟨m̂ₓ²⟩ + ⟨m̂ᵣ²⟩ + ⟨(∂u/∂x)²⟩_outlet + ⟨v²⟩_outlet          (+ ⟨ĉ²⟩ ≡ 0, logged only)
```

`m̂ₓ, m̂ᵣ` are equations (2)–(3) divided by the local scale
`U_m²/R + U_m/(Re R²)`. Without this normalisation the d_t = 0.4 throat,
whose residuals are about 100× larger, would drown out the other designs.
The continuity residual is still computed and logged as a check; it sits at
round-off (~1e-32).

Every derivative is computed by **automatic differentiation**
(`torch.autograd.grad(..., create_graph=True)`). Velocities are first
derivatives of `ψ`, so the momentum equations need derivatives of `ψ` up to
third order. Derivatives with respect to `x` and `r` flow through the network
*and* through the geometry `η = r/R(x)`, so they are exact derivatives in
physical coordinates. There is no mesh and no finite differences.

**No CFD data appears in the loss.** OpenFOAM results are used only by the
optional `monitor` callback. It prints the current error every 500 epochs and
never contributes a gradient.

### 5.3 Optimisation

| stage | settings | purpose |
|---|---|---|
| Adam | 9,000 epochs, lr 1e-3 × 0.8 every 1,500 | robust global descent |
| L-BFGS | up to 1,500 iterations (1,642 function evaluations), 4,000 fixed residual-adapted points, history 50, strong-Wolfe line search | fast local convergence to high accuracy |

Other settings:
- float64 throughout. The `1/r` terms near the axis cancel, and third
  derivatives amplify round-off; both are catastrophic in float32.
- Seed 1234 for everything.
- 4 CPU threads.
- Checkpoints every 5,000 epochs.
- Wall-clock time is recorded per stage in `results/pinn/history.json`.
- `checkpoints/adam_final.pt` is saved before L-BFGS starts.

**Memory and resuming.** Third-derivative autograd graphs are large: about
5.5 GB at 3,000–4,000 points. The first full run was killed (out of memory)
at the start of L-BFGS, because the adaptive-sampling candidate pool was
evaluated in one batch. Pool residuals are now evaluated in 2,000-point
chunks. The run was continued from its epoch-5,000 checkpoint:

```bash
python scripts/train.py --resume-from results/pinn/adam_005000_part1.pt \
    --start-epoch 5000 --prev-log results/pinn/train_log_part1.txt
```

`--prev-log` restores the loss/monitor history and the elapsed wall-clock
time, so the reported training time covers the whole run. The learning-rate
schedule is fast-forwarded to epoch 5,000; only Adam's moment estimates
restart.

---

## 6. Inference and amortisation

`nozzle_pinn/inference.py::field_grid(model, d_t, nx, neta)` evaluates the
trained network on a structured `(x, η)` grid of any nozzle: 501 × 61 =
30,561 points by default. It returns `u, v, p, |U|, q, p0, ψ`.

`ψ` is integrated across each cross-section with the trapezoidal rule. This
needs `R² ∫₀^η u η' dη'` because `dr = R dη`.

`scripts/infer_and_postprocess.py` sweeps **121 throat diameters** and times
every inference. It then compares the timings against the OpenFOAM wall-clock
times to compute the break-even number of designs:

```
break-even N* = (PINN training time) / (OpenFOAM time per design − PINN inference time per design)
```

---

## 7. OpenFOAM reference solver

`nozzle_pinn/openfoam.py` writes, runs and reads a complete case per design.

- **Mesh (`blockMeshDict`).** One hex block whose axis edge is collapsed: the
  standard 5° **wedge** for axisymmetric flow. The wall edges follow
  `R(x; d_t)` via `polyLine`. There are 625 × 50 = 31,250 cells (50 per
  inlet diameter axially), graded 4:1 toward the wall.
- **Solver.** `simpleFoam` (steady SIMPLEC), `simulationType laminar`,
  `ν = 1/Re`. Convection is second-order `linearUpwind`; Laplacians are
  second-order. The run stops at a pressure residual of 1e-8.
- **Inlet.** The Poiseuille profile is written as a nonuniform `fixedValue`
  list. Inlet face centres are computed from the mesh in NumPy, because the
  packaged v1912's `postProcess -func` needs dynamic code compilation, which
  is unavailable.
- **Reading.** `read_mesh` parses `points/faces/owner/neighbour/boundary`
  and computes centres. `read_solution` parses the final `U` and `p` and
  projects `U` onto (axial, radial) components.

### 7.1 Validation of the reference itself

- **Exact solution.** For the straight pipe (`d_t = 1`), OpenFOAM reproduces
  the exact Hagen–Poiseuille solution to **0.18 % (u)** and **0.39 % (p)**
  (relative L2 over all cells).
- **Mesh study.** We re-ran d_t = 0.4, 0.7 and 1.0 on a 4× finer mesh
  (1250 × 100 = 125,000 cells, `run_openfoam.py --nx 1250 --nr 100 --tag _fine`).
  - Inlet pressure changed by **≤ 0.018 %**.
  - Centre-line velocity changed by **≤ 0.11 %** (relative L2).

The reference is therefore mesh-converged to well below the PINN error level.
Each 31k-cell case takes about 30–37 s of wall-clock time (meshing + about
300 SIMPLE iterations on one core).

### 7.2 Why the outlet pipe is 6 D long and Re = 50 (outlet-length study)

The first version used Re = 100 and a 2 D outlet pipe. OpenFOAM showed that
for narrow throats the jet was still far from fully developed at the outlet:

| case | backflow region | centre-line u at old outlet (fully developed: 2.0) |
|---|---|---|
| Re 100, d_t 0.4 | x = 4.9 → 11.0 | 6.5, and backflow *through* the outlet plane |
| Re 100, d_t 0.5 | x = 5.1 → 8.1 | 4.0 |
| Re 100, d_t 0.6 | x = 5.6 → 6.4 | 2.9 |

An outflow condition ("fully developed", "zero gradient") that is not true
makes the answer depend on how each solver implements it. In that setup the
PINN and OpenFOAM converged to *different* solutions, and the disagreement
grew exactly as d_t shrank.

Lengthening the outlet in OpenFOAM showed what each option needs:
- **Re = 100:** the jet re-develops only at x ≈ 17 (d_t = 0.4), i.e. a ~20 D
  pipe, too large for a CPU-trained PINN.
- **Re = 50:** the recirculation closes at x = 7.7 and the flow is fully
  developed by x = 11.0.

The final set-up is therefore **Re = 50 with a 6 D outlet pipe (L = 12.5)**.
At the outlet every design now has centre-line u = 2.00–2.02 and no backflow,
so the outflow condition is *physically true*.

### 7.3 Why a stream function (mass-leak diagnosis)

With primitive outputs `(u, v, p)` and continuity as a soft loss term, the
d_t = 0.4 solution after 5,000 epochs had a centre-line velocity of 1.55 at
the outlet instead of 2.0. The network was **losing ~20 % of the mass flux**
through the jet region.
- Less flow through the throat means less pressure drop: the inlet pressure
  was 56–75 instead of OpenFOAM's 89.
- The leak was cheap for the optimiser because the continuity residual is
  normalised by the large local scale `U_m/R` in the throat.

The stream-function ansatz (§4.2) removes this failure mode entirely: the
flow rate is exact at every section for any weights.

---

## 8. Post-processing

`scripts/infer_and_postprocess.py` produces:

| output | content |
|---|---|
| `results/figures/fields_dtXXX.png` | static pressure, velocity magnitude and **dynamic pressure** contours with streamlines, full pipe (mirrored) |
| `results/figures/compare_openfoam_dtXXX.png` | OpenFOAM (top half) vs PINN (bottom half) on one colour scale, plus absolute-error maps |
| `results/figures/axial_profiles.png` | centre-line u, p, q along the pipe, PINN lines vs OpenFOAM markers |
| `results/figures/design_sweep.png` | throat dynamic pressure, velocity, static pressure, total Δp, min u vs d_t (121 PINN designs + OpenFOAM) |
| `results/figures/throat_sweep_animation.gif` | throat opening d1 → 1.0, closing → 0.4, back to d1: pressure + dynamic pressure + streamlines |
| `results/figures/training_history.png` | loss components, plus monitor-only error vs OpenFOAM |
| `results/figures/timing_amortisation.png` | cumulative cost vs number of designs: OpenFOAM vs PINN |
| `results/tables/*.csv, timing.json` | sweep data (non-dim + SI), PINN-vs-OpenFOAM errors, timings |
| `results/fields/*.npz` | full field arrays of the plotted designs |
| `results/paraview/nozzle_sweep.pvd` | 25-design ParaView series (git-ignored, regenerate) — open it and press **Play** to sweep d_t |

<!-- RESULTS_START -->
## 9. Results

All numbers below come from the committed run. Hardware: one 4-core CPU,
**no GPU**, float64. Tables: `results/tables/`; training log:
`results/pinn/train_log_part1.txt` + `train_log.txt`.

### 9.1 What the solver shows: throat diameter vs dynamic pressure

![baseline fields](results/figures/fields_dt070.png)

*Baseline design d1 (`d_t = 0.70`): static pressure, velocity magnitude and
dynamic pressure with streamlines (iso-ψ, equal flux between lines).*

![narrowest throat](results/figures/fields_dt040.png)

*Narrowest throat (`d_t = 0.40`).*
- The streamlines crowd into a fast jet: centre-line velocity ≈ 11.5 × inlet.
- Static pressure drops below the outlet value just downstream of the throat.
- The diffuser separates: the white dashed iso-ψ > 1 loops are recirculating
  fluid.

The question in the brief, *"when the throat diameter increases the dynamic
pressure decreases, and when it goes below d1 it increases"*, is answered
quantitatively by the sweep:

![design sweep](results/figures/design_sweep.png)

| throat ratio d_t | D_t (mm, water case) | throat centre-line dynamic pressure q/ρU² (PINN / OpenFOAM) | vs d1 | throat centre-line velocity | total Δp (PINN / OpenFOAM) | recirculation bubble (PINN / OF) |
|---|---|---|---|---|---|---|
| 0.40 | 4 | 66.2 / 65.5 | **× 9.0** | 11.51 | 85.9 / 88.9 | 2.55 / 2.66 D |
| 0.50 | 5 | 27.30 / 27.30 | × 3.7 | 7.39 | 39.5 / 40.0 | 0.88 / 0.98 D |
| 0.60 | 6 | 13.36 / 13.42 | × 1.8 | 5.17 | 22.46 / 22.47 | none / none |
| **0.70 (d1)** | 7 | **7.38 / 7.41** | 1 | 3.84 | 14.95 / 14.92 | none / none |
| 0.80 | 8 | 4.45 / 4.47 | × 0.60 | 2.98 | 11.20 / 11.21 | none / none |
| 0.90 | 9 | 2.89 / 2.90 | × 0.39 | 2.40 | 9.14 / 9.20 | none / none |
| 1.00 (straight pipe) | 10 | 1.99 / 2.00 | × 0.27 | 2.00 | 7.93 / 8.03 | none / none |

- **Below d1** the throat dynamic pressure rises steeply. It scales roughly
  like `d_t⁻⁴`, because velocity scales like `d_t⁻²` by mass conservation.
- **Above d1** it falls, and at `d_t = 1` it reaches the Poiseuille value
  (centre-line q = 2).
- The **static** pressure needed to drive the flow rises even faster:
  `Δp = 86` at `d_t = 0.4` versus 15 at d1. Viscous losses in the throat and
  the energy dumped by the jet into the separated diffuser account for this.
- **Below `d_t ≈ 0.53`** the diffuser separates.
- In the water case (ρU² = 0.025 Pa per unit), `d_t = 0.4` needs 2.1 Pa
  against 0.37 Pa at d1. SI columns are in `results/tables/design_sweep_pinn.csv`.

The animation `results/figures/throat_sweep_animation.gif` sweeps the throat
d1 → 1.0 (opening) → 0.4 (closing) → d1, showing the pressure and
dynamic-pressure fields with streamlines evolving continuously. Every frame
is a PINN inference.

### 9.2 Accuracy vs OpenFOAM (no CFD data used in training)

Error of the PINN evaluated at every OpenFOAM cell centre (31,250 cells per
design), relative L2 norm:

| d_t | u | v (rel. to ‖U‖) | p | inlet pressure (PINN / OF) | throat q error | total Δp error |
|---|---|---|---|---|---|---|
| 0.40 | 1.25 % | 0.07 % | 3.9 % | 85.9 / 88.9 | 1.1 % | 3.4 % |
| 0.50 | 0.22 % | 0.03 % | 1.3 % | 39.5 / 40.0 | 0.02 % | 1.2 % |
| 0.60 | 0.24 % | 0.03 % | 0.26 % | 22.46 / 22.47 | 0.42 % | 0.05 % |
| 0.70 | 0.23 % | 0.03 % | 0.37 % | 14.95 / 14.92 | 0.39 % | 0.22 % |
| 0.80 | 0.25 % | 0.03 % | 0.11 % | 11.20 / 11.21 | 0.34 % | 0.05 % |
| 0.90 | 0.22 % | 0.03 % | 0.55 % | 9.14 / 9.20 | 0.39 % | 0.60 % |
| 1.00 | 0.36 % | 0.05 % | 1.2 % | 7.93 / 8.03 | 0.52 % | 1.3 % |
| **mean** | **0.40 %** | **0.04 %** | **1.09 %** | | **0.44 %** | **1.0 %** |

For context, OpenFOAM's own error against the exact straight-pipe solution
is 0.18 % (u) / 0.39 % (p). From `d_t = 0.5` to 1.0 the PINN therefore agrees
with OpenFOAM to within a few times OpenFOAM's own discretisation error.

The hardest design is the narrowest throat (`d_t = 0.4`), with a strong jet
and a 2.7 D separation bubble. There the PINN error is 1.3 % in velocity and
3.9 % in pressure. The error map shows it sits where the jet decays
(x ≈ 4.5–7).

![compare 0.4](results/figures/compare_openfoam_dt040.png)
![compare d1](results/figures/compare_openfoam_dt070.png)
![axial profiles](results/figures/axial_profiles.png)

Mass conservation is exact by construction: the logged continuity residual is
3e-32 and the flux test passes to 1e-6. The final mean squared scaled momentum
residuals are 8.5e-5 (x) and 1.7e-5 (r).

![training history](results/figures/training_history.png)

*Left: the physics loss, which is the only training signal. Right: error vs
OpenFOAM, monitored every 500 iterations but never used for training.*

### 9.3 Timing and amortisation (same 4-core CPU, no GPU)

| item | measured |
|---|---|
| PINN training (21,698 parameters) | **12,914 s = 3.59 h** (Adam 9,000 epochs 9,251 s + L-BFGS 1,642 evaluations 3,663 s) |
| PINN inference, full field of one design (30,561 points, u, v, p, q, ψ) | **0.26 s** |
| PINN inference, 121-design sweep | 32 s |
| OpenFOAM, one design (31,250 cells, mesh + ~300 SIMPLE iterations, 1 core) | **34.6 s** |
| OpenFOAM, 4 designs concurrently on the 4 cores (throughput) | **13.7 s / design** |

| designs explored | PINN (train once + infer) | OpenFOAM, 1 case at a time | OpenFOAM, 4 in parallel |
|---|---|---|---|
| 7 (the comparison set) | 3.6 h | 4 min | 1.6 min |
| 121 (our sweep) | 3.60 h | 1.16 h | 0.46 h |
| **~376** | = break-even vs serial OpenFOAM | | |
| 1,000 | **3.66 h** | 9.6 h (**2.6× slower**) | 3.8 h (break-even ~962) |
| 10,000 | **4.32 h** | 96 h (**22× slower**) | 38 h (**8.8× slower**) |

![timing](results/figures/timing_amortisation.png)

**Honest reading of these numbers:**

- **This case is a best case for OpenFOAM.** It is 2-D axisymmetric, laminar
  and only 31k cells, so each CFD run takes seconds. On these CPUs the PINN
  only pays off beyond a few hundred to ~1,000 designs.
- **The PINN's cost is front-loaded.** Training is paid once, and every
  further design costs a quarter of a second. OpenFOAM's cost grows linearly
  with the number of designs.
- **A GPU changes the PINN side most.** Training is dense linear algebra plus
  autograd; a data-centre GPU typically trains this 10–50× faster, i.e. in
  minutes, and batches inference over thousands of designs. `simpleFoam`
  remains a CPU solver. GPU linear-solver back-ends exist, but give little at
  31k cells. With a GPU the break-even moves from hundreds of designs to
  tens.
- **Bigger CFD problems favour the surrogate even more.** For 3-D turbulent
  flow with millions of cells each OpenFOAM design costs hours plus meshing,
  so the PINN wins much earlier. See §11 for the limits of plain PINNs there.
- **The surrogate also gives what CFD does not.** It is a continuous,
  differentiable map `d_t ↦ (u, v, p)`. Any intermediate design, a
  derivative like `∂Δp/∂d_t`, or an optimisation loop is a cheap
  forward/backward pass.

The total training time includes the restart from the epoch-5,000 checkpoint
after an out-of-memory kill at the start of L-BFGS (since fixed, see §5.3).

It excludes the earlier development runs described in §7.2–7.3.
<!-- RESULTS_END -->

---

## 10. Code walkthrough, file by file

### `configs/nozzle.yaml`
The single source of every number: segment lengths, design range, Re, SI
mapping, network size, sampling densities, optimiser schedule, inference
grid, and OpenFOAM mesh and convergence settings. Code never hard-codes these.
A run is fully defined by (config, seed, code version).

### `nozzle_pinn/config.py`
`load_config(path)` reads the YAML into a dict. `REPO_ROOT` lets scripts find
`results/` regardless of the working directory.

### `nozzle_pinn/geometry.py`
- `smootherstep`, `smootherstep_dt` — the C² step `S(t)` and its derivative,
  written to work identically on NumPy arrays and torch tensors (`_clip01`
  dispatches on type). That way the same geometry feeds OpenFOAM meshing,
  plotting and autograd.
- `Nozzle` (frozen dataclass) — built from the config by `from_config`.
  - Properties `x1..x4`, `length` and `x_throat_mid` are the segment
    junctions.
  - `phi`, `dphi_dx` give the shape function and its slope.
  - `radius(x, dt)` and `dradius_dx(x, dt)` give `R` and `R'`.
  - `mean_velocity` gives `U_m = (R_in/R)²` from mass conservation.

### `nozzle_pinn/model.py`
- `MLP` — plain tanh network, Xavier init (kept for comparison:
  `architecture: mlp`).
- `ModifiedMLP` — the Wang et al. architecture (§4.3). `forward` computes the
  two encoders `U, V` once, then mixes them into each hidden layer.
- `NozzlePINN.__init__`
  - Builds the chosen network and **zeros its last layer**, so the initial
    prediction equals the engineering prior.
  - Stores Gauss–Legendre nodes and weights as buffers. Buffers are saved with
    the model and move with `.to(device)`.
- `NozzlePINN.p_prior` — the quasi-1-D pressure `p_1D`. It maps the 64 GL
  nodes onto `[x, L]` for every point at once (`xq` has shape `N × 64`),
  evaluates the Poiseuille friction integrand there, and sums. Because the
  integration limits depend on `x`, autograd differentiates the quadrature
  formula itself, giving `∂p_1D/∂x = −32 U_m/(Re D²) − ∂(U_m²)/∂x`
  automatically.
- `NozzlePINN.normalise` — scales the inputs to [−1, 1]³.
- `NozzlePINN.stream_H` — runs the network on `(x, s, d_t)` and returns
  `H = ⅛[(2 − s) + ξ²(1 − s)² N_ψ]` and the raw pressure output `N_p`.
- `NozzlePINN.forward`
  - Computes `R, R'` and `s = (r/R)²`.
  - Creates the separate handles `xh, sh` and calls `stream_H`.
  - Takes **one** autograd call for the partials `H_x, H_s`.
  - Builds `u = 2F_s/R²` and `v = −(r/R²)(H_x − 2R'F_s/R)` (the axis-regular
    form of `(1/r)∂ψ/∂r` and `−(1/r)∂ψ/∂x`) and `p = p_1D + (1 − ξ)d_t⁻⁴N_p`.
  - Runs inside `torch.enable_grad()`. Velocities are themselves derivatives,
    so even inference (`predict_points` under `no_grad`) needs this one
    gradient. During training the graph is kept (`create_graph=True`), so
    `physics.py` can differentiate `u, v` further.
  - Returns `u, v, p, ψ, R, U_m`.

### `nozzle_pinn/physics.py`
- `_grad(f, X)` — one autograd call with `create_graph=True`, so the result
  can be differentiated again. That is needed for second derivatives, and for
  back-propagating the loss through the derivatives.
- `ns_residuals(model, X, dt, Re)` — `X = (x, r)` is a leaf tensor with
  `requires_grad=True`.
  - One `_grad` per field gives both `∂/∂x` and `∂/∂r`.
  - Four more give `u_xx, u_rr, v_xx, v_rr`. Since `u, v` are already first
    derivatives of `ψ`, these are third derivatives of the network. `p` only
    needs first derivatives.
  - It then writes equations (1)–(3) literally and divides by the local
    scales (`scale="local"`), or by per-design throat scales
    (`scale="design"`, an option kept for experiments). Continuity is
    computed as a check; it is zero to round-off.
- `outlet_residuals` — `∂u/∂x` and `v` on the outlet plane: the soft
  "fully developed outflow" condition, matching OpenFOAM's `zeroGradient`.

### `nozzle_pinn/sampling.py`
- `sample_interior` — mixtures for `x` (uniform + throat region) and `η²`
  (uniform + wall layer). Each is shuffled independently with
  `torch.randperm` so the mixtures decorrelate, then mapped to
  `r = R(x; d_t)·η`.
- `sample_outlet` — points on `x = L`.
- `draw` — one complete set from the config, with `requires_grad=True` on
  the coordinates. Given the current `model`, it implements **RAD**:
  1. Draws a candidate pool `adaptive_pool` times larger.
  2. Evaluates the squared momentum residual there.
  3. Picks `adaptive_fraction` of the points with `torch.multinomial`, using
     probability `∝ e/mean(e) + 1`.

  Every resample after the first is adaptive, and so is the L-BFGS point
  set.

### `nozzle_pinn/train.py`
- `build_model(cfg)` — sets float64 and constructs `NozzlePINN` from the
  config.
- `loss_terms` / `total_loss` — mean-squared residuals and their weighted sum.
- `train(cfg, out_dir, monitor)`
  - Seeds everything.
  - **Adam stage:** resamples every `resample_every` epochs, applies a
    `StepLR` decay, logs every `log_every`, calls the monitor (if any) and
    checkpoints.
  - **L-BFGS stage:** one `LBFGS.step(closure)` on a larger fixed point set.
    The closure recomputes the loss and gradients for every line-search
    evaluation and logs as it goes.
  - Finally stores the final loss, per-stage wall-clock times and the
    parameter count in `history.json`, and saves `model_final.pt` with the
    config embedded. The checkpoint is then self-describing.
- `load_model(path)` — rebuilds the model from that file.

### `nozzle_pinn/inference.py`
- `predict_points` — `torch.no_grad()` evaluation at arbitrary points. It is
  used for the OpenFOAM comparison (evaluating at OpenFOAM cell centres)
  and is cheap because no graph is built.
- `field_grid` — structured `(x, η)` grid, derived fields, and the stream
  function by cumulative trapezoidal integration, normalised by
  `Q/2π = 1/8`.
- `axial_profiles` — centre-line and area-averaged quantities and the
  integrated mass flux (≈ 1 everywhere, which is a conservation check).

### `nozzle_pinn/compare.py`
- `load_openfoam` — reads all `results/openfoam/dt_XXXX.npz`.
- `compare_case` — evaluates the PINN *at the OpenFOAM cell centres*, so no
  interpolation error is involved. It returns relative L2 errors of
  `u, v, |U|, p` and max-abs errors. The v error is normalised by `|U|`,
  because `v ≡ 0` in the straight pipe.
- `make_monitor` — wraps `compare_case` for use during training.

### `nozzle_pinn/openfoam.py`
See §7.
- `write_block_mesh` writes the mesh dictionary.
- `write_case` writes `controlDict`, `fvSchemes`, `fvSolution`,
  `transportProperties`, `turbulenceProperties` and `0/p`.
- `_write_velocity` writes the inlet profile.
- `run_case` meshes and solves, recording wall-clock times and the iteration
  count.
- `latest_time` finds the newest time directory.
- `_parse_list` and `read_internal` are ASCII OpenFOAM list parsers.
- `read_mesh` handles polyMesh parsing and face and cell centres.
- `read_solution` returns the final cell-centre fields.
- `final_residuals` scrapes the log.

### `nozzle_pinn/postprocess.py`
- `_mirror` / `_full` — reflect the half-plane solution to draw the whole pipe.
- `_streamlines` — iso-ψ contours at 10 equal flux levels. White dashed
  levels appear when ψ > 1 (recirculation).
- `plot_design_fields` — p, |U| and q contour panels with streamlines.
- `plot_pinn_vs_openfoam` — split-pipe comparison and error maps.
  `_of_grid` reshapes OpenFOAM's x-fastest cell order to `(nr, nx)`.
- `plot_axial_profiles`, `plot_sweep`, `plot_history`, `plot_timing` — the
  graphs listed in §8.
- `make_animation` — evaluates the surrogate on each frame's nozzle and
  writes a GIF with `PillowWriter`. Colour limits are fixed across frames so
  colours are comparable.
- `write_vts` / `write_pvd` — ParaView structured-grid files and a collection
  whose "time" is `d_t`.

### `scripts/`
- `run_openfoam.py` — loops over the configured designs.
  - `--nx/--nr/--tag` run the mesh-refinement study.
  - `--jobs 4` runs cases concurrently (one core each) and writes
    `timings_parallel.json`, the fair multi-core OpenFOAM baseline used in
    §9.3.
- `train.py` — CLI wrapper around `train()` with overrides and automatic
  monitoring when OpenFOAM results exist. `--resume-from/--start-epoch/--prev-log`
  continue an interrupted run (`parse_log` rebuilds its history).
- `infer_and_postprocess.py` — the sweep, comparisons, timing, figures and
  exports. `section_metrics` computes throat quantities identically for the
  PINN grid and for OpenFOAM cells.
- `run_all.sh` — the whole pipeline.

### `tests/test_nozzle_pinn.py`
1. Geometry: end radii, throat radius, and analytic slope = finite difference.
2. Every hard boundary condition holds to 1e-12 for random network weights.
3. Continuity is satisfied to round-off for random weights (stream function).
4. The volume flux through 15 cross-sections of 3 nozzles equals π/4 to 1e-6,
   for random weights.
5. Exact Poiseuille flow (the prior at d_t = 1) gives zero residual in all
   three equations. This validates the `1/r` terms, the third-derivative
   autograd chain and the pressure prior.
6. The integrated stream function is 0 on the axis and 1 on the wall.
7. The OpenFOAM list parser handles both scalar and vector lists.

---

## 11. Limitations and next steps

- **Laminar only (Re = 50).** A real 0.5 m water pipe at 2 m/s runs at
  Re ≈ 10⁶ and is turbulent. That needs RANS closures (e.g. k-ω SST residuals
  in the loss) or wall-modelled formulations. Vanilla PINNs struggle at high
  Re; this is an open research problem.
- **Hardest design.** The `d_t = 0.4` design (strong jet + separation) is the
  least accurate, at 3.9 % in p. A longer training budget, more points in the
  jet, a larger network or a GPU would reduce it.
- **Axisymmetric, one design parameter.** Throat length, diffuser angle or
  non-circular sections mean more inputs (`normalise`), more config ranges and
  more sampling dimensions. 3-D requires the full equations.
- **Complex geometry → neural operators.** For many-parameter or free-form 3-D
  shapes, **Geo-FNO, GINO, Transolver** and graph networks (MeshGraphNets) can
  take the geometry directly as a point cloud or signed-distance field, with
  no meshing per design. They are *data-driven*, typically trained on a few
  hundred CFD solutions, often produced by OpenFOAM itself. So OpenFOAM
  becomes the data generator rather than the competitor.
  - A **physics-informed neural operator (PINO)** adds the same Navier–Stokes
    residual loss used here, to cut the data requirement.
  - That is the natural next step: OpenFOAM sparse data + physics loss + point
    cloud geometry for 3-D turbulent nozzles.
- **Hardware.** All timings are on a 4-core CPU without a GPU (see §9.3 for
  how a GPU shifts the comparison).

## 12. References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis, *Physics-informed neural networks*, J. Comput. Phys. 378 (2019) 686–707.
2. L. Sun, H. Gao, S. Pan, J.-X. Wang, *Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data*, Comput. Methods Appl. Mech. Eng. 361 (2020) 112732. Parametric geometries, hard BCs, no data.
3. S. Wang, Y. Teng, P. Perdikaris, *Understanding and mitigating gradient flow pathologies in physics-informed neural networks*, SIAM J. Sci. Comput. 43 (2021) A3055–A3081. The modified MLP.
4. F. M. White, *Viscous Fluid Flow*, 3rd ed., McGraw-Hill (2006). Hagen–Poiseuille flow and the axisymmetric Navier–Stokes equations.
5. OpenFOAM v1912 user guide: `simpleFoam`, wedge boundary conditions.
