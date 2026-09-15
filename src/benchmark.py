"""THE ARGUMENT FOR A PINN SURROGATE, MEASURED — not just asserted.

OBJECTIVE OF THIS FILE
-----------------------
Section 2 of README.md makes a claim: "train once, expensively; then answer
unlimited queries almost for free." This file is where that claim gets
checked against a clock, on two things that answer the exact same
question (given a throat diameter Dt and an inlet velocity V_in, what is
V(x) and p(x)?) by two completely different mechanisms:

  1. `classical_solver.solve_nozzle_rk45` -- one fresh adaptive
     Runge-Kutta integration PER QUERY. Every new (Dt, V_in) pair means
     marching the ODE system from x=0 to x=L all over again; nothing from
     a previous query is reused.
  2. A trained PINN's `forward()` -- one ordinary batched matrix-multiply
     pass through ~21,000 fixed numbers, for as many queries as fit in
     the batch AT ONCE. No optimizer, no gradient, no `.backward()` --
     training already happened, once, beforehand (train.py), and
     `forward()` never touches any of that machinery again.

WHY "PER QUERY" VS "BATCHED" IS THE ENTIRE STORY
------------------------------------------------------
This is not an apples-to-apples "which single evaluation is faster"
benchmark -- it is deliberately structured to expose the actual shape of
each method's cost:

    classical_solver total time  ~  N_queries * (cost of one RK45 solve)
    PINN inference total time    ~  (fixed per-call overhead) + tiny * N_queries

The classical solver's cost is *linear* in the number of queries with no
way to amortize, because each query is physically a different ODE that
has to be integrated from scratch. The PINN's cost is *nearly flat* once
you're past the one-time training cost, because a batched matrix multiply
gets cheaper per-row as the batch grows (GPU/CPU vectorization) -- and
critically, that one-time training cost is paid exactly ONCE regardless
of whether you query the model once or a billion times afterward.

HONEST CAVEAT -- READ BEFORE QUOTING A SPEEDUP NUMBER
-----------------------------------------------------------
This network is tiny (21,122 parameters) and this comparison typically
runs on a shared, CPU-only machine with no GPU. At small N_queries, both
Python-level call overhead and system noise (other processes, thermal
throttling, cache state) can be a meaningful fraction of the measured
time, so the speedup ratio measured here will NOT scale linearly or
perfectly predictably with N_queries the way the argument above suggests
in the limit -- it is a real, reproducible measurement of an actual
speedup on this actual hardware, not a clean theoretical curve. Report
the numbers this file actually measures, not an idealized extrapolation.
Training cost itself (one run of `train.py`, tens of minutes) is NOT
included in the inference-time numbers below -- it is a one-time cost
paid up front, separately, and is exactly why this argument only pays off
once you need "unlimited queries," not for a single one-off evaluation.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.classical_solver import solve_nozzle_rk45


def time_classical_solver(
    cfg: dict, n_queries: int, seed: int = 1234, rtol: float = 1e-6, atol: float = 1e-9
) -> dict:
    """Time N_queries independent RK45 solves, one fresh integration each.

    Draws random (Dt, V_in) pairs across the design space, exactly the
    kind of "unlimited different queries" scenario the amortization
    argument is about, and marches a fresh ODE solve for every single one.

    `rtol`/`atol` here are looser than Section 4's cross-validation defaults
    (1e-10/1e-12) -- Part 4 needs the tightest tolerance available to make
    "these two independent methods agree" a meaningful statement, but a
    fair speed comparison should use tolerances an engineer would actually
    run for routine queries, not the tightest tolerance the solver supports.
    """
    g, ph = cfg["geometry"], cfg["physics"]
    rng = np.random.default_rng(seed)
    dts = rng.uniform(g["dt_min"], g["dt_max"], size=n_queries)
    v_ins = rng.uniform(0.5 * ph["V_in"], 1.5 * ph["V_in"], size=n_queries)

    t0 = time.perf_counter()
    for dt, v_in in zip(dts, v_ins):
        solve_nozzle_rk45(float(dt), float(v_in), cfg, n_report=201, rtol=rtol, atol=atol)
    elapsed = time.perf_counter() - t0

    return {
        "n_queries": n_queries,
        "total_seconds": elapsed,
        "seconds_per_query": elapsed / n_queries,
    }


def time_pinn_inference(model, cfg: dict, n_queries: int, seed: int = 1234) -> dict:
    """Time ONE batched forward pass answering all N_queries at once.

    `torch.no_grad()` is essential here, not decorative: it's what makes
    this an inference-only pass (see physics.py's docstring on why
    `create_graph=True` is needed during TRAINING) -- no computation graph
    is built, so there is nothing for a `.backward()` to walk, which is
    exactly why this is fast: matrix multiplications only, no autodiff.
    """
    g = cfg["geometry"]
    rng = np.random.default_rng(seed)
    dts = rng.uniform(g["dt_min"], g["dt_max"], size=n_queries)
    xt = torch.as_tensor(rng.uniform(0.0, 1.0, size=(n_queries, 1)), dtype=torch.float64)
    dtt = torch.as_tensor((dts / g["D_in"]).reshape(-1, 1), dtype=torch.float64)

    device = next(model.parameters()).device
    xt, dtt = xt.to(device), dtt.to(device)

    with torch.no_grad():  # inference: no gradient graph, no autodiff machinery at all
        model(xt, dtt)  # warm-up call, not timed (first call can pay one-time setup cost)
        t0 = time.perf_counter()
        model(xt, dtt)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    return {
        "n_queries": n_queries,
        "total_seconds": elapsed,
        "seconds_per_query": elapsed / n_queries,
    }


def run_benchmark(model, cfg: dict, n_queries_list: list[int] | None = None) -> list[dict]:
    """Run both timers across a range of query counts and report the ratio.

    Returns one dict per n_queries with both timings and the speedup, so
    the caller can plot/print how the ratio changes with query count --
    that trend, not any single number, is the actual evidence for the
    amortization argument.
    """
    n_queries_list = n_queries_list or [1, 10, 100, 1000, 10000]
    rows = []
    for n in n_queries_list:
        classical = time_classical_solver(cfg, n)
        pinn = time_pinn_inference(model, cfg, n)
        rows.append({
            "n_queries": n,
            "classical_total_s": classical["total_seconds"],
            "classical_per_query_s": classical["seconds_per_query"],
            "pinn_total_s": pinn["total_seconds"],
            "pinn_per_query_s": pinn["seconds_per_query"],
            "speedup_x": classical["total_seconds"] / pinn["total_seconds"],
        })
    return rows
