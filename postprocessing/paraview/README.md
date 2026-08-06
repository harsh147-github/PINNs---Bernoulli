# Viewing the PINN surrogate output in ParaView

`scripts/run_parametric_sweep.py` writes one `.vtu` file per throat diameter to
`results/paraview/nozzle_Dt_###.vtu` (sequential zero-padded index, e.g.
`nozzle_Dt_000.vtu` ... `nozzle_Dt_024.vtu` for a 25-point sweep). Each file is
a 2D axisymmetric reconstruction of the 1D surrogate: for every axial station
`x`, the duct cross-section `y in [-D(x)/2, D(x)/2]` carries point data `V`
(m/s) and `p` (Pa), constant across `y` (plug flow — see README.md Section 11).

## Basic viewing

1. **File > Open**, select `results/paraview/nozzle_Dt_000.vtu` (or the whole
   sequence — ParaView will detect `nozzle_Dt_###.vtu` as a time series because
   of the shared prefix and sequential index).
2. Click **Apply**.
3. In the **Coloring** dropdown, pick `p` (pressure) or `V` (velocity) to color
   the mesh by field.

## Suggested filters

- **Calculator**: create a derived field, e.g. `V_mag = abs(V)` or a dynamic
  pressure proxy `q = 0.5*1.225*V^2` (Filters > Alphabetical > Calculator).
- **Warp By Vector**: exaggerate the duct profile for visual clarity — first
  use Calculator to build a `(0, V, 0)` vector field, then Warp By Vector with
  a small scale factor.
- **Stream Tracer**: seed streamlines along the inlet cross-section using the
  same `(V, 0, 0)`-style vector field built via Calculator; since the flow is
  plug flow this mainly illustrates the axial acceleration through the throat.

## Animating the parametric sweep

Because the 25 files share a common name pattern with a sequential index,
opening them as a **group** (select all `nozzle_Dt_*.vtu` files in the File >
Open dialog) lets ParaView step through them with the **time** controls (the
"time" here just indexes throat diameter, not physical time) — press Play to
watch the pressure/velocity field respond continuously as the throat narrows.
