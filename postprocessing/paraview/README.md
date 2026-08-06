# ParaView post-processing

Training/sweep exports `results/paraview/nozzle_Dt_000.vtu ... nozzle_Dt_024.vtu`
(2D axisymmetric reconstruction of the nozzle, point data: `V` [m/s], `p` [Pa], `q_dyn` [Pa]).

## Quick start

1. Open ParaView -> File -> Open -> select all `nozzle_Dt_*.vtu` (they load as a time series).
2. Apply. Color by `p` (pressure) or `V` (velocity). The solid duct shape is the mesh itself.
3. Press Play to animate the flow field as the throat diameter sweeps the design space.

## Useful filters

- **Calculator**: `q_dyn = 0.5*1.225*V^2` (already exported), or a Mach proxy `V/340`.
- **Warp by Vector / by Scalar**: exaggerate the pressure field visually.
- **Stream Tracer**: seed a line at the inlet for streamlines (plug-flow field).
- **Plot Over Line**: sample `p` or `V` along the nozzle axis — compare with
  `results/figures/fields_vs_exact.png`.

## What to look for

- Throat = minimum pressure, maximum velocity (Bernoulli).
- Smaller `D_t` files show deeper suction peaks at the throat.
- Inlet and outlet recover to identical conditions (same area).
