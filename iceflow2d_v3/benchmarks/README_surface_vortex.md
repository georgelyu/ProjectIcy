# Surface-vortex diagnostic controls

These scripts call the current v3 CUDA solver. Stage diagnostics use the same
flow, body and thermal helpers as the production step.
Run from the repository root with the Python environment providing Taichi,
NumPy, and Matplotlib. Results must go into a new directory.

```bash
python iceflow2d_v3/benchmarks/check_surface_vortex.py --case baseline --output iceflow2d_v3/outputs/iceflow2d_v3/my_checks/baseline
python iceflow2d_v3/benchmarks/check_surface_vortex.py --case no_buoyancy --output iceflow2d_v3/outputs/iceflow2d_v3/my_checks/no_buoyancy
python iceflow2d_v3/benchmarks/check_surface_vortex.py --case shear_065 --output iceflow2d_v3/outputs/iceflow2d_v3/my_checks/shear_065
python iceflow2d_v3/benchmarks/check_surface_vortex.py --case width_4 --output iceflow2d_v3/outputs/iceflow2d_v3/my_checks/width_4
python iceflow2d_v3/benchmarks/check_surface_vortex.py --case cutoff_1e5 --output iceflow2d_v3/outputs/iceflow2d_v3/my_checks/cutoff_1e5
python iceflow2d_v3/benchmarks/check_surface_vortex.py --case grid_200 --output iceflow2d_v3/outputs/iceflow2d_v3/my_checks/grid_200
python iceflow2d_v3/benchmarks/analyze_surface_vortex.py iceflow2d_v3/outputs/iceflow2d_v3/my_checks --frame-time 1.8
```

Defaults: end at 3.8 s; field output every 0.02 s from 1.8 s; stage budgets
sample one full thermal cycle preceding each 0.1 s output. These sampled
budgets are not continuous mechanical-energy integrals. NPZ axes are x,y.
Do not confuse speed magnitude with positive horizontal velocity.

The grid control preserves physical wall and diffuse-interface thickness,
phase mobility, shear-viscosity floor, thermal cadence, angular damping time,
and Smagorinsky filter scale. The sharp-ice stabilization stencil remains
one cell, so its physical thickness changes. This is a sensitivity check,
not proof of convergence. The 20260915 grid run failed at 2.0322125 s.

A complete checkpoint is saved at 1.8 s by the current diagnostic script.
For paired restarts, `--source` must end in one of the case names above:

```bash
python iceflow2d_v3/benchmarks/restart_surface_vortex.py --source iceflow2d_v3/outputs/iceflow2d_v3/my_checks/shear_065 --mode same
python iceflow2d_v3/benchmarks/restart_surface_vortex.py --source iceflow2d_v3/outputs/iceflow2d_v3/my_checks/shear_065 --mode cutoff
python iceflow2d_v3/benchmarks/restart_surface_vortex.py --source iceflow2d_v3/outputs/iceflow2d_v3/my_checks/shear_065 --mode flow_cutoff
python iceflow2d_v3/benchmarks/restart_surface_vortex.py --source iceflow2d_v3/outputs/iceflow2d_v3/my_checks/shear_065 --mode gauge
```

`flow_cutoff` leaves the thermal wet-cell threshold at its original value.
`gauge` changes the frozen pressure decomposition with population and half-force
compensation. These are diagnostic interventions, not production fixes.

The isolated perturbed-pool script uses a fixed output root under the dated
check directory and refuses to overwrite existing directories:

```bash
python iceflow2d_v3/benchmarks/perturbed_pool_vortex.py --cutoff 0.001
python iceflow2d_v3/benchmarks/perturbed_pool_vortex.py --cutoff 0.00001
```

Historical 20260915 reports and saved checkpoints describe the previous model.
Use newly generated v3 checkpoints for current restarts; old state fields and
stage names do not describe the disappearing-melt solver.
