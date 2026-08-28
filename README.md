# IceFlow2D

IceFlow2D is a two-dimensional air-water phase-field lattice Boltzmann solver.
Its mechanical path couples the flow to one sharp rigid ice body. An optional
fixed-body thermal path adds conservative finite-volume enthalpy transport,
melting/freezing, a liquid-fraction-driven ice boundary, and density-aware
water exchange.

For full-width hydrostatic pools, the single
`well_balanced_hydrostatics` option enables the complete frozen-reference
scheme: residual-gravity forcing in the bulk, dynamic-pressure populations,
and matching hydrostatic cut-link traction on a moving ice body. These parts
are one coupled discretization rather than independently selectable features.

The phase-changing ice is held at a fixed pose in the first coupled model;
moving-body melting, deformation, and fracture are not included. The LBM
solver requires an NVIDIA CUDA-capable GPU. A separate NumPy two-dimensional
Stefan benchmark remains available as a CPU reference for the enthalpy method.

## Installation

Python 3.10 or newer is recommended. Install the runtime dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Quick start

Run the rigid-ice dam-break example through the package entry point:

```bash
python -m iceflow2d --frames 120 --steps-per-frame 100 --progress
```

For a small entry-point and output smoke run:

```bash
python -m iceflow2d \
  --frames 2 --steps-per-frame 5 \
  --resolution-x 120 --resolution-y 60 \
  --reference-velocity-m-s 0.001 --no-progress
```

Run the free-falling ice example with:

```bash
python iceflow2d/examples/ice_fall_2d.py --progress
```

Run the fixed-ice two-dimensional phase-change validation with:

```bash
python iceflow2d/examples/stefan_melting_2d.py
```

The default benchmark uses a `30 x 30 mm` water domain resolved by a
`120 x 120` finite-volume grid. A centred `12 x 12 mm` ice square starts at
`0 °C`; the water and all four walls remain at `20 °C` for `180 s`. About
`44.3%` of the initial solid area melts, corresponding to a `1.52 mm`
equivalent uniform melt depth, while the core remains solid. Validation checks
four-wall energy balance, D4 rotational/reflection symmetry, monotone solid
area, and grid-refinement convergence. The CPU example writes `history.csv`,
`metadata.json`, `fields.npz`, and a two-dimensional summary figure below
`outputs/iceflow2d/stefan_melting_2d/`; it does not initialize Taichi or
require CUDA.

Run the first end-to-end LBM/thermal phase-change example with:

```bash
python iceflow2d/examples/coupled_fixed_ice_melting_2d.py
```

Its default domain is `0.025 x 0.035 m` on a `200 x 280` grid with square
`1.25e-4 m` cells. A fixed
`0.008 x 0.008 m` ice square starts at `0 °C` in `60 °C` water below a free surface.
The fixed reference mapping is `0.001 m/s <-> 0.1 LU`, giving
`dt=0.0125 s`; the `2.0 s` run therefore covers 160 LBM steps. Ice and water use their physical
densities, `917` and `1000 kg/m³`; melting therefore adds only `0.917` water
cells per lost solid cell, with the remaining contraction taken up by the
water-air interface. Gravity and surface tension are zero in this controlled
verification so the result isolates heat transfer, phase-boundary updates,
and conservative mass exchange. The higher-resolution default supersedes the
earlier `80 x 112` coupling baseline; quantitative melt fractions should be
read from the generated metadata for the exact configuration being run.

The example writes water-velocity PNGs to `velocity_frames/` with an animated
`velocity_field.gif`, and the corresponding out-of-plane vorticity sequence to
`vorticity_frames/` with `vorticity_field.gif`. Temperature is streamed as a
third sequence to `temperature_frames/frame_*.png` and
`temperature_field.gif`. Every frame and every temperature case in one
invocation use the same speed-color and quiver reference scale, whose default
maximum is `9e-6 m/s`; override it with
`--velocity-visualization-max-m-s`. Vorticity uses the signed definition
`omega_z=dv/dx-du/dy` and a shared symmetric default range of `[-0.2, 0.2] s^-1`,
configurable with `--vorticity-visualization-max-s-1`. The temperature sequence
and final figure use one fixed scan-wide range,
`[Tm, max(T_inf)]`, from the melting temperature to the largest requested
far-field water temperature. Temperature frames show water and phase-change
material while masking the thermally inactive air cap and container walls. The
controlled default case is intentionally
quiescent because its gravity, surface tension, and initial velocity are all
zero; the same output becomes a flow visualization when the example is
configured with nonzero forcing. `--no-plot` disables the final summary figure
and all three PNG/GIF sequences. At every sampling time, the velocity,
vorticity, and temperature PNGs are rendered immediately and appended to their
GIFs, so the sequences appear while the simulation is running instead of being
cached until completion. The default outputs retain `history.csv`,
`metadata.json`, the final temperature/liquid-fraction figure, and the three
velocity, vorticity, and temperature PNG/GIF sequences, but no longer include a
raw field archive. Use
`--save-npz` only when full-field postprocessing is required; the resulting
`fields.npz` keeps the legacy raw `velocity_lattice` field and the
half-force-corrected `physical_velocity_lattice` used by the plots. Consequently,
the default peak memory is `O(Nx Ny)`, whereas enabling the archive requires
`O(Nf Nx Ny)` memory for `Nf` sampled fields.

To exercise freshwater natural convection at several far-field temperatures in
one command, use:

```bash
python iceflow2d/examples/coupled_fixed_ice_melting_2d.py \
  --water-temperatures-c 4 5.6 8 --gravity-m-s2 9.8 \
  --reference-velocity-m-s 0.4 \
  --output-dir outputs/iceflow2d/freshwater_temperature_sweep
```

The example uses the quadratic freshwater equation of state by default,
`rho(T)=rho_star[1-beta(T-T_star)^2]`, with `T_star=4 degC`; each bath
temperature is also its far-field buoyancy reference. Nonzero gravity enables
the well-balanced hydrostatic formulation automatically. Every temperature is
written to a distinct `T_*C` directory with its own three PNG/GIF sequences, and
`temperature_sweep.json` summarizes melting, peak velocity, and the shared
visualization ranges across cases. For the command above, all speed plots use
`[0, 9e-6] m/s`, all vorticity plots use `[-0.2, 0.2] s^-1`, and all
temperature plots use `[0, 8] °C`.
The larger explicit reference velocity in this gravity-driven command is a
stability requirement of the present explicit two-phase LBM.  Configuration no
longer rejects lower values, but the `0.001 m/s` default maps full gravity to
`g_LU=12.25` and is numerically unstable; retain it only for the zero-gravity
coupled case.

By default, generated frames and run data are written below
`outputs/iceflow2d/`. Use `--output-dir` to select another location. See
[iceflow2d/README.md](iceflow2d/README.md) for model details, configuration,
and validation notes.

## Python API

```python
from iceflow2d import IceFlow2D, create_iceflow_config

config = create_iceflow_config(resolution=(600, 300))
simulation = IceFlow2D(config)
simulation.step(100)
simulation.save_frame("frame.png")
```

## Tests

Run the repository-level tests with:

```bash
python -m unittest discover -s tests
```

The numerical solver tests under `iceflow2d/` require the CUDA backend.

## Reference and license

The fluid formulation is derived from the 2D reference path accompanying
[Volume-Preserving LBM-MPM Coupling for Air-Water-Sand Mixtures](https://geometry.caltech.edu/pubs/XWYDL26.pdf),
with a rigid, impermeable ice coupling replacing the porous MPM solid model.

This project is released under the [MIT License](LICENSE).
