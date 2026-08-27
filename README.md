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
  --reference-length-cells 300 --no-progress
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

Its default domain is `2.5 x 3.5 mm` on an `80 x 112` grid. A fixed
`1 x 1 mm` ice square starts at `0 °C` in `60 °C` water below a free surface.
The run covers `0.30 s` (35,560 LBM steps). Ice and water use their physical
densities, `917` and `1000 kg/m³`; melting therefore adds only `0.917` water
cells per lost solid cell, with the remaining contraction taken up by the
water-air interface. Gravity and surface tension are zero in this controlled
verification so the result isolates heat transfer, phase-boundary updates,
and conservative mass exchange. A Taichi reference execution melts about
`49.0%` of the initial solid area, an equivalent uniform melt depth of
`0.143 mm` (about 4.6 cells), while retaining a solid core.

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
