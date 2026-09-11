# IceFlow2D

IceFlow2D is a CUDA/Taichi simulation of one two-dimensional scenario: a rigid
ice block falls through an air–water free surface and melts in a hot-water
bath. The retained solver couples:

- a D2Q9 pressure–momentum LBM and conservative Allen–Cahn water phase field;
- a unified sharp moving boundary with rigid translation, rotation, and wall contact;
- well-balanced hydrostatics for the full-width water pool;
- material-coordinate ice enthalpy, world-grid water enthalpy, conservative ALE
  remapping, and melt mass/momentum transfer.

The runtime requires an NVIDIA CUDA-capable GPU. Python 3.10 or newer is
recommended.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

The package entry point and the example script run the same case:

```bash
python -m iceflow2d
```

```bash
python iceflow2d/examples/coupled_falling_ice_melting_2d.py
```

The default domain is `0.025 m × 0.050 m` on a `100 × 200` grid. A
`0.008 m × 0.008 m` ice block starts at `0 °C`, rotated by `5°`, with its
lowest corner `0.0015 m` above a `0.030 m`-deep, `90 °C` water bath. The
reference mapping is `4 m/s ↔ 0.1 LU`, so `dx = 2.5e-4 m` and
`dt = 6.25e-6 s`. Conduction and phase change are updated every eight LBM
steps; aperture remapping and thermal advection remain coupled every step.
All four walls and the air interfaces are adiabatic, so the water cools as
it supplies the ice's sensible and latent heat. Complete melting adds water
equal to 91.7% of the ice volume; the default bath's equilibrium is about
76.64 °C, calculated using the actual water volume inside the wall cells.

The default physical duration is 3 seconds, sampled every `0.01 s`: 480,000
LBM steps and 301 snapshots including the initial state.
Use this short output smoke run when checking an installation:

```bash
python -m iceflow2d \
  --resolution-x 100 --resolution-y 200 \
  --end-time-s 0.00005 --output-interval-s 0.00005 \
  --no-plot --save-npz --no-progress
```

Useful output controls are `--output-dir`, `--no-plot`, `--save-npz`,
`--no-progress`, and `--quiet`. Run `python -m iceflow2d --help` for all
physical and numerical parameters.

## Outputs

Results are written to
`outputs/iceflow2d/coupled_falling_ice_melting_2d/` by default:

- `history.csv`: sampled motion, melt, water-volume, energy, ALE, and momentum diagnostics;
- `metadata.json`: requested settings, complete validated configuration, scales,
  output manifest, and final diagnostics;
- `coupled_falling_ice_melting_2d.png`: final summary when plotting is enabled;
- `velocity_frames/`, `vorticity_frames/`, and `temperature_frames/`, plus one
  GIF for each sequence, when plotting is enabled;
- `fields.npz`: sampled raw arrays only when `--save-npz` is selected.

Field frames are streamed to disk. Without `--save-npz`, sampled arrays are not
retained, so output memory remains proportional to one grid rather than the
number of frames.

Container contact is a whole-body geometric constraint. After each free rigid
step, the current melted ice support is translated back inside the physical
wall planes if necessary. A left/right hit removes only the velocity component
that points through that wall; a bottom/top hit does the same for the vertical
component. Tangential velocity and angular velocity are unchanged. There is no
restitution, dry friction, near-wall film force, or contact-point impulse solve;
wall contact is a purely kinematic position and velocity constraint.

## Python API

Use the example's validated geometry builder so programmatic runs stay on the
supported solver path:

```python
from iceflow2d import IceFlow2D
from iceflow2d.examples.coupled_falling_ice_melting_2d import create_config, parse_args

config = create_config(parse_args(["--end-time-s", "0.001"]))
simulation = IceFlow2D(config)
# End chunks on a multiple of config.thermal.update_interval_lbm_steps.
simulation.step(8)
```

## Tests

CPU-only configuration and output contracts:

```bash
python -m unittest discover -s tests -v
```

CUDA numerical regressions:

```bash
python -m unittest discover -s iceflow2d -p 'test_*.py' -v
```

## Scope

The model is two-dimensional and reports extensive quantities per unit
out-of-plane depth. It assumes one connected rigid ice remnant that melts
monotonically. It does not model fracture, multiple rigid fragments,
re-freezing onto the body, evaporation, boiling, vapour transport, or bubbles.

See [iceflow2d/README.md](iceflow2d/README.md) for the coupling sequence and
conservation diagnostics.

This project is released under the [MIT License](LICENSE).

## GPU state and performance review

All compute kernels now live in `iceflow2d/simulator.py`; configuration and unit
conversions live in `iceflow2d/config.py`. Cheap derived quantities use read-only
views instead of persistent GPU fields. See the
[field audit and validation record](iceflow2d/docs/gpu_state_review.md) for each
retained allocation, API name changes, numerical checks, and measured timings.
Run `python benchmarks/benchmark_solver.py` to repeat the warmed step benchmark.
