# Mixture2D: Volume-Preserving LBM-MPM Coupling for Air-Water-Sand Mixtures

[[**Paper**](https://geometry.caltech.edu/pubs/XWYDL26.pdf)] [[**Video**](https://www.youtube.com/watch?v=LPqutmyjDyM)]

## Project Overview

This is the 2D Python/Taichi CUDA code release for "Volume-Preserving LBM-MPM Coupling for Air-Water-Sand Mixtures".

The repository contains a compact 2D dam-break implementation of the paper's air-water-sand coupling model. It is intended as an open-source reference version of the core solver, with the full production system reduced to a single reproducible 2D example.

## Key Features

- 2D dam-break air-water-sand mixture simulation
- Two-phase air/water LBM fluid solver
- MPM sand solver with APIC transfer and Drucker-Prager plasticity
- Two-way LBM-MPM coupling through porosity, drag, buoyancy, pressure correction, and phase-source correction
- Optional water-retention model for wet sand
- Standalone `fluid`, `sand`, and `coupled` modes
- PNG frame output, optional particle `.npz` output, and per-run metadata

NOTE: This release only targets the 2D dam-break case. It does not include 3D support, mesh SDF collision, rigid-body interaction, remapping, CPU fallback, or the full production renderer.

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **GPU Memory**: A few GB is sufficient for the default 2D examples; larger resolutions or long output sequences require more memory and disk space

### Software Dependencies

#### Required Dependencies

- **Python**: Python 3.10 or newer is recommended
- **Taichi**: Version 1.7.4
- **CUDA**: A working NVIDIA driver and CUDA-capable runtime visible to Taichi
- **NumPy**: Array processing
- **imageio**: PNG output
- **matplotlib**: Fallback image writing and colormap support

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

#### Optional Dependencies

- **ffmpeg**: Useful if you want to convert generated PNG frames into videos

The code initializes Taichi with `ti.cuda`. If CUDA is unavailable, the program exits instead of falling back to CPU.

## Build and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DacianShaw/LBM-MPM-Air-Water-Sand.git
cd LBM-MPM-Air-Water-Sand
```

### 2. Install Dependencies

```bash
python -m venv .venv
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Pure 2D LBM Dam Break

```bash
python examples/dam_break_2d.py --mode fluid --resolution-x 600 --resolution-y 300 --frames 150 --steps-per-frame 1000
```

### Standalone 2D Sand MPM

```bash
python examples/dam_break_2d.py --mode sand --resolution-x 600 --resolution-y 300 --frames 400 --steps-per-frame 1000
```

### Coupled 2D Air-Water-Sand Dam Break

```bash
python examples/dam_break_2d.py --mode coupled --resolution-x 600 --resolution-y 300 --frames 400 --steps-per-frame 1000
```

### Coupled Simulation with Water Retention

```bash
python examples/dam_break_2d.py --mode coupled --water-retention --resolution-x 600 --resolution-y 300 --frames 400 --steps-per-frame 1000
```

### Useful Output Options

```bash
python examples/dam_break_2d.py --mode coupled --output-dir outputs/my_run
python examples/dam_break_2d.py --mode sand --save-npz
python examples/dam_break_2d.py --mode fluid --show-gui
python examples/dam_break_2d.py --mode fluid --no-progress
```

By default, frames are written to `outputs/dam_break_2d/<mode>/`; the retention run uses `outputs/dam_break_2d/coupled_water_retention/` so it does not overwrite the coupled result without retention. Each run also writes `metadata.json` with the mode, configuration, Taichi version, backend, and step count.

The command-line demo shows frame progress, elapsed time, and ETA automatically in an interactive terminal. Use `--no-progress` to disable it, or `--progress` to force it when stderr is redirected.

## Python API

```python
from mixture2d import Simulator2D, create_dambreak_config

cfg = create_dambreak_config(
    mode="coupled",
    resolution=(600, 300),
    water_retention=True,
)

sim = Simulator2D(cfg)
sim.step(300)
sim.save_frame("frame.png")
```

The main configuration object is `DamBreakConfig` in `mixture2d/config.py`.

## Paper Correspondence

The main solver implementation is in `mixture2d/simulator.py`. Code comments mark the corresponding paper sections and equations:

- Sec. 4.1, Eqs. 8-23: velocity-based LBM fluid solver and force construction
- Sec. 4.2, Eqs. 24-30: MPM sand transfer, grid update, and particle advection
- Sec. 4.3, Eq. 31 and Eqs. 32-40: drag, buoyancy, and pressure coupling
- Sec. 4.4, Eqs. 45-52: phase-field source term and volume controller
- Sec. 4.5, Eqs. 53-59: water retention and saturation-dependent cohesion
- Algorithm 1: coupled LBM-MPM stepping order

The volume controller uses the pore-volume-weighted form of Eq. 52,
`sum_x epsilon * S_phi` (or `epsilon_hat` with retention), because this is
the source-term contribution to the change in physical free-water volume.

The paper uses D3Q27/D3Q7 lattices and the original production implementation contains additional simulation cases and optimizations. This repository uses D2Q9 and explicit double buffers for the 2D release.

When water retention is disabled, the sand cohesion force uses the cohesion parameter setting from [Multi-species Simulation of Porous Sand and Water Mixtures](https://dl.acm.org/doi/10.1145/3072959.3073651).

## Tests

Run the CUDA smoke tests with:

```bash
python -m unittest discover -s tests
```

The tests cover small fluid, sand, coupled, and water-retention runs.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{xiao2026volume,
  title={Volume-Preserving LBM-MPM Coupling for Air-Water-Sand Mixtures},
  author={Xiao, Xiaoyu and Wang, Haoxiang and Yang, Xiaokang and Desbrun, Mathieu and Li, Wei},
  journal={ACM Transactions on Graphics},
  volume={45},
  number={4},
  articleno={77},
  year={2026},
  doi={10.1145/3811302}
}
```
