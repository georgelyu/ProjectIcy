"""Isothermal pool with no ice, identical initial solenoidal perturbation.
Check whether interface clipping can amplify pre-existing flow by itself.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow2d import IceFlow2D
from iceflow2d.examples.coupled_falling_ice_melting_2d import create_config, parse_args

p = argparse.ArgumentParser()
p.add_argument("--cutoff", type=float, required=True)
a = p.parse_args()
s = IceFlow2D(
    replace(
        create_config(parse_args(["--thermal-expansion-water-1-k", "0"])),
        volume_projection_interface_cutoff=a.cutoff,
    )
)
nx, ny = s.nx, s.ny
dx = s.config.dx
dt = s.scales.dt_s
wall = s.wall_mask.to_numpy().astype(bool)
# Same analytic phase field for both controls, far above the initial reference.
y = (np.arange(ny) + 0.5) * dx
x = (np.arange(nx) + 0.5) * dx
ph1 = 1 / (1 + np.exp(4 * (y - 0.0325) / 0.00125))
ph = np.broadcast_to(ph1, (nx, ny)).copy().astype(np.float32)
ph[wall] = 0
s.solid_mask.fill(0)
s.body_mass_lattice[None] = 0
s.body_inertia_lattice[None] = 0
s.water_phase.from_numpy(ph)
s._build_hydrostatic_reference()
s._initialize_hydrostatic_equilibrium()
# Streamfunction vanishes with its normal derivative at x walls; Gaussian in y.
X = (x - 0.00075) / 0.0235
Y = (y - 0.0307) / 0.0012
fx = np.sin(np.pi * np.clip(X, 0, 1)) ** 2
dfx = np.pi / 0.0235 * np.sin(2 * np.pi * np.clip(X, 0, 1))
fy = np.exp(-Y * Y)
dfy = -2 * Y / 0.0012 * fy
v = np.stack(
    np.broadcast_arrays(fx[:, None] * dfy[None, :], -dfx[:, None] * fy[None, :]),
    axis=-1,
)
v *= 0.005 / np.max(np.linalg.norm(v, axis=-1))
v[wall] = 0
u = v / 40
rho = 0.00125 + 0.99875 * ph
c = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
)
w = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
f = np.zeros((nx, ny, 9))
h = f.copy()
for q in range(9):
    cu = u @ c[q]
    poly = 3 * cu + 4.5 * cu**2 - 1.5 * np.sum(u * u, axis=-1)
    f[:, :, q] = w[q] * rho * poly
    h[:, :, q] = w[q] * ph * (1 + poly)
f[wall] = 0
h[wall] = 0
s.momentum_populations.from_numpy(f.astype(np.float32))
s.phase_populations.from_numpy(h.astype(np.float32))
s.momentum_velocity_lattice.from_numpy(u.astype(np.float32))
s.fluid_acceleration_lattice.fill(0)
s.water_volume_target[None] = float(ph.sum(dtype=np.float64))
out = Path("iceflow2d/outputs/iceflow2d/vortex_checks_20260915") / f"perturbed_pool_{a.cutoff:g}"
out.mkdir(exist_ok=False)
rows = []
start = time.monotonic()
for n in range(480001):
    if n % 16000 == 0:
        ph = s.water_phase.to_numpy()
        v = s.physical_velocity_lattice.to_numpy() * 40
        speed = np.linalg.norm(v, axis=-1)
        wet = (ph > 0.5) & ~wall
        row = {
            "time_s": n * dt,
            "umax": float(speed[wet].max()),
            "water_ke": float(
                (0.5 * 1000 * np.clip(ph, 0, 1) * speed**2)[~wall].sum() * dx * dx
            ),
            "wall_s": time.monotonic() - start,
        }
        rows.append(row)
        print(a.cutoff, json.dumps(row), flush=True)
    if n == 480000:
        break
    s._collide_momentum()
    s._collide_phase()
    s._stream()
    s._update_streamed_macroscopic_fields()
    s._update_pressure()
    s._correct_water_volume()
with (out / "history.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
np.savez_compressed(out / "final.npz", phi=ph, velocity=v, dx=dx)
