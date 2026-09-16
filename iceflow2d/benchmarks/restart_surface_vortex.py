"""Paired short controls from the same saved state, without changing source solver."""

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from check_surface_vortex import (
    configuration,
    IceFlow2D,
    fields,
    kinetic,
    diagnostic_step,
)

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument(
    "--mode", choices=["same", "cutoff", "flow_cutoff", "gauge"], required=True
)
p.add_argument("--duration", type=float, default=0.1)
a = p.parse_args()
c = configuration(a.source.name)
if a.mode in ["cutoff", "flow_cutoff"]:
    c = replace(c, volume_projection_interface_cutoff=1e-5)
s = IceFlow2D(c)
if a.mode == "flow_cutoff":
    s.thermal._water_phase_cutoff = 1e-3
data = np.load(a.source / "checkpoint_1p8.npz")
host = json.loads((a.source / "checkpoint_1p8_host.json").read_text())
for key in data.files:
    prefix, name = key.split("__", 1)
    owner = s if prefix == "flow" else s.thermal
    getattr(owner, name).from_numpy(data[key])
for name in ["steps", "_thermal_lbm_steps_pending", "_moving_melt_momentum_pending"]:
    setattr(s, name, host["flow"][name])
for name in ["steps", "time_s"]:
    setattr(s.thermal, name, host["thermal"][name])
if a.mode == "gauge":
    # Change only the pressure decomposition, leaving physical pressure unchanged.
    ph = np.clip(s.water_phase.to_numpy(), 0, 1)
    mask = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
    count = mask.sum(axis=0)
    profile = np.divide(
        np.where(mask, ph, 0).sum(axis=0), count, out=np.zeros(s.ny), where=count > 0
    )
    rho = 0.00125 + 0.99875 * profile
    new = np.zeros(s.ny)
    for j in range(s.ny - 2, -1, -1):
        new[j] = new[j + 1] - 0.5 * s._gravity_lattice[1] * (rho[j] + rho[j + 1])
    old = s.reference_pressure_by_row.to_numpy()
    oldrho = s.reference_density_by_row.to_numpy()
    pop = s.momentum_populations.to_numpy()
    w = [4 / 9] + [1 / 9] * 4 + [1 / 36] * 4
    for q in range(9):
        pop[:, :, q] += 3 * (w[q] - (q == 0)) * (old - new)[None, :]
    # f represents momentum velocity; compensate the gauge-induced half-force change.
    density = 0.00125 + 0.99875 * ph
    delta_a = (oldrho - rho)[None, :] * s._gravity_lattice[1] / density
    velocity = s.momentum_velocity_lattice.to_numpy()
    velocity[:, :, 1] -= 0.5 * delta_a
    directions = np.array(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    for q in range(9):
        pop[:, :, q] += -1.5 * w[q] * directions[q, 1] * density * delta_a
    force = s.fluid_acceleration_lattice.to_numpy()
    force[:, :, 1] += delta_a
    s.momentum_populations.from_numpy(pop)
    s.momentum_velocity_lattice.from_numpy(velocity)
    s.fluid_acceleration_lattice.from_numpy(force)
    s.reference_pressure_by_row.from_numpy(new.astype(np.float32))
    s.reference_density_by_row.from_numpy(rho.astype(np.float32))
out = a.source.parent / ("restart_" + a.source.name + "_" + a.mode)
out.mkdir(exist_ok=False)
rows = []
stages = []
start = time.monotonic()
for n in range(round(a.duration / 0.01) + 1):
    if n:
        s.step(round(0.01 / s.scales.dt_s) - c.thermal.update_interval_lbm_steps)
        maps = {}
        for _ in range(c.thermal.update_interval_lbm_steps):
            diagnostic_step(s, stages, maps, True)
    state = fields(s)
    v = state["velocity"]
    ph = state["phi"]
    x = (np.arange(s.nx) + 0.5) * c.dx
    y = (np.arange(s.ny) + 0.5) * c.dx
    mask = (
        (ph > 0.5)
        & (state["solid"] == 0)
        & (x[:, None] > 0.009)
        & (x[:, None] < 0.0238)
        & (y[None, :] > 0.029)
        & (y[None, :] < 0.0318)
    )
    rows.append(
        {
            "time_s": s.physical_time_s,
            **kinetic(s, state),
            "jet_max": float(np.linalg.norm(v, axis=-1)[mask].max()),
            "jet_ux_mean": float(v[:, :, 0][mask].mean()),
        }
    )
    print(a.mode, json.dumps(rows[-1]), flush=True)
np.savez_compressed(out / "final.npz", **state, time_s=s.physical_time_s)
for name, r in [("metrics.csv", rows), ("stages.csv", stages)]:
    if r:
        with (out / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=r[0])
            w.writeheader()
            w.writerows(r)
(out / "manifest.json").write_text(
    json.dumps(
        {
            "source": str(a.source),
            "mode": a.mode,
            "duration": a.duration,
            "wall_s": time.monotonic() - start,
        },
        indent=2,
    )
)
