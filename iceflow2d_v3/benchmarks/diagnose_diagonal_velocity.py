"""Reproduce diagonal contact-line streaks and run same-state ablations.

This is a diagnostic, not a production solver modification. Frozen-state and
altered-viscosity cases deliberately change the problem and are not fixes.
Arrays use (x, y[, component]); velocities and force densities use SI units.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.benchmarks.check_surface_vortex import sample_force
from iceflow2d_v3.examples.coupled_falling_ice_melting_2d import create_config
from iceflow2d_v3.lattice import lattice_direction, mixture_density
from iceflow2d_v3.simulator import IceFlow2D


def checkpoint(s, folder):
    folder.mkdir()
    state, seen = {}, set()
    for prefix, owner in (("flow", s), ("thermal", s.thermal)):
        for name, value in vars(owner).items():
            if isinstance(value, (ti.Field, ti.MatrixField)) and id(value) not in seen:
                state[prefix + "__" + name] = value.to_numpy()
                seen.add(id(value))
    np.savez_compressed(folder / "state.npz", **state)
    host = {
        "flow": {k: getattr(s, k) for k in ("steps", "_thermal_lbm_steps_pending")},
        "thermal": {k: getattr(s.thermal, k) for k in ("steps", "time_s")},
        "config": s.config.to_dict(),
    }
    (folder / "host.json").write_text(json.dumps(host, indent=2) + "\n")


def restore(s, folder):
    host = json.loads((folder / "host.json").read_text())
    # Case ablations may change forces/relaxation, but never grid or time units.
    current_config = json.loads(json.dumps(s.config.to_dict()))
    for key in ("resolution", "dx", "reference_velocity"):
        if host["config"][key] != current_config[key]:
            raise ValueError(f"Restart changes {key}")
    with np.load(folder / "state.npz") as state:
        for key in state.files:
            prefix, name = key.split("__", 1)
            owner = s if prefix == "flow" else s.thermal
            getattr(owner, name).from_numpy(state[key])
    for prefix, owner in (("flow", s), ("thermal", s.thermal)):
        for name, value in host[prefix].items():
            setattr(owner, name, value)


@ti.kernel
def recover_frozen_phase(s: ti.template()):
    """Keep phase fixed, but recover the streamed momentum with that density."""
    for i, j in s.water_phase:
        if s._is_fluid_cell(i, j):
            momentum = ti.Vector.zero(ti.f32, 2)
            for q in range(9):
                momentum += ti.cast(lattice_direction(q), ti.f32) * s.momentum_populations[i, j, q]
            rho = mixture_density(s.water_phase[i, j], s._water_density_lattice, s._air_density_lattice)
            s.momentum_velocity_lattice[i, j] = momentum / rho
        else:
            s.momentum_velocity_lattice[i, j] = ti.Vector.zero(ti.f32, 2)
            s.fluid_acceleration_lattice[i, j] = ti.Vector.zero(ti.f32, 2)


def snapshot(s):
    c = s.config
    force = np.zeros((s.nx, s.ny, 9), dtype=np.float32)
    sample_force(s, force)
    force_scale = c.rho_water * c.dx / s.scales.dt_s**2
    return dict(
        phi=s.water_phase.to_numpy(), solid=s.solid_mask.to_numpy(), wall=s.wall_mask.to_numpy(),
        velocity=s.physical_velocity_lattice.to_numpy() * s.scales.velocity_scale_m_s,
        pressure=s.pressure_lattice.to_numpy() * c.rho_water * s.scales.velocity_scale_m_s**2,
        temperature=s.thermal.water_temperature.to_numpy(),
        force_capillary=force[..., 4:6] * force_scale,
        force_buoyancy=force[..., 2:4] * force_scale,
        force_gravity=force[..., :2] * force_scale,
        shear_tau=force[..., 6], density=force[..., 8] * c.rho_water,
        stress=s.capillary_stress_lattice.to_numpy() * c.rho_water * s.scales.velocity_scale_m_s**2,
        time_s=s.physical_time_s, dx_m=c.dx, dt_s=s.scales.dt_s,
        body_center=np.array(s.body_center[None]) * c.dx,
        body_velocity=np.array(s.body_velocity[None]) * s.scales.velocity_scale_m_s,
        body_angle=float(s.body_angle[None]), body_mass=float(s.body_mass_lattice[None]),
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    c = create_config()
    if args.case == "no_sigma":
        c = replace(c, sigma=0.0)
    elif args.case == "no_buoyancy":
        c = replace(c, thermal=replace(c.thermal, thermal_expansion_water_1_k=0.0))
    elif args.case == "acoustic_one":
        c = replace(c, cm_acoustic_relaxation_rate=1.0)
    elif args.case == "viscosity_x10":
        c = replace(c, viscosity_water=3.14173e-6)
    source = Path(__file__).resolve().parents[1] / "simulator.py"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps(dict(
        case=args.case, config=c.to_dict(), simulator_sha256=sha,
        restart=str(args.restart) if args.restart else None,
        frozen=args.frozen, reset_flow=args.reset_flow,
    ), indent=2) + "\n")
    s = IceFlow2D(c)
    if args.restart:
        restore(s, args.restart)
    if args.frozen:
        s.body_velocity.fill(0.0)
        s.body_angular_velocity.fill(0.0)
    if args.reset_flow:
        s._initialize_hydrostatic_equilibrium()
        s.fluid_acceleration_lattice.fill(0.0)
    end = round(args.end / s.scales.dt_s)
    if end <= s.steps:
        raise ValueError("End time must be after the restart time")
    stride = max(1, round(args.interval / s.scales.dt_s))
    targets = sorted(set([s.steps, end] + list(range((s.steps // stride + 1) * stride, end, stride))))
    checks = {round(t / s.scales.dt_s) for t in args.checkpoint_times}
    targets = sorted(set(targets) | {t for t in checks if s.steps <= t <= end})
    started, rows = time.monotonic(), []
    for target in targets:
        if args.frozen:
            while s.steps < target:
                s._collide_momentum()
                s._stream()
                recover_frozen_phase(s)
                s._update_pressure()
                s.steps += 1
        else:
            s.step(target - s.steps)
        d = snapshot(s)
        if not all(np.isfinite(d[k]).all() for k in ("phi", "velocity", "pressure", "temperature")):
            raise RuntimeError(f"Nonfinite field at step {target}")
        water = (d["phi"] >= 0.5) & (d["solid"] == 0) & (d["wall"] == 0)
        speed = np.linalg.norm(d["velocity"], axis=-1)
        row = dict(time_s=s.physical_time_s, elapsed_s=time.monotonic() - started,
                   water_speed_max=float(speed[water].max()),
                   body_center=d["body_center"].tolist(), body_mass=d["body_mass"])
        rows.append(row)
        np.savez_compressed(args.output / f"fields_{target:07d}.npz", **d)
        if target in checks:
            checkpoint(s, args.output / f"checkpoint_{target:07d}")
        (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if sha != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Production source changed during the diagnostic")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--restart", type=Path)
    p.add_argument("--case", choices=("baseline", "no_sigma", "no_buoyancy", "acoustic_one", "viscosity_x10"), default="baseline")
    p.add_argument("--end", type=float, default=0.8)
    p.add_argument("--interval", type=float, default=0.02)
    p.add_argument("--checkpoint-times", type=float, nargs="*", default=[0.74])
    p.add_argument("--frozen", action="store_true", help="freeze geometry, phase and temperature; advance momentum only")
    p.add_argument("--reset-flow", action="store_true", help="start with zero velocity and zero dynamic pressure")
    a = p.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (a.end, a.interval)):
        p.error("end and interval must be finite and positive")
    if a.reset_flow and not a.frozen:
        p.error("reset-flow is only supported for frozen-state diagnostics")
    run(a)


if __name__ == "__main__":
    main()
