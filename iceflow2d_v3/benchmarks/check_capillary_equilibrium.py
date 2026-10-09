"""Evolve a stationary drop or a neutral ice contact line at physical viscosity.

Refinement keeps the physical domain, diffuse-interface width and phase mobility
fixed by default. No heat or rigid-body update is performed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from iceflow2d_v3.config import ThermalConfig, create_iceflow_config
from iceflow2d_v3.simulator import IceFlow2D


def make_case(case, refinement=1, interface_width_cells=None):
    n = refinement
    nx, ny = (96 * n, 112 * n) if case == "drop" else (80 * n, 96 * n)
    boundary = 2 * n
    width = 5.0 * n if interface_width_cells is None else interface_width_cells
    config = create_iceflow_config(
        resolution=(nx, ny),
        dx=0.00025 / n,
        reference_velocity=4.0,
        rho_water=1000.0,
        rho_air=1.25,
        rho_ice=917.0,
        viscosity_water=None,
        viscosity_air=None,
        sigma=0.072,
        gravity=(0.0, 0.0),
        interface_width=width,
        # dx and dt both halve: M_LU must double to keep M_LU*dx**2/dt.
        mobility=0.1 * n,
        phase_warmup_steps=0,
        water_width_fraction=(nx - boundary) / nx,
        water_height_fraction=40 * n / ny,
        ice_width_fraction=(8 if case == "drop" else 16) * n / nx,
        ice_height_fraction=(8 if case == "drop" else 32) * n / ny,
        ice_base_y_cells=(90 if case == "drop" else 24) * n,
        boundary_cells=boundary,
        ice_initial_angle=0.0,
        thermal=ThermalConfig(initial_water_temperature_c=90.0),
    )
    x, y = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="ij")
    radius = np.hypot(x - 48 * n, y - 45 * n)
    signed_distance = radius - 20 * n if case == "drop" else y - 40 * n
    phase = (0.5 * (1.0 - np.tanh(2.0 * signed_distance / width))).astype(np.float32)
    return config, phase, radius


def initialize_rest(simulation, phase, drop_radius_cells=None):
    simulation.water_phase.from_numpy(phase)
    simulation._build_hydrostatic_reference()
    simulation._initialize_hydrostatic_equilibrium()
    simulation.thermal._initialize_water_state(
        simulation.water_phase, simulation.wall_mask, simulation.solid_mask
    )
    if drop_radius_cells is not None:
        # The two-dimensional Laplace jump is sigma/R. This is an initial
        # continuous approximation, not a precomputed discrete equilibrium.
        pressure = simulation._surface_tension_lattice / drop_radius_cells * phase
        weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
        populations = 3 * pressure[..., None] * (weights - np.eye(9)[0])
        simulation.momentum_populations.from_numpy(populations.astype(np.float32))
        simulation.pressure_lattice.from_numpy(pressure)


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config, phase, radius = make_case(
        args.case, args.refinement, args.interface_width_cells
    )
    source = Path(__file__).resolve().parents[1] / "simulator.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "config": config.to_dict(),
                "simulator_sha256": source_hash,
                "body_and_heat_frozen": True,
            },
            indent=2,
        )
    )
    s = IceFlow2D(config)
    drop_radius = 20 * args.refinement
    initialize_rest(s, phase, drop_radius if args.case == "drop" else None)
    fluid = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
    initial_volume = phase[fluid].sum(dtype=float)
    end = round(args.end_time_s / s.scales.dt_s)
    stride = max(1, round(0.05 / s.scales.dt_s))
    targets = sorted(set([0, end] + list(range(stride, end, stride))))
    rows = []
    started = time.monotonic()
    for target in targets:
        while s.steps < target:
            s._advance_fluid()
            s.steps += 1
        phase = s.water_phase.to_numpy()
        velocity = s.physical_velocity_lattice.to_numpy() * s.scales.velocity_scale_m_s
        pressure = (
            s.pressure_lattice.to_numpy()
            * config.rho_water
            * s.scales.velocity_scale_m_s**2
        )
        if not all(np.isfinite(a).all() for a in (phase, velocity, pressure)):
            raise RuntimeError(
                f"Non-finite equilibrium state at {s.physical_time_s:g} s"
            )
        speed = np.linalg.norm(velocity, axis=-1)
        water = fluid & (phase >= 0.5)
        row = {
            "time_s": s.physical_time_s,
            "water_speed_max_m_s": float(speed[water].max()),
            "air_speed_max_m_s": float(speed[fluid & ~water].max()),
            "relative_water_volume_error": float(
                phase[fluid].sum(dtype=float) / initial_volume - 1
            ),
        }
        if args.case == "drop":
            inside = fluid & (radius < 0.6 * drop_radius)
            outside = (
                fluid & (radius > 1.5 * drop_radius) & (radius < 1.9 * drop_radius)
            )
            row["pressure_jump_pa"] = float(
                pressure[inside].mean() - pressure[outside].mean()
            )
            row["expected_pressure_jump_pa"] = config.sigma / (drop_radius * config.dx)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (output / "metrics.json").write_text(json.dumps(rows, indent=2))
        if target in (0, end):
            np.savez_compressed(
                output / f"fields_{target:09d}.npz",
                phi=phase,
                velocity=velocity,
                pressure=pressure,
                solid=s.solid_mask.to_numpy(),
                wall=s.wall_mask.to_numpy(),
                time_s=s.physical_time_s,
                dx_m=config.dx,
            )
    if source_hash != hashlib.sha256(source.read_bytes()).hexdigest():
        raise RuntimeError("Solver source changed during the benchmark")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "final": rows[-1],
                "elapsed_s": time.monotonic() - started,
                "simulator_sha256": source_hash,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("drop", "contact"), default="drop")
    parser.add_argument("--refinement", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--interface-width-cells", type=float)
    parser.add_argument("--end-time-s", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not np.isfinite(args.end_time_s) or args.end_time_s <= 0:
        parser.error("--end-time-s must be positive and finite")
    if args.interface_width_cells is not None and (
        not np.isfinite(args.interface_width_cells) or args.interface_width_cells <= 0
    ):
        parser.error("--interface-width-cells must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
