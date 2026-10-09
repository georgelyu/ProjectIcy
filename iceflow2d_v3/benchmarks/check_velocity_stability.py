"""Sample the actual example at high cadence to expose velocity flicker.

    python -m iceflow2d_v3.benchmarks.check_velocity_stability --output-dir /tmp/ice-stability

The temporal metric is the RMS vector change over ``sample_steps`` LBM steps,
not a mass/energy balance or a filtered rendering of the velocity field.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from iceflow2d_v3.examples.coupled_falling_ice_melting_2d import create_config
from iceflow2d_v3.simulator import IceFlow2D


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    config = create_config()
    if args.case == "cold_water":
        config = replace(
            config,
            thermal=replace(
                config.thermal,
                initial_water_temperature_c=0.0,
                buoyancy_reference_temperature_c=0.0,
            ),
        )
    elif args.case == "no_surface_tension":
        config = replace(config, sigma=0.0)
    simulation = IceFlow2D(config)
    if args.case == "fixed_ice":

        def discard_loads():
            simulation.hydrodynamic_impulse.fill(0.0)
            simulation.hydrodynamic_torque.fill(0.0)

        simulation._integrate_rigid_ice = discard_loads

    wall = simulation.wall_mask.to_numpy() != 0
    end_step = int(np.ceil(args.end_time_s / simulation.scales.dt_s))
    targets = list(range(0, end_step, args.sample_steps)) + [end_step]
    previous_velocity = previous_wet = None
    rows = []
    started = time.perf_counter()
    for target in targets:
        simulation.step(target - simulation.steps)
        phase = simulation.water_phase.to_numpy()
        solid = simulation.solid_mask.to_numpy() != 0
        velocity = (
            simulation.physical_velocity_lattice.to_numpy().astype(np.float64)
            * simulation.scales.velocity_scale_m_s
        )
        pressure = simulation.pressure_lattice.to_numpy()
        temperature = simulation.thermal.water_temperature.to_numpy()
        if not all(
            np.isfinite(field).all()
            for field in (phase, velocity, pressure, temperature)
        ):
            raise RuntimeError(f"Non-finite simulation state at step {target}")
        wet = ~wall & ~solid & (phase > 0.5)
        speed2 = np.sum(velocity**2, axis=-1)
        row = {
            "step": target,
            "time_s": simulation.physical_time_s,
            "water_max_speed_m_s": float(np.sqrt(speed2[wet].max())),
            "water_rms_speed_m_s": float(np.sqrt(speed2[wet].mean())),
            "melted_fraction": 1.0
            - float(simulation.body_mass_lattice[None]) / config.ice_mass_lattice,
        }
        if previous_velocity is not None:
            delta2 = np.sum((velocity - previous_velocity) ** 2, axis=-1)
            row["velocity_delta_rms_m_s"] = float(np.sqrt(delta2[wet].mean()))
            row["persistent_water_delta_rms_m_s"] = float(
                np.sqrt(delta2[wet & previous_wet].mean())
            )
        rows.append(row)
        previous_velocity, previous_wet = velocity, wet

    measured = [row for row in rows if "velocity_delta_rms_m_s" in row]
    summary = {
        "end_time_s": rows[-1]["time_s"],
        "sample_interval_s": args.sample_steps * simulation.scales.dt_s,
        "max_water_speed_m_s": max(row["water_max_speed_m_s"] for row in rows),
        "mean_velocity_delta_rms_m_s": float(
            np.mean([row["velocity_delta_rms_m_s"] for row in measured])
        ),
        "final_melted_fraction": rows[-1]["melted_fraction"],
    }
    source_root = Path(__file__).resolve().parents[1]
    results = {
        "case": args.case,
        "config": config.to_dict(),
        "source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in (
                "simulator.py",
                "lattice.py",
                "config.py",
                "examples/coupled_falling_ice_melting_2d.py",
            )
        },
        "wall_time_s": time.perf_counter() - started,
        "summary": summary,
        "samples": rows,
    }
    (output / "velocity_stability.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    np.savez_compressed(
        output / "final_fields.npz",
        phase=phase,
        solid=solid,
        velocity_m_s=velocity,
        pressure_lattice=pressure,
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True, help="new directory for measurements"
    )
    parser.add_argument(
        "--case",
        choices=("default", "cold_water", "no_surface_tension", "fixed_ice"),
        default="default",
    )
    parser.add_argument("--end-time-s", type=float, default=0.2)
    parser.add_argument("--sample-steps", type=int, default=32)
    args = parser.parse_args()
    if not np.isfinite(args.end_time_s) or args.end_time_s <= 0.0:
        parser.error("--end-time-s must be positive and finite")
    if args.sample_steps < 1:
        parser.error("--sample-steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
