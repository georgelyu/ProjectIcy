"""Sample lateral motion with the actual falling/melting production step."""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from iceflow2d_v3.examples.coupled_falling_ice_melting_2d import (
    create_config,
    parse_args,
)
from iceflow2d_v3.simulator import IceFlow2D


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    angle = {"mirror": -5.0, "symmetric": 0.0}.get(args.case, 5.0)
    config = create_config(parse_args(["--ice-angle-degrees", str(angle)]))
    if args.case == "no_surface_tension":
        config = replace(config, sigma=0.0)
    elif args.case == "cold_water":
        config = replace(
            config,
            thermal=replace(
                config.thermal,
                initial_water_temperature_c=0.0,
                buoyancy_reference_temperature_c=0.0,
            ),
        )
    source = Path(__file__).resolve().parents[1] / "simulator.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "simulator_sha256": source_hash}, indent=2
        )
    )
    simulation = IceFlow2D(config)
    initial_mass = float(simulation.body_mass_lattice[None])
    started = time.perf_counter()
    end_step = round(args.end_time_s / simulation.scales.dt_s)
    stride = max(1, round(0.01 / simulation.scales.dt_s))
    targets = list(range(0, end_step, stride)) + [end_step]
    rows = []
    with (output / "history.csv").open("w") as file:
        writer = None
        for target in targets:
            simulation.step(target - simulation.steps)
            center = np.asarray(simulation.body_center[None], dtype=float) * config.dx
            velocity = (
                np.asarray(simulation.body_velocity[None], dtype=float)
                * simulation.scales.velocity_scale_m_s
            )
            phase = simulation.water_phase.to_numpy()
            solid = simulation.solid_mask.to_numpy()
            wall = simulation.wall_mask.to_numpy()
            water_velocity = (
                simulation.physical_velocity_lattice.to_numpy()
                * simulation.scales.velocity_scale_m_s
            )
            temperature = simulation.thermal.water_temperature.to_numpy()
            pressure = simulation.pressure_lattice.to_numpy()
            if not all(
                np.isfinite(a).all()
                for a in (
                    phase,
                    water_velocity,
                    temperature,
                    pressure,
                    center,
                    velocity,
                )
            ):
                raise RuntimeError(f"Non-finite state at step {target}")
            if phase.min() < 0.0 or phase.max() > 1.0:
                raise RuntimeError(f"Unbounded phase at step {target}")
            wet = (wall == 0) & (solid == 0) & (phase > 0.5)
            row = {
                "time_s": simulation.physical_time_s,
                "x_m": float(center[0]),
                "y_m": float(center[1]),
                "vx_m_s": float(velocity[0]),
                "vy_m_s": float(velocity[1]),
                "angle_degrees": float(np.degrees(simulation.body_angle[None])),
                "melted_fraction": 1.0
                - simulation.body_mass_lattice[None] / initial_mass,
                "water_max_speed_m_s": float(
                    np.linalg.norm(water_velocity[wet], axis=1).max()
                ),
                "water_volume_cells": float(
                    phase[(wall == 0) & (solid == 0)].sum(dtype=np.float64)
                ),
            }
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            file.flush()
            rows.append(row)
            if target % (10 * stride) == 0 or target == end_step:
                np.savez_compressed(
                    output / f"fields_{target:09d}.npz",
                    phase=phase,
                    solid=solid,
                    wall=wall,
                    velocity_m_s=water_velocity,
                    temperature_c=temperature,
                    time_s=simulation.physical_time_s,
                )
                print(json.dumps(row), flush=True)
    summary = {
        "case": args.case,
        "wall_time_s": time.perf_counter() - started,
        "peak_absolute_vx_m_s": max(abs(r["vx_m_s"]) for r in rows),
        "peak_absolute_vx_after_0p2_m_s": max(
            (abs(r["vx_m_s"]) for r in rows if r["time_s"] >= 0.2), default=0.0
        ),
        "final": rows[-1],
        "source_unchanged": source_hash
        == hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("default", "mirror", "symmetric", "no_surface_tension", "cold_water"),
        default="default",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--end-time-s", type=float, default=1.2)
    args = parser.parse_args()
    if not np.isfinite(args.end_time_s) or args.end_time_s <= 0:
        parser.error("--end-time-s must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
