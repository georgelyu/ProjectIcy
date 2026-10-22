"""Reproducible full-case/ablation diagnostics, with a failure volume.

Run from the repository root. --source-root can select an immutable copy of
the iceflow3d package, keeping a failed experiment reproducible after fixes.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument("--source-root", type=Path)
    extra.add_argument("--sample-steps", type=int, default=200)
    extra.add_argument("--fixed-ice", action="store_true")
    extra.add_argument("--no-thermal", action="store_true")
    extra.add_argument("--no-capillary", action="store_true")
    extra.add_argument("--snapshot-interval-s", type=float, default=0.0)
    extra.add_argument("--checkpoint-interval-s", type=float, default=0.0)
    extra.add_argument("--restart", type=Path)
    extra.add_argument("--allow-source-change", action="store_true")
    extra.add_argument("--interface-width", type=float)
    extra.add_argument("--unlimited-density-source", action="store_true")
    extra.add_argument("--unlimited-phase-flux", action="store_true")
    diagnostic, argv = extra.parse_known_args()
    source = diagnostic.source_root or Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source.resolve()))
    from iceflow3d import IceFlow3D
    from iceflow3d.examples.dambreak_four_ice_melting_3d import (
        create_config,
        parse_args,
        snapshot_targets,
    )

    args = parse_args(argv)
    if diagnostic.sample_steps <= 0:
        raise ValueError("sample-steps must be positive")
    for interval in (
        diagnostic.snapshot_interval_s,
        diagnostic.checkpoint_interval_s,
    ):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError(
                "snapshot/checkpoint intervals must be finite and nonnegative"
            )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "diagnostic.json").exists():
        raise FileExistsError(output)
    config = create_config(args)
    if diagnostic.interface_width is not None:
        config = replace(config, interface_width=diagnostic.interface_width)
    if diagnostic.unlimited_density_source:
        config = replace(config, density_gradient_source_limit=0.0)
    if diagnostic.unlimited_phase_flux:
        config = replace(config, phase_flux_acceleration_limit=0.0)
    if diagnostic.no_capillary:
        config = replace(config, surface_tension_n_m=0.0)
    sim = None
    if diagnostic.restart is not None:
        from iceflow3d.checkpoint import load_checkpoint

        sim = load_checkpoint(
            diagnostic.restart, allow_source_change=diagnostic.allow_source_change
        )
        config = sim.config
    target = snapshot_targets(config, args)[-1]
    record = {
        "status": "running",
        "configuration": config.to_dict(),
        "controls": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(diagnostic).items()
        },
        "target_steps": target,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source / "iceflow3d").glob("*.py")
        },
    }
    metadata = output / "diagnostic.json"
    write_json(metadata, record)
    start = time.perf_counter()
    if sim is None:
        sim = IceFlow3D(config)
    if diagnostic.fixed_ice:
        sim.rigid.advance = lambda: None
    if diagnostic.no_thermal:
        sim._advance_advection = lambda: None
        sim._advance_conduction = lambda dt: None
    last_print = 0.0
    # Physical output times are integer LBM steps. Repeated float additions
    # can otherwise defer an exact target (e.g. 1.6 s) to the next sample.
    snapshot_stride = (
        max(1, round(diagnostic.snapshot_interval_s / sim.scales.dt_s))
        if diagnostic.snapshot_interval_s > 0
        else None
    )
    checkpoint_stride = (
        max(1, round(diagnostic.checkpoint_interval_s / sim.scales.dt_s))
        if diagnostic.checkpoint_interval_s > 0
        else None
    )
    next_snapshot = sim.steps if snapshot_stride is not None else None
    next_checkpoint = (
        sim.steps + checkpoint_stride if checkpoint_stride is not None else None
    )
    checkpoint_slot = 0

    def measure():
        row = sim.diagnostics()
        u = sim.velocity_lattice.to_numpy()
        phi = sim.water_phase.to_numpy()
        ids = sim.solid_body_id.to_numpy()
        speed = np.abs(u).sum(axis=-1)
        speed[ids != -1] = 0
        index = np.unravel_index(np.argmax(speed), sim.shape)
        row.update(
            wall_time_s=time.perf_counter() - start,
            peak_cell=[int(i) for i in index],
            peak_phi=float(phi[index]),
            peak_velocity_lattice=u[index].tolist(),
            peak_coverage=float(sim.coverage[index]),
            peak_force_lattice=np.asarray(sim.force_density[index]).tolist(),
            peak_pressure_lattice=float(sim.pressure_lattice[index]),
            maximum_water_speed_lattice=float(
                speed[(ids == -1) & (phi >= 0.5)].max(initial=0)
            ),
            maximum_gas_speed_lattice=float(
                speed[(ids == -1) & (phi < 0.01)].max(initial=0)
            ),
            phase_volume_m3=float(phi[ids == -1].sum(dtype=float) * sim.cell_volume),
        )
        return row

    def snapshot(name):
        np.savez_compressed(
            output / name,
            **sim.snapshot(),
            force_density=sim.force_density.to_numpy(),
            phase_gradient=sim.phase_gradient_lattice.to_numpy(),
            coverage_body_id=sim.coverage_body_id.to_numpy(),
            body_origin=sim.rigid.origin,
            body_quaternion_wxyz=sim.rigid.q,
            body_velocity=sim.rigid.velocity,
            body_omega=sim.rigid.omega,
            body_mass=sim.rigid.mass,
            dx_m=sim.scales.dx_m,
            dt_s=sim.scales.dt_s,
            step=sim.steps,
            time_s=sim.physical_time_s,
        )

    error = None
    with (output / "timeline.jsonl").open("w") as stream:
        try:
            while True:
                if sim.steps >= target:
                    sim.flush_thermal()
                row = measure()
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                record["last_sample"] = row
                write_json(metadata, record)
                now = time.perf_counter()
                if now - last_print >= 15 or sim.steps == target:
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "step",
                                    "time_s",
                                    "wall_time_s",
                                    "maximum_fluid_speed_lattice",
                                    "peak_cell",
                                    "peak_phi",
                                    "water_volume_change_ratio",
                                    "gas_repair_count",
                                )
                            }
                        ),
                        flush=True,
                    )
                    last_print = now
                if next_snapshot is not None and sim.steps >= next_snapshot:
                    snapshot(f"frame_{sim.steps:09d}.npz")
                    next_snapshot += snapshot_stride
                if next_checkpoint is not None and sim.steps >= next_checkpoint:
                    from iceflow3d.checkpoint import save_checkpoint

                    name = f"checkpoint_{checkpoint_slot % 2}.npz"
                    save_checkpoint(sim, output / name)
                    write_json(
                        output / "latest_checkpoint.json",
                        {
                            "path": name,
                            "step": sim.steps,
                            "time_s": sim.physical_time_s,
                        },
                    )
                    checkpoint_slot += 1
                    next_checkpoint += checkpoint_stride
                if sim.steps >= target:
                    record["status"] = "completed"
                    snapshot("final.npz")
                    break
                stop = min(sim.steps + diagnostic.sample_steps, target)
                for scheduled in (next_snapshot, next_checkpoint):
                    if scheduled is not None:
                        stop = min(stop, scheduled)
                sim.step(stop - sim.steps)
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["failure"] = measure()
            snapshot("failure.npz")
            error = exc
        finally:
            record["wall_time_s"] = time.perf_counter() - start
            write_json(metadata, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "error": record.get("error"),
                "wall_time_s": record["wall_time_s"],
            }
        ),
        flush=True,
    )
    if error is not None:
        raise error


if __name__ == "__main__":
    main()
