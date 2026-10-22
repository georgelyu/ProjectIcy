"""Run the physical case and save render geometry inputs on exact sample steps."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import os

import numpy as np


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_npz(path, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--resolution-x", type=int, default=450)
    p.add_argument("--end-time-s", type=float, default=5.0)
    p.add_argument("--output-interval-s", type=float, default=0.02)
    p.add_argument("--checkpoint-interval-s", type=float, default=0.1)
    p.add_argument("--resume", type=Path)
    p.add_argument("--allow-source-change", action="store_true")
    p.add_argument("--phase-warmup-steps", type=int, default=2000)
    p.add_argument("--interface-width", type=float, default=5.0)
    p.add_argument("--body-contact-iterations", type=int, help="explicit contact iteration budget, including on checkpoint resume")
    p.add_argument("--time-step-scale", type=float, default=1.0, help="physical timestep multiplier; material properties and grid/interface widths remain fixed")
    p.add_argument("--geometry-only", action="store_true", help="omit the unused 3D velocity field from render snapshots")
    p.add_argument(
        "--momentum-collision-model", choices=("cm_mrt", "cm_hrr"), default=None
    )
    args = p.parse_args()
    source = args.source_root or Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(source.resolve()))
    from iceflow3d import IceFlow3D
    from iceflow3d.checkpoint import load_checkpoint, save_checkpoint, source_hashes
    from iceflow3d.examples.dambreak_four_ice_melting_3d import (
        create_config,
        parse_args,
    )
    from iceflow3d.rigid import rotation_matrix

    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("fields", "checkpoints", "sessions"):
        (root / folder).mkdir(exist_ok=True)
    path = root / "simulation.json"
    if path.exists() and args.resume is None:
        raise FileExistsError(
            f"{path}: choose a new directory or an explicit checkpoint"
        )
    previous = json.loads(path.read_text()) if path.exists() else None
    started = time.perf_counter()
    session = {
        "started_utc": utc_now(),
        "pid": os.getpid(),
        "source_root": str(source.resolve()),
        "source_sha256": source_hashes(),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "timing_s": {
            "initialization": 0.0,
            "stepping": 0.0,
            "diagnostics": 0.0,
            "field_output": 0.0,
            "checkpoint_output": 0.0,
        },
    }
    case_args = [
        "--resolution-x",
        str(args.resolution_x),
        "--phase-warmup-steps",
        str(args.phase_warmup_steps),
        "--interface-width",
        str(args.interface_width),
    ]
    if args.momentum_collision_model is not None:
        case_args += ["--momentum-collision-model", args.momentum_collision_model]
    config = create_config(parse_args(case_args))
    if not math.isfinite(args.time_step_scale) or args.time_step_scale <= 0:
        raise ValueError("time-step scale must be finite and positive")
    config = replace(config, reference_lattice_velocity=config.reference_lattice_velocity * args.time_step_scale)
    if args.body_contact_iterations is not None:
        config = replace(config, body_contact_iterations=args.body_contact_iterations)
    record = {
        "status": "initializing",
        "configuration": config.to_dict(),
        "sessions": previous["sessions"] if previous else [],
        "frames": previous["frames"] if previous else [],
        "frame_interval_s": args.output_interval_s,
        "requested_end_time_s": args.end_time_s,
        "source_sha256": session["source_sha256"],
    }
    record["sessions"].append(session)
    session_path = root / "sessions" / f"session_{len(record['sessions']):03d}.json"
    write_json(path, record)
    sim = None
    try:
        sim = (
            load_checkpoint(args.resume, allow_source_change=args.allow_source_change)
            if args.resume
            else IceFlow3D(config)
        )
        if args.resume:
            old_budget = sim.config.body_contact_iterations
            if args.body_contact_iterations is None:
                config = replace(config, body_contact_iterations=old_budget)
            else:
                sim.config = replace(sim.config, body_contact_iterations=args.body_contact_iterations)
                session["contact_iteration_budget_change"] = {
                    "from": old_budget, "to": args.body_contact_iterations,
                    "note": "Iteration cap only; contact residual tolerances and physical parameters unchanged.",
                }
            record["configuration"] = config.to_dict()
        if sim.config.to_dict() != config.to_dict():
            raise ValueError("checkpoint configuration differs from requested case")
        session["timing_s"]["initialization"] = time.perf_counter() - started
        session["start_step"] = sim.steps
        dt = sim.scales.dt_s
        end = round(args.end_time_s / dt)
        stride = round(args.output_interval_s / dt)
        checkpoint_stride = round(args.checkpoint_interval_s / dt)
        if min(end, stride, checkpoint_stride) < 1 or end % stride:
            raise ValueError(
                "end time must be an exact multiple of the output interval"
            )
        if not np.isclose(stride * dt, args.output_interval_s, rtol=0, atol=1e-12):
            raise ValueError(
                "output interval does not align with the physical timestep"
            )
        record.update(
            status="running",
            target_steps=end,
            dt_s=dt,
            dx_m=sim.scales.dx_m,
            frame_count=end // stride + 1,
            current_step=sim.steps,
        )
        if args.resume:
            stale = [f for f in record["frames"] if f["step"] > sim.steps]
            archive = root / "sessions" / f"superseded_{len(record['sessions']):03d}"
            if stale:
                archive.mkdir(exist_ok=True)
                for frame in stale:
                    old = root / frame["path"]
                    if old.exists():
                        old.replace(archive / old.name)
            record["frames"] = [f for f in record["frames"] if f["step"] <= sim.steps]
            session["restart_sha256"] = hashlib.sha256(
                args.resume.read_bytes()
            ).hexdigest()

        def render_fields(include_velocity=False):
            fields = {
                "water_phase": sim.water_phase.to_numpy(),
                "solid_body_id": sim.solid_body_id.to_numpy().astype(np.int8),
                "ice_material_fraction": (
                    sim.material_mass.to_numpy() / sim.initial_cell_mass
                ).astype(np.float32),
                "material_size": sim.material_size.to_numpy(),
                "body_origin_m": sim.rigid.origin * sim.scales.dx_m,
                "body_rotation": np.array([rotation_matrix(q) for q in sim.rigid.q]),
                "body_quaternion_wxyz": sim.rigid.q.copy(),
                "body_mass_kg": sim.rigid.mass * sim.scales.mass_kg,
                "body_velocity_m_s": sim.rigid.velocity * sim.scales.velocity_m_s,
                "body_angular_velocity_s_1": sim.rigid.omega / dt,
                "dx_m": sim.scales.dx_m,
                "dt_s": dt,
                "step": sim.steps,
                "time_s": sim.physical_time_s,
                "domain_size_m": config.domain_size_m,
                "boundary_cells": config.boundary_cells,
            }
            if include_velocity or not args.geometry_only:
                fields["velocity_m_s"] = sim.velocity_lattice.to_numpy() * sim.scales.velocity_m_s
            return fields

        next_frame = ((sim.steps + stride - 1) // stride) * stride
        next_checkpoint = (
            sim.steps
            if not args.resume
            else (sim.steps // checkpoint_stride + 1) * checkpoint_stride
        )
        checkpoint_slot = len(record.get("checkpoints", []))
        timeline = root / "timeline.jsonl"
        if args.resume and timeline.exists():
            old_rows = [json.loads(line) for line in timeline.read_text().splitlines()]
            timeline.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in old_rows
                    if row["step"] < sim.steps
                )
            )
        with timeline.open("a") as stream:
            while True:
                if sim.steps == end:
                    sim.flush_thermal()
                if sim.steps == next_frame:
                    t0 = time.perf_counter()
                    row = sim.diagnostics()
                    if not all(np.isfinite(value).all() for value in row.values()):
                        raise RuntimeError("nonfinite diagnostic")
                    session["timing_s"]["diagnostics"] += time.perf_counter() - t0
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    t0 = time.perf_counter()
                    fields = render_fields()
                    if not all(np.isfinite(value).all() for value in fields.values()):
                        raise RuntimeError("nonfinite render field")
                    index = sim.steps // stride
                    name = f"fields/frame_{index:06d}.npz"
                    save_npz(root / name, fields)
                    frame = {
                        "index": index,
                        "step": sim.steps,
                        "time_s": sim.physical_time_s,
                        "path": name,
                        "sha256": hashlib.sha256(
                            (root / name).read_bytes()
                        ).hexdigest(),
                    }
                    record["frames"] = [
                        f for f in record["frames"] if f["step"] != sim.steps
                    ] + [frame]
                    record["last_diagnostics"] = row
                    session["timing_s"]["field_output"] += time.perf_counter() - t0
                    next_frame += stride
                    print(
                        json.dumps(
                            {
                                "frame": index,
                                "step": sim.steps,
                                "time_s": sim.physical_time_s,
                                "elapsed_s": time.perf_counter() - started,
                                "peak_speed_lattice": row["peak_fluid_speed_lattice"],
                                "phase_volume_relative_error": row[
                                    "phase_volume_change_ratio"
                                ],
                            }
                        ),
                        flush=True,
                    )
                if sim.steps == next_checkpoint or sim.steps == end:
                    t0 = time.perf_counter()
                    name = f"checkpoints/checkpoint_{checkpoint_slot % 2}.npz"
                    save_checkpoint(sim, root / name)
                    checkpoint_slot += 1
                    record["latest_checkpoint"] = {
                        "path": name,
                        "step": sim.steps,
                        "time_s": sim.physical_time_s,
                        "source_sha256": source_hashes(),
                    }
                    session["timing_s"]["checkpoint_output"] += time.perf_counter() - t0
                    next_checkpoint += checkpoint_stride
                record.update(
                    current_step=sim.steps, current_time_s=sim.physical_time_s
                )
                session["wall_time_s"] = time.perf_counter() - started
                write_json(path, record)
                if sim.steps == end:
                    break
                stop = min(end, next_frame, next_checkpoint, sim.steps + 120)
                t0 = time.perf_counter()
                try:
                    sim.step(stop - sim.steps)
                finally:
                    session["timing_s"]["stepping"] += time.perf_counter() - t0
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        if sim is not None and "render_fields" in locals():
            record.update(current_step=sim.steps, current_time_s=sim.physical_time_s)
            # A partly executed failure step is never used as a restart state.
            save_npz(root / "failure_fields.npz", render_fields(include_velocity=True))
        raise
    finally:
        session["ended_utc"] = utc_now()
        session["wall_time_s"] = time.perf_counter() - started
        session["end_step"] = sim.steps if sim is not None else 0
        session["status"] = record["status"]
        record["total_simulation_wall_time_s"] = sum(
            item.get("wall_time_s", 0) for item in record["sessions"]
        )
        write_json(session_path, session)
        write_json(path, record)
        write_json(
            root / "timing.json",
            {
                "status": record["status"],
                "total_simulation_wall_time_s": record["total_simulation_wall_time_s"],
                "includes": "initialization, phase warmup, physical steps, diagnostics, field and checkpoint output",
                "excludes": "meshing and Blender rendering",
                "sessions": record["sessions"],
            },
        )


if __name__ == "__main__":
    main()
