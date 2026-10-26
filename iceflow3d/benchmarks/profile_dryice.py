"""Reproducible profiling of the complete, coupled dry-ice step.

Run from the repository root. --source-root selects a frozen package snapshot;
the benchmark itself stays in the working tree. Initialization, JIT, checkpoint
transfers, diagnostics and output are excluded. Each repeat restores all device
fields and rigid state. A multiple-of-eight block includes every thermal update.
Use separate --mode wall and --mode kernels processes for final measurements.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--mode", choices=("wall", "kernels"), default="kernels")
    parser.add_argument(
        "--case", choices=("bottom", "drop", "moving"), default="bottom"
    )
    parser.add_argument("--resolution", type=int, default=216)
    parser.add_argument("--warmup-steps", type=int, default=63)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--validation-steps", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.steps % 8 or args.warmup_steps % 8 != 7:
        parser.error(
            "steps must be a positive multiple of 8; warmup_steps % 8 must be 7"
        )
    sys.path.insert(0, str(args.source_root.resolve()))

    import numpy as np
    import taichi as ti
    from iceflow3d.benchmarks.profile_step import capture, restore, timed_step
    from iceflow3d.dryice.simulator import DryIceFlow3D
    from iceflow3d.examples.dry_ice_in_hot_water_3d import build_parser, create_config

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        default_fp=ti.f32,
        offline_cache=True,
        kernel_profiler=args.mode == "kernels",
    )
    example_args = build_parser().parse_args(["--resolution", str(args.resolution)])
    example_args.initial_position = "above" if args.case == "drop" else "bottom"
    config = create_config(example_args)
    if args.case == "moving":
        body = config.bodies[0]
        config = replace(
            config,
            ice_bodies=(
                replace(
                    body,
                    center_cells=(
                        body.center_cells[0],
                        body.center_cells[1] + 4,
                        body.center_cells[2],
                    ),
                    quaternion_wxyz=(np.cos(0.08), 0.0, np.sin(0.08), 0.0),
                    velocity_lattice=(0.003, 0.0, 0.002),
                    angular_velocity_lattice=(0.0, 0.0002, 0.0),
                ),
            ),
        )
    package = args.source_root.resolve() / "iceflow3d"
    sources = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*package.glob("*.py"), *package.joinpath("dryice").glob("*.py")]
    }
    started = time.perf_counter()
    sim = DryIceFlow3D(config)
    sim.step(args.warmup_steps)
    state = capture(sim)
    cache_valid = sim._gradient_cache_valid

    def reset():
        restore(sim, state)
        sim._gradient_cache_valid = cache_valid
        # Scratch boundary classifications may have changed during the repeat.
        sim._classify_home()
        ti.sync()

    # Compile every branch exercised by the measurement before any timer.
    sim.step(args.steps)
    reset()
    print(
        f"ready {args.case}, {sim.shape}, setup {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    times, full_thermal = [], []
    kernels = defaultdict(lambda: [0, 0.0])
    resources = {}
    for repeat in range(args.repeats):
        reset()
        if args.mode == "kernels":
            ti.profiler.clear_kernel_profiler_info()
        times.append(timed_step(sim, args.steps))
        assert sim._thermal_pending == 7
        if args.mode == "kernels":
            program = ti.lang.impl.get_runtime().prog
            program.sync_kernel_profiler()
            program.update_kernel_profiler()
            for record in program.get_kernel_profiler_records():
                name = re.split(r"_c\d+_\d+", record.name, maxsplit=1)[0]
                kernels[name][0] += 1
                kernels[name][1] += record.kernel_time
                resources[record.name] = {
                    key: getattr(record, key)
                    for key in (
                        "block_size",
                        "grid_size",
                        "register_per_thread",
                        "shared_mem_per_block",
                        "active_blocks_per_multiprocessor",
                    )
                }
        reset()
        full_thermal.append(timed_step(sim, 1))
        assert sim._thermal_pending == 0
        print(
            f"repeat {repeat + 1}: average {times[-1]:.4f}, thermal step {full_thermal[-1]:.4f} ms",
            flush=True,
        )
    result = dict(
        mode=args.mode,
        case=args.case,
        source_root=str(args.source_root.resolve()),
        configuration=config.to_dict(),
        source_sha256=sources,
        dt_s=sim.scales.dt_s,
        warmup_steps=args.warmup_steps,
        steps_per_repeat=args.steps,
        repeats=args.repeats,
        wall_ms_per_step=times,
        wall_median_ms=statistics.median(times),
        full_thermal_step_wall_ms=full_thermal,
        full_thermal_step_wall_median_ms=statistics.median(full_thermal),
        kernel_resources=resources,
        kernels=sorted(
            [
                dict(
                    name=k,
                    calls_per_step=v[0] / (args.steps * args.repeats),
                    gpu_ms_per_step=v[1] / (args.steps * args.repeats),
                )
                for k, v in kernels.items()
            ],
            key=lambda r: -r["gpu_ms_per_step"],
        ),
    )
    if args.validation_steps:
        reset()
        sim.step(args.validation_steps)
        sim.flush_thermal()
        result["validation"] = sim.diagnostics()
        result["validation_steps"] = args.validation_steps
        np.savez_compressed(
            args.output.with_suffix(".npz"),
            phi=sim.water_phase.to_numpy(),
            velocity=sim.velocity_lattice.to_numpy(),
            solid_id=sim.solid_body_id.to_numpy(),
            water_energy=sim.water_energy.to_numpy(),
            water_volume=sim.water_volume.to_numpy(),
            body_origin=sim.rigid.origin,
            body_q=sim.rigid.q,
            body_velocity=sim.rigid.velocity,
            body_mass=sim.rigid.mass,
        )
    assert sources == {
        name: hashlib.sha256((package / name).read_bytes()).hexdigest()
        for name in sources
    }, "source changed while benchmark was running"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["kernels"][:12], indent=2), flush=True)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
