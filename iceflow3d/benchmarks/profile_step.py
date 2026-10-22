"""Reproducible profiling of the current coupled IceFlow3D.step.

Examples (run from the repository root):
    python iceflow3d/benchmarks/profile_step.py --mode wall --output /tmp/wall.json
    python iceflow3d/benchmarks/profile_step.py --mode kernels --output /tmp/gpu.json
    python iceflow3d/benchmarks/profile_step.py --mode host --output /tmp/host.json

All repeats start from the same complete in-memory checkpoint. Initialization,
phase relaxation, JIT, checkpoint transfers, diagnostics and output are untimed.
Wall mode compares the untouched step with nested synchronized method timers;
their overhead is reported explicitly. CUDA event times and cProfile attribution
are separate modes: host waits must not be added to GPU execution times.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import cProfile
import ctypes
from dataclasses import replace
from datetime import datetime, timezone
import functools
import io
import json
from pathlib import Path
import platform
import pstats
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import taichi as ti

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iceflow3d import IceFlow3D  # noqa: E402
from iceflow3d.checkpoint import _fields, load_checkpoint, source_hashes  # noqa: E402
from iceflow3d.examples.dambreak_four_ice_melting_3d import (  # noqa: E402
    create_config,
    parse_args,
)


METHODS = (
    "_gradients",
    "_gradients_regular",
    "_gradients_ice",
    "_refill_gradients",
    "_classify_refill_gradients",
    "_refill_gradients_regular",
    "_refill_gradients_ice",
    "_collide_phase",
    "_forces",
    "_classify_home",
    "_forces_bulk",
    "_forces_tank",
    "_forces_boundary",
    "_collide_momentum",
    "_stream_home",
    "_stream_home_bulk",
    "_stream_home_shared",
    "_stream_home_boundary",
    "_repair_phase_layout",
    "_recover_pressure",
    "_recover_phase",
    "check_stability",
    "_scan_stability",
    "_refresh_geometry",
    "_rasterize",
    "_seed_refill",
    "_find_refill_changes",
    "_seed_refill_region",
    "_propagate_refill",
    "_accept_refill_iteration",
    "_commit_refill",
    "_project_refill",
    "_refill_forces",
    "_write_refill_equilibrium",
    # Global phase-volume correction kernels are currently commented out.
    # "_conserve_phase_volume",
    # "_measure_phase_balance",
    # "_project_phase_volume",
    "_advance_advection",
    "_measure_outflow",
    "_thermal_fluxes",
    "_apply_thermal_fluxes",
    "_synchronize_water",
    "_build_water_targets",
    "_commit_water_targets",
    "_advance_conduction",
    "_body_conduction",
    "_exchange_heat",
    "_apply_melting",
    "_reduce_mass",
    "_check_heat",
)
RIGID_METHODS = ("advance", "upload", "update_mass", "resolve_contacts")
SCALARS = ("steps", "_thermal_pending", "pending_cells")


def capture(sim):
    fields = [(field, field.to_numpy()) for field in _fields(sim).values()]
    rigid = {
        k: v.copy() for k, v in vars(sim.rigid).items() if isinstance(v, np.ndarray)
    }
    scalars = {key: getattr(sim, key) for key in SCALARS}
    ti.sync()
    return fields, rigid, scalars


def restore(sim, state):
    fields, rigid, scalars = state
    for field, array in fields:
        field.from_numpy(array)
    for key, array in rigid.items():
        setattr(sim.rigid, key, array.copy())
    for key, value in scalars.items():
        setattr(sim, key, value)
    ti.sync()


class StageTimer:
    def __init__(self, sim):
        self.sim = sim
        self.stack = []
        self.records = defaultdict(lambda: [0, 0.0, 0.0])
        self.originals = []

    def install(self):
        for owner, names, prefix in (
            (self.sim, METHODS, "flow"),
            (self.sim.rigid, RIGID_METHODS, "rigid"),
        ):
            for name in names:
                if not hasattr(owner, name):
                    continue  # Allow the same timer to inspect frozen older solvers.
                original = getattr(owner, name)
                self.originals.append((owner, name, name in vars(owner), original))
                setattr(owner, name, self.wrap(f"{prefix}.{name}", original))

    def wrap(self, name, original):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            path = tuple(frame[0] for frame in self.stack) + (name,)
            frame = [name, 0.0]
            self.stack.append(frame)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                ti.sync()
                elapsed = (time.perf_counter() - start) * 1000
                record = self.records[path]
                record[0] += 1
                record[1] += elapsed
                record[2] += elapsed - frame[1]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed

        return measured

    def remove(self):
        for owner, name, existed, original in reversed(self.originals):
            if existed:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def timed_step(sim, steps):
    ti.sync()
    start = time.perf_counter()
    sim.step(steps)
    ti.sync()
    return (time.perf_counter() - start) * 1000 / steps


def wall_profile(sim, state, steps, repeats):
    baseline, instrumented = [], []
    timer = StageTimer(sim)
    for repeat in range(repeats):
        for instrument in (False, True) if repeat % 2 == 0 else (True, False):
            restore(sim, state)
            if instrument:
                timer.install()
            try:
                elapsed = timed_step(sim, steps)
            finally:
                if instrument:
                    timer.remove()
                    timer.originals.clear()
            (instrumented if instrument else baseline).append(elapsed)
        print(
            f"repeat {repeat + 1}: baseline {baseline[-1]:.4f}, stages {instrumented[-1]:.4f} ms/step",
            flush=True,
        )
    denominator = steps * repeats
    details = [
        {
            "path": list(path),
            "calls_per_step": row[0] / denominator,
            "inclusive_ms_per_step": row[1] / denominator,
            "exclusive_ms_per_step": row[2] / denominator,
        }
        for path, row in timer.records.items()
    ]
    methods = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in details:
        aggregate = methods[row["path"][-1]]
        for i, key in enumerate(
            ("calls_per_step", "inclusive_ms_per_step", "exclusive_ms_per_step")
        ):
            aggregate[i] += row[key]
    return {
        "baseline_ms_per_step": baseline,
        "baseline_median_ms_per_step": statistics.median(baseline),
        "instrumented_ms_per_step": instrumented,
        "instrumentation_overhead_percent": (
            statistics.median(instrumented) / statistics.median(baseline) - 1
        )
        * 100,
        "unattributed_ms_per_step": statistics.mean(instrumented)
        - sum(row["inclusive_ms_per_step"] for row in details if len(row["path"]) == 1),
        "stages": sorted(
            (r for r in details if len(r["path"]) == 1),
            key=lambda r: -r["inclusive_ms_per_step"],
        ),
        "details": sorted(details, key=lambda r: -r["inclusive_ms_per_step"]),
        "methods": sorted(
            [
                {
                    "name": name,
                    "calls_per_step": row[0],
                    "inclusive_ms_per_step": row[1],
                    "exclusive_ms_per_step": row[2],
                }
                for name, row in methods.items()
            ],
            key=lambda r: -r["exclusive_ms_per_step"],
        ),
    }


def condition_device(sim):
    # Restoring a multi-GB state can leave SM clocks idle. Only overwrite
    # collision caches that step() recomputes before reading; physical inputs
    # and the pending thermal interval remain exactly those of the snapshot.
    for _ in range(32):
        sim._collide_momentum()
    ti.sync()


def kernel_profile(sim, state, steps, repeats, *, condition=False):
    aggregate = defaultdict(lambda: [0, 0.0])
    tasks = defaultdict(lambda: [0, 0.0])
    resources = {}
    elapsed = []
    for repeat in range(repeats):
        restore(sim, state)
        if condition:
            condition_device(sim)
        ti.profiler.clear_kernel_profiler_info()
        elapsed.append(timed_step(sim, steps))
        # Taichi 1.7.4 runtime records backing the public profiler. Times are ms.
        program = ti.lang.impl.get_runtime().prog
        program.sync_kernel_profiler()
        program.update_kernel_profiler()
        for record in program.get_kernel_profiler_records():
            tasks[record.name][0] += 1
            tasks[record.name][1] += record.kernel_time
            name = re.split(r"_c\d+_\d+", record.name, maxsplit=1)[0]
            aggregate[name][0] += 1
            aggregate[name][1] += record.kernel_time
            resources[record.name] = {
                "name": name,
                **{
                    key: getattr(record, key)
                    for key in (
                        "block_size",
                        "grid_size",
                        "register_per_thread",
                        "shared_mem_per_block",
                        "active_blocks_per_multiprocessor",
                    )
                },
            }
        print(
            f"repeat {repeat + 1}: profiled wall {elapsed[-1]:.4f} ms/step", flush=True
        )
    total = sum(row[1] for row in aggregate.values())
    return {
        "profiled_wall_ms_per_step": elapsed,
        "gpu_ms_per_step": total / (steps * repeats),
        "launches_per_step": sum(row[0] for row in aggregate.values())
        / (steps * repeats),
        "kernel_resources": list(resources.values()),
        "kernel_tasks": [
            dict(
                name=k,
                launches_per_step=v[0] / (steps * repeats),
                gpu_ms_per_step=v[1] / (steps * repeats),
            )
            for k, v in sorted(tasks.items(), key=lambda item: -item[1][1])
        ],
        "kernels": sorted(
            [
                {
                    "name": name,
                    "launches_per_step": row[0] / (steps * repeats),
                    "gpu_ms_per_step": row[1] / (steps * repeats),
                    "gpu_percent": 100 * row[1] / total,
                }
                for name, row in aggregate.items()
            ],
            key=lambda r: -r["gpu_ms_per_step"],
        ),
    }


def host_profile(sim, state, steps, repeats, output):
    profiler = cProfile.Profile()
    elapsed = []
    for _ in range(repeats):
        restore(sim, state)
        profiler.enable()
        elapsed.append(timed_step(sim, steps))
        profiler.disable()
    profiler.dump_stats(str(output.with_suffix(".prof")))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(60)
    output.with_suffix(".txt").write_text(stream.getvalue())
    rows = [
        {
            "file": key[0],
            "line": key[1],
            "function": key[2],
            "calls_per_step": value[1] / (steps * repeats),
            "self_ms_per_step": value[2] * 1000 / (steps * repeats),
            "cumulative_ms_per_step": value[3] * 1000 / (steps * repeats),
        }
        for key, value in stats.stats.items()
    ]
    return {
        "profiled_wall_ms_per_step": elapsed,
        "functions": sorted(rows, key=lambda r: -r["cumulative_ms_per_step"]),
    }


def cuda_capture(sim, steps):
    # CUDA driver profiler API lets nsys/ncu capture only the warmed step window.
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuProfilerStart() != 0:
        raise RuntimeError("cuProfilerStart failed")
    try:
        elapsed = timed_step(sim, steps)
    finally:
        if cuda.cuProfilerStop() != 0:
            raise RuntimeError("cuProfilerStop failed")
    return {"profiled_wall_ms_per_step": [elapsed]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("wall", "kernels", "host", "capture"), default="wall"
    )
    parser.add_argument("--resolution-x", type=int, default=450)
    parser.add_argument("--interface-width", type=float, default=5)
    parser.add_argument(
        "--momentum-collision-model",
        choices=("cm_mrt",),
        help="override the case/checkpoint model before compiling collision",
    )
    parser.add_argument("--phase-warmup-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--allow-checkpoint-source-change",
        action="store_true",
        help="explicitly profile a historical checkpoint against the current source",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.repeats <= 0
        or args.warmup_steps < 8
        or args.phase_warmup_steps < 0
    ):
        parser.error("steps/repeats must be positive; warmup >=8; phase warmup >=0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    ti.init(
        arch=ti.cuda,
        enable_fallback=False,
        default_fp=ti.f32,
        offline_cache=True,
        kernel_profiler=args.mode == "kernels",
    )
    started = time.perf_counter()
    checkpoint_metadata = None
    if args.checkpoint:
        with np.load(args.checkpoint, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
        checkpoint_metadata = {
            "collision_model": metadata["configuration"]["momentum_collision_model"],
            "source_sha256": metadata["source_sha256"],
            "changed_source_files": sorted(
                key
                for key in set(hashes) | set(metadata["source_sha256"])
                if hashes.get(key) != metadata["source_sha256"].get(key)
            ),
            "source_change_allowed": args.allow_checkpoint_source_change,
        }
        sim = load_checkpoint(
            args.checkpoint,
            backend="cuda",
            allow_source_change=args.allow_checkpoint_source_change,
        )
        if args.momentum_collision_model:
            # load_checkpoint initializes/restores fields without any momentum
            # collision. Its ti.static model branches are first compiled below.
            sim.config = replace(
                sim.config, momentum_collision_model=args.momentum_collision_model
            )
    else:
        case_args = [
            "--resolution-x",
            str(args.resolution_x),
            "--interface-width",
            str(args.interface_width),
            "--phase-warmup-steps",
            str(args.phase_warmup_steps),
        ]
        if args.momentum_collision_model:
            case_args += ["--momentum-collision-model", args.momentum_collision_model]
        sim = IceFlow3D(create_config(parse_args(case_args)))
    sim.step(args.warmup_steps)
    ti.sync()
    state = capture(sim)
    # Warm every conditional path reached in the full measured window, then rewind.
    sim.step(args.steps)
    ti.sync()
    restore(sim, state)
    print(
        f"ready: shape={sim.shape}, model={sim.config.momentum_collision_model}, "
        f"start step={sim.steps}, setup={time.perf_counter() - started:.2f}s",
        flush=True,
    )
    start_state = {
        "step": sim.steps,
        "physical_time_s": sim.physical_time_s,
        "ice_mass_kg": (sim.rigid.mass * sim.scales.mass_kg).tolist(),
    }
    if args.mode == "wall":
        measurement = wall_profile(sim, state, args.steps, args.repeats)
    elif args.mode == "kernels":
        measurement = kernel_profile(sim, state, args.steps, args.repeats)
    elif args.mode == "host":
        measurement = host_profile(sim, state, args.steps, args.repeats, args.output)
    else:
        measurement = cuda_capture(sim, args.steps)
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
    )
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "taichi": list(ti.__version__),
        "numpy": np.__version__,
        "gpu": gpu.stdout.strip(),
        "mode": args.mode,
        "steps_per_repeat": args.steps,
        "repeats": 1 if args.mode == "capture" else args.repeats,
        "source_sha256": hashes,
        "source_unchanged": hashes == source_hashes(),
        "configuration": sim.config.to_dict(),
        "dt_s": sim.scales.dt_s,
        "start_state": start_state,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_metadata": checkpoint_metadata,
        "measurement": measurement,
        "final_diagnostics": sim.diagnostics(),
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
