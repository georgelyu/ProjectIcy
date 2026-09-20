"""Profile the actual v3 step, excluding initialization, JIT and output I/O.

Run from the repository root, for example:
    python -m iceflow2d_v3.benchmarks.profile_step --output /tmp/step-wall.json
    python -m iceflow2d_v3.benchmarks.profile_step --mode kernels --output /tmp/step-gpu.json

Wall mode synchronizes at stage boundaries and also measures the untouched
step at the same checkpoint. Kernel mode uses Taichi's CUDA event profiler.
Only this benchmark installs wrappers; simulator.py is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
from pathlib import Path
import platform
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

from iceflow2d_v3.config import IceFlowConfig  # noqa: E402
from iceflow2d_v3.simulator import IceFlow2D  # noqa: E402


STAGES = ("_advance_fluid", "_advance_body", "_advance_thermal")


def checkpoint(sim):
    """Capture unique fields and mutable Python scalar state, outside timing."""
    fields, scalars, seen = [], [], set()
    for owner in (sim, sim.thermal):
        for name, value in vars(owner).items():
            if isinstance(value, (ti.Field, ti.MatrixField)) and id(value) not in seen:
                seen.add(id(value))
                fields.append((value, value.to_numpy()))
            elif isinstance(value, (int, float, bool, str, type(None))):
                scalars.append((owner, name, value))
    ti.sync()
    return fields, scalars


def restore(state):
    fields, scalars = state
    for field, values in fields:
        field.from_numpy(values)
    for owner, name, value in scalars:
        setattr(owner, name, value)
    ti.sync()


class StageTimer:
    def __init__(self, sim):
        self.sim = sim
        self.depth = 0
        self.records = defaultdict(list)
        self.originals = {}

    def install(self):
        for name in STAGES:
            original = getattr(self.sim, name)
            self.originals[name] = original
            setattr(self.sim, name, self.wrap(name, original))

    def wrap(self, name, original):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            # Calls inside SLOW belong to SLOW, so rows do not double count.
            if self.depth:
                return original(*args, **kwargs)
            self.depth += 1
            started = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
                ti.sync()
                return result
            finally:
                elapsed = (time.perf_counter_ns() - started) * 1e-6
                self.records[name].append(elapsed)
                self.depth -= 1

        return measured

    def remove(self):
        for name in self.originals:
            delattr(self.sim, name)


def state_summary(sim):
    return {
        "step": sim.steps,
        "physical_time_s": sim.physical_time_s,
        "body_center_cells": np.asarray(sim.body_center[None]).tolist(),
        "body_mass_lattice": float(sim.body_mass_lattice[None]),
        "melted_mass_lattice": float(sim.cumulative_melted_mass_lattice[None]),
    }


def profile_wall(sim, state, steps, repeats):
    baseline, instrumented, grouped = [], [], defaultdict(list)
    # Warm up host-to-device restore kernels and the exact measured window.
    restore(state)
    sim.step(steps)
    ti.sync()
    for repeat in range(repeats):
        # Alternate order to reduce bias from clocks and temperature drift.
        order = (False, True) if repeat % 2 == 0 else (True, False)
        for instrument in order:
            restore(state)
            timer = StageTimer(sim)
            if instrument:
                timer.install()
            try:
                started = time.perf_counter_ns()
                sim.step(steps)
                ti.sync()
                elapsed = (time.perf_counter_ns() - started) * 1e-6
            finally:
                if instrument:
                    timer.remove()
            (instrumented if instrument else baseline).append(elapsed / steps)
            if instrument:
                for name, samples in timer.records.items():
                    grouped[name].append(samples)
        print(
            f"wall repeat {repeat + 1}/{repeats}: baseline={baseline[-1]:.4f}, instrumented={instrumented[-1]:.4f} ms/step",
            flush=True,
        )
    total_stage_ms = sum(
        sum(sum(samples) for samples in batches) for batches in grouped.values()
    )
    rows = []
    for name, batches in grouped.items():
        values = [x for samples in batches for x in samples]
        elapsed = sum(values)
        rows.append(
            {
                "name": name,
                "source_line": inspect.getsourcelines(getattr(type(sim), name))[1],
                "calls": len(values),
                "calls_per_step": len(values) / (steps * repeats),
                "mean_ms_per_call": statistics.mean(values),
                "median_ms_per_call": statistics.median(values),
                "amortized_ms_per_step": elapsed / (steps * repeats),
                "percent_of_stage_time": elapsed / total_stage_ms * 100,
                "repeat_ms_per_step": [sum(samples) / steps for samples in batches],
            }
        )
    rows.sort(key=lambda row: row["amortized_ms_per_step"], reverse=True)
    return {
        "baseline_ms_per_step": baseline,
        "baseline_median_ms_per_step": statistics.median(baseline),
        "instrumented_ms_per_step": instrumented,
        "instrumented_median_ms_per_step": statistics.median(instrumented),
        "stage_total_ms_per_step": total_stage_ms / (steps * repeats),
        "stages": rows,
    }


def profile_kernels(sim, state, steps, repeats):
    # Use the runtime record API that backs Taichi 1.7.4's public print/query
    # helpers; its kernel_time values are milliseconds, not seconds.
    aggregate = defaultdict(lambda: {"calls": 0, "total_ms": 0.0})
    python_calls = Counter()
    elapsed_per_repeat = []
    restore(state)
    sim.step(steps)
    ti.sync()

    def counted(owner, name, key):
        original = getattr(owner, name)

        @functools.wraps(original)
        def invoke(*args, **kwargs):
            python_calls[key] += 1
            return original(*args, **kwargs)

        setattr(owner, name, invoke)

    counted_methods = [(sim, name, "flow." + name) for name in STAGES] + [
        (sim.thermal, name, "thermal." + name)
        for name in ("synchronize_water_aperture", "advance_fast", "advance_slow")
    ]
    for repeat in range(repeats):
        restore(state)
        ti.profiler.clear_kernel_profiler_info()
        for owner, name, key in counted_methods:
            counted(owner, name, key)
        try:
            started = time.perf_counter_ns()
            sim.step(steps)
            ti.sync()
            elapsed_per_repeat.append((time.perf_counter_ns() - started) * 1e-6 / steps)
        finally:
            for owner, name, _ in counted_methods:
                delattr(owner, name)
        program = ti.lang.impl.get_runtime().prog
        program.sync_kernel_profiler()
        program.update_kernel_profiler()
        for record in program.get_kernel_profiler_records():
            # A single Taichi kernel can launch several CUDA tasks.
            name = re.split(r"_c\d+_\d+", record.name, maxsplit=1)[0]
            aggregate[name]["calls"] += 1
            aggregate[name]["total_ms"] += record.kernel_time
        print(
            f"kernel repeat {repeat + 1}/{repeats}: {elapsed_per_repeat[-1]:.4f} ms/step including profiling",
            flush=True,
        )
    total = sum(item["total_ms"] for item in aggregate.values())
    rows = [
        dict(
            name=name,
            **item,
            amortized_ms_per_step=item["total_ms"] / (steps * repeats),
            percent_of_gpu_time=100 * item["total_ms"] / total,
        )
        for name, item in aggregate.items()
    ]
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return {
        "profiled_wall_ms_per_step": elapsed_per_repeat,
        "python_call_counts": dict(python_calls),
        "gpu_kernel_ms_per_step": total / (steps * repeats),
        "kernels": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("wall", "kernels"), default="wall")
    parser.add_argument("--warmup-steps", type=int, default=32)
    parser.add_argument("--start-step", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.warmup_steps) < 1 or args.start_step < 0:
        parser.error(
            "steps, repeats and warmup must be positive; start must be nonnegative"
        )
    ti.init(
        arch=ti.cuda,
        default_fp=ti.f32,
        default_ip=ti.i32,
        enable_fallback=False,
        kernel_profiler=args.mode == "kernels",
    )
    sim = IceFlow2D(IceFlowConfig())
    remaining = max(args.start_step, args.warmup_steps)
    while remaining:
        chunk = min(256, remaining)
        sim.step(chunk)
        remaining -= chunk
        ti.sync()
        print(f"preparing checkpoint: step={sim.steps}", flush=True)
        if args.mode == "kernels":
            ti.profiler.clear_kernel_profiler_info()
    start = state_summary(sim)
    state = checkpoint(sim)
    source = Path(inspect.getfile(IceFlow2D)).resolve()
    output = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "python": sys.version,
        "platform": platform.platform(),
        "taichi_version": list(ti.__version__),
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "solver_source": str(source),
        "solver_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "config": sim.config.to_dict(),
        "start_state": start,
        "steps_per_repeat": args.steps,
        "repeats": args.repeats,
        "measurement": (profile_wall if args.mode == "wall" else profile_kernels)(
            sim, state, args.steps, args.repeats
        ),
        "end_state": state_summary(sim),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for row in output["measurement"].get(
        "stages", output["measurement"].get("kernels", [])
    ):
        print(f"{row['amortized_ms_per_step']:9.5f} ms/step  {row['name']}")
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
