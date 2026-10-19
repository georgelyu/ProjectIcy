"""Profile the actual v3 step, excluding initialization, JIT and output I/O.

Run from the repository root, for example:
    python -m iceflow2d_v3.benchmarks.profile_step --output /tmp/step-wall.json
    python -m iceflow2d_v3.benchmarks.profile_step --case single-ice-dambreak --output /tmp/single-wall.json
    python -m iceflow2d_v3.benchmarks.profile_step --mode kernels --output /tmp/step-gpu.json

Wall mode synchronizes at nested method boundaries and also measures the
untouched step at the same checkpoint. Inclusive times contain child calls;
exclusive times can be added without double counting. Kernel mode uses
Taichi's CUDA event profiler.
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
from iceflow2d_v3.body_state import BodyField  # noqa: E402
from iceflow2d_v3.simulator import IceFlow2D  # noqa: E402


STAGES = ("_prepare_body_motion", "_advance_fluid", "_advance_body", "_advance_thermal")
FLOW_METHODS = STAGES + (
    "_collide_momentum",
    "_prepare_phase_gradients",
    "_prepare_phase_gradients_kernel",
    "_prepare_capillary_stress",
    "_cache_stability_velocity",
    "_collide_phase",
    "_measure_phase_mass_flux",
    "_collide_momentum_kernel",
    "_stream",
    "_update_streamed_macroscopic_fields",
    "_bound_water_phase",
    "_update_pressure",
    "_integrate_rigid_ice",
    "_refresh_fluid_domain",
    "_solve_contact_and_rasterize",
    "_calculate_moving_body_fraction",
    "_prepare_moving_body_contact_support_kernel",
    "_project_moving_body_inside_container",
    "_commit_moving_body_raster",
    "_update_solid_mask",
    "_prepare_phase_geometry",
    "_refill_connected_nodes",
    "_seed_connected_refill",
    "_propagate_connected_refill",
    "_commit_connected_refill",
    "_refill_changed_nodes",
    "_advance_moving_thermal_fast",
    "_measure_water_outflow_rate",
    "_advance_moving_thermal_slow",
    "_refresh_eroded_body",
    "_update_moving_body_mass_properties",
    "_reduce_moving_body_mass_properties",
    "_apply_moving_body_mass_properties",
)
THERMAL_METHODS = (
    "advance_fast",
    "advance_slow",
    "synchronize_water_aperture",
    "_build_phase_aperture_targets",
    "_apply_phase_aperture_transfer",
    "_compute_water_advection_flux_x",
    "_compute_water_advection_flux_y",
    "_update_water_advection",
    "_accumulate_body_conduction",
    "_compute_water_conduction_flux_x",
    "_compute_water_conduction_flux_y",
    "_update_water_conduction",
    "_exchange_interface_heat",
    "_apply_body_heat_and_phase_change",
)


def owners(sim):
    """Include nested solver storage, without traversing back references."""
    return [("flow", sim), ("thermal", sim.thermal)] + [
        (prefix, owner)
        for prefix, owner in (
            ("d2q5", sim.thermal._d2q5),
            ("contacts", sim.contacts),
            ("body_motion", sim.body_motion),
        )
        if owner is not None
    ]


def measured_methods(sim):
    for prefix, owner in owners(sim):
        names = {
            "flow": FLOW_METHODS,
            "thermal": THERMAL_METHODS,
            "d2q5": ("collide", "stream", "synchronize"),
            "contacts": ("detect", "project"),
            "body_motion": (
                "prepare", "solve", "check", "commit", "commit_raster",
                "refresh_material_bounds",
            ),
        }[prefix]
        for name in names:
            yield owner, name, f"{prefix}.{name}"


def checkpoint(sim):
    """Capture unique fields and mutable Python scalar state, outside timing."""
    fields, scalars, seen = [], [], set()
    for _, owner in owners(sim):
        for name, value in vars(owner).items():
            if isinstance(value, BodyField):
                value = value.field
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
        self.stack = []
        self.records = defaultdict(list)
        self.exclusive_records = defaultdict(list)
        self.sources = {}
        self.originals = []

    def install(self):
        for owner, name, key in measured_methods(self.sim):
            original = getattr(owner, name)
            source = inspect.unwrap(getattr(type(owner), name))
            self.originals.append((owner, name, name in vars(owner), original))
            self.sources[key] = {
                "source_file": inspect.getsourcefile(source),
                "source_line": inspect.getsourcelines(source)[1],
            }
            setattr(owner, name, self.wrap(key, original, source))

    def wrap(self, name, original, source):
        @functools.wraps(source)
        def measured(*args, **kwargs):
            path = (*[frame[0] for frame in self.stack], name)
            frame = [name, 0.0]
            self.stack.append(frame)
            started = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
                ti.sync()
                return result
            finally:
                elapsed = (time.perf_counter_ns() - started) * 1e-6
                self.records[path].append(elapsed)
                self.exclusive_records[path].append(elapsed - frame[1])
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed

        return measured

    def remove(self):
        for owner, name, had_override, original in reversed(self.originals):
            if had_override:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def state_summary(sim):
    return {
        "step": sim.steps,
        "physical_time_s": sim.physical_time_s,
        "body_center_cells": np.asarray(sim.body_center.to_numpy()).tolist(),
        "body_mass_lattice": sim.body_mass_lattice.to_numpy().tolist(),
        "melted_mass_lattice": sim.cumulative_melted_mass_lattice.to_numpy().tolist(),
    }


def profile_wall(sim, state, steps, repeats):
    baseline, instrumented, grouped = [], [], defaultdict(list)
    exclusive = defaultdict(list)
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
                    exclusive[name].extend(timer.exclusive_records[name])
        print(
            f"wall repeat {repeat + 1}/{repeats}: baseline={baseline[-1]:.4f}, instrumented={instrumented[-1]:.4f} ms/step",
            flush=True,
        )
    total_stage_ms = sum(
        sum(sum(samples) for samples in batches)
        for path, batches in grouped.items()
        if len(path) == 1
    )
    rows = []
    for path, batches in grouped.items():
        values = [x for samples in batches for x in samples]
        elapsed = sum(values)
        rows.append(
            {
                "name": path[-1],
                "path": list(path),
                "depth": len(path) - 1,
                **timer.sources[path[-1]],
                "calls": len(values),
                "calls_per_step": len(values) / (steps * repeats),
                "mean_ms_per_call": statistics.mean(values),
                "median_ms_per_call": statistics.median(values),
                "amortized_ms_per_step": elapsed / (steps * repeats),
                "exclusive_ms_per_step": sum(exclusive[path]) / (steps * repeats),
                "exclusive_percent_of_stage_time": sum(exclusive[path])
                / total_stage_ms
                * 100,
                "percent_of_stage_time": elapsed / total_stage_ms * 100,
                "repeat_ms_per_step": [sum(samples) / steps for samples in batches],
            }
        )
    rows.sort(key=lambda row: row["amortized_ms_per_step"], reverse=True)
    methods = defaultdict(lambda: {"calls": 0, "inclusive": 0.0, "exclusive": 0.0})
    for row in rows:
        method = methods[row["name"]]
        method["calls"] += row["calls"]
        method["inclusive"] += row["amortized_ms_per_step"]
        method["exclusive"] += row["exclusive_ms_per_step"]
    return {
        "baseline_ms_per_step": baseline,
        "baseline_median_ms_per_step": statistics.median(baseline),
        "instrumented_ms_per_step": instrumented,
        "instrumented_median_ms_per_step": statistics.median(instrumented),
        "stage_total_ms_per_step": total_stage_ms / (steps * repeats),
        "instrumentation_overhead_percent": (
            statistics.median(instrumented) / statistics.median(baseline) - 1
        )
        * 100,
        "unattributed_ms_per_step": statistics.mean(instrumented)
        - total_stage_ms / (steps * repeats),
        "stages": [
            dict(row, name=row["name"].removeprefix("flow."))
            for row in rows
            if row["depth"] == 0
        ],
        "details": rows,
        "methods": sorted(
            [
                {
                    "name": name,
                    **timer.sources[name],
                    "calls": item["calls"],
                    "calls_per_step": item["calls"] / (steps * repeats),
                    "inclusive_ms_per_step": item["inclusive"],
                    "exclusive_ms_per_step": item["exclusive"],
                    "exclusive_percent_of_stage_time": item["exclusive"]
                    / (total_stage_ms / (steps * repeats))
                    * 100,
                }
                for name, item in methods.items()
            ],
            key=lambda row: row["exclusive_ms_per_step"],
            reverse=True,
        ),
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

        @functools.wraps(inspect.unwrap(getattr(type(owner), name)))
        def invoke(*args, **kwargs):
            python_calls[key] += 1
            return original(*args, **kwargs)

        setattr(owner, name, invoke)

    counted_methods = list(measured_methods(sim))
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
            cuda_tasks_per_step=item["calls"] / (steps * repeats),
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
    parser.add_argument(
        "--case", choices=("default", "single-ice-dambreak"), default="default"
    )
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
    if args.case == "single-ice-dambreak":
        from iceflow2d_v3.examples.dambreak_single_ice_melting_2d import create_config

        config = create_config()
    else:
        config = IceFlowConfig()
    print(
        f"Initializing {args.case}: grid={config.resolution}, phase warmup={config.phase_warmup_steps}",
        flush=True,
    )
    sim = IceFlow2D(config)
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
        "case": args.case,
        "timing_notes": {
            "excluded": "initialization, phase warmup, JIT, checkpoint copies and output I/O",
            "wall": "perf_counter_ns with ti.sync after every measured method; inclusive parents contain children; only exclusive times may be summed across depths",
            "kernels": "CUDA event task times; excludes Python dispatch, CPU work and host synchronization; profiler adds overhead",
        },
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
    measurement = output["measurement"]
    if args.mode == "wall":
        print(
            f"Uninstrumented step: {measurement['baseline_median_ms_per_step']:.5f} ms/step (median); "
            f"instrumentation overhead: {measurement['instrumentation_overhead_percent']:.1f}%"
        )
    for row in output["measurement"].get(
        "stages", output["measurement"].get("kernels", [])
    ):
        print(f"{row['amortized_ms_per_step']:9.5f} ms/step  {row['name']}")
    if args.mode == "wall":
        print("Per-method exclusive times (including synchronization):")
        for row in measurement["methods"]:
            print(
                f"{row['exclusive_ms_per_step']:9.5f} ms/step  "
                f"{row['calls_per_step']:5.3f} calls/step  {row['name']}"
            )
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
