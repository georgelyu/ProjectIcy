"""Isolate phase-balance reduction cost without changing solver precision.

Synthetic inputs; this is a microbenchmark, not a coupled-step speedup claim.
Compare explicit atomic_add, +=, and a two-pass FP64 reduction. Partial sums
use strided reads so adjacent lanes still read adjacent cells.
"""

import argparse
import json
from pathlib import Path
import statistics
import time
from collections import defaultdict

import numpy as np
import taichi as ti


@ti.data_oriented
class Probe:
    def __init__(self, n, lanes):
        self.n, self.lanes = n, lanes
        self.phi = ti.field(ti.f32, shape=n)
        self.ids = ti.field(ti.i32, shape=n)
        self.total = ti.field(ti.f64, shape=())
        self.capacity = ti.field(ti.f64, shape=())
        self.partial = ti.Vector.field(2, ti.f64, shape=lanes)

    @ti.kernel
    def initialize(self):
        for i in self.phi:
            self.phi[i] = ti.cast(i % 1024, ti.f32) / 1024
            self.ids[i] = -1 if i % 17 != 0 else -2

    @ti.kernel
    def atomic(self):
        self.total[None], self.capacity[None] = 0.0, 0.0
        for i in self.phi:
            if self.ids[i] == -1:
                p = ti.cast(self.phi[i], ti.f64)
                ti.atomic_add(self.total[None], p)
                ti.atomic_add(self.capacity[None], p * (1 - p))

    @ti.kernel
    def augmented(self):
        self.total[None], self.capacity[None] = 0.0, 0.0
        for i in self.phi:
            if self.ids[i] == -1:
                p = ti.cast(self.phi[i], ti.f64)
                self.total[None] += p
                self.capacity[None] += p * (1 - p)

    @ti.kernel
    def partials(self):
        for lane in range(self.lanes):
            total, capacity = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
            for k in range((self.n + self.lanes - 1) // self.lanes):
                i = lane + k * self.lanes
                if i < self.n and self.ids[i] == -1:
                    p = ti.cast(self.phi[i], ti.f64)
                    total += p
                    capacity += p * (1 - p)
            self.partial[lane] = ti.Vector([total, capacity])

    @ti.kernel
    def finish(self):
        self.total[None], self.capacity[None] = 0.0, 0.0
        for lane in self.partial:
            self.total[None] += self.partial[lane][0]
            self.capacity[None] += self.partial[lane][1]

    def staged(self):
        self.partials()
        self.finish()

    @ti.kernel
    def finish_serial(self):
        total, capacity = ti.cast(0.0, ti.f64), ti.cast(0.0, ti.f64)
        ti.loop_config(serialize=True)
        for lane in range(self.lanes):
            total += self.partial[lane][0]
            capacity += self.partial[lane][1]
        self.total[None], self.capacity[None] = total, capacity

    def staged_serial(self):
        self.partials()
        self.finish_serial()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=6075000)
    parser.add_argument("--lanes", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.cells, args.lanes, args.steps, args.repeats) <= 0:
        parser.error("cells, lanes, steps and repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ti.init(arch=ti.cuda, enable_fallback=False, kernel_profiler=True)
    probe = Probe(args.cells, args.lanes)
    probe.initialize()
    arrays = probe.phi.to_numpy().astype(np.float64)
    arrays = arrays[probe.ids.to_numpy() == -1]
    reference = np.array([arrays.sum(), (arrays * (1 - arrays)).sum()])
    results = []
    for name in ("atomic", "augmented", "staged", "staged_serial"):
        function = getattr(probe, name)
        for _ in range(10):
            function()
        ti.sync()
        wall, gpu = [], []
        for _ in range(args.repeats):
            ti.profiler.clear_kernel_profiler_info()
            started = time.perf_counter()
            for _ in range(args.steps):
                function()
            ti.sync()
            wall.append((time.perf_counter() - started) * 1000 / args.steps)
            gpu.append(ti.profiler.get_kernel_profiler_total_time() * 1000 / args.steps)
        actual = np.array([probe.total[None], probe.capacity[None]])
        per_kernel = defaultdict(float)
        for record in ti.lang.impl.get_runtime().prog.get_kernel_profiler_records():
            per_kernel[record.name] += record.kernel_time / args.steps
        result = {
            "method": name,
            "wall_ms": wall,
            "gpu_ms": gpu,
            "median_gpu_ms": statistics.median(gpu),
            "reference": reference.tolist(),
            "actual": actual.tolist(),
            "max_abs_error": float(np.max(np.abs(actual - reference))),
            "last_repeat_kernel_ms": dict(per_kernel),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "cells": args.cells,
                "lanes": args.lanes,
                "steps": args.steps,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
