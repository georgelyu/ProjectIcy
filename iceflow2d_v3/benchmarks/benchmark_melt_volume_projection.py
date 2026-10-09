"""Time the reference correction and a device-only, known-component fast path.

This is a microbenchmark, not a production conservation implementation. The
GPU path handles one capacity-sufficient layer. Component identification and
the target/seed construction are outside its timing; the CPU reference still
does its original connected-component labeling. No solver kernels are edited.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy import ndimage
import taichi as ti

from iceflow2d_v3.benchmarks.check_simple_melt_volume import (
    _apply_phase,
    correct_full_melt_volume,
)
from iceflow2d_v3.examples.four_falling_ice_melting_2d import create_config, parse_args
from iceflow2d_v3.simulator import IceFlow2D
from iceflow2d_v3.test_iceflow import _falling_config


@ti.data_oriented
class DeviceProjection:
    def __init__(self, s, seeds, component, target):
        self.s = s
        self.nx, self.ny = s.nx, s.ny
        self.lanes = min(2048, s.nx * s.ny)
        shape = (s.nx, s.ny)
        self.seeds = ti.field(ti.i8, shape=shape)
        self.component = ti.field(ti.i8, shape=shape)
        self.bands = ti.field(ti.i8, shape=shape)
        self.candidate = ti.ndarray(ti.f32, shape=shape)
        self.saved_phi = ti.field(ti.f32, shape=shape)
        self.saved_h = ti.field(ti.f32, shape=(*shape, 9))
        self.saved_g = ti.field(ti.f32, shape=(*shape, 9))
        self.target = ti.field(ti.f64, shape=())
        self.before = ti.field(ti.f64, shape=())
        self.after = ti.field(ti.f64, shape=())
        self.capacity_minus = ti.field(ti.f64, shape=())
        self.capacity_plus = ti.field(ti.f64, shape=())
        self.repair_index = ti.field(ti.i32, shape=())
        self.failed = ti.field(ti.i32, shape=())
        self.seeds.from_numpy(seeds.astype(np.int8))
        self.component.from_numpy(component.astype(np.int8))
        self.target[None] = target
        self.capture()

    @ti.kernel
    def capture(self):
        for i, j in self.saved_phi:
            self.saved_phi[i, j] = self.s.water_phase[i, j]
            for q in range(9):
                self.saved_h[i, j, q] = self.s.phase_populations[i, j, q]
                self.saved_g[i, j, q] = self.s.momentum_populations[i, j, q]

    @ti.kernel
    def restore(self):
        for i, j in self.saved_phi:
            self.s.water_phase[i, j] = self.saved_phi[i, j]
            for q in range(9):
                self.s.phase_populations[i, j, q] = self.saved_h[i, j, q]
                self.s.momentum_populations[i, j, q] = self.saved_g[i, j, q]

    @ti.func
    def seeded_phase(self, i, j):
        value = self.s.water_phase[i, j]
        if self.seeds[i, j] != 0:
            value = 1.0
        return value

    @ti.func
    def prepare_cell(self, i, j, candidate: ti.template()):
        values = ti.Vector.zero(ti.f64, 3)
        phi = self.seeded_phase(i, j)
        candidate[i, j] = phi
        bits = 0
        if self.s._is_fluid_cell(i, j):
            values[0] = ti.cast(phi, ti.f64)
            near_air = False
            near_component = False
            for di, dj in ti.static(((1, 0), (-1, 0), (0, 1), (0, -1))):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.nx and 0 <= nj < self.ny:
                    if self.s._is_fluid_cell(ni, nj):
                        neighbor = self.seeded_phase(ni, nj)
                        near_air = near_air or neighbor == 0.0
                        near_component = near_component or (
                            self.component[ni, nj] != 0 and neighbor > 0.0
                        )
            partial = self.component[i, j] != 0 and 0.0 < phi < 1.0
            if self.component[i, j] != 0 and phi > 0 and (partial or near_air):
                bits += 1
                values[1] = ti.cast(phi, ti.f64)
            if partial or (phi == 0.0 and near_component):
                bits += 2
                values[2] = 1.0 - ti.cast(phi, ti.f64)
        self.bands[i, j] = ti.cast(bits, ti.i8)
        return values

    @ti.kernel
    def prepare(self, candidate: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        self.before[None] = 0.0
        self.capacity_minus[None] = 0.0
        self.capacity_plus[None] = 0.0
        self.failed[None] = 0
        # Explicit lane-local accumulation bounds contended f64 atomic writes
        # by the lane count, instead of updating shared scalars for every cell.
        # Adjacent lanes read adjacent cells on each pass.
        for lane in range(self.lanes):
            values = ti.Vector.zero(ti.f64, 3)
            for chunk in range((self.nx * self.ny + self.lanes - 1) // self.lanes):
                k = lane + chunk * self.lanes
                if k < self.nx * self.ny:
                    values += self.prepare_cell(k // self.ny, k % self.ny, candidate)
            self.before[None] += values[0]
            self.capacity_minus[None] += values[1]
            self.capacity_plus[None] += values[2]

    @ti.kernel
    def project(self, candidate: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        difference = self.target[None] - self.before[None]
        increasing = difference >= 0
        capacity = self.capacity_minus[None]
        if increasing:
            capacity = self.capacity_plus[None]
        fraction = ti.min(1.0, ti.abs(difference) / ti.max(capacity, 1e-30))
        if ti.abs(difference) > capacity + 1e-8:
            self.failed[None] = 1
        self.after[None] = 0.0
        self.repair_index[None] = 2 * self.nx * self.ny
        for lane in range(self.lanes):
            total = ti.cast(0.0, ti.f64)
            minimum_rank = 2 * self.nx * self.ny
            for chunk in range((self.nx * self.ny + self.lanes - 1) // self.lanes):
                k = lane + chunk * self.lanes
                if k < self.nx * self.ny:
                    i, j = k // self.ny, k % self.ny
                    if self.s._is_fluid_cell(i, j):
                        phi = ti.cast(candidate[i, j], ti.f64)
                        active = self.bands[i, j] & 1 != 0
                        if increasing:
                            active = self.bands[i, j] & 2 != 0
                        if active:
                            if increasing:
                                phi += fraction * (1.0 - phi)
                            else:
                                phi *= 1.0 - fraction
                        new = ti.cast(phi, ti.f32)
                        candidate[i, j] = new
                        total += ti.cast(new, ti.f64)
                        if active and 0 < new < 1:
                            rank = k
                            if not (0.1 < new < 0.9):
                                rank += self.nx * self.ny
                            minimum_rank = ti.min(minimum_rank, rank)
            self.after[None] += total
            ti.atomic_min(self.repair_index[None], minimum_rank)

    @ti.kernel
    def close_rounding(self, candidate: ti.types.ndarray(dtype=ti.f32, ndim=2)):
        residual = self.target[None] - self.after[None]
        index = self.repair_index[None]
        if self.nx * self.ny <= index < 2 * self.nx * self.ny:
            index -= self.nx * self.ny
        if index < self.nx * self.ny:
            i, j = index // self.ny, index % self.ny
            adjusted = ti.cast(candidate[i, j], ti.f64) + residual
            if 0 <= adjusted <= 1:
                candidate[i, j] = ti.cast(adjusted, ti.f32)
            else:
                self.failed[None] = 2
        elif ti.abs(residual) > 1e-7:
            self.failed[None] = 3

    def run(self):
        self.prepare(self.candidate)
        self.project(self.candidate)
        self.close_rounding(self.candidate)
        _apply_phase(self.s, self.candidate)


def stats(values):
    a = np.asarray(values)
    return {
        "median_ms": float(np.median(a)),
        "p10_ms": float(np.percentile(a, 10)),
        "p90_ms": float(np.percentile(a, 90)),
        "samples_ms": a.tolist(),
    }


def make_fixture(kind, nx, ny):
    boundary = 2 if nx == 32 else (3 if nx == 450 else 6)
    s = IceFlow2D(
        _falling_config(
            resolution=(nx, ny),
            boundary_cells=boundary,
            water_width_fraction=(nx - boundary + 0.5) / nx,
            water_height_fraction=(ny / 2 + 0.5) / ny,
            ice_width_fraction=8.5 / nx,
            ice_height_fraction=8.5 / ny,
            ice_base_y_cells=(30.0 if kind == "sharp_air" else 8.0)
            if nx == 32
            else 20.0,
            ice_initial_angle=0.0,
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            sigma=0.0,
        )
    )
    seeds = s.solid_mask.to_numpy().astype(bool)
    s.thermal._body_heat_delta.fill(0)
    s.thermal.body_sensible_energy.fill(
        2 * s.thermal._initial_body_cell_mass * s.thermal._latent_heat_j_kg
    )
    s.thermal._apply_body_heat_and_phase_change()
    s._refresh_eroded_body()
    fluid = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
    if kind == "saved_w5":
        p = Path(
            "iceflow2d_v3/outputs/water_volume_budget_w5_20260922/consistent_flux/final.npz"
        )
        phi = np.load(p)["phi"]
        if nx == 900:
            phi = phi.repeat(2, axis=0).repeat(2, axis=1)
        _apply_phase(s, np.ascontiguousarray(phi))
    source = s.water_phase.to_numpy()
    seeded = source.copy()
    seeded[seeds] = 1
    labels, _ = ndimage.label(
        fluid & (seeded > 0), ndimage.generate_binary_structure(2, 1)
    )
    component = fluid & np.isin(labels, np.unique(labels[seeds]))
    return s, source, fluid, seeds, component


def measure_case(kind, nx, ny, count):
    s, source, fluid, seeds, component = make_fixture(kind, nx, ny)
    seeded = source.copy()
    seeded[seeds] = 1
    target = float(seeded[fluid].sum(dtype=np.float64)) - 5.312
    p = DeviceProjection(s, seeds, component, target)
    reference, ref_info = correct_full_melt_volume(source, fluid, seeds, target)
    _apply_phase(s, reference)  # compile the NumPy argument specialization
    for _ in range(25):
        p.restore()
        p.run()
    ti.sync()
    result = s.water_phase.to_numpy()
    validation = {
        "water_residual_cells": float(result[fluid].sum(dtype=np.float64) - target),
        "min_phi": float(result[fluid].min()),
        "max_phi": float(result[fluid].max()),
        "device_error_flag": int(p.failed[None]),
        "reference_interface_layers": ref_info["interface_layers"],
        "changed_cells": int(np.count_nonzero(result[fluid] != source[fluid])),
    }
    assert validation["device_error_flag"] == 0
    assert abs(validation["water_residual_cells"]) <= 1e-7
    assert 0 <= validation["min_phi"] <= validation["max_phi"] <= 1
    np.testing.assert_array_equal(
        result[fluid & ~component], source[fluid & ~component]
    )
    s._update_streamed_macroscopic_fields()
    np.testing.assert_array_equal(s.water_phase.to_numpy(), result)
    # Also verify the opposite sign, outside the timed negative-correction loop.
    positive_target = float(seeded[fluid].sum(dtype=np.float64)) + 5.312
    p.target[None] = positive_target
    p.restore()
    p.run()
    ti.sync()
    positive = s.water_phase.to_numpy()
    validation["positive_water_residual_cells"] = float(
        positive[fluid].sum(dtype=np.float64) - positive_target
    )
    assert p.failed[None] == 0
    assert abs(validation["positive_water_residual_cells"]) <= 1e-7
    assert positive[fluid].min() >= 0 and positive[fluid].max() <= 1
    p.target[None] = target
    gpu, cpu, download, projection, upload = [], [], [], [], []
    for _ in range(count):
        p.restore()
        ti.sync()
        start = time.perf_counter()
        p.run()
        ti.sync()
        gpu.append((time.perf_counter() - start) * 1000)
        p.restore()
        ti.sync()
        start = time.perf_counter()
        phase = s.water_phase.to_numpy()
        first = time.perf_counter()
        corrected, _ = correct_full_melt_volume(phase, fluid, seeds, target)
        second = time.perf_counter()
        _apply_phase(s, corrected)
        ti.sync()
        end = time.perf_counter()
        cpu.append((end - start) * 1000)
        download.append((first - start) * 1000)
        projection.append((second - first) * 1000)
        upload.append((end - second) * 1000)
    row = {
        "kind": kind,
        "grid": [nx, ny],
        "grid_cells": nx * ny,
        "fluid_cells": int(fluid.sum()),
        "melted_cells": int(seeds.sum()),
        "known_component_cells": int(component.sum()),
        "validation": validation,
        "device_only_known_component": stats(gpu),
        "cpu_reference_end_to_end": stats(cpu),
        "cpu_reference_parts": {
            "download": stats(download),
            "numpy_scipy": stats(projection),
            "upload_and_population_lift": stats(upload),
        },
    }
    print(
        json.dumps(
            {
                "kind": kind,
                "grid": [nx, ny],
                "validation": validation,
                "gpu_ms": row["device_only_known_component"]["median_ms"],
                "cpu_ms": row["cpu_reference_end_to_end"]["median_ms"],
            }
        ),
        flush=True,
    )
    return row


def baseline(prepare_steps, repeats, batch):
    config = create_config(
        parse_args(
            ["--interface-width", "5", "--phase-momentum-coupling", "consistent_flux"]
        )
    )
    s = IceFlow2D(config)
    s._thermal_stability_check_enabled = True
    s.step(prepare_steps)
    ti.sync()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        s.step(batch)
        ti.sync()
        timings.append((time.perf_counter() - start) * 1000 / batch)
    return {
        "grid": list(config.resolution),
        "phase_momentum_coupling": "consistent_flux",
        "width": 5,
        "thermal_interval_steps": config.thermal.update_interval_lbm_steps,
        "start_time_s": prepare_steps * s.scales.dt_s,
        "end_time_s": s.physical_time_s,
        "steps_per_batch": batch,
        "per_lbm_step": stats(timings),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--prepare-steps", type=int, default=18000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
        text=True,
    )
    print("Measuring actual coupled W=5 LBM step...", flush=True)
    base = baseline(args.prepare_steps, 5, 512)
    print(json.dumps({"baseline": base}), flush=True)
    rows = []
    for kind, nx, ny in (
        ("sharp_pool", 32, 48),
        ("sharp_air", 32, 48),
        ("sharp_pool", 450, 150),
        ("saved_w5", 450, 150),
        ("saved_w5", 900, 300),
    ):
        print(f"Measuring {kind} {nx}x{ny}...", flush=True)
        rows.append(measure_case(kind, nx, ny, args.samples))
        (args.output / "partial.json").write_text(
            json.dumps({"baseline": base, "rows": rows}, indent=2) + "\n"
        )
    step_ms = base["per_lbm_step"]["median_ms"]
    for row in rows:
        if row["grid"] == [450, 150]:
            row["estimated_extra_step_time_percent"] = {
                name: {
                    str(interval): 100 * row[name]["median_ms"] / (interval * step_ms)
                    for interval in (1, 8)
                }
                for name in ("device_only_known_component", "cpu_reference_end_to_end")
            }
    result = {
        "scope": "Isolated correction event timings; one capacity-sufficient layer; known GPU component; no thermal remap; no production integration or multi-GPU scaling test.",
        "timing": "Warmed, restore excluded; each event synchronized before and after; CPU path includes phi download, original CPU labeling/projection, upload and h/g lift. Masks and target cached on both paths.",
        "gpu": gpu_info.strip(),
        "python": platform.python_version(),
        "taichi": ti.__version__,
        "source_sha256": {
            str(Path(__file__)): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "iceflow2d_v3/simulator.py": hashlib.sha256(
                Path("iceflow2d_v3/simulator.py").read_bytes()
            ).hexdigest(),
        },
        "baseline": base,
        "rows": rows,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"results": str(args.output / "results.json"), "baseline_ms": step_ms}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
