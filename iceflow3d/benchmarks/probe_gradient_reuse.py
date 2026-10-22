"""Benchmark reusing the last geometry gradient at the next step's entrance.

Only this process installs a wrapper. No production solver files are modified.
This is short-window evidence, not a long-run numerical acceptance test.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d.benchmarks.profile_step import capture, restore, timed_step  # noqa: E402
from iceflow3d.checkpoint import load_checkpoint, source_hashes  # noqa: E402


FIELDS = (
    "water_phase",
    "velocity_lattice",
    "pressure_lattice",
    "water_volume",
    "water_energy",
    "material_mass",
    "material_energy",
    "solid_body_id",
    "body_origin",
    "body_velocity",
    "body_rotation",
    "body_angular_velocity",
)


def snapshot(sim):
    return {name: getattr(sim, name).to_numpy() for name in FIELDS}


def compare(left, right):
    return {
        name: {
            "max_abs_error": float(
                np.max(np.abs(left[name].astype(np.float64) - right[name]))
            ),
            "identical": bool(np.array_equal(left[name], right[name])),
        }
        for name in FIELDS
    }


def install(sim):
    original = sim._gradients
    last_step = None

    def cached():
        nonlocal last_step
        if sim.steps != last_step:
            last_step = sim.steps
            return
        original()

    sim._gradients = cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.repeats < 2:
        parser.error("steps must be positive and repeats must be at least 2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    ti.init(arch=ti.cuda, enable_fallback=False, default_fp=ti.f32)
    sim = load_checkpoint(args.checkpoint, backend="cuda")
    sim.step(32)
    state = capture(sim)
    # Verify that the stored gradient already equals a fresh computation here.
    before = [sim.phase_gradient_lattice.to_numpy(), sim.capillary_stress.to_numpy()]
    sim._gradients()
    cached_equal = [
        np.array_equal(a, field.to_numpy())
        for a, field in zip(before, (sim.phase_gradient_lattice, sim.capillary_stress))
    ]
    del before
    sim.step(args.steps)
    ti.sync()
    restore(sim, state)
    baseline, optimized = [], []
    control_error = variant_error = None
    reference = None
    for repeat in range(args.repeats):
        for variant in (False, True) if repeat % 2 == 0 else (True, False):
            restore(sim, state)
            if variant:
                install(sim)
            try:
                elapsed = timed_step(sim, args.steps)
            finally:
                if variant:
                    del sim._gradients
            (optimized if variant else baseline).append(elapsed)
            if repeat == 0:
                if not variant:
                    reference = snapshot(sim)
                else:
                    variant_error = compare(reference, snapshot(sim))
            if repeat == 1 and not variant:
                control_error = compare(reference, snapshot(sim))
        print(
            f"repeat {repeat + 1}: baseline {baseline[-1]:.4f}, reuse {optimized[-1]:.4f} ms/step",
            flush=True,
        )
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "steps": args.steps,
                "repeats": args.repeats,
                "baseline_ms": baseline,
                "reuse_ms": optimized,
                "speedup": statistics.median(baseline) / statistics.median(optimized),
                "cached_gradient_and_stress_identical": [bool(x) for x in cached_equal],
                "baseline_replay_error": control_error,
                "variant_error": variant_error,
                "source_sha256": hashes,
                "source_unchanged": hashes == source_hashes(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
