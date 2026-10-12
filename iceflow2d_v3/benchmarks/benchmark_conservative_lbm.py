"""Reproducible CUDA timing and independent mass audit of the dual-LBM paths.

All field downloads/reductions are outside timed batches. Run one mode per
process, on an otherwise idle GPU, using the same preparation and sample count.
"""

import argparse
import json
import hashlib
from pathlib import Path
import statistics
import time

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as single
from iceflow2d_v3.examples import four_falling_ice_melting_2d as four
from iceflow2d_v3.benchmarks.benchmark_solver import field_inventory


def audit(s):
    mode = getattr(s.config, "water_mass_conservation", "legacy")
    if mode == "conservative_lbm":
        water = s.phase_populations.to_numpy().sum(dtype=np.float64)
        phase = s.water_phase.to_numpy()
        capacity = s.conservative_phase.capacity.to_numpy()
        active = capacity > 1e-14
        ice_volume = s.conservative_phase.ice_volume.to_numpy().sum(dtype=np.float64)
    else:
        phase = s.water_phase.to_numpy()
        active = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
        water = phase[active].sum(dtype=np.float64)
        ice_volume = None
    ice = s.thermal.body_solid_mass.to_numpy().sum(dtype=np.float64) / (s.config.rho_water * s.config.dx**2)
    return dict(water_volume_cells=float(water), ice_mass_water_equivalent_cells=float(ice), total_water_equivalent_cells=float(water + ice), projected_ice_volume_cells=ice_volume, phase_min=float(phase[active].min()), phase_max=float(phase[active].max()))


def inventory(s):
    result = field_inventory(s)
    if hasattr(s, "conservative_phase"):
        for name, field in vars(s.conservative_phase).items():
            if isinstance(field, (ti.Field, ti.MatrixField)):
                count = getattr(field, "n", 1) * getattr(field, "m", 1)
                for size in field.shape:
                    count *= size
                result.append(dict(name="conservative_phase." + name, bytes=count * int(str(field.dtype)[1:]) // 8))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("legacy", "conservative_lbm"), default="conservative_lbm")
    parser.add_argument("--scenario", choices=("single", "four"), default="single")
    parser.add_argument("--prepare-steps", type=int, default=512)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--audit-steps", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare_steps < 32 or args.steps < 1 or args.repeats < 1 or args.audit_steps < 0:
        parser.error("prepare>=32, steps/repeats>=1 and audit>=0 required")
    example = four if args.scenario == "four" else single
    argv = ["--phase-momentum-coupling", "consistent_flux"]
    if args.scenario == "single":
        argv += ["--resolution-x", "100", "--resolution-y", "200"]
    cfg = example.create_config(example.parse_args(argv))
    source_dir = Path(__file__).resolve().parents[1]
    source_hashes = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("simulator.py", "config.py", "lattice.py", "conservative_phase.py")
        if (source_dir / name).exists()
    }
    # Also runs from the archived pre-change source to time the actual baseline.
    if hasattr(cfg, "water_mass_conservation"):
        cfg.water_mass_conservation = args.mode
    elif args.mode != "legacy":
        parser.error("this checkout does not implement conservative_lbm")
    s = IceFlow2D(cfg)
    initial = audit(s)
    print("initial", initial, flush=True)
    history = []
    remaining = args.prepare_steps
    while remaining:
        count = min(1024, remaining)
        s.step(count)
        remaining -= count
        if args.prepare_steps > 1024:
            print("prepared", s.steps, flush=True)
    ti.sync()
    prepared = audit(s)
    durations = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        s.step(args.steps)
        ti.sync()
        durations.append((time.perf_counter() - started) / args.steps)
    history.append(dict(step=s.steps, **audit(s)))
    fields = inventory(s)
    baseline = initial["total_water_equivalent_cells"]
    result = dict(mode=args.mode, scenario=args.scenario, grid=cfg.resolution,
                  preparation_steps=args.prepare_steps, timed_steps_per_repeat=args.steps,
                  repeats=args.repeats, seconds_per_step=durations,
                  median_seconds_per_step=statistics.median(durations),
                  field_payload_bytes=sum(f["bytes"] for f in fields), fields=fields,
                  initial=initial, prepared=prepared, history=history,
                  source_sha256=source_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["maximum_sampled_relative_mass_error"] = max(
            abs(r["total_water_equivalent_cells"] / baseline - 1)
            for r in [prepared, *history])
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    # Keep the completed timing even if a subsequent long audit fails.
    save()
    print("timing", result["median_seconds_per_step"], flush=True)
    remaining = args.audit_steps
    while remaining:
        count = min(4096, remaining)
        try:
            s.step(count)
        except Exception as exc:
            result["audit_failure"] = dict(step=s.steps, error=str(exc))
            save()
            raise
        remaining -= count
        row = dict(step=s.steps, **audit(s))
        history.append(row)
        save()
        print("audit", row, flush=True)
        if not np.isfinite(s.momentum_populations.to_numpy()).all():
            raise RuntimeError("nonfinite momentum populations")
    if any(hashlib.sha256((source_dir / name).read_bytes()).hexdigest() != digest for name, digest in source_hashes.items()):
        raise RuntimeError("solver source changed during benchmark")
    save()
    print(json.dumps({k: v for k, v in result.items() if k not in ("fields", "history")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
