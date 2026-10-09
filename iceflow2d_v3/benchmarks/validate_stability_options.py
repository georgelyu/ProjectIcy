"""Run each stability option independently on the unchanged W=5 four-ice case.

Example: python -m iceflow2d_v3.benchmarks.validate_stability_options
    --variant hrr --output /tmp/four-ice-hrr --end 5
Failures are recorded with completed=false, not presented as successful runs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.benchmarks.diagnose_four_ice_instability import fields, metrics
from iceflow2d_v3.examples.four_falling_ice_melting_2d import create_config, parse_args
from iceflow2d_v3.simulator import IceFlow2D, _MAX_D2Q9_LATTICE_VELOCITY_L1

VARIANTS = {
    "baseline": [],
    "hrr": ["--momentum-collision-model", "cm_hrr"],
    "consistent_flux": ["--phase-momentum-coupling", "consistent_flux"],
    "refill": ["--exposed-node-refill", "non_equilibrium"],
    "lagged_phase_only": ["--phase-momentum-coupling", "consistent_flux"],
}


@ti.data_oriented
class LaggedPhaseOnly(IceFlow2D):
    """Ablation: same phase timing as consistent_flux, but no flux force."""

    @ti.func
    def _phase_flux_force_density(self, i, j):
        return ti.Vector.zero(ti.f32, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--end", type=float, default=5.0)
    p.add_argument("--sample-steps", type=int, default=600)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = create_config(parse_args(VARIANTS[args.variant]))
    assert config.interface_width == 5 and config.resolution == (450, 150)
    paths = [
        Path(__file__).parents[1] / name
        for name in (
            "config.py",
            "simulator.py",
            "stability_options.py",
            "lattice.py",
            "thermal_d2q5.py",
        )
    ]
    manifest = {
        "variant": args.variant,
        "config": config.to_dict(),
        "speed_guard_l1": _MAX_D2Q9_LATTICE_VELOCITY_L1,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
        "requested_end_s": args.end,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    solver = LaggedPhaseOnly if args.variant == "lagged_phase_only" else IceFlow2D
    s = solver(config)
    s._thermal_stability_check_enabled = True
    end_step = round(args.end / s.scales.dt_s)
    rows, error = [], None
    started = time.monotonic()
    with (args.output / "metrics.jsonl").open("w") as out:
        while True:
            d = fields(s)
            row = metrics(s, d, s.steps + (1 if error else 0))
            active = (d["solid"] == 0) & (d["wall"] == 0)
            gas = active & (d["phi"] < 0.01)
            row["gas_mean_phi"] = float(d["phi"][gas].mean()) if gas.any() else None
            row["temperature_min_c"] = float(d["temperature"][active].min())
            row["temperature_max_c"] = float(d["temperature"][active].max())
            rows.append(row)
            row["phase_relative_change"] = (
                row["phase_amount"] / rows[0]["phase_amount"] - 1
            )
            out.write(json.dumps(row) + "\n")
            out.flush()
            if len(rows) == 1 or s.steps % 6000 == 0 or error or s.steps >= end_step:
                print(json.dumps(row), flush=True)
            if error or s.steps >= end_step:
                np.savez_compressed(args.output / "final.npz", **d)
                break
            try:
                s.step(min(args.sample_steps, end_step - s.steps))
            except RuntimeError as exc:
                error = str(exc)
    summary = {
        "variant": args.variant,
        "completed": error is None and s.steps == end_step,
        "requested_end_s": args.end,
        "error": error,
        "last": rows[-1],
        "peak_sampled_l1": max(r["max_l1"] for r in rows),
        "max_abs_phase_relative_change": max(
            abs(r["phase_relative_change"]) for r in rows
        ),
        "elapsed_s": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
