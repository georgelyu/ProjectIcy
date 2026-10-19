"""Compare the production cut-link ratio with the former [0.05, 0.95] clamp.

Both modes start the default four-ice dam break from its initial condition and
use the production step() and its per-step stability guard. Only the diagnostic
subclass restores the old clamp. Sparse samples include the raw crossing range;
they are not an exhaustive record of every link at every time step.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import taichi as ti

from iceflow2d_v3.examples.dambreak_four_ice_melting_2d import create_config
from iceflow2d_v3.simulator import IceFlow2D


@ti.data_oriented
class ClampedCutLinks(IceFlow2D):
    @ti.func
    def _cut_link_fraction(self, i, j, ni, nj):
        return ti.min(
            0.95, ti.max(0.05, IceFlow2D._cut_link_fraction(self, i, j, ni, nj))
        )


def crossing_statistics(s, active, solid):
    coverage = s._body_fractions.to_numpy()
    owner = s.solid_body_id.to_numpy()
    i, j = np.nonzero(active)
    fractions = []
    threshold = s.config.thermal.solid_liquid_threshold
    for di, dj in (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
    ):
        ni, nj = i + di, j + dj
        inside = (ni >= 0) & (ni < s.nx) & (nj >= 0) & (nj < s.ny)
        fi, fj, ni, nj = i[inside], j[inside], ni[inside], nj[inside]
        cut = solid[ni, nj] == 1
        fi, fj, ni, nj = fi[cut], fj[cut], ni[cut], nj[cut]
        b = np.maximum(owner[ni, nj], 0)
        df = np.maximum(
            (threshold - coverage[b, fi, fj]) * s.config.dx, s.config.dx * 1e-6
        )
        ds = (threshold - coverage[b, ni, nj]) * s.config.dx
        fractions.append(df / (df - ds + 1e-12))
    values = np.concatenate(fractions)
    return {
        "cut_links": int(values.size),
        "raw_q_min": float(values.min()) if values.size else None,
        "raw_q_max": float(values.max()) if values.size else None,
        "links_below_005": int(np.count_nonzero(values < 0.05)),
        "links_above_095": int(np.count_nonzero(values > 0.95)),
        "invalid_q_count": int(
            np.count_nonzero(~np.isfinite(values) | (values < 0) | (values > 1))
        ),
    }


def sample(s):
    fields = {
        name: getattr(s, name).to_numpy()
        for name in (
            "water_phase",
            "solid_mask",
            "wall_mask",
            "physical_velocity_lattice",
            "pressure_lattice",
            "temperature",
            "thermal_enthalpy",
            "momentum_populations",
            "phase_populations",
        )
    }
    active = (fields["solid_mask"] == 0) & (fields["wall_mask"] == 0)
    phi = fields["water_phase"]
    u = fields["physical_velocity_lattice"].astype(float)
    speed_l1 = np.abs(u).sum(axis=-1)
    speed_m_s = np.linalg.norm(u, axis=-1) * s.scales.velocity_scale_m_s
    ranked = np.where(active, speed_l1, -np.inf)
    peak = np.unravel_index(int(np.argmax(ranked)), ranked.shape)
    water, air = active & (phi >= 0.5), active & (phi < 0.5)
    body = {
        "body_mass_lattice": s.body_mass_lattice.array(),
        "body_center": s.body_center.to_numpy(),
        "body_velocity_lattice": s.body_velocity.to_numpy(),
        "body_angle": s.body_angle.to_numpy(),
        "body_angular_velocity_lattice": s.body_angular_velocity.to_numpy(),
    }
    invalid = [
        name for name, values in fields.items() if not np.isfinite(values[active]).all()
    ]
    invalid += [name for name, values in body.items() if not np.isfinite(values).all()]
    row = {
        "completed_step": s.steps,
        "completed_time_s": s.physical_time_s,
        "finite": not invalid,
        "nonfinite_fields": invalid,
        "speed_l1_max_lattice": float(ranked[peak]),
        "peak_cell": [int(v) for v in peak],
        "peak_phi": float(phi[peak]),
        "water_max_speed_m_s": float(speed_m_s[water].max()) if water.any() else 0.0,
        "air_max_speed_m_s": float(speed_m_s[air].max()) if air.any() else 0.0,
        "phi_min": float(phi[active].min()),
        "phi_max": float(phi[active].max()),
        "temperature_min_c": float(fields["temperature"][active].min()),
        "temperature_max_c": float(fields["temperature"][active].max()),
        **{name: values.tolist() for name, values in body.items()},
        **crossing_statistics(s, active, fields["solid_mask"]),
    }
    return row, fields | body


def run(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = (
        list(root.glob("*.py"))
        + list((root / "examples").glob("*.py"))
        + [Path(__file__).resolve()]
    )
    hashes = {}
    for path in sources:
        data = path.read_bytes()
        relative = path.relative_to(root)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    config = create_config()
    result = {
        "mode": args.mode,
        "requested_time_s": args.end_time_s,
        "sample_steps": args.sample_steps,
        "configuration": config.to_dict(),
        "source_sha256": hashes,
        "stability_check_enabled": True,
        "speed_l1_limit_lattice": 1.0,
        "status": "initializing",
    }

    def save():
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    started = time.monotonic()
    s = (ClampedCutLinks if args.mode == "clamped" else IceFlow2D)(config)
    s._thermal_stability_check_enabled = True
    target = round(args.end_time_s / s.scales.dt_s)
    result.update(target_step=target, dt_s=s.scales.dt_s, status="running")
    rows = []
    error = None
    with (out / "history.jsonl").open("w") as history:
        while True:
            row, fields = sample(s)
            rows.append(row)
            history.write(json.dumps(row) + "\n")
            history.flush()
            result.update(last=row, wall_time_s=time.monotonic() - started)
            save()
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "step": s.steps,
                        "time_s": s.physical_time_s,
                        "speed_l1": row["speed_l1_max_lattice"],
                        "q_min": row["raw_q_min"],
                        "q_max": row["raw_q_max"],
                        "error": error,
                    }
                ),
                flush=True,
            )
            if error is None and (not row["finite"] or row["invalid_q_count"]):
                error = f"Invalid sampled state at step {s.steps}"
            if error is not None or s.steps >= target:
                break
            try:
                s.step(min(args.sample_steps, target - s.steps))
            except RuntimeError as exc:
                error = str(exc)
    match = re.search(r"at step (\d+)", error or "")
    failure_step = int(match[1]) if match else None
    result.update(
        status="failed" if error else "completed",
        error=error,
        last_state_includes_failed_attempt=error is not None,
        attempted_failure_step=failure_step,
        attempted_failure_time_s=None
        if failure_step is None
        else failure_step * s.scales.dt_s,
        sampled_raw_q_min=min(
            (r["raw_q_min"] for r in rows if r["raw_q_min"] is not None), default=None
        ),
        sampled_raw_q_max=max(
            (r["raw_q_max"] for r in rows if r["raw_q_max"] is not None), default=None
        ),
        all_samples_finite=all(r["finite"] for r in rows),
        sampled_invalid_q_count=sum(r["invalid_q_count"] for r in rows),
        sampled_max_speed_l1_lattice=max(r["speed_l1_max_lattice"] for r in rows),
        wall_time_s=time.monotonic() - started,
        source_unchanged=all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in hashes.items()
        ),
    )
    np.savez_compressed(out / "final_fields.npz", **fields)
    save()
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("configuration", "source_sha256", "last")
            }
        ),
        flush=True,
    )
    return 1 if error else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("unclamped", "clamped"), default="unclamped")
    parser.add_argument("--end-time-s", type=float, default=8.0)
    parser.add_argument("--sample-steps", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (
        not np.isfinite(args.end_time_s)
        or args.end_time_s <= 0
        or args.sample_steps < 1
    ):
        parser.error("end-time-s and sample-steps must be positive and finite")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
