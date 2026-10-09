"""Compare refill operators with identical prescribed ice trajectories.

No thermal update, gravity or free body integration is used. A square is
translated downwards through the same W=5 interface. Thus feedback from
different hydrodynamic forces cannot change the geometry in this comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from iceflow2d_v3.config import IceBodyConfig
from iceflow2d_v3.simulator import IceFlow2D
from iceflow2d_v3.test_iceflow import _falling_config


def run(option, steps):
    speed = 0.003
    s = IceFlow2D(
        _falling_config(
            resolution=(64, 96),
            water_width_fraction=62 / 64,
            water_height_fraction=0.5,
            dx=1 / 3000,
            ice_bodies=(IceBodyConfig(size_cells=(8, 8), center_cells=(32.0, 54.0)),),
            phase_warmup_steps=2000,
            interface_width=5,
            gravity=(0, 0),
            sigma=0,
            angular_damping=1,
            viscosity_water=3.141732579601075e-7,
            viscosity_air=1.4506451972115707e-5,
            exposed_node_refill=option,
        )
    )
    s.body_velocity[None] = (0.0, -speed)
    s.body_angular_velocity[None] = 0
    rows = []
    for n in range(steps + 1):
        if n % 20 == 0 or n == steps:
            active = (s.wall_mask.to_numpy() == 0) & (s.solid_mask.to_numpy() == 0)
            phase = s.water_phase.to_numpy()
            u = s.physical_velocity_lattice.to_numpy()
            pressure = s.pressure_lattice.to_numpy()
            finite = np.isfinite(u).all() and np.isfinite(pressure).all()
            rows.append(
                {
                    "step": n,
                    "max_l1": float(abs(u[active]).sum(axis=-1).max()),
                    "pressure_rms": float(np.sqrt(np.mean(pressure[active] ** 2))),
                    "pressure_max_abs": float(abs(pressure[active]).max()),
                    "phase_amount": float(phase[active].sum(dtype=np.float64)),
                    "finite": bool(finite),
                }
            )
            if not finite or rows[-1]["max_l1"] > 1:
                break
        if n < steps:
            s._advance_fluid()
            center = (32.0, 54.0 - speed * (n + 1))
            s.body_center[None] = center
            s.body_reference_origin[None] = center
            s._refresh_fluid_domain(resolve_contact=False)
            s.steps += 1
    summary = {
        "option": option,
        "completed": rows[-1]["step"] == steps,
        "last": rows[-1],
        "peak_l1": max(r["max_l1"] for r in rows),
        "peak_pressure_rms": max(r["pressure_rms"] for r in rows),
        "peak_pressure_max_abs": max(r["pressure_max_abs"] for r in rows),
        "phase_relative_change": rows[-1]["phase_amount"] / rows[0]["phase_amount"] - 1,
        "final_center": np.asarray(s.body_center[None]).tolist(),
        "rows": rows,
    }
    return summary, s.solid_mask.to_numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000)
    args = p.parse_args()
    original, mask0 = run("equilibrium", args.steps)
    candidate, mask1 = run("non_equilibrium", args.steps)
    result = {
        "same_final_geometry": bool(np.array_equal(mask0, mask1)),
        "baseline": original,
        "candidate": candidate,
        "trajectory": {
            "center_initial": [32, 54],
            "velocity_lattice": [0, -0.003],
            "steps": args.steps,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {a: b for a, b in v.items() if a != "rows"}
                if isinstance(v, dict)
                else v
                for k, v in result.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
