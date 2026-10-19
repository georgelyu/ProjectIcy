"""Isolate missing underside traction without changing the production solver.

Run from the repository root::

    python -m iceflow2d_v3.benchmarks.diagnose_floor_buoyancy

The four production-size blocks are placed in a resting full-width pool with
different floor clearances. Only fluid and rigid motion advance; heat/melting
is deliberately frozen. This is a causal control, not a dam-break validation.
Forces are per metre of out-of-plane depth and positive upwards.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example


ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    paths = (
        "simulator.py",
        "body_motion.py",
        "config.py",
        "contact.py",
        "phase_reconstruction.py",
        "stability_options.py",
        "examples/dambreak_four_ice_melting_2d.py",
        "benchmarks/diagnose_floor_buoyancy.py",
    )
    return {
        p: hashlib.sha256((ROOT / "iceflow2d_v3" / p).read_bytes()).hexdigest()
        for p in paths
    }


def interface_inventory(s):
    """Independently count resolved fluid/ice links using NumPy masks."""
    solid = s.solid_mask.to_numpy()
    wall = s.wall_mask.to_numpy()
    owner = s.solid_body_id.to_numpy()
    phase = s.water_phase.to_numpy()
    fluid = (solid == 0) & (wall == 0)
    result = []
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))
    for b in range(s.body_count):
        normal_sum = np.zeros(2)
        bottom_links = 0
        bottom_phase = None
        for di, dj in directions:
            ii, jj = np.nonzero(fluid)
            ni, nj = ii + di, jj + dj
            valid = (ni >= 0) & (ni < s.nx) & (nj >= 0) & (nj < s.ny)
            ni, nj = ni[valid], nj[valid]
            selected = (solid[ni, nj] == 1) & (owner[ni, nj] == b)
            count = int(np.count_nonzero(selected))
            weight = 1 / 9 if di == 0 or dj == 0 else 1 / 36
            normal_sum += 6 * weight * count * np.array([di, dj])
            if (di, dj) == (0, 1):
                bottom_links = count
                if count:
                    bottom_phase = float(
                        phase[ni[selected] - di, nj[selected] - dj].mean()
                    )
        result.append(
            {
                "body_id": b,
                "vertical_underside_links": bottom_links,
                "underside_mean_water_phase": bottom_phase,
                "sum_link_normals_m": (normal_sum * s.config.dx).tolist(),
            }
        )
    return result


def history_summary(path):
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    result = []
    for b in sorted({int(r["body_id"]) for r in rows}):
        body = [r for r in rows if int(r["body_id"]) == b]
        result.append(
            {
                "body_id": b,
                "time_end_s": float(body[-1]["physical_time_s"]),
                "y_min_mm": 1000 * min(float(r["body_center_y_m"]) for r in body),
                "y_max_mm": 1000 * max(float(r["body_center_y_m"]) for r in body),
                "angle_max_abs_deg": float(
                    np.rad2deg(max(abs(float(r["body_angle_rad"])) for r in body))
                ),
                "final_melted_fraction": float(body[-1]["body_melted_fraction"]),
            }
        )
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bodies": result,
        "water_volume_relative_change": float(rows[-1]["water_volume_current_cells"])
        / float(rows[0]["water_volume_current_cells"])
        - 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument(
        "--scenario",
        choices=("pool", "dry_dambreak", "production", "rough_floor"),
        default="pool",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "iceflow2d_v3/outputs/floor_buoyancy_diagnosis",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    base = example.create_config()
    gaps = (0.0, 0.25, 1.0, 2.0)
    config = replace(
        base,
        ice_bodies=tuple(
            replace(b, center_cells=(b.center_cells[0], b.center_cells[1] + gap))
            for b, gap in zip(base.bodies, gaps, strict=True)
        ),
        water_width_fraction=(base.nx - base.boundary_cells + 0.5) / base.nx
        if args.scenario == "pool"
        else base.water_width_fraction,
        water_height_fraction=75.5 / base.ny
        if args.scenario == "pool"
        else base.water_height_fraction,
        phase_warmup_steps=0,
        thermal=replace(base.thermal, thermal_expansion_water_1_k=0.0),
    )
    if args.scenario == "production":
        config = base
        gaps = (0.0,) * len(base.bodies)
    if args.scenario == "rough_floor":
        gaps = (2.0,) * len(base.bodies)
        config = replace(
            base,
            ice_bodies=tuple(
                replace(b, center_cells=(b.center_cells[0], b.center_cells[1] + gap))
                for b, gap in zip(base.bodies, gaps, strict=True)
            ),
        )
    s = IceFlow2D(config)
    if args.scenario == "rough_floor":
        # Sparse mechanical asperities keep a two-cell hydraulic gap open.
        # This changes the floor model, not buoyancy; it is a diagnostic
        # surrogate, NOT a calibrated subgrid lubrication model.
        s.body_motion.solve(True, 0)
        limits = s.body_motion.wall_limit.to_numpy()
        limits[:, 0] = base.boundary_cells + 2.0
        s.body_motion.wall_limit.from_numpy(limits)
    # Keep the production per-step fluid scan while freezing heat transport.
    if args.scenario in ("pool", "dry_dambreak"):
        s._advance_thermal = lambda: s._check_thermal_advection_stability(
            target_step=s.steps + 1
        )
    s._thermal_stability_check_enabled = True
    dt = s.scales.dt_s
    force_scale = config.rho_water * config.dx**3 / dt**2
    force_expected = config.rho_water * abs(config.gravity[1]) * (30 * config.dx) ** 2
    weight = (
        s.body_mass_lattice.array()
        * config.rho_water
        * config.dx**2
        * abs(config.gravity[1])
    )
    inventory = interface_inventory(s)
    start_center = s.body_center.to_numpy().copy()
    # Probe the actual production momentum boundary operator in exact rest.
    s._prepare_body_motion()
    s._advance_fluid()
    force = s.hydrodynamic_impulse.to_numpy() * force_scale
    torque = s.hydrodynamic_torque.array() * force_scale * config.dx
    s._advance_body()
    s._advance_thermal()
    s.steps += 1
    first = [
        {
            **item,
            "gap_cells": gap,
            "gap_mm": 1000 * gap * config.dx,
            "force_y_N_per_m": float(force[b, 1]),
            "weight_N_per_m": float(weight[b]),
            "free_net_force_y_N_per_m": float(force[b, 1] - weight[b]),
            "archimedes_force_N_per_m": force_expected,
            "torque_Nm_per_m": float(torque[b]),
            "first_kick_velocity_y_m_s": float(
                s.body_velocity.to_numpy()[b, 1] * s.scales.velocity_scale_m_s
            ),
        }
        for b, (item, gap) in enumerate(zip(inventory, gaps, strict=True))
    ]
    print(json.dumps({"first_step": first}, indent=2), flush=True)
    samples = []
    sampled_force = force.copy()
    sampled_inventory = inventory

    def sample():
        centers = s.body_center.to_numpy()
        velocities = s.body_velocity.to_numpy() * s.scales.velocity_scale_m_s
        for b in range(s.body_count):
            samples.append(
                {
                    "step": s.steps,
                    "time_s": s.steps * dt,
                    "body_id": b,
                    "gap_cells_initial": gaps[b],
                    "center_y_mm": float(centers[b, 1] * config.dx * 1000),
                    "rise_mm": float(
                        (centers[b, 1] - start_center[b, 1]) * config.dx * 1000
                    ),
                    "velocity_y_m_s": float(velocities[b, 1]),
                    "force_y_N_per_m": float(sampled_force[b, 1]),
                    "underside_links_before_step": sampled_inventory[b][
                        "vertical_underside_links"
                    ],
                    "contact_steps": int(s.body_motion.contact_steps[b]),
                    "underside_mean_water_phase": sampled_inventory[b][
                        "underside_mean_water_phase"
                    ],
                    "held_steps": int(s.body_motion.held_steps[b]),
                }
            )

    sample()
    while s.steps < args.steps:
        s.step(min(120, args.steps - s.steps) - 1)
        sampled_inventory = interface_inventory(s)
        s._prepare_body_motion()
        s._advance_fluid()
        sampled_force = s.hydrodynamic_impulse.to_numpy() * force_scale
        s._advance_body()
        s._advance_thermal()
        s.steps += 1
        sample()
    with (args.output / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)
    checks = (
        {
            "sealed_bottom_has_no_vertical_links": first[0]["vertical_underside_links"]
            == 0,
            "subcell_gap_still_has_no_vertical_links": first[1][
                "vertical_underside_links"
            ]
            == 0,
            "sealed_bottom_load_is_downward": first[0]["force_y_N_per_m"] < 0,
            "resolved_gap_restores_upward_net_force": all(
                r["free_net_force_y_N_per_m"] > 0 for r in first[2:]
            ),
            "resolved_gap_force_within_two_percent_of_archimedes": all(
                abs(r["force_y_N_per_m"] / force_expected - 1) < 0.02 for r in first[2:]
            ),
        }
        if args.scenario == "pool"
        else {
            "all_bodies_fall_initially": all(
                r["free_net_force_y_N_per_m"] < 0 for r in first
            ),
        }
    )
    result = {
        "scope": "short dam-break run with melting; rough_floor changes the floor model"
        if args.scenario in ("production", "rough_floor")
        else "static hydrostatic force and short no-melt release control; not a full dam-break rerun",
        "scenario": args.scenario,
        "floor_model": "sparse support at two cells above the hydraulic wall; experimental"
        if args.scenario == "rough_floor"
        else "production smooth wall",
        "thermal_frozen": args.scenario in ("pool", "dry_dambreak"),
        "configuration": config.to_dict(),
        "source_sha256": source_hashes(),
        "first_step": first,
        "last_samples": samples[-s.body_count :],
        "checks": checks,
        "gas_repair_count": int(s.gas_repair_count[None]),
        "angular_damping_rate_per_s": -float(np.log(base.angular_damping)) / dt,
        "production_history": history_summary(
            ROOT / "iceflow2d_v3/outputs/dambreak_four_ice_melting_2d/history.csv"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    labels = (
        [str(g) for g in gaps]
        if args.scenario in ("pool", "dry_dambreak")
        else [str(b) for b in range(s.body_count)]
    )
    axes[0].bar(labels, force[:, 1], color=["#b64a43"] * 2 + ["#26827e"] * 2)
    axes[0].axhline(
        float(weight[0]), color="#555555", linestyle="--", label="Ice weight"
    )
    axes[0].axhline(
        force_expected, color="#26827e", linestyle=":", label="Archimedes: 0.98 N/m"
    )
    axes[0].axhline(0, color="#999999", linewidth=0.7)
    axes[0].set(
        xlabel="Initial floor gap (cells)"
        if args.scenario in ("pool", "dry_dambreak")
        else "Body ID",
        ylabel="Fluid force upward (N/m)",
        title="Same ice, different access beneath it",
    )
    axes[0].legend(fontsize=8)
    for b, gap in enumerate(gaps):
        values = [r for r in samples if r["body_id"] == b]
        axes[1].plot(
            [r["time_s"] for r in values],
            [r["rise_mm"] for r in values],
            label=f"{gap:g} cell gap"
            if args.scenario in ("pool", "dry_dambreak")
            else f"Body {b}",
        )
    axes[1].set(
        xlabel="Time (s)",
        ylabel="Rise from initial center (mm)",
        title="Production motion solver, heat frozen"
        if args.scenario in ("pool", "dry_dambreak")
        else "Production dam break with melting",
    )
    axes[1].legend(fontsize=8)
    fig.savefig(args.output / "comparison.png", dpi=170)
    plt.close(fig)
    print(
        json.dumps(
            {
                "checks": checks,
                "last_samples": samples[-s.body_count :],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not all(checks.values()):
        raise SystemExit("A diagnostic expectation failed; inspect summary.json")


if __name__ == "__main__":
    main()
