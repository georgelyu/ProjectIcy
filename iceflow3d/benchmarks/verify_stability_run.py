"""Audit a completed diagnostic run and optionally render its saved volumes."""

import argparse
from contextlib import closing
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--end-time-s", type=float, default=8.0)
    parser.add_argument("--resolution-x", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--without-phase-conservation",
        action="store_true",
        help="audit the disabled-projection experiment and report volume drift",
    )
    parser.add_argument(
        "--prefix-run",
        type=Path,
        help="Audit the earlier segment ending at this run's complete restart checkpoint.",
    )
    args = parser.parse_args()
    record = json.loads((args.run / "diagnostic.json").read_text())
    config = record["configuration"]
    rows = [
        json.loads(line)
        for line in (args.run / "timeline.jsonl").read_text().splitlines()
    ]
    segments = [(args.run, record)]
    frames = sorted(args.run.glob("frame_*.npz"))
    restart_evidence = None
    if args.prefix_run is not None:
        prefix = json.loads((args.prefix_run / "diagnostic.json").read_text())
        assert prefix["configuration"] == config, "restart configuration changed"
        restart_path = Path(record["controls"]["restart"])
        with np.load(restart_path, allow_pickle=False) as state:
            restart = json.loads(str(state["metadata"]))
        restart_step = restart["steps"]
        assert restart["configuration"] == config
        assert restart["source_sha256"] == prefix["source_sha256"]
        assert rows[0]["step"] == restart_step
        prefix_rows = [
            json.loads(line)
            for line in (args.prefix_run / "timeline.jsonl").read_text().splitlines()
        ]
        checkpoint_row = next(row for row in prefix_rows if row["step"] == restart_step)
        for key, value in rows[0].items():
            if key != "wall_time_s":
                np.testing.assert_allclose(
                    value,
                    checkpoint_row[key],
                    rtol=1e-9,
                    atol=1e-12,
                    err_msg=f"restart mismatch: {key}",
                )
        rows = [row for row in prefix_rows if row["step"] < restart_step] + rows
        frames = [
            p
            for p in sorted(args.prefix_run.glob("frame_*.npz"))
            if int(p.stem.split("_")[-1]) < restart_step
        ] + frames
        segments.insert(0, (args.prefix_run, prefix))
        restart_evidence = {
            "prefix_run": str(args.prefix_run.resolve()),
            "checkpoint": str(restart_path.resolve()),
            "checkpoint_sha256": hashlib.sha256(restart_path.read_bytes()).hexdigest(),
            "step": restart_step,
            "changed_source_files": [
                name
                for name, digest in record["source_sha256"].items()
                if prefix["source_sha256"].get(name) != digest
            ],
            "first_resumed_sample_matches_checkpoint_sample": True,
        }
    assert rows[0]["step"] == 0, (
        "restart runs require --prefix-run for a full-time audit"
    )
    assert record["status"] == "completed", record["status"]
    assert config["check_stability"] and not config["gas_state_repair"]
    assert config["conserve_phase_volume"] == (not args.without_phase_conservation)
    if args.without_phase_conservation:
        for row in rows:
            assert row["phase_volume_projection_m3"] == 0.0
            for axis in "xyz":
                assert row[f"phase_projection_momentum_{axis}_n_s"] == 0.0
    for _, segment in segments:
        for name in ("fixed_ice", "no_thermal", "no_capillary"):
            assert not segment["controls"][name], name
    if args.resolution_x is not None:
        assert config["nx"] == args.resolution_x
    assert rows[-1]["step"] == record["target_steps"]
    assert abs(rows[-1]["time_s"] - args.end_time_s) < 1e-10
    assert all(b["step"] > a["step"] for a, b in zip(rows, rows[1:]))
    assert all(np.isfinite(v).all() for row in rows for v in row.values())
    assert all(row["gas_repair_count"] == 0 for row in rows)
    assert all(row["pending_refill_cells"] == 0 for row in rows)
    peak = rows[-1]["peak_fluid_speed_lattice"]
    assert peak < config["maximum_velocity_lattice"]
    assert all(
        b["ice_mass_kg"] <= a["ice_mass_kg"] + 1e-12 for a, b in zip(rows, rows[1:])
    )
    volume_error = max(abs(row["phase_volume_change_ratio"]) for row in rows)
    if not args.without_phase_conservation:
        assert volume_error < 2e-7, volume_error
    initial = rows[0]
    latent = config["thermal"]["properties"]["latent_heat_j_kg"]
    energy = [
        row["water_energy_j"]
        + row["ice_sensible_energy_j"]
        + row["melted_mass_kg"] * latent
        + row["discarded_melt_heat_j"]
        - row["thermal_remap_energy_j"]
        - initial["water_energy_j"]
        - initial["ice_sensible_energy_j"]
        for row in rows
    ]
    # Remapping is an explicitly measured energy source, not claimed to vanish.
    energy_residual = max(abs(v) for v in energy)
    assert energy_residual < 1e-6 * initial["water_energy_j"], energy_residual
    source = Path(
        record["controls"]["source_root"] or Path(__file__).resolve().parents[2]
    )
    for _, segment in segments:
        segment_source = Path(
            segment["controls"]["source_root"] or Path(__file__).resolve().parents[2]
        )
        assert all(
            hashlib.sha256(
                (segment_source / "iceflow3d" / name).read_bytes()
            ).hexdigest()
            == digest
            for name, digest in segment["source_sha256"].items()
        ), "source changed after the run"
    final = args.run / "final.npz"
    frame_summary = []
    initial_temperatures = [
        config["thermal"][f"initial_{phase}_temperature_c"]
        for phase in ("water", "air", "ice")
    ] + [
        body["temperature_c"]
        for body in config["ice_bodies"]
        if body["temperature_c"] is not None
    ]
    temperature_min, temperature_max = np.inf, -np.inf
    for path in [*frames, final]:
        with np.load(path, allow_pickle=False) as fields:
            assert all(np.isfinite(fields[name]).all() for name in fields.files), path
            phi = fields["water_phase"]
            assert phi.shape == (config["nx"], config["ny"], config["nz"])
            assert phi.min() >= 0 and phi.max() <= 1, path
            assert (fields["body_mass"] >= 0).all(), path
            temperature_min = min(temperature_min, float(fields["temperature_c"].min()))
            temperature_max = max(temperature_max, float(fields["temperature_c"].max()))
            np.testing.assert_allclose(
                np.linalg.norm(fields["body_quaternion_wxyz"], axis=-1),
                1,
                atol=1e-6,
            )
            frame_summary.append(
                {
                    "file": path.name,
                    "path": str(path.resolve()),
                    "step": int(fields["step"]),
                    "time_s": float(fields["time_s"]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    assert temperature_min >= min(initial_temperatures) - 1e-3
    assert temperature_max <= max(initial_temperatures) + 1e-3
    checkpoint = json.loads((args.run / "latest_checkpoint.json").read_text())
    assert checkpoint["step"] == rows[-1]["step"], "final complete checkpoint required"
    sys.path.insert(0, str(source.resolve()))
    from iceflow3d.rigid import RigidBodies, rotation_matrix

    with np.load(args.run / checkpoint["path"], allow_pickle=False) as state:
        for name in ("home", "home_post", "h"):
            assert np.isfinite(state["field/" + name]).all(), name
        rigid = RigidBodies.__new__(RigidBodies)
        rigid.s = SimpleNamespace(
            shape=phi.shape,
            body_count=len(config["ice_bodies"]),
            config=SimpleNamespace(boundary_cells=config["boundary_cells"]),
        )
        for name in json.loads(str(state["metadata"]))["rigid"]:
            setattr(rigid, name, state["rigid/" + name])
        penetration, closing_speed = rigid.contact_residuals()
        assert penetration < 1e-7 and closing_speed < 1e-7, (penetration, closing_speed)
        active_bodies = np.flatnonzero(state["rigid/mass"] > 1e-15)
        for body in active_bodies:
            rotation = rotation_matrix(state["rigid/q"][body])
            lower, upper = state["rigid/lower"][body], state["rigid/upper"][body]
            center = state["rigid/origin"][body] + rotation @ (0.5 * (lower + upper))
            extent = abs(rotation) @ (0.5 * (upper - lower))
            assert np.all(center - extent >= config["boundary_cells"] - 1e-5)
            assert np.all(
                center + extent <= np.array(phi.shape) - config["boundary_cells"] + 1e-5
            )
    with (args.run / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.run / "verified_timeline.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = {
        "status": "passed",
        "configuration": config,
        "completed_steps": rows[-1]["step"],
        "time_s": rows[-1]["time_s"],
        "sample_count": len(rows),
        "maximum_speed_over_every_step_lattice": peak,
        "maximum_sampled_phase_volume_relative_error": volume_error,
        "phase_volume_conservation_required": not args.without_phase_conservation,
        "final_phase_volume_change_ratio": rows[-1]["phase_volume_change_ratio"],
        "maximum_sampled_accounted_energy_residual_j": energy_residual,
        "final_thermal_remap_energy_j": rows[-1]["thermal_remap_energy_j"],
        "final_ice_mass_kg": rows[-1]["ice_mass_kg"],
        "gas_repairs": 0,
        "saved_temperature_range_c": [temperature_min, temperature_max],
        "final_active_bodies_inside_container": int(len(active_bodies)),
        "final_contact_penetration_cells": penetration,
        "final_contact_closing_speed_lattice": closing_speed,
        "restart": restart_evidence,
        "source_segments": [
            {"run": str(path.resolve()), "source_sha256": segment["source_sha256"]}
            for path, segment in segments
        ],
        "source_sha256": record["source_sha256"],
        "volumes": frame_summary,
    }
    (args.run / "verification.json").write_text(json.dumps(result, indent=2))
    if args.render:
        sys.path.insert(0, str(source.resolve()))
        from iceflow3d.config import (
            IceBodyConfig,
            IceFlowConfig,
            LatticeScales,
            ThermalConfig,
            PhaseChangeProperties,
        )
        from iceflow3d.reporting import StreamingGif, render_frame

        values = dict(config)
        thermal = dict(values["thermal"])
        thermal["properties"] = PhaseChangeProperties(**thermal["properties"])
        values["thermal"] = ThermalConfig(**thermal)
        values["ice_bodies"] = tuple(IceBodyConfig(**b) for b in values["ice_bodies"])
        cfg = IceFlowConfig(**values)
        scales = LatticeScales.from_iceflow_config(cfg)
        images = args.run / "rendered"
        images.mkdir(exist_ok=True)
        with closing(StreamingGif(args.run / "dambreak_four_ice_8s.gif")) as gif:
            for path in frames:
                with np.load(path, allow_pickle=False) as fields:
                    image = images / (path.stem + ".png")
                    render_frame(image, cfg, scales, fields, float(fields["time_s"]))
                    gif.append(image)
            with np.load(final, allow_pickle=False) as fields:
                render_frame(
                    args.run / "final.png", cfg, scales, fields, float(fields["time_s"])
                )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("configuration", "volumes", "source_sha256")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
