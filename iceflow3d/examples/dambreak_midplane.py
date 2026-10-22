"""Run the dam break and save only its exact z-midplane, never 3D fields.

Render afterwards with ``python -m iceflow3d.postprocessing.midplane_fields RUN``.
"""

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow3d import IceFlow3D  # noqa: E402
from iceflow3d.config import LatticeScales  # noqa: E402
from iceflow3d.examples.dambreak_four_ice_melting_3d import (  # noqa: E402
    build_parser,
    create_config,
    snapshot_targets,
)


def write_json(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def midplane(fields, config, dx):
    """Interpolate cell-centered fields; categorical solid IDs are not averaged."""
    coordinate = config.domain_size_m[2] / (2 * dx) - 0.5
    k0 = int(np.floor(coordinate + 1e-10))
    k1 = int(np.ceil(coordinate - 1e-10))
    weight = coordinate - k0 if k1 != k0 else 0.0

    def interpolate(name):
        array = fields[name]
        return ((1 - weight) * array[:, :, k0] + weight * array[:, :, k1]).copy()

    ids0 = fields["solid_body_id"][:, :, k0]
    ids1 = fields["solid_body_id"][:, :, k1]
    ids = np.where(
        (ids0 < -1) | (ids1 < -1), -2, np.where(ids0 >= 0, ids0, ids1)
    ).astype(np.int32)
    phi = interpolate("water_phase")
    liquid_temp = interpolate("temperature_c")
    temp = liquid_temp.copy()
    # The thermal solver evolves liquid and ice energy, not gas energy.
    # Show gas at its configured temperature instead of plotting the liquid
    # temperature extension into cells with very small liquid fractions.
    temp[(ids == -1) & (phi < 0.5)] = config.thermal.initial_air_temperature_c
    ice0, ice1 = ids0 >= 0, ids1 >= 0
    ice_weight = (1 - weight) * ice0 + weight * ice1
    ice = ice_weight > 0
    ice_temp = (1 - weight) * ice0 * fields["temperature_c"][
        :, :, k0
    ] + weight * ice1 * fields["temperature_c"][:, :, k1]
    temp[ice] = ice_temp[ice] / ice_weight[ice]
    return dict(
        velocity_m_s=interpolate("velocity_m_s"),
        water_phase=phi,
        temperature_c=temp,
        # Retain the liquid-temperature extension separately for constitutive
        # audits; this is still a 2D field and is not a solved gas temperature.
        liquid_temperature_c=liquid_temp,
        solid_body_id=ids,
        ice_fraction=interpolate("ice_fraction"),
    )


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.set_defaults(
        resolution_x=150,
        end_time_s=4.0,
        output_interval_s=0.02,
        output_dir=Path("iceflow3d/outputs/dambreak_150_4s_midplane"),
    )
    args = parser.parse_args()
    if args.save_npz or args.save_vtk:
        parser.error(
            "this entry point saves 2D slices only; omit --save-npz/--save-vtk"
        )
    config = create_config(args)
    targets = snapshot_targets(config, args)
    scales = LatticeScales.from_iceflow_config(config)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(output)
    (output / "slices").mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    coordinate = config.nz / 2 - 0.5
    record = dict(
        status="running",
        configuration=config.to_dict(),
        scales=asdict(scales),
        target_steps=targets[-1],
        output_interval_s=args.output_interval_s,
        plane=dict(
            axis="z",
            coordinate_m=config.domain_size_m[2] / 2,
            cell_indices_zero_based=[
                int(np.floor(coordinate)),
                int(np.ceil(coordinate)),
            ],
            interpolation="Linear interpolation between adjacent cell centers; solid mask is their union, with wall priority.",
            temperature="Liquid/ice temperature; gas shown at prescribed air temperature. For mixed solid/fluid interpolation layers use the solid temperature in the solid mask.",
        ),
        output_3d_fields=False,
        checkpoint_output=False,
        frames=[],
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.glob("*.py")
        },
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    meta = output / "metadata.json"
    write_json(meta, record)
    start = time.perf_counter()
    sim = None
    last_print = 0.0
    try:
        sim = IceFlow3D(config)
        with (output / "history.csv").open("w", newline="") as history:
            writer = None
            for index, target in enumerate(targets):
                sim.step(target - sim.steps)
                if target == targets[-1]:
                    sim.flush_thermal()
                row = sim.diagnostics()
                assert all(np.isfinite(v).all() for v in row.values())
                assert row["gas_repair_count"] == row["pending_refill_cells"] == 0
                assert row["peak_fluid_speed_lattice"] < config.maximum_velocity_lattice
                if writer is None:
                    writer = csv.DictWriter(history, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                history.flush()
                # Full arrays exist temporarily in RAM only. No volume or
                # checkpoint writer is called, including on failure.
                fields = sim.snapshot()
                assert all(np.isfinite(v).all() for v in fields.values())
                assert fields["water_phase"].min() >= 0
                assert fields["water_phase"].max() <= 1
                section = midplane(fields, config, scales.dx_m)
                name = f"slices/frame_{index:06d}.npz"
                np.savez_compressed(
                    output / name,
                    **section,
                    step=target,
                    time_s=sim.physical_time_s,
                    dx_m=scales.dx_m,
                    z_m=config.domain_size_m[2] / 2,
                )
                fluid = section["solid_body_id"] == -1
                speed = np.linalg.norm(section["velocity_m_s"], axis=-1)
                record["frames"].append(
                    dict(
                        index=index,
                        step=target,
                        time_s=sim.physical_time_s,
                        path=name,
                        max_speed_m_s=float(speed[fluid].max(initial=0)),
                        temperature_min_c=float(fields["temperature_c"].min()),
                        temperature_max_c=float(fields["temperature_c"].max()),
                    )
                )
                del fields, section
                record.update(last_sample=row, wall_time_s=time.perf_counter() - start)
                write_json(meta, record)
                if time.perf_counter() - last_print >= 15 or target == targets[-1]:
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in (
                                    "step",
                                    "time_s",
                                    "maximum_fluid_speed_lattice",
                                    "peak_fluid_speed_lattice",
                                    "phase_volume_change_ratio",
                                )
                            }
                        ),
                        flush=True,
                    )
                    last_print = time.perf_counter()
            for name in ("f", "f_post", "h", "h_post"):
                assert np.isfinite(getattr(sim, name).to_numpy()).all(), name
            record["final_contact_residuals"] = list(sim.rigid.contact_residuals())
            assert max(record["final_contact_residuals"]) < 1e-7
            assert all(
                hashlib.sha256((source / name).read_bytes()).hexdigest() == value
                for name, value in record["source_sha256"].items()
            )
            record["status"] = "completed"
    except KeyboardInterrupt:
        record["status"] = "stopped_by_user"
        raise
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["completed_steps"] = sim.steps if sim else 0
        record["completed_time_s"] = sim.physical_time_s if sim else 0
        record["wall_time_s"] = time.perf_counter() - start
        write_json(meta, record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "status",
                    "completed_steps",
                    "completed_time_s",
                    "wall_time_s",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
