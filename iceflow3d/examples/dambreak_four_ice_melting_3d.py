"""Released hot-water column hitting four independently moving ice cubes."""

from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import json
import math
import sys
import time
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow3d.config import (
    IceBodyConfig,
    IceFlowConfig,
    LatticeScales,
    ThermalConfig,
    finite,
    integer,
)
from iceflow3d.rigid import quaternion_multiply, rotation_matrix


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution-x", type=int, default=450)
    p.add_argument(
        "--resolution-y",
        type=int,
        default=None,
        help="inferred from cubic cells if omitted",
    )
    p.add_argument(
        "--resolution-z",
        type=int,
        default=None,
        help="inferred from cubic cells if omitted",
    )
    p.add_argument("--domain-width-m", type=float, default=0.15)
    p.add_argument("--domain-height-m", type=float, default=0.05)
    p.add_argument("--domain-depth-m", type=float, default=0.03)
    p.add_argument(
        "--boundary-cells",
        type=int,
        default=None,
        help="default: 1 mm, rounded to at least one cell",
    )
    p.add_argument("--water-column-width-m", type=float, default=0.065)
    p.add_argument("--water-column-height-m", type=float, default=0.045)
    p.add_argument("--floor-water-depth-m", type=float, default=None)
    p.add_argument(
        "--ice-floor-clearance-cells",
        type=int,
        default=None,
        help="default: 1 mm rounded to cells",
    )
    p.add_argument("--ice-size-m", type=float, default=0.01)
    p.add_argument(
        "--ice-centers-x-m", type=float, nargs=4, default=[0.085, 0.103, 0.121, 0.139]
    )
    p.add_argument(
        "--ice-centers-z-m",
        type=float,
        nargs=4,
        default=None,
        help="default: tank mid-depth",
    )
    p.add_argument(
        "--ice-angle-degrees",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="initial extrinsic xyz Euler angles",
    )
    p.add_argument("--density-water", type=float, default=1000.0)
    p.add_argument("--density-air", type=float, default=1.25)
    p.add_argument("--density-ice", type=float, default=917.0)
    p.add_argument("--water-temperature-c", type=float, default=90.0)
    p.add_argument("--ice-temperature-c", type=float, default=0.0)
    p.add_argument("--air-temperature-c", type=float, default=20.0)
    p.add_argument("--reference-velocity-m-s", type=float, default=4.0)
    p.add_argument("--interface-width", type=float, default=5.0)
    p.add_argument("--cm-acoustic-relaxation-rate", type=float, default=0.6)
    p.add_argument(
        "--momentum-collision-model", choices=("cm_mrt",), default="cm_mrt"
    )
    p.add_argument("--phase-warmup-steps", type=int, default=2000)
    p.add_argument("--thermal-update-interval", type=int, default=8)
    p.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    p.add_argument(
        "--check-stability", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--gas-state-repair", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--conserve-phase-volume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="global correction is currently commented out; keep this disabled",
    )
    p.add_argument("--end-time-s", type=float, default=8.0)
    p.add_argument("--output-interval-s", type=float, default=0.01)
    p.add_argument(
        "--steps", type=int, help="explicit step count, overrides --end-time-s"
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--save-npz",
        action="store_true",
        help="one compressed volume per sample, streamed to disk",
    )
    p.add_argument(
        "--save-vtk", action="store_true", help="3D ImageData and ParaView time series"
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this example's generated files in output-dir",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow3d/outputs/dambreak_four_ice_melting_3d"),
    )
    return p


def aligned(name, length, dx, *, allow_zero=False):
    cells = finite(name, length, positive=not allow_zero, nonnegative=allow_zero) / dx
    if not math.isclose(cells, round(cells), rel_tol=1e-10, abs_tol=1e-9):
        raise ValueError(f"{name} must align with cubic grid spacing {dx:g} m")
    return int(round(cells))


def create_config(args=None):
    args = build_parser().parse_args([]) if args is None else args
    nx = integer("resolution_x", args.resolution_x)
    dx = finite("domain_width_m", args.domain_width_m, positive=True) / nx
    ny = (
        aligned("domain_height_m", args.domain_height_m, dx)
        if args.resolution_y is None
        else args.resolution_y
    )
    nz = (
        aligned("domain_depth_m", args.domain_depth_m, dx)
        if args.resolution_z is None
        else args.resolution_z
    )
    wall = (
        max(1, round(0.001 / dx))
        if args.boundary_cells is None
        else integer("boundary_cells", args.boundary_cells)
    )
    clearance = (
        max(1, round(0.001 / dx))
        if args.ice_floor_clearance_cells is None
        else integer("ice_floor_clearance_cells", args.ice_floor_clearance_cells, 0)
    )
    size = aligned("ice_size_m", args.ice_size_m, dx)
    if size < 4:
        raise ValueError("ice cubes must span at least four cells")
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for a, angle in enumerate(args.ice_angle_degrees):
        radians = math.radians(finite("ice_angle_degrees", angle))
        axis_q = np.zeros(4)
        axis_q[0], axis_q[a + 1] = math.cos(radians / 2), math.sin(radians / 2)
        q = quaternion_multiply(axis_q, q)
    extent = abs(rotation_matrix(q)) @ np.full(3, size / 2)
    if args.floor_water_depth_m is None:
        water, air, ice = [
            finite("density", v, positive=True)
            for v in (args.density_water, args.density_air, args.density_ice)
        ]
        if not air < ice < water:
            raise ValueError("automatic pool depth requires air < ice < water density")
        floor = math.ceil(clearance + 2 * extent[1] * (ice - air) / (water - air))
    else:
        floor = aligned(
            "floor_water_depth_m", args.floor_water_depth_m, dx, allow_zero=True
        )
    if floor and floor <= clearance:
        raise ValueError("floor water must reach the ice above its clearance")
    width = wall + aligned("water_column_width_m", args.water_column_width_m, dx)
    height = wall + aligned("water_column_height_m", args.water_column_height_m, dx)
    zs = args.ice_centers_z_m or [args.domain_depth_m / 2] * 4
    bodies = []
    for x, z in zip(args.ice_centers_x_m, zs):
        x = finite("ice_center_x_m", x, positive=True) / dx
        z = finite("ice_center_z_m", z, positive=True) / dx
        if x - extent[0] < width:
            raise ValueError("ice must initially be downstream of the column")
        bodies.append(
            IceBodyConfig(
                (x, wall + clearance + extent[1], z), (size, size, size), tuple(q)
            )
        )
    thermal = ThermalConfig(
        initial_water_temperature_c=args.water_temperature_c,
        initial_ice_temperature_c=args.ice_temperature_c,
        initial_air_temperature_c=args.air_temperature_c,
        update_interval_lbm_steps=args.thermal_update_interval,
    )
    return IceFlowConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        domain_size_m=(args.domain_width_m, args.domain_height_m, args.domain_depth_m),
        boundary_cells=wall,
        water_width=width,
        water_height=height,
        floor_water_depth_cells=floor,
        ice_bodies=tuple(bodies),
        rho_water=args.density_water,
        rho_air=args.density_air,
        rho_ice=args.density_ice,
        reference_velocity_m_s=args.reference_velocity_m_s,
        cm_acoustic_relaxation_rate=args.cm_acoustic_relaxation_rate,
        momentum_collision_model=args.momentum_collision_model,
        interface_width=args.interface_width,
        phase_warmup_steps=args.phase_warmup_steps,
        thermal=thermal,
        backend=args.backend,
        check_stability=args.check_stability,
        gas_state_repair=args.gas_state_repair,
        conserve_phase_volume=args.conserve_phase_volume,
    )


def snapshot_targets(config, args):
    dt = LatticeScales.from_iceflow_config(config).dt_s
    end = (
        integer("steps", args.steps, 0)
        if args.steps is not None
        else max(0, round(finite("end_time_s", args.end_time_s, nonnegative=True) / dt))
    )
    interval = max(
        1,
        round(finite("output_interval_s", args.output_interval_s, positive=True) / dt),
    )
    return sorted(set([0, *range(interval, end + 1, interval), end]))


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = create_config(args)
        snapshot_targets(config, args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    return args


def run_case(args):
    from iceflow3d import IceFlow3D
    from iceflow3d.reporting import StreamingGif, render_frame, write_pvd, write_vti

    config = create_config(args)
    targets = snapshot_targets(config, args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    products = [
        output / "history.csv",
        output / "metadata.json",
        output / "run_failure.json",
        output / "dambreak_four_ice_melting_3d.gif",
        output / "dambreak_four_ice_melting_3d.png",
        output / "fields.pvd",
    ]
    for folder, pattern in (
        ("frames", "frame_*.png"),
        ("fields", "frame_*.npz"),
        ("vtk", "frame_*.vti"),
    ):
        products.extend((output / folder).glob(pattern))
    existing = [p for p in products if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output} contains a previous run; choose --output-dir or --overwrite"
        )
    for path in existing:
        path.unlink()
    metadata = {
        "status": "running",
        "configuration": config.to_dict(),
        "scales": asdict(LatticeScales.from_iceflow_config(config)),
        "requested": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "target_steps": targets,
        "samples": [],
        "model": f"D3Q27 HOME {config.momentum_collision_model}, D3Q7 AA central-moment Allen-Cahn, 6-DOF eroding OBBs, material FV enthalpy; melt disappears",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    sim, gif, start = None, None, time.perf_counter()
    try:
        if not args.quiet:
            cell_count = config.nx * config.ny * config.nz
            print(
                f"Grid {config.nx} x {config.ny} x {config.nz}; four cubes; "
                f"population storage {cell_count * 27 * 4 * 4 / 2**30:.2f} GiB; "
                f"{targets[-1]} steps ({targets[-1] * metadata['scales']['dt_s']:.6g} s)",
                flush=True,
            )
        sim = IceFlow3D(config)
        if not args.no_plot:
            (output / "frames").mkdir(exist_ok=True)
            gif = StreamingGif(output / "dambreak_four_ice_melting_3d.gif")
        if args.save_npz:
            (output / "fields").mkdir(exist_ok=True)
        if args.save_vtk:
            (output / "vtk").mkdir(exist_ok=True)
        vtk_samples = []
        next_progress = time.perf_counter()
        with (output / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = None
            for frame, target in enumerate(targets):
                while sim.steps < target:
                    sim.step(min(16, target - sim.steps))
                    if (
                        not args.quiet
                        and not args.no_progress
                        and time.perf_counter() >= next_progress
                    ):
                        print(
                            f"\rstep {sim.steps}/{targets[-1]}  t={sim.physical_time_s:.6f} s",
                            end="",
                            flush=True,
                        )
                        next_progress = time.perf_counter() + 1.0
                if target == targets[-1]:
                    sim.flush_thermal()
                row = sim.diagnostics()
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                sample = {"step": target, "time_s": sim.physical_time_s}
                if not args.no_plot or args.save_npz or args.save_vtk:
                    fields = sim.snapshot()
                    if args.save_npz:
                        name = f"fields/frame_{frame:06d}.npz"
                        np.savez_compressed(
                            output / name,
                            **fields,
                            step=target,
                            time_s=sim.physical_time_s,
                            dx_m=sim.scales.dx_m,
                            body_centers_m=sim.body_center.to_numpy() * sim.scales.dx_m,
                            body_quaternion_wxyz=sim.rigid.q,
                            body_mass_kg=sim.rigid.mass * sim.scales.mass_kg,
                        )
                        sample["npz"] = name
                    if args.save_vtk:
                        name = f"vtk/frame_{frame:06d}.vti"
                        write_vti(output / name, fields, sim.scales.dx_m)
                        vtk_samples.append((sim.physical_time_s, name))
                        write_pvd(output / "fields.pvd", vtk_samples)
                        sample["vtk"] = name
                    if not args.no_plot:
                        name = f"frames/frame_{frame:06d}.png"
                        render_frame(
                            output / name,
                            config,
                            sim.scales,
                            fields,
                            sim.physical_time_s,
                        )
                        gif.append(output / name)
                        sample["png"] = name
                        if target == targets[-1]:
                            (output / "dambreak_four_ice_melting_3d.png").write_bytes(
                                (output / name).read_bytes()
                            )
                metadata["samples"].append(sample)
                metadata["final_diagnostics"] = row
                (output / "metadata.json").write_text(
                    json.dumps(metadata, indent=2), encoding="utf-8"
                )
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_steps": sim.steps if sim is not None else 0,
            "completed_time_s": sim.physical_time_s if sim is not None else 0.0,
        }
        metadata["failure"] = failure
        (output / "run_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        raise
    finally:
        if gif is not None:
            gif.close()
        metadata["wall_time_s"] = time.perf_counter() - start
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    if not args.quiet:
        print(
            f"\nCompleted {sim.steps} steps, {sim.physical_time_s:.6g} s; outputs: {output}"
        )
    return metadata


def main(argv=None):
    return run_case(parse_args(argv))


if __name__ == "__main__":
    main()
