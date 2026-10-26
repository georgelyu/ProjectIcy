"""Dry ice dropped into an adiabatic open tank inside a CO2 reservoir."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow3d.config import IceBodyConfig, finite, integer
from iceflow3d.dryice.config import DryIceConfig, DryIceThermalConfig
from iceflow3d.dryice.materials import co2_density


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--resolution",
        "--resolution-x",
        type=int,
        default=216,
        help="physical cells per side, excluding the one-cell halo; multiple of 12",
    )
    p.add_argument("--domain-size-m", type=float, default=0.072)
    p.add_argument("--water-depth-m", type=float, default=0.016)
    p.add_argument("--wall-thickness-m", type=float, default=0.001)
    p.add_argument("--ice-size-m", type=float, default=0.008)
    p.add_argument("--drop-height-m", type=float, default=0.005)
    p.add_argument("--water-temperature-c", type=float, default=60.0)
    p.add_argument("--gas-temperature-c", type=float, default=20.0)
    p.add_argument("--film-thickness-m", type=float, default=0.0001)
    p.add_argument(
        "--initial-position", choices=("above", "submerged", "bottom"), default="above"
    )
    p.add_argument("--fixed-body", action="store_true")
    p.add_argument("--no-sublimation", action="store_true")
    p.add_argument("--phase-warmup-steps", type=int, default=200)
    p.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--end-time-s", type=float, default=1.0)
    p.add_argument("--steps", type=int)
    p.add_argument("--output-interval-s", type=float, default=0.01)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--save-npz", action="store_true", help="save full 3D snapshots (large)"
    )
    p.add_argument(
        "--save-slices",
        action="store_true",
        help="save central z velocity/temperature/phase slices as compact NPZ files",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("iceflow3d/outputs/dry_ice_in_hot_water_3d"),
    )
    return p


def create_config(args=None):
    args = build_parser().parse_args([]) if args is None else args
    n = integer("resolution", args.resolution, 24)
    if n % 12:
        raise ValueError(
            "resolution must be a multiple of 12 for exact tank proportions"
        )
    length = finite("domain_size_m", args.domain_size_m, positive=True)
    dx = length / n
    wall = max(
        1, round(finite("wall_thickness_m", args.wall_thickness_m, positive=True) / dx)
    )
    size = max(2, round(finite("ice_size_m", args.ice_size_m, positive=True) / dx))
    lo, hi = (1 + n / 4, 1.0, 1 + n / 4), (1 + 3 * n / 4, 1 + n / 3, 1 + 3 * n / 4)
    water = 1 + wall + finite("water_depth_m", args.water_depth_m, positive=True) / dx
    y = (
        water
        + finite("drop_height_m", args.drop_height_m, nonnegative=True) / dx
        + size / 2
    )
    if args.initial_position == "submerged":
        y = 1 + wall + size / 2 + max(0.5, (water - 1 - wall - size) / 2)
    elif args.initial_position == "bottom":
        y = 1 + wall + size / 2
    if size + 2 * wall >= n / 2:
        raise ValueError("dry ice must fit within the tank opening")
    thermal = DryIceThermalConfig(
        initial_water_temperature_c=args.water_temperature_c,
        initial_air_temperature_c=args.gas_temperature_c,
    )
    return DryIceConfig(
        nx=n + 2,
        ny=n + 2,
        nz=n + 2,
        domain_size_m=((n + 2) * dx,) * 3,
        water_width=n,
        water_height=round(water),
        floor_water_depth_cells=0,
        tank_lower_cells=lo,
        tank_upper_cells=hi,
        tank_wall_cells=wall,
        water_surface_cells=water,
        rho_ice=1560.0,
        rho_air=co2_density(args.gas_temperature_c),
        thermal=thermal,
        ice_bodies=(IceBodyConfig((1 + n / 2, y, 1 + n / 2), (size,) * 3),),
        film_thickness_m=args.film_thickness_m,
        backend=args.backend,
        phase_warmup_steps=args.phase_warmup_steps,
        body_motion_enabled=not args.fixed_body,
        sublimation_enabled=not args.no_sublimation,
        interface_width=5.0,
        reference_velocity_m_s=4.0,
        reference_lattice_velocity=0.1,
    )


def render_frame(path, sim, fields):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    z = sim.nz // 2
    phi = fields["water_phase"][1:-1, 1:-1, z].T
    ids = fields["solid_body_id"][1:-1, 1:-1, z].T
    temp = fields["temperature_c"][1:-1, 1:-1, z].T
    extent = (
        0,
        sim.config.physical_size_m[0] * 1000,
        0,
        sim.config.physical_size_m[1] * 1000,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), constrained_layout=True)
    axes[0].imshow(
        np.ma.masked_where(ids != -1, phi),
        origin="lower",
        extent=extent,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    t = axes[1].imshow(
        np.ma.masked_where(ids == -2, temp),
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        vmin=-78.5,
        vmax=sim.config.thermal.initial_water_temperature_c,
    )
    fig.colorbar(t, ax=axes[1], label="Temperature (deg C)")
    for ax in axes:
        ax.imshow(
            np.ma.masked_where(ids != -2, np.ones_like(phi)),
            origin="lower",
            extent=extent,
            cmap=ListedColormap(["#60646c"]),
            vmin=0,
            vmax=1,
        )
        ax.contour(
            (np.arange(phi.shape[1]) + 0.5) * sim.dx * 1000,
            (np.arange(phi.shape[0]) + 0.5) * sim.dx * 1000,
            ids >= 0,
            levels=[0.5],
            colors=["#222222"],
            linewidths=1,
        )
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
    axes[0].set_title("Water / CO2, central slice")
    axes[1].set_title("Adiabatic tank; isothermal CO2")
    fig.suptitle(f"Dry ice in hot water | t = {sim.physical_time_s:.4f} s")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_case(args):
    from iceflow3d.dryice.simulator import DryIceFlow3D

    config = create_config(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(
            f"output already contains a run: {output}; choose another --output-dir"
        )
    interval = finite("output_interval_s", args.output_interval_s, positive=True)
    end = finite("end_time_s", args.end_time_s, nonnegative=True)
    metadata = {
        "status": "initializing",
        "model": "HOME / D3Q7 AA / isothermal CO2 / film-resistance sublimation",
        "configuration": asdict(config),
        "physical_size_m": config.physical_size_m,
        "array_origin_m": [-config.domain_size_m[0] / config.nx] * 3,
        "assumptions": [
            "fixed adiabatic tank",
            "isothermal CO2 reservoir",
            "prescribed subgrid film thickness",
            "no water freezing, dissolution or fog",
        ],
        "samples": [],
    }
    source_root = Path(__file__).resolve().parents[1]
    metadata["source_sha256"] = {
        str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *source_root.glob("*.py"),
            *source_root.joinpath("dryice").glob("*.py"),
            Path(__file__),
        ]
    }
    start, sim = time.perf_counter(), None
    try:
        sim = DryIceFlow3D(config)
        steps = (
            integer("steps", args.steps, 0)
            if args.steps is not None
            else math.ceil(end / sim.scales.dt_s)
        )
        stride = max(1, round(interval / sim.scales.dt_s))
        targets = sorted(set([0, *range(stride, steps + 1, stride), steps]))
        metadata["dt_s"] = sim.scales.dt_s
        metadata["target_steps"] = steps
        metadata["requested_end_time_s"] = end
        metadata["output_interval_s"] = interval
        metadata["output_stride_steps"] = stride
        if args.save_slices:
            (output / "slices").mkdir(exist_ok=True)
            metadata["slice"] = {
                "axis": "z",
                "requested_coordinate_m": config.physical_size_m[2] / 2,
                "coordinate_m": (sim.nz // 2 - 0.5) * sim.dx,
                "index_with_halo": sim.nz // 2,
                "sampling": "upper central cell layer; no spatial interpolation",
                "array_order": "y,x; velocity components x,y,z",
                "shape": [sim.ny - 2, sim.nx - 2],
                "gas_temperature_model": "fixed isothermal reservoir",
            }
        metadata["status"] = "running"
        writer = None
        with (output / "history.csv").open("w", newline="") as stream:
            for frame, target in enumerate(targets):
                sim.step(target - sim.steps)
                if target == steps:
                    sim.flush_thermal()
                row = sim.diagnostics()
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                sample = {"step": sim.steps, "time_s": sim.physical_time_s}
                if args.save_slices:
                    from iceflow3d.dryice.slices import central_z_slice

                    name = f"slices/slice_{frame:05d}.npz"
                    temporary = output / (name + ".tmp")
                    with temporary.open("wb") as saved_slice:
                        np.savez_compressed(saved_slice, **central_z_slice(sim))
                    temporary.replace(output / name)
                    sample["slice_npz"] = name
                if args.save_npz or not args.no_plot:
                    fields = sim.snapshot()
                    if args.save_npz:
                        name = f"fields_{frame:05d}.npz"
                        np.savez_compressed(
                            output / name,
                            **fields,
                            time_s=sim.physical_time_s,
                            dx_m=sim.dx,
                        )
                        sample["npz"] = name
                    if not args.no_plot:
                        name = f"frame_{frame:05d}.png"
                        render_frame(output / name, sim, fields)
                        sample["png"] = name
                metadata["samples"].append(sample)
                metadata["final_diagnostics"] = row
                metadata["wall_time_s"] = time.perf_counter() - start
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
                if not args.quiet:
                    print(
                        f"step {sim.steps}/{steps}; t={sim.physical_time_s:.6f}s; solid={row['solid_co2_mass_kg']:.6g} kg; speed={row['maximum_fluid_speed_lattice']:.4g} LU",
                        flush=True,
                    )
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["failure"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "completed_steps": sim.steps if sim else 0,
        }
        (output / "run_failure.json").write_text(
            json.dumps(metadata["failure"], indent=2)
        )
        raise
    finally:
        metadata["wall_time_s"] = time.perf_counter() - start
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main(argv=None):
    return run_case(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
