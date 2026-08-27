"""Run the fixed-ice two-dimensional phase-change validation.

The example uses only NumPy: a centred square of ice is held mechanically
fixed in a warm, square water bath.  A conservative finite-volume enthalpy
method advances heat conduction and melting from all four sides.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iceflow2d.stefan2d import (  # noqa: E402
    Stefan2DConfig,
    Stefan2DSnapshot,
    Stefan2DSolver,
)


DEFAULT_OUTPUT_DIR = "outputs/iceflow2d/stefan_melting_2d"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU two-dimensional fixed-ice phase-change validation using a "
            "conservative finite-volume enthalpy method"
        )
    )
    parser.add_argument("--domain-width-mm", type=float, default=30.0)
    parser.add_argument("--domain-height-mm", type=float, default=30.0)
    parser.add_argument("--cells-x", type=int, default=120)
    parser.add_argument("--cells-y", type=int, default=120)
    parser.add_argument("--ice-width-mm", type=float, default=12.0)
    parser.add_argument("--ice-height-mm", type=float, default=12.0)
    parser.add_argument(
        "--end-time-s",
        type=float,
        default=180.0,
        help="physical end time in seconds (default: 180)",
    )
    parser.add_argument(
        "--output-interval-s",
        type=float,
        default=30.0,
        help="field sampling interval in seconds (default: 30)",
    )
    parser.add_argument("--melting-temperature-c", type=float, default=0.0)
    parser.add_argument("--bath-temperature-c", type=float, default=20.0)
    parser.add_argument("--density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--water-specific-heat", type=float, default=4186.0)
    parser.add_argument("--ice-specific-heat", type=float, default=2100.0)
    parser.add_argument("--water-conductivity", type=float, default=0.60)
    parser.add_argument("--ice-conductivity", type=float, default=2.20)
    parser.add_argument("--latent-heat", type=float, default=334000.0)
    parser.add_argument("--fourier-number", type=float, default=0.15)
    parser.add_argument(
        "--time-step-s",
        type=float,
        default=None,
        help="optional explicit time step; must not exceed the reported limit",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the summary PNG (CSV and metadata are still written)",
    )
    parser.add_argument(
        "--no-npz",
        action="store_true",
        help="skip the compressed two-dimensional field archive",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def create_config(args: argparse.Namespace) -> Stefan2DConfig:
    return Stefan2DConfig(
        domain_width_m=float(args.domain_width_mm) * 1.0e-3,
        domain_height_m=float(args.domain_height_mm) * 1.0e-3,
        cells_x=args.cells_x,
        cells_y=args.cells_y,
        ice_width_m=float(args.ice_width_mm) * 1.0e-3,
        ice_height_m=float(args.ice_height_mm) * 1.0e-3,
        end_time_s=args.end_time_s,
        output_interval_s=args.output_interval_s,
        melting_temperature_c=args.melting_temperature_c,
        bath_temperature_c=args.bath_temperature_c,
        density_kg_m3=args.density_kg_m3,
        specific_heat_water_j_kg_k=args.water_specific_heat,
        specific_heat_ice_j_kg_k=args.ice_specific_heat,
        conductivity_water_w_m_k=args.water_conductivity,
        conductivity_ice_w_m_k=args.ice_conductivity,
        latent_heat_j_kg=args.latent_heat,
        fourier_number=args.fourier_number,
        time_step_s=args.time_step_s,
    )


def write_history_csv(path: Path, snapshots: list[Stefan2DSnapshot]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "time_s",
                "ice_area_m2",
                "ice_area_mm2",
                "equivalent_square_side_m",
                "equivalent_square_half_width_m",
                "equivalent_uniform_melt_depth_m",
                "horizontal_midline_solid_width_m",
                "vertical_midline_solid_height_m",
                "melted_fraction",
                "total_enthalpy_j_m",
                "boundary_heat_input_j_m",
                "energy_residual_j_m",
                "liquid_fraction_symmetry_error",
                "temperature_symmetry_error_c",
                "enthalpy_symmetry_error_j_m3",
                "normalized_symmetry_error",
            )
        )
        for snapshot in snapshots:
            writer.writerow(
                (
                    f"{snapshot.time_s:.17g}",
                    f"{snapshot.ice_area_m2:.17g}",
                    f"{snapshot.ice_area_m2 * 1.0e6:.17g}",
                    f"{snapshot.equivalent_square_side_m:.17g}",
                    f"{snapshot.equivalent_square_half_width_m:.17g}",
                    f"{snapshot.equivalent_uniform_melt_depth_m:.17g}",
                    f"{snapshot.horizontal_midline_solid_width_m:.17g}",
                    f"{snapshot.vertical_midline_solid_height_m:.17g}",
                    f"{snapshot.melted_fraction:.17g}",
                    f"{snapshot.total_enthalpy_j_m:.17g}",
                    f"{snapshot.boundary_heat_input_j_m:.17g}",
                    f"{snapshot.energy_residual_j_m:.17g}",
                    f"{snapshot.liquid_fraction_symmetry_error:.17g}",
                    f"{snapshot.temperature_symmetry_error_c:.17g}",
                    f"{snapshot.enthalpy_symmetry_error_j_m3:.17g}",
                    f"{snapshot.normalized_symmetry_error:.17g}",
                )
            )


def write_fields_npz(
    path: Path,
    solver: Stefan2DSolver,
    snapshots: list[Stefan2DSnapshot],
) -> None:
    np.savez_compressed(
        path,
        x_m=np.asarray(solver.x_m, dtype=np.float64),
        y_m=np.asarray(solver.y_m, dtype=np.float64),
        time_s=np.asarray([item.time_s for item in snapshots], dtype=np.float64),
        temperature_c=np.stack([item.temperature_c for item in snapshots]),
        liquid_fraction=np.stack([item.liquid_fraction for item in snapshots]),
        enthalpy_j_m3=np.stack([item.enthalpy_j_m3 for item in snapshots]),
        ice_area_m2=np.asarray(
            [item.ice_area_m2 for item in snapshots], dtype=np.float64
        ),
        equivalent_square_side_m=np.asarray(
            [item.equivalent_square_side_m for item in snapshots], dtype=np.float64
        ),
        equivalent_square_half_width_m=np.asarray(
            [item.equivalent_square_half_width_m for item in snapshots],
            dtype=np.float64,
        ),
        equivalent_uniform_melt_depth_m=np.asarray(
            [item.equivalent_uniform_melt_depth_m for item in snapshots],
            dtype=np.float64,
        ),
        horizontal_midline_solid_width_m=np.asarray(
            [item.horizontal_midline_solid_width_m for item in snapshots],
            dtype=np.float64,
        ),
        vertical_midline_solid_height_m=np.asarray(
            [item.vertical_midline_solid_height_m for item in snapshots],
            dtype=np.float64,
        ),
        melted_fraction=np.asarray(
            [item.melted_fraction for item in snapshots], dtype=np.float64
        ),
        total_enthalpy_j_m=np.asarray(
            [item.total_enthalpy_j_m for item in snapshots], dtype=np.float64
        ),
        boundary_heat_input_j_m=np.asarray(
            [item.boundary_heat_input_j_m for item in snapshots], dtype=np.float64
        ),
        energy_residual_j_m=np.asarray(
            [item.energy_residual_j_m for item in snapshots], dtype=np.float64
        ),
        liquid_fraction_symmetry_error=np.asarray(
            [item.liquid_fraction_symmetry_error for item in snapshots],
            dtype=np.float64,
        ),
        temperature_symmetry_error_c=np.asarray(
            [item.temperature_symmetry_error_c for item in snapshots],
            dtype=np.float64,
        ),
        enthalpy_symmetry_error_j_m3=np.asarray(
            [item.enthalpy_symmetry_error_j_m3 for item in snapshots],
            dtype=np.float64,
        ),
        normalized_symmetry_error=np.asarray(
            [item.normalized_symmetry_error for item in snapshots], dtype=np.float64
        ),
    )


def write_summary_plot(
    path: Path,
    snapshots: list[Stefan2DSnapshot],
    config: Stefan2DConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent_mm = (
        0.0,
        config.domain_width_m * 1.0e3,
        0.0,
        config.domain_height_m * 1.0e3,
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.6, 8.4), constrained_layout=True)
    phase_axes = list(axes.flat[:4])
    selected = np.unique(
        np.linspace(0, len(snapshots) - 1, min(4, len(snapshots)), dtype=int)
    )
    phase_image = None
    for axis, index in zip(phase_axes, selected):
        snapshot = snapshots[index]
        phase_image = axis.imshow(
            snapshot.liquid_fraction,
            origin="lower",
            extent=extent_mm,
            vmin=0.0,
            vmax=1.0,
            cmap="Blues",
            interpolation="nearest",
            aspect="equal",
        )
        axis.contour(
            snapshot.liquid_fraction,
            levels=(0.5,),
            colors="orange",
            linewidths=1.1,
            origin="lower",
            extent=extent_mm,
        )
        axis.set_title(f"Liquid fraction, t = {snapshot.time_s:g} s")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    for axis in phase_axes[len(selected) :]:
        axis.set_visible(False)
    if phase_image is not None:
        figure.colorbar(
            phase_image,
            ax=phase_axes[: len(selected)],
            label="liquid fraction",
            shrink=0.85,
        )

    final = snapshots[-1]
    temperature_axis = axes.flat[4]
    temperature_image = temperature_axis.imshow(
        final.temperature_c,
        origin="lower",
        extent=extent_mm,
        vmin=config.melting_temperature_c,
        vmax=config.bath_temperature_c,
        cmap="inferno",
        interpolation="nearest",
        aspect="equal",
    )
    temperature_axis.contour(
        final.liquid_fraction,
        levels=(0.5,),
        colors="cyan",
        linewidths=1.1,
        origin="lower",
        extent=extent_mm,
    )
    temperature_axis.set_title(f"Temperature, t = {final.time_s:g} s")
    temperature_axis.set_xlabel("x (mm)")
    temperature_axis.set_ylabel("y (mm)")
    figure.colorbar(temperature_image, ax=temperature_axis, label="temperature (°C)")

    history_axis = axes.flat[5]
    time_s = np.asarray([item.time_s for item in snapshots])
    melted_percent = 100.0 * np.asarray([item.melted_fraction for item in snapshots])
    melt_depth_mm = 1.0e3 * np.asarray(
        [item.equivalent_uniform_melt_depth_m for item in snapshots]
    )
    history_axis.plot(time_s, melted_percent, marker="o", label="melted ice area")
    history_axis.set_xlabel("time (s)")
    history_axis.set_ylabel("melted area (%)", color="C0")
    history_axis.tick_params(axis="y", labelcolor="C0")
    history_axis.grid(alpha=0.25)
    depth_axis = history_axis.twinx()
    depth_axis.plot(
        time_s,
        melt_depth_mm,
        color="C1",
        marker="s",
        label="equivalent melt depth",
    )
    depth_axis.set_ylabel("equivalent melt depth (mm)", color="C1")
    depth_axis.tick_params(axis="y", labelcolor="C1")
    history_axis.set_title("Integral phase-change measures")

    relative_energy_residual = abs(final.energy_residual_j_m) / max(
        1.0, abs(final.boundary_heat_input_j_m)
    )
    figure.suptitle(
        "Fixed 2D ice melting: "
        f"{100.0 * final.melted_fraction:.2f}% area melted, "
        f"energy residual {relative_energy_residual:.2e}, "
        f"symmetry error {final.normalized_symmetry_error:.2e}"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_metadata(
    path: Path,
    config: Stefan2DConfig,
    solver: Stefan2DSolver,
    snapshots: list[Stefan2DSnapshot],
) -> None:
    final = snapshots[-1]
    relative_energy_residual = abs(final.energy_residual_j_m) / max(
        1.0, abs(final.boundary_heat_input_j_m)
    )
    data = {
        "model": "two-dimensional fixed-ice phase-change validation",
        "solver": "cell-centred conservative finite-volume enthalpy, explicit",
        "backend": "numpy-cpu",
        "mechanics": "fixed ice; no fluid motion",
        "thermal_boundaries": {
            "left": "constant-temperature water bath",
            "right": "constant-temperature water bath",
            "bottom": "constant-temperature water bath",
            "top": "constant-temperature water bath",
        },
        "assumptions": [
            "centred rectangular ice initially at the melting temperature",
            "equal ice/water density; no phase-change volume source",
            "two-dimensional heat conduction and phase change",
            "no natural convection or rigid-body motion",
        ],
        "validation": [
            "global enthalpy change equals heat supplied through all four walls",
            "the centred benchmark preserves its geometric reflection/rotation symmetries",
            "ice area decreases monotonically and the grid-refinement trend converges",
        ],
        "units": {
            "length": "m",
            "time": "s",
            "temperature": "degC",
            "volumetric_enthalpy": "J/m^3",
            "energy_per_out_of_plane_depth": "J/m",
        },
        "config": asdict(config),
        "derived": {
            "dx_m": config.dx_m,
            "dy_m": config.dy_m,
            "maximum_diffusivity_m2_s": config.maximum_diffusivity_m2_s,
            "maximum_time_step_s": config.maximum_time_step_s,
            "actual_time_step_s": config.actual_time_step_s,
            "water_diffusivity_m2_s": config.conductivity_water_w_m_k
            / (config.density_kg_m3 * config.specific_heat_water_j_kg_k),
            "ice_diffusivity_m2_s": config.conductivity_ice_w_m_k
            / (config.density_kg_m3 * config.specific_heat_ice_j_kg_k),
            "initial_ice_area_m2": solver.initial_ice_area_m2,
        },
        "results": {
            "steps": solver.steps,
            "ice_area_m2": final.ice_area_m2,
            "equivalent_square_side_m": final.equivalent_square_side_m,
            "equivalent_square_half_width_m": (final.equivalent_square_half_width_m),
            "equivalent_uniform_melt_depth_m": (final.equivalent_uniform_melt_depth_m),
            "horizontal_midline_solid_width_m": (
                final.horizontal_midline_solid_width_m
            ),
            "vertical_midline_solid_height_m": (final.vertical_midline_solid_height_m),
            "melted_fraction": final.melted_fraction,
            "total_enthalpy_j_m": final.total_enthalpy_j_m,
            "boundary_heat_input_j_m": final.boundary_heat_input_j_m,
            "energy_residual_j_m": final.energy_residual_j_m,
            "relative_energy_residual": relative_energy_residual,
            "liquid_fraction_symmetry_error": (final.liquid_fraction_symmetry_error),
            "temperature_symmetry_error_c": final.temperature_symmetry_error_c,
            "enthalpy_symmetry_error_j_m3": (final.enthalpy_symmetry_error_j_m3),
            "normalized_symmetry_error": final.normalized_symmetry_error,
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = create_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = Stefan2DSolver(config)
    snapshots = solver.run()
    write_history_csv(output_dir / "history.csv", snapshots)
    write_metadata(output_dir / "metadata.json", config, solver, snapshots)
    if not args.no_npz:
        write_fields_npz(output_dir / "fields.npz", solver, snapshots)
    if not args.no_plot:
        write_summary_plot(output_dir / "stefan_melting_2d.png", snapshots, config)

    if not args.quiet:
        final = snapshots[-1]
        relative_energy_residual = abs(final.energy_residual_j_m) / max(
            1.0, abs(final.boundary_heat_input_j_m)
        )
        print("Fixed-ice 2D phase-change validation complete")
        print(
            f"  domain: {config.domain_width_m * 1.0e3:.3f} x "
            f"{config.domain_height_m * 1.0e3:.3f} mm, "
            f"grid={config.cells_x} x {config.cells_y}, "
            f"time={config.end_time_s:.3f} s"
        )
        print(
            f"  ice: {100.0 * final.melted_fraction:.3f}% area melted, "
            "equivalent uniform melt depth="
            f"{final.equivalent_uniform_melt_depth_m * 1.0e3:.6f} mm"
        )
        print(f"  relative energy residual: {relative_energy_residual:.3e}")
        print(f"  normalized symmetry error: {final.normalized_symmetry_error:.3e}")
        print(f"  output: {output_dir}")


if __name__ == "__main__":
    main()
