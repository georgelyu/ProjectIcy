"""Coupled LBM/enthalpy example for a fixed two-dimensional melting ice block.

The short millimetre-scale case is intended as the first end-to-end thermal
coupling check.  A 1 mm square ice block is held below a water/air free surface
inside a 2.5 mm by 3.5 mm container.  The real ice/water density ratio changes
the conservative water-volume target as the ice melts, so the free surface can
absorb the associated volume contraction.

Configuration and command-line helpers deliberately do not import Taichi.  The
CUDA-dependent :class:`iceflow2d.IceFlow2D` symbol is imported only in
``main()``, which keeps this module importable on CPU-only hosts for geometry
and configuration tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iceflow2d.config import IceFlowConfig, create_iceflow_config  # noqa: E402
from iceflow2d.thermal import (  # noqa: E402
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
)


DEFAULT_OUTPUT_DIR = "outputs/iceflow2d/coupled_fixed_ice_melting_2d"
DEFAULT_BOUNDARY_CELLS = 3


@dataclass(slots=True)
class CoupledSnapshot:
    """Host copy of one synchronized LBM/thermal state."""

    physical_time_s: float
    thermal_time_s: float
    lbm_steps: int
    thermal_substeps: int
    temperature_c: np.ndarray
    liquid_fraction: np.ndarray
    enthalpy_j_m3: np.ndarray
    water_phase: np.ndarray
    solid: np.ndarray
    sdf: np.ndarray
    phase_change_material: np.ndarray
    velocity_lattice: np.ndarray
    solid_volume_cells: float
    sharp_geometry_volume_cells: float
    solid_area_m2: float
    melted_fraction: float
    equivalent_square_side_m: float
    equivalent_uniform_melt_depth_m: float
    generated_water_volume_cells: float
    phase_change_contraction_cells: float
    water_volume_target_cells: float
    water_volume_current_cells: float
    water_volume_residual_cells: float
    total_enthalpy_j_m: float
    boundary_heat_input_j_m: float
    energy_residual_j_m: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Taichi CUDA air-water LBM coupled to finite-volume enthalpy "
            "melting of a fixed two-dimensional ice block"
        )
    )
    parser.add_argument("--domain-width-mm", type=float, default=2.5)
    parser.add_argument("--domain-height-mm", type=float, default=3.5)
    parser.add_argument("--resolution-x", type=int, default=80)
    parser.add_argument("--resolution-y", type=int, default=112)
    parser.add_argument("--water-level-mm", type=float, default=2.75)
    parser.add_argument("--ice-size-mm", type=float, default=1.0)
    parser.add_argument("--ice-center-y-mm", type=float, default=1.5)
    parser.add_argument("--boundary-cells", type=int, default=DEFAULT_BOUNDARY_CELLS)

    parser.add_argument("--end-time-s", type=float, default=0.30)
    parser.add_argument("--output-interval-s", type=float, default=0.05)
    parser.add_argument("--water-temperature-c", type=float, default=60.0)
    parser.add_argument("--ice-temperature-c", type=float, default=0.0)
    parser.add_argument("--air-temperature-c", type=float, default=40.0)
    parser.add_argument("--melting-temperature-c", type=float, default=0.0)

    parser.add_argument("--density-water", type=float, default=1000.0)
    parser.add_argument("--density-ice", type=float, default=917.0)
    parser.add_argument("--density-air", type=float, default=1.25)
    # IceFlow2D converts these as kinematic viscosities.  The former large-scale
    # numerical defaults would give tau > 2 at the present 31.25 um spacing.
    parser.add_argument("--kinematic-viscosity-water", type=float, default=1.0e-6)
    parser.add_argument("--kinematic-viscosity-air", type=float, default=1.5e-5)
    parser.add_argument(
        "--surface-tension",
        type=float,
        default=0.0,
        help="water-air surface tension; zero isolates the phase-change check",
    )

    parser.add_argument("--water-specific-heat", type=float, default=4186.0)
    parser.add_argument("--ice-specific-heat", type=float, default=2100.0)
    parser.add_argument("--air-specific-heat", type=float, default=1005.0)
    parser.add_argument("--water-conductivity", type=float, default=0.60)
    parser.add_argument("--ice-conductivity", type=float, default=2.20)
    parser.add_argument("--air-conductivity", type=float, default=0.026)
    parser.add_argument("--latent-heat", type=float, default=334000.0)

    parser.add_argument("--phase-warmup-steps", type=int, default=100)
    parser.add_argument("--thermal-update-interval", type=int, default=10)
    parser.add_argument("--interface-width", type=float, default=5.0)
    parser.add_argument("--mobility", type=float, default=0.1)
    parser.add_argument(
        "--no-advection",
        action="store_true",
        help="disable velocity advection in the thermal finite-volume step",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _aligned_cell_count(name: str, length_m: float, spacing_m: float) -> int:
    cells = length_m / spacing_m
    rounded = round(cells)
    if not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must align with lattice-cell faces")
    return int(rounded)


def _integer_bin_fraction(count: int, total: int) -> float:
    """Select the midpoint of the fraction bin that discretizes to ``count``."""

    return (int(count) + 0.5) / int(total)


def create_config(args: argparse.Namespace | None = None) -> IceFlowConfig:
    """Create the default short-scale coupled configuration without Taichi.

    Passing no arguments constructs the documented 80 by 112 benchmark.  A
    parsed namespace can be supplied by ``main`` or by tests exercising custom
    resolutions.  Geometry lengths must remain aligned with cell faces because
    the initial sharp ice mask is an analytic rectangle.
    """

    if args is None:
        args = parse_args([])

    nx = int(args.resolution_x)
    ny = int(args.resolution_y)
    boundary = int(args.boundary_cells)
    if nx <= 2 * boundary + 4 or ny <= 2 * boundary + 4:
        raise ValueError("resolution is too small for the requested boundary layers")

    width_m = _positive("domain_width_mm", args.domain_width_mm) * 1.0e-3
    height_m = _positive("domain_height_mm", args.domain_height_mm) * 1.0e-3
    dx = width_m / nx
    dy = height_m / ny
    if not math.isclose(dx, dy, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("the coupled LBM/thermal example requires square cells")

    water_level_m = _positive("water_level_mm", args.water_level_mm) * 1.0e-3
    ice_size_m = _positive("ice_size_mm", args.ice_size_mm) * 1.0e-3
    ice_center_y_m = _positive("ice_center_y_mm", args.ice_center_y_mm) * 1.0e-3
    if water_level_m >= height_m:
        raise ValueError("water level must leave an air layer below the top wall")

    water_height_cells = _aligned_cell_count("water level", water_level_m, dx)
    ice_width_cells = _aligned_cell_count("ice size in x", ice_size_m, dx)
    ice_height_cells = _aligned_cell_count("ice size in y", ice_size_m, dy)
    center_y_cells = ice_center_y_m / dy
    base_y_cells = center_y_cells - 0.5 * ice_height_cells
    if not math.isclose(base_y_cells, round(base_y_cells), abs_tol=1.0e-9):
        raise ValueError("ice base must align with lattice-cell faces")
    base_y_cells = float(round(base_y_cells))

    if ice_width_cells < 4 or ice_height_cells < 4:
        raise ValueError("ice must span at least four cells in each direction")
    if water_height_cells >= ny - boundary:
        raise ValueError("water level must lie below the top wall")
    if base_y_cells < boundary:
        raise ValueError("ice overlaps the bottom wall")
    ice_top_cells = base_y_cells + ice_height_cells
    if ice_top_cells >= water_height_cells:
        raise ValueError("the fixed ice must start fully below the water surface")
    interface_clearance = max(2, int(math.ceil(float(args.interface_width))))
    if water_height_cells - ice_top_cells < interface_clearance:
        raise ValueError(
            "the ice needs at least one diffuse-interface width of water cover"
        )

    properties = PhaseChangeProperties(
        melting_temperature_c=args.melting_temperature_c,
        specific_heat_water_j_kg_k=args.water_specific_heat,
        specific_heat_ice_j_kg_k=args.ice_specific_heat,
        specific_heat_air_j_kg_k=args.air_specific_heat,
        conductivity_water_w_m_k=args.water_conductivity,
        conductivity_ice_w_m_k=args.ice_conductivity,
        conductivity_air_w_m_k=args.air_conductivity,
        latent_heat_j_kg=args.latent_heat,
    )
    hot_wall = ThermalBoundary.dirichlet(args.water_temperature_c)
    thermal = ThermalConfig(
        properties=properties,
        boundaries=ThermalBoundarySet(
            left=hot_wall,
            right=hot_wall,
            bottom=hot_wall,
            top=ThermalBoundary.adiabatic(),
        ),
        initial_water_temperature_c=args.water_temperature_c,
        initial_ice_temperature_c=args.ice_temperature_c,
        initial_air_temperature_c=args.air_temperature_c,
        advection_enabled=not args.no_advection,
        # The air cap is hydrodynamic volume accommodation.  Treating its
        # interface as thermally insulating removes the restrictive air
        # diffusivity without changing the short-time submerged-ice result.
        water_air_interface_adiabatic=True,
        update_interval_lbm_steps=args.thermal_update_interval,
        solid_liquid_threshold=0.5,
        max_fourier_number=0.15,
        max_courant_number=0.50,
        buoyancy_reference_temperature_c=args.melting_temperature_c,
    )

    # A zero mechanical gravity makes this first benchmark a controlled
    # zero-flow coupling check.  LatticeScales intentionally retains the
    # solver's 9.8 m/s^2 fallback for time-unit conversion when gravity is zero.
    return create_iceflow_config(
        resolution=(nx, ny),
        dx=dx,
        reference_length_cells=ny,
        rho_water=args.density_water,
        rho_air=args.density_air,
        rho_ice=args.density_ice,
        viscosity_water=args.kinematic_viscosity_water,
        viscosity_air=args.kinematic_viscosity_air,
        sigma=args.surface_tension,
        gravity=(0.0, 0.0),
        interface_width=args.interface_width,
        mobility=args.mobility,
        phase_warmup_steps=args.phase_warmup_steps,
        well_balanced_hydrostatics=False,
        water_width_fraction=_integer_bin_fraction(nx - boundary, nx),
        water_height_fraction=_integer_bin_fraction(water_height_cells, ny),
        ice_width_fraction=_integer_bin_fraction(ice_width_cells, nx),
        ice_height_fraction=_integer_bin_fraction(ice_height_cells, ny),
        ice_base_y_cells=base_y_cells,
        ice_fixed=True,
        thermal=thermal,
        volume_projection_tolerance=1.0e-8,
        volume_projection_max_shift=0.50,
        output_dir=str(args.output_dir),
        show_gui=False,
        save_npz=False,
    )


def _requested_snapshot_times(end_time_s: float, interval_s: float) -> list[float]:
    end = _positive("end_time_s", end_time_s)
    interval = _positive("output_interval_s", interval_s)
    times = [0.0]
    count = int(math.floor(end / interval + 1.0e-12))
    times.extend(index * interval for index in range(1, count + 1))
    if not math.isclose(times[-1], end, rel_tol=0.0, abs_tol=1.0e-12):
        times.append(end)
    return times


def _snapshot_step_targets(
    config: IceFlowConfig, end_time_s: float, interval_s: float
) -> list[int]:
    scales = LatticeScales.from_iceflow_config(config)
    update_interval = int(config.thermal.update_interval_lbm_steps)
    targets = [0]
    for requested_time in _requested_snapshot_times(end_time_s, interval_s)[1:]:
        updates = math.ceil(requested_time / (scales.dt_s * update_interval) - 1.0e-12)
        target = max(update_interval, int(updates) * update_interval)
        if target > targets[-1]:
            targets.append(target)
    return targets


def _capture_snapshot(
    simulation,
    *,
    initial_total_enthalpy_j_m: float,
    initial_solid_volume_cells: float,
) -> CoupledSnapshot:
    """Synchronize diagnostics and copy Taichi's ``(nx, ny)`` fields to host."""

    solid_volume = float(simulation.phase_change_solid_volume_cells())
    sharp_geometry_volume = float(simulation.phase_change_geometry_volume_cells())
    initial_solid = float(initial_solid_volume_cells)
    melted_solid = max(0.0, initial_solid - solid_volume)
    density_ratio = float(simulation.cfg.rho_ice / simulation.cfg.rho_water)
    melted_fraction = 0.0 if initial_solid <= 0.0 else melted_solid / initial_solid
    dx = float(simulation.cfg.dx)
    solid_area = solid_volume * dx * dx
    initial_side = math.sqrt(initial_solid * dx * dx)
    equivalent_side = math.sqrt(max(0.0, solid_area))

    total_enthalpy = float(simulation.thermal.total_enthalpy_j_m(simulation.wall))
    boundary_heat = float(simulation.thermal.boundary_heat_input_j_m[None])
    energy_residual = total_enthalpy - float(initial_total_enthalpy_j_m) - boundary_heat
    target = float(simulation.water_volume_target[None])
    current = float(simulation.water_volume_current[None])

    # Taichi uses (nx, ny); files and Matplotlib use conventional (ny, nx).
    def scalar_host(field, dtype=None):
        values = np.asarray(field.to_numpy(), dtype=dtype)
        return np.ascontiguousarray(values.T)

    velocity = np.asarray(simulation.u.to_numpy(), dtype=np.float32)
    velocity = np.ascontiguousarray(np.transpose(velocity, (1, 0, 2)))
    return CoupledSnapshot(
        physical_time_s=float(simulation.physical_time_s),
        thermal_time_s=float(simulation.thermal.time_s),
        lbm_steps=int(simulation.steps),
        thermal_substeps=int(simulation.thermal.steps),
        temperature_c=scalar_host(simulation.temperature, np.float64),
        liquid_fraction=scalar_host(simulation.liquid_fraction, np.float32),
        enthalpy_j_m3=scalar_host(simulation.thermal_enthalpy, np.float64),
        water_phase=scalar_host(simulation.phi, np.float32),
        solid=scalar_host(simulation.solid, np.int8),
        sdf=scalar_host(simulation.sdf, np.float32),
        phase_change_material=scalar_host(simulation.phase_change_material, np.int8),
        velocity_lattice=velocity,
        solid_volume_cells=solid_volume,
        sharp_geometry_volume_cells=sharp_geometry_volume,
        solid_area_m2=solid_area,
        melted_fraction=melted_fraction,
        equivalent_square_side_m=equivalent_side,
        equivalent_uniform_melt_depth_m=0.5 * (initial_side - equivalent_side),
        generated_water_volume_cells=density_ratio * melted_solid,
        phase_change_contraction_cells=(1.0 - density_ratio) * melted_solid,
        water_volume_target_cells=target,
        water_volume_current_cells=current,
        water_volume_residual_cells=current - target,
        total_enthalpy_j_m=total_enthalpy,
        boundary_heat_input_j_m=boundary_heat,
        energy_residual_j_m=energy_residual,
    )


def write_history_csv(path: Path, snapshots: list[CoupledSnapshot]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "physical_time_s",
                "thermal_time_s",
                "lbm_steps",
                "thermal_substeps",
                "solid_volume_cells",
                "sharp_geometry_volume_cells",
                "solid_area_m2",
                "solid_area_mm2",
                "melted_fraction",
                "equivalent_square_side_m",
                "equivalent_uniform_melt_depth_m",
                "generated_water_volume_cells",
                "phase_change_contraction_cells",
                "water_volume_target_cells",
                "water_volume_current_cells",
                "water_volume_residual_cells",
                "total_enthalpy_j_m",
                "boundary_heat_input_j_m",
                "energy_residual_j_m",
            )
        )
        for item in snapshots:
            writer.writerow(
                (
                    f"{item.physical_time_s:.17g}",
                    f"{item.thermal_time_s:.17g}",
                    item.lbm_steps,
                    item.thermal_substeps,
                    f"{item.solid_volume_cells:.17g}",
                    f"{item.sharp_geometry_volume_cells:.17g}",
                    f"{item.solid_area_m2:.17g}",
                    f"{item.solid_area_m2 * 1.0e6:.17g}",
                    f"{item.melted_fraction:.17g}",
                    f"{item.equivalent_square_side_m:.17g}",
                    f"{item.equivalent_uniform_melt_depth_m:.17g}",
                    f"{item.generated_water_volume_cells:.17g}",
                    f"{item.phase_change_contraction_cells:.17g}",
                    f"{item.water_volume_target_cells:.17g}",
                    f"{item.water_volume_current_cells:.17g}",
                    f"{item.water_volume_residual_cells:.17g}",
                    f"{item.total_enthalpy_j_m:.17g}",
                    f"{item.boundary_heat_input_j_m:.17g}",
                    f"{item.energy_residual_j_m:.17g}",
                )
            )


def write_fields_npz(
    path: Path, config: IceFlowConfig, snapshots: list[CoupledSnapshot]
) -> None:
    x_m = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx
    y_m = (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dx
    np.savez_compressed(
        path,
        x_m=x_m,
        y_m=y_m,
        physical_time_s=np.asarray([item.physical_time_s for item in snapshots]),
        thermal_time_s=np.asarray([item.thermal_time_s for item in snapshots]),
        lbm_steps=np.asarray([item.lbm_steps for item in snapshots], dtype=np.int64),
        temperature_c=np.stack([item.temperature_c for item in snapshots]),
        liquid_fraction=np.stack([item.liquid_fraction for item in snapshots]),
        enthalpy_j_m3=np.stack([item.enthalpy_j_m3 for item in snapshots]),
        water_phase=np.stack([item.water_phase for item in snapshots]),
        solid=np.stack([item.solid for item in snapshots]),
        sdf=np.stack([item.sdf for item in snapshots]),
        phase_change_material=np.stack(
            [item.phase_change_material for item in snapshots]
        ),
        velocity_lattice=np.stack([item.velocity_lattice for item in snapshots]),
        solid_volume_cells=np.asarray(
            [item.solid_volume_cells for item in snapshots], dtype=np.float64
        ),
        sharp_geometry_volume_cells=np.asarray(
            [item.sharp_geometry_volume_cells for item in snapshots],
            dtype=np.float64,
        ),
        solid_area_m2=np.asarray(
            [item.solid_area_m2 for item in snapshots], dtype=np.float64
        ),
        melted_fraction=np.asarray(
            [item.melted_fraction for item in snapshots], dtype=np.float64
        ),
        generated_water_volume_cells=np.asarray(
            [item.generated_water_volume_cells for item in snapshots], dtype=np.float64
        ),
        phase_change_contraction_cells=np.asarray(
            [item.phase_change_contraction_cells for item in snapshots],
            dtype=np.float64,
        ),
        water_volume_target_cells=np.asarray(
            [item.water_volume_target_cells for item in snapshots], dtype=np.float64
        ),
        water_volume_current_cells=np.asarray(
            [item.water_volume_current_cells for item in snapshots], dtype=np.float64
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
    )


def write_final_plot(
    path: Path, config: IceFlowConfig, snapshot: CoupledSnapshot
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent_mm = (
        0.0,
        config.nx * config.dx * 1.0e3,
        0.0,
        config.ny * config.dx * 1.0e3,
    )
    wall = snapshot.solid.copy()
    # The static container is not included in the snapshot dataclass.  Its
    # known boundary thickness is enough to mask plotting-only wall cells.
    boundary = int(config.boundary_cells)
    wall[:, :] = 0
    wall[:boundary, :] = 1
    wall[-boundary:, :] = 1
    wall[:, :boundary] = 1
    wall[:, -boundary:] = 1

    temperature = np.ma.masked_where(wall != 0, snapshot.temperature_c)
    phase = np.ma.masked_where(
        (wall != 0) | (snapshot.phase_change_material == 0),
        snapshot.liquid_fraction,
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 6.0), constrained_layout=True)
    temperature_image = axes[0].imshow(
        temperature,
        origin="lower",
        extent=extent_mm,
        cmap="inferno",
        vmin=config.thermal.properties.melting_temperature_c,
        vmax=config.thermal.initial_water_temperature_c,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].contour(
        phase,
        levels=(config.thermal.solid_liquid_threshold,),
        colors="cyan",
        linewidths=1.1,
        origin="lower",
        extent=extent_mm,
    )
    axes[0].contour(
        snapshot.water_phase,
        levels=(0.5,),
        colors="white",
        linestyles="--",
        linewidths=0.9,
        origin="lower",
        extent=extent_mm,
    )
    axes[0].set_title("Final temperature")
    figure.colorbar(temperature_image, ax=axes[0], label="temperature (°C)")

    phase_image = axes[1].imshow(
        phase,
        origin="lower",
        extent=extent_mm,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].contour(
        phase,
        levels=(config.thermal.solid_liquid_threshold,),
        colors="orange",
        linewidths=1.1,
        origin="lower",
        extent=extent_mm,
    )
    axes[1].contour(
        snapshot.water_phase,
        levels=(0.5,),
        colors="black",
        linestyles="--",
        linewidths=0.9,
        origin="lower",
        extent=extent_mm,
    )
    axes[1].set_title("Final liquid fraction")
    figure.colorbar(phase_image, ax=axes[1], label="liquid fraction")

    for axis in axes:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.suptitle(
        "Coupled fixed-ice melting, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%, "
        f"water residual={snapshot.water_volume_residual_cells:.2e} cells"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    config: IceFlowConfig,
    scales: LatticeScales,
    targets: list[int],
    snapshots: list[CoupledSnapshot],
) -> None:
    initial = snapshots[0]
    final = snapshots[-1]
    density_ratio = config.rho_ice / config.rho_water
    data = {
        "model": "coupled two-dimensional fixed-ice melting",
        "backend": "taichi-cuda",
        "hydrodynamics": "air-water phase-field D2Q9 LBM",
        "thermal_solver": "explicit conservative finite-volume enthalpy with optional advection",
        "mechanics": "fixed pose; thermally changing sharp ice mask",
        "phase_change_mass_conversion": (
            "delta_V_water=(rho_ice/rho_water)*(V_solid_initial-V_solid_current)"
        ),
        "active_lbm_water_target": (
            "Vw0+(1-rho_ice/rho_water)*(Vs-Vs0)-(Vsharp-Vsharp0)"
        ),
        "thermal_boundaries": {
            "left": f"Dirichlet {config.thermal.boundaries.left.value:g} degC",
            "right": f"Dirichlet {config.thermal.boundaries.right.value:g} degC",
            "bottom": f"Dirichlet {config.thermal.boundaries.bottom.value:g} degC",
            "top": "adiabatic",
            "water_air_interface": "adiabatic",
        },
        "assumptions": [
            "one fixed connected ice body",
            "real ice/water density ratio with free-surface volume accommodation",
            "zero mechanical gravity for the first controlled coupling check",
            "zero surface tension to suppress unrelated capillary transients",
            "air is hydrodynamic but excluded from heat conduction",
            "two-dimensional unit out-of-plane depth",
        ],
        "requested": {
            "end_time_s": float(args.end_time_s),
            "output_interval_s": float(args.output_interval_s),
        },
        "geometry": {
            "domain_width_m": config.nx * config.dx,
            "domain_height_m": config.ny * config.dx,
            "water_level_m": config.water_height * config.dx,
            "ice_width_m": config.ice_width * config.dx,
            "ice_height_m": config.ice_height * config.dx,
            "ice_initial_center_m": [
                config.ice_initial_center[0] * config.dx,
                config.ice_initial_center[1] * config.dx,
            ],
        },
        "lattice_scaling": {
            "dx_m": scales.dx_m,
            "dt_s": scales.dt_s,
            "velocity_scale_m_s": scales.velocity_scale_m_s,
            "reference_velocity_m_s": scales.reference_velocity_m_s,
            "scheduled_snapshot_steps": targets,
        },
        "density_ratio_ice_to_water": density_ratio,
        "config": config.to_dict(),
        "results": {
            "snapshots": len(snapshots),
            "lbm_steps": final.lbm_steps,
            "thermal_substeps": final.thermal_substeps,
            "physical_time_s": final.physical_time_s,
            "thermal_time_s": final.thermal_time_s,
            "initial_solid_volume_cells": initial.solid_volume_cells,
            "final_solid_volume_cells": final.solid_volume_cells,
            "initial_sharp_geometry_volume_cells": (
                initial.sharp_geometry_volume_cells
            ),
            "final_sharp_geometry_volume_cells": final.sharp_geometry_volume_cells,
            "melted_fraction": final.melted_fraction,
            "solid_area_m2": final.solid_area_m2,
            "equivalent_square_side_m": final.equivalent_square_side_m,
            "equivalent_uniform_melt_depth_m": (final.equivalent_uniform_melt_depth_m),
            "generated_water_volume_cells": final.generated_water_volume_cells,
            "total_liquid_water_volume_cells": (
                initial.water_volume_target_cells + final.generated_water_volume_cells
            ),
            "phase_change_contraction_cells": (final.phase_change_contraction_cells),
            "water_volume_target_cells": final.water_volume_target_cells,
            "water_volume_current_cells": final.water_volume_current_cells,
            "water_volume_residual_cells": final.water_volume_residual_cells,
            "total_enthalpy_j_m": final.total_enthalpy_j_m,
            "boundary_heat_input_j_m": final.boundary_heat_input_j_m,
            "energy_residual_j_m": final.energy_residual_j_m,
        },
        "outputs": {
            "history": "history.csv",
            "fields": None if args.no_npz else "fields.npz",
            "plot": None if args.no_plot else "coupled_fixed_ice_melting_2d.png",
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = create_config(args)
        targets = _snapshot_step_targets(
            config, args.end_time_s, args.output_interval_s
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is the first import in the module that loads simulator.py/Taichi.
    from iceflow2d import IceFlow2D

    simulation = IceFlow2D(config)
    scales = LatticeScales.from_iceflow_config(config)
    initial_total_enthalpy = float(
        simulation.thermal.total_enthalpy_j_m(simulation.wall)
    )
    initial_solid_volume = float(simulation.phase_change_solid_volume_cells())
    snapshots = [
        _capture_snapshot(
            simulation,
            initial_total_enthalpy_j_m=initial_total_enthalpy,
            initial_solid_volume_cells=initial_solid_volume,
        )
    ]

    for target in targets[1:]:
        simulation.step(target - simulation.steps)
        snapshot = _capture_snapshot(
            simulation,
            initial_total_enthalpy_j_m=initial_total_enthalpy,
            initial_solid_volume_cells=initial_solid_volume,
        )
        snapshots.append(snapshot)
        if not args.quiet:
            print(
                f"t={snapshot.physical_time_s:.4f} s | "
                f"steps={snapshot.lbm_steps} | "
                f"melted={100.0 * snapshot.melted_fraction:6.2f}% | "
                "water residual="
                f"{snapshot.water_volume_residual_cells:+.3e} cells",
                flush=True,
            )

    write_history_csv(output_dir / "history.csv", snapshots)
    if not args.no_npz:
        write_fields_npz(output_dir / "fields.npz", config, snapshots)
    if not args.no_plot:
        write_final_plot(
            output_dir / "coupled_fixed_ice_melting_2d.png", config, snapshots[-1]
        )
    write_metadata(
        output_dir / "metadata.json",
        args=args,
        config=config,
        scales=scales,
        targets=targets,
        snapshots=snapshots,
    )

    if not args.quiet:
        final = snapshots[-1]
        print("Coupled fixed-ice melting example complete")
        print(
            f"  domain: {config.nx * config.dx * 1.0e3:.3f} x "
            f"{config.ny * config.dx * 1.0e3:.3f} mm, "
            f"grid={config.nx} x {config.ny}"
        )
        print(
            f"  simulated: {final.physical_time_s:.6f} s in {final.lbm_steps} LBM steps"
        )
        print(
            f"  ice melted: {100.0 * final.melted_fraction:.3f}%, "
            "equivalent melt depth="
            f"{final.equivalent_uniform_melt_depth_m * 1.0e3:.6f} mm"
        )
        print(f"  output: {output_dir}")


if __name__ == "__main__":
    main()
