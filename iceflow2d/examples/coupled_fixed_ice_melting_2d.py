"""Coupled LBM/enthalpy example for a fixed two-dimensional melting ice block.

The short centimetre-scale case is intended as the first end-to-end thermal
coupling check.  A 0.008 m square ice block is held below a water/air free surface
inside a 0.025 m by 0.035 m container.  The real ice/water density ratio changes
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

from iceflow2d.config import (  # noqa: E402
    DEFAULT_REFERENCE_VELOCITY_M_S,
    IceFlowConfig,
    create_iceflow_config,
)
from iceflow2d.thermal import (  # noqa: E402
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
)


DEFAULT_OUTPUT_DIR = "outputs/iceflow2d/coupled_fixed_ice_melting_2d"
DEFAULT_BOUNDARY_CELLS = 3
DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S = 9e-6
DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1 = 0.2


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
    physical_velocity_lattice: np.ndarray
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


@dataclass(frozen=True, slots=True)
class VelocitySequenceSummary:
    """Files and the fixed scale used by the velocity visualization."""

    frame_directory: str
    frame_pattern: str
    frame_count: int
    animation: str
    observed_max_m_s: float
    color_max_m_s: float
    quiver_stride_cells: int


@dataclass(frozen=True, slots=True)
class VorticitySequenceSummary:
    """Files and the fixed scale used by the vorticity visualization."""

    frame_directory: str
    frame_pattern: str
    frame_count: int
    animation: str
    observed_abs_max_s_1: float
    color_abs_max_s_1: float
    quiver_stride_cells: int


@dataclass(frozen=True, slots=True)
class TemperatureSequenceSummary:
    """Files and the fixed scale used by the temperature visualization."""

    frame_directory: str
    frame_pattern: str
    frame_count: int
    animation: str
    observed_min_c: float
    observed_max_c: float
    color_min_c: float
    color_max_c: float


@dataclass(frozen=True, slots=True)
class VisualizationLimits:
    """Color and vector scales shared by every case in one invocation."""

    velocity_max_m_s: float
    vorticity_abs_max_s_1: float
    temperature_min_c: float
    temperature_max_c: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Taichi CUDA air-water LBM coupled to finite-volume enthalpy "
            "melting of a fixed two-dimensional ice block"
        )
    )
    parser.add_argument("--domain-width-m", type=float, default=0.025)
    parser.add_argument("--domain-height-m", type=float, default=0.035)
    parser.add_argument("--resolution-x", type=int, default=200)
    parser.add_argument("--resolution-y", type=int, default=280)
    parser.add_argument("--water-level-m", type=float, default=0.0275)
    parser.add_argument("--ice-size-m", type=float, default=0.008)
    parser.add_argument("--ice-center-y-m", type=float, default=0.015)
    parser.add_argument("--boundary-cells", type=int, default=DEFAULT_BOUNDARY_CELLS)
    parser.add_argument(
        "--reference-velocity-m-s",
        type=float,
        default=DEFAULT_REFERENCE_VELOCITY_M_S,
        help="physical speed represented by the fixed reference speed 0.1 LU",
    )

    parser.add_argument("--end-time-s", type=float, default=30.0)
    parser.add_argument(
        "--output-interval-s",
        type=float,
        default=0.1,
        help=(
            "physical-time spacing requested for history rows and streamed "
            "velocity/vorticity/temperature frames"
        ),
    )
    parser.add_argument(
        "--velocity-visualization-max-m-s",
        type=float,
        default=DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S,
        help=(
            "shared speed colorbar maximum and quiver reference speed for "
            "every frame and temperature case"
        ),
    )
    parser.add_argument(
        "--vorticity-visualization-max-s-1",
        type=float,
        default=DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
        help=(
            "shared absolute vorticity colorbar maximum for every frame and "
            "temperature case"
        ),
    )
    parser.add_argument("--water-temperature-c", type=float, default=60.0)
    parser.add_argument(
        "--water-temperatures-c",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help=(
            "run one case per listed far-field water temperature; this takes "
            "precedence over --water-temperature-c and writes T_*C subdirectories"
        ),
    )
    parser.add_argument("--ice-temperature-c", type=float, default=0.0)
    parser.add_argument("--air-temperature-c", type=float, default=40.0)
    parser.add_argument("--melting-temperature-c", type=float, default=0.0)
    parser.add_argument(
        "--gravity-m-s2",
        type=float,
        default=0.0,
        help="non-negative gravitational acceleration directed downward",
    )
    parser.add_argument(
        "--well-balanced-hydrostatics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable/disable well-balanced hydrostatics (default: on when gravity > 0)",
    )
    parser.add_argument(
        "--water-buoyancy-model",
        choices=("linear", "freshwater_quadratic"),
        default="freshwater_quadratic",
    )
    parser.add_argument(
        "--thermal-expansion-water-1-k",
        type=float,
        default=2.1e-4,
        help="linear-EOS expansion coefficient",
    )
    parser.add_argument(
        "--freshwater-density-max-temperature-c",
        type=float,
        default=4.0,
        help="temperature of maximum freshwater density for the quadratic EOS",
    )
    parser.add_argument(
        "--freshwater-density-quadratic-coefficient-1-k2",
        type=float,
        default=8.0e-6,
        help="beta in rho=rho_star*[1-beta*(T-T_star)^2]",
    )

    parser.add_argument("--density-water", type=float, default=1000.0)
    parser.add_argument("--density-ice", type=float, default=917.0)
    parser.add_argument("--density-air", type=float, default=1.25)
    # IceFlow2D converts these physical kinematic viscosities using the fixed
    # reference speed and the present 1.25e-4 m spacing.
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help=(
            "disable the final plot and all velocity/vorticity/temperature "
            "PNG and GIF sequences"
        ),
    )
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "retain all sampled fields and write fields.npz; disabled by "
            "default because it requires memory proportional to the frame count"
        ),
    )
    parser.add_argument(
        "--no-npz",
        dest="save_npz",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(save_npz=False)
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _non_negative(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _shared_visualization_limits(
    args: argparse.Namespace,
    temperatures_c: list[float] | tuple[float, ...],
) -> VisualizationLimits:
    """Resolve plot limits once so every temperature case uses the same scale."""

    if not temperatures_c:
        raise ValueError("at least one water temperature is required")
    temperatures = [float(value) for value in temperatures_c]
    if not all(math.isfinite(value) for value in temperatures):
        raise ValueError("visualization temperatures must be finite")
    temperature_min = float(args.melting_temperature_c)
    if not math.isfinite(temperature_min):
        raise ValueError("melting_temperature_c must be finite")
    temperature_max = max(temperatures)
    if temperature_max < temperature_min:
        raise ValueError(
            "temperature visualization maximum must not be below "
            "the melting temperature"
        )
    velocity_max = _positive(
        "velocity_visualization_max_m_s",
        args.velocity_visualization_max_m_s,
    )
    vorticity_max = _positive(
        "vorticity_visualization_max_s_1",
        args.vorticity_visualization_max_s_1,
    )
    return VisualizationLimits(
        velocity_max_m_s=velocity_max,
        vorticity_abs_max_s_1=vorticity_max,
        temperature_min_c=temperature_min,
        temperature_max_c=temperature_max,
    )


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
    """Create the default centimetre-scale configuration without Taichi.

    Passing no arguments constructs the documented 200 by 280 benchmark.  A
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

    width_m = _positive("domain_width_m", args.domain_width_m)
    height_m = _positive("domain_height_m", args.domain_height_m)
    dx = width_m / nx
    dy = height_m / ny
    if not math.isclose(dx, dy, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("the coupled LBM/thermal example requires square cells")

    water_level_m = _positive("water_level_m", args.water_level_m)
    ice_size_m = _positive("ice_size_m", args.ice_size_m)
    ice_center_y_m = _positive("ice_center_y_m", args.ice_center_y_m)
    reference_velocity_m_s = _positive(
        "reference_velocity_m_s", args.reference_velocity_m_s
    )
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
        water_buoyancy_model=args.water_buoyancy_model,
        thermal_expansion_water_1_k=args.thermal_expansion_water_1_k,
        # The bath temperature is T_inf in the anomalous-buoyancy source.
        buoyancy_reference_temperature_c=args.water_temperature_c,
        freshwater_density_max_temperature_c=(
            args.freshwater_density_max_temperature_c
        ),
        freshwater_density_quadratic_coefficient_1_k2=(
            args.freshwater_density_quadratic_coefficient_1_k2
        ),
    )

    gravity_m_s2 = _non_negative("gravity_m_s2", args.gravity_m_s2)
    well_balanced = args.well_balanced_hydrostatics
    if well_balanced is None:
        well_balanced = gravity_m_s2 > 0.0

    # Zero gravity remains the legacy controlled coupling default.  Supplying
    # gravity activates a vertical pool and well-balanced hydrostatics unless
    # the caller explicitly disables the latter.
    return create_iceflow_config(
        resolution=(nx, ny),
        dx=dx,
        reference_velocity=reference_velocity_m_s,
        rho_water=args.density_water,
        rho_air=args.density_air,
        rho_ice=args.density_ice,
        viscosity_water=args.kinematic_viscosity_water,
        viscosity_air=args.kinematic_viscosity_air,
        sigma=args.surface_tension,
        gravity=(0.0, -gravity_m_s2),
        interface_width=args.interface_width,
        mobility=args.mobility,
        phase_warmup_steps=args.phase_warmup_steps,
        well_balanced_hydrostatics=bool(well_balanced),
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

    momentum_velocity = np.asarray(simulation.u.to_numpy(), dtype=np.float32)
    force = np.asarray(simulation.fluid_force.to_numpy(), dtype=np.float32)
    # The physical velocity in the forced LBM is the momentum velocity plus
    # the standard half-force correction.  Keep the legacy raw-u field while
    # storing this physical field for visualization and thermal postprocessing.
    physical_velocity = momentum_velocity + np.float32(0.5) * force
    momentum_velocity = np.ascontiguousarray(np.transpose(momentum_velocity, (1, 0, 2)))
    physical_velocity = np.ascontiguousarray(np.transpose(physical_velocity, (1, 0, 2)))
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
        velocity_lattice=momentum_velocity,
        physical_velocity_lattice=physical_velocity,
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


HISTORY_COLUMNS = (
    "physical_time_s",
    "thermal_time_s",
    "lbm_steps",
    "thermal_substeps",
    "solid_volume_cells",
    "sharp_geometry_volume_cells",
    "solid_area_m2",
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


def _history_row(item: CoupledSnapshot) -> tuple[object, ...]:
    return (
        f"{item.physical_time_s:.17g}",
        f"{item.thermal_time_s:.17g}",
        item.lbm_steps,
        item.thermal_substeps,
        f"{item.solid_volume_cells:.17g}",
        f"{item.sharp_geometry_volume_cells:.17g}",
        f"{item.solid_area_m2:.17g}",
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


def write_history_csv(path: Path, snapshots: list[CoupledSnapshot]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HISTORY_COLUMNS)
        writer.writerows(_history_row(item) for item in snapshots)


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
        physical_velocity_lattice=np.stack(
            [item.physical_velocity_lattice for item in snapshots]
        ),
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


def _static_wall_mask(config: IceFlowConfig) -> np.ndarray:
    """Return the plotting mask for the fixed container walls."""

    wall = np.zeros((config.ny, config.nx), dtype=bool)
    boundary = int(config.boundary_cells)
    wall[:boundary, :] = True
    wall[-boundary:, :] = True
    wall[:, :boundary] = True
    wall[:, -boundary:] = True
    return wall


def _physical_water_velocity(
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical velocity and the visible liquid-water mask for one frame."""

    velocity = np.asarray(snapshot.physical_velocity_lattice, dtype=np.float64)
    expected_vector_shape = (config.ny, config.nx, 2)
    if velocity.shape != expected_vector_shape:
        raise ValueError(
            f"velocity frame {frame_index} has shape {velocity.shape}, "
            f"expected {expected_vector_shape}"
        )
    if not np.isfinite(velocity).all():
        raise ValueError(f"velocity frame {frame_index} contains NaN or infinity")

    solid = np.asarray(snapshot.solid)
    water_phase = np.asarray(snapshot.water_phase)
    expected_scalar_shape = (config.ny, config.nx)
    if (
        solid.shape != expected_scalar_shape
        or water_phase.shape != expected_scalar_shape
    ):
        raise ValueError(
            f"velocity frame {frame_index} masks must have shape "
            f"{expected_scalar_shape}"
        )
    water = (~_static_wall_mask(config)) & (solid == 0) & (water_phase >= 0.5)
    return velocity * scales.velocity_scale_m_s, water


def velocity_sequence_limits_m_s(
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshots: list[CoupledSnapshot],
    *,
    display_max_m_s: float | None = None,
) -> tuple[float, float]:
    """Return the observed maximum and the requested positive display limit."""

    if not snapshots:
        raise ValueError("at least one snapshot is required for velocity plotting")

    observed_max = 0.0
    for frame_index, snapshot in enumerate(snapshots):
        velocity_m_s, visible_water = _physical_water_velocity(
            config, scales, snapshot, frame_index=frame_index
        )
        if np.any(visible_water):
            speed_m_s = np.hypot(velocity_m_s[..., 0], velocity_m_s[..., 1])
            observed_max = max(observed_max, float(np.max(speed_m_s[visible_water])))

    if display_max_m_s is None:
        # Preserve the standalone helper's former automatic behavior.  The
        # example entry point always supplies one shared invocation-wide limit.
        color_max = observed_max if observed_max > 0.0 else 1.0e-12
    else:
        color_max = _positive("display_max_m_s", display_max_m_s)
    return observed_max, color_max


def _scaled_quiver_components(
    velocity_m_s: np.ndarray,
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    *,
    display_max_m_s: float,
    reference_arrow_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical velocity to plot-space arrows using one shared scale."""

    maximum = _positive("display_max_m_s", display_max_m_s)
    reference_length = _positive(
        "reference_arrow_length_m", reference_arrow_length_m
    )
    sampled = velocity_m_s[np.ix_(y_indices, x_indices)]
    factor = reference_length / maximum
    return sampled[..., 0] * factor, sampled[..., 1] * factor


def _masked_centered_axis_derivative(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    spacing_m: float,
    axis: int,
) -> np.ndarray:
    """Apply a centered derivative only when both neighbors are valid water."""

    scalar = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if scalar.shape != mask.shape or scalar.ndim != 2:
        raise ValueError(
            "masked derivative values and mask must be matching 2-D arrays"
        )
    if axis not in (0, 1):
        raise ValueError("masked derivative axis must be 0 or 1")
    spacing = _positive("spacing_m", spacing_m)

    forward_value = np.roll(scalar, -1, axis=axis)
    backward_value = np.roll(scalar, 1, axis=axis)
    forward_valid = np.roll(mask, -1, axis=axis)
    backward_valid = np.roll(mask, 1, axis=axis)
    forward_boundary = [slice(None), slice(None)]
    backward_boundary = [slice(None), slice(None)]
    forward_boundary[axis] = -1
    backward_boundary[axis] = 0
    forward_valid[tuple(forward_boundary)] = False
    backward_valid[tuple(backward_boundary)] = False

    derivative = np.full(scalar.shape, np.nan, dtype=np.float64)
    centered = mask & forward_valid & backward_valid
    derivative[centered] = (forward_value[centered] - backward_value[centered]) / (
        2.0 * spacing
    )
    return derivative


def _water_vorticity_s_1(
    velocity_m_s: np.ndarray,
    visible_water: np.ndarray,
    *,
    spacing_m: float,
) -> np.ndarray:
    r"""Return :math:`\omega_z=\partial_x v-\partial_y u` in inverse seconds."""

    velocity = np.asarray(velocity_m_s, dtype=np.float64)
    water = np.asarray(visible_water, dtype=bool)
    if velocity.shape != (*water.shape, 2) or water.ndim != 2:
        raise ValueError("velocity and visible-water mask shapes are incompatible")
    if not np.isfinite(velocity).all():
        raise ValueError("velocity contains NaN or infinity")
    dv_dx = _masked_centered_axis_derivative(
        velocity[..., 1], water, spacing_m=spacing_m, axis=1
    )
    du_dy = _masked_centered_axis_derivative(
        velocity[..., 0], water, spacing_m=spacing_m, axis=0
    )
    vorticity = dv_dx - du_dy
    vorticity[~water] = np.nan
    return vorticity


def vorticity_sequence_limits_s_1(
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshots: list[CoupledSnapshot],
    *,
    display_abs_max_s_1: float | None = None,
) -> tuple[float, float]:
    """Return observed and display maxima of absolute water vorticity."""

    if not snapshots:
        raise ValueError("at least one snapshot is required for vorticity plotting")

    observed_abs_max = 0.0
    for frame_index, snapshot in enumerate(snapshots):
        velocity_m_s, visible_water = _physical_water_velocity(
            config, scales, snapshot, frame_index=frame_index
        )
        vorticity_s_1 = _water_vorticity_s_1(
            velocity_m_s,
            visible_water,
            spacing_m=config.dx,
        )
        finite = np.isfinite(vorticity_s_1)
        if np.any(finite):
            observed_abs_max = max(
                observed_abs_max,
                float(np.max(np.abs(vorticity_s_1[finite]))),
            )

    if display_abs_max_s_1 is None:
        color_abs_max = observed_abs_max if observed_abs_max > 0.0 else 1.0e-12
    else:
        color_abs_max = _positive(
            "display_abs_max_s_1",
            display_abs_max_s_1,
        )
    return observed_abs_max, color_abs_max


def _contains_contour(values: np.ndarray, level: float) -> bool:
    finite = np.asarray(values)[np.isfinite(values)]
    return bool(finite.size and np.min(finite) < level < np.max(finite))


@dataclass(frozen=True, slots=True)
class _FlowPlotGeometry:
    extent_m: tuple[float, float, float, float]
    x_indices: np.ndarray
    y_indices: np.ndarray
    sample_x_m: np.ndarray
    sample_y_m: np.ndarray
    reference_arrow_length_m: float
    stride_cells: int


def _flow_plot_geometry(config: IceFlowConfig) -> _FlowPlotGeometry:
    extent_m = (
        0.0,
        config.nx * config.dx,
        0.0,
        config.ny * config.dx,
    )
    x_m = (np.arange(config.nx, dtype=np.float64) + 0.5) * config.dx
    y_m = (np.arange(config.ny, dtype=np.float64) + 0.5) * config.dx
    stride = max(1, int(math.ceil(max(config.nx, config.ny) / 24.0)))
    boundary = int(config.boundary_cells)
    x_indices = np.arange(boundary, config.nx - boundary, stride)
    y_indices = np.arange(boundary, config.ny - boundary, stride)
    sample_x, sample_y = np.meshgrid(x_m[x_indices], y_m[y_indices])
    return _FlowPlotGeometry(
        extent_m=extent_m,
        x_indices=x_indices,
        y_indices=y_indices,
        sample_x_m=sample_x,
        sample_y_m=sample_y,
        reference_arrow_length_m=max(
            0.75 * stride * config.dx,
            1.0e-12,
        ),
        stride_cells=stride,
    )


def _draw_flow_boundaries(
    axis: object,
    snapshot: CoupledSnapshot,
    extent_m: tuple[float, float, float, float],
    *,
    water_surface_color: str,
) -> None:
    signed_distance = getattr(snapshot, "sdf", None)
    if signed_distance is not None and _contains_contour(signed_distance, 0.0):
        axis.contour(
            signed_distance,
            levels=(0.0,),
            colors="cyan",
            linewidths=1.2,
            origin="lower",
            extent=extent_m,
        )
    else:
        solid = np.asarray(snapshot.solid)
        if _contains_contour(solid, 0.5):
            axis.contour(
                solid,
                levels=(0.5,),
                colors="cyan",
                linewidths=1.2,
                origin="lower",
                extent=extent_m,
            )
    water_phase = np.asarray(snapshot.water_phase)
    if _contains_contour(water_phase, 0.5):
        axis.contour(
            water_phase,
            levels=(0.5,),
            colors=water_surface_color,
            linestyles="--",
            linewidths=0.9,
            origin="lower",
            extent=extent_m,
        )


class _StreamingGif:
    """Append PNG frames to a GIF while retaining at most one decoded frame."""

    def __init__(self, path: Path, *, duration_s: float = 0.8) -> None:
        self.path = path
        self._duration_ms = int(round(1000.0 * _positive("duration_s", duration_s)))
        self._stream = path.open("wb")
        self._frame_count = 0
        self._closed = False

    def append_png(self, path: Path) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed GIF stream")
        from PIL import GifImagePlugin, Image

        with Image.open(path) as source:
            frame = source.convert("RGB").convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=256,
            )
        try:
            frame_info = {
                "duration": self._duration_ms,
                "disposal": 2,
                "optimize": False,
            }
            if self._frame_count == 0:
                frame_info["loop"] = 0
                for block in GifImagePlugin._get_global_header(frame, frame_info):
                    self._stream.write(block)
            else:
                frame_info["include_color_table"] = True
            GifImagePlugin._write_frame_data(
                self._stream,
                frame,
                (0, 0),
                frame_info,
            )
            self._stream.flush()
            self._frame_count += 1
        finally:
            frame.close()

    def close(self) -> None:
        if not self._closed:
            if self._frame_count > 0:
                self._stream.write(b";")
            self._stream.close()
            if self._frame_count == 0:
                self.path.unlink(missing_ok=True)
            self._closed = True


def _render_velocity_frame(
    path: Path,
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
    color_max_m_s: float,
    geometry: _FlowPlotGeometry,
) -> float:
    """Render one velocity frame and return its observed water-speed maximum."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    velocity_m_s, visible_water = _physical_water_velocity(
        config, scales, snapshot, frame_index=frame_index
    )
    speed_m_s = np.hypot(velocity_m_s[..., 0], velocity_m_s[..., 1])
    frame_observed_max = (
        float(np.max(speed_m_s[visible_water])) if np.any(visible_water) else 0.0
    )
    plotted_speed = np.ma.masked_where(~visible_water, speed_m_s)
    normalization = Normalize(vmin=0.0, vmax=color_max_m_s)
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#aeb4ba")

    figure, axis = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    speed_image = axis.imshow(
        plotted_speed,
        origin="lower",
        extent=geometry.extent_m,
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
        aspect="equal",
    )

    sampled_mask = visible_water[np.ix_(geometry.y_indices, geometry.x_indices)]
    arrow_u, arrow_v = _scaled_quiver_components(
        velocity_m_s,
        geometry.x_indices,
        geometry.y_indices,
        display_max_m_s=color_max_m_s,
        reference_arrow_length_m=geometry.reference_arrow_length_m,
    )
    sampled_speed = np.hypot(arrow_u, arrow_v)
    arrow_mask = sampled_mask & (sampled_speed > 0.0)
    arrow_u = np.ma.masked_where(~arrow_mask, arrow_u)
    arrow_v = np.ma.masked_where(~arrow_mask, arrow_v)
    if frame_observed_max > 0.0:
        quiver = axis.quiver(
            geometry.sample_x_m,
            geometry.sample_y_m,
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            color="white",
            width=0.004,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        axis.quiverkey(
            quiver,
            X=0.92,
            Y=0.04,
            U=geometry.reference_arrow_length_m,
            label=f"{color_max_m_s:.3e} m/s",
            labelpos="W",
            labelcolor="white",
            coordinates="axes",
        )
    else:
        axis.text(
            0.97,
            0.04,
            "all water velocities = 0",
            color="white",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    _draw_flow_boundaries(
        axis,
        snapshot,
        geometry.extent_m,
        water_surface_color="white",
    )
    figure.colorbar(
        speed_image,
        ax=axis,
        label="water speed |u| (m/s), shared across all cases",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        "Water velocity during fixed-ice melting\n"
        f"T_inf={config.thermal.buoyancy_reference_temperature_c:g} degC, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return frame_observed_max


class VelocitySequenceStream:
    """Write velocity PNG/GIF frames incrementally with constant frame memory."""

    def __init__(
        self,
        output_dir: Path,
        config: IceFlowConfig,
        scales: LatticeScales,
        *,
        display_max_m_s: float,
    ) -> None:
        self.config = config
        self.scales = scales
        self.color_max_m_s = _positive("display_max_m_s", display_max_m_s)
        self.geometry = _flow_plot_geometry(config)
        self.frame_directory = output_dir / "velocity_frames"
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frame_directory.glob("frame_*.png"):
            stale_frame.unlink()
        self.animation_path = output_dir / "velocity_field.gif"
        self._gif = _StreamingGif(self.animation_path)
        self.frame_count = 0
        self.observed_max_m_s = 0.0
        self._closed = False
        self._summary: VelocitySequenceSummary | None = None

    def append(self, snapshot: CoupledSnapshot) -> Path:
        if self._closed:
            raise RuntimeError("cannot append to a closed velocity sequence")
        frame_path = self.frame_directory / f"frame_{self.frame_count:05d}.png"
        frame_observed_max = _render_velocity_frame(
            frame_path,
            self.config,
            self.scales,
            snapshot,
            frame_index=self.frame_count,
            color_max_m_s=self.color_max_m_s,
            geometry=self.geometry,
        )
        self._gif.append_png(frame_path)
        self.observed_max_m_s = max(
            self.observed_max_m_s,
            frame_observed_max,
        )
        self.frame_count += 1
        return frame_path

    def close(self) -> VelocitySequenceSummary:
        if not self._closed:
            self._gif.close()
            self._summary = VelocitySequenceSummary(
                frame_directory=self.frame_directory.name,
                frame_pattern="frame_*.png",
                frame_count=self.frame_count,
                animation=self.animation_path.name,
                observed_max_m_s=self.observed_max_m_s,
                color_max_m_s=self.color_max_m_s,
                quiver_stride_cells=self.geometry.stride_cells,
            )
            self._closed = True
        assert self._summary is not None
        return self._summary


def write_velocity_sequence(
    output_dir: Path,
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshots: list[CoupledSnapshot],
    *,
    display_max_m_s: float | None = None,
) -> VelocitySequenceSummary:
    """Compatibility wrapper that streams an existing snapshot iterable."""

    if not snapshots:
        raise ValueError("at least one snapshot is required for velocity plotting")
    if display_max_m_s is None:
        _, display_max_m_s = velocity_sequence_limits_m_s(
            config,
            scales,
            snapshots,
        )
    stream = VelocitySequenceStream(
        output_dir,
        config,
        scales,
        display_max_m_s=display_max_m_s,
    )
    try:
        for snapshot in snapshots:
            stream.append(snapshot)
    except BaseException:
        stream.close()
        raise
    return stream.close()


def _render_vorticity_frame(
    path: Path,
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
    color_abs_max_s_1: float,
    quiver_speed_max_m_s: float,
    geometry: _FlowPlotGeometry,
) -> float:
    """Render one vorticity frame and return its observed absolute maximum."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    velocity_m_s, visible_water = _physical_water_velocity(
        config, scales, snapshot, frame_index=frame_index
    )
    vorticity_s_1 = _water_vorticity_s_1(
        velocity_m_s,
        visible_water,
        spacing_m=config.dx,
    )
    finite_vorticity = np.isfinite(vorticity_s_1)
    frame_observed_abs_max = (
        float(np.max(np.abs(vorticity_s_1[finite_vorticity])))
        if np.any(finite_vorticity)
        else 0.0
    )
    speed_m_s = np.hypot(velocity_m_s[..., 0], velocity_m_s[..., 1])
    frame_observed_velocity_max = (
        float(np.max(speed_m_s[visible_water])) if np.any(visible_water) else 0.0
    )
    plotted_vorticity = np.ma.masked_invalid(vorticity_s_1)
    normalization = Normalize(
        vmin=-color_abs_max_s_1,
        vmax=color_abs_max_s_1,
    )
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#aeb4ba")

    figure, axis = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    vorticity_image = axis.imshow(
        plotted_vorticity,
        origin="lower",
        extent=geometry.extent_m,
        cmap=colormap,
        norm=normalization,
        interpolation="nearest",
        aspect="equal",
    )

    sampled_mask = finite_vorticity[np.ix_(geometry.y_indices, geometry.x_indices)]
    arrow_u, arrow_v = _scaled_quiver_components(
        velocity_m_s,
        geometry.x_indices,
        geometry.y_indices,
        display_max_m_s=quiver_speed_max_m_s,
        reference_arrow_length_m=geometry.reference_arrow_length_m,
    )
    sampled_speed = np.hypot(arrow_u, arrow_v)
    arrow_mask = sampled_mask & (sampled_speed > 0.0)
    arrow_u = np.ma.masked_where(~arrow_mask, arrow_u)
    arrow_v = np.ma.masked_where(~arrow_mask, arrow_v)
    if frame_observed_velocity_max > 0.0:
        quiver = axis.quiver(
            geometry.sample_x_m,
            geometry.sample_y_m,
            arrow_u,
            arrow_v,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="mid",
            color="black",
            edgecolor="white",
            linewidth=0.35,
            width=0.004,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        axis.quiverkey(
            quiver,
            X=0.92,
            Y=0.04,
            U=geometry.reference_arrow_length_m,
            label=f"{quiver_speed_max_m_s:.3e} m/s",
            labelpos="W",
            labelcolor="black",
            coordinates="axes",
        )
    else:
        axis.text(
            0.97,
            0.04,
            "all water velocities = 0",
            color="white",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )

    _draw_flow_boundaries(
        axis,
        snapshot,
        geometry.extent_m,
        water_surface_color="black",
    )
    figure.colorbar(
        vorticity_image,
        ax=axis,
        label=(
            r"water vorticity $\omega_z=\partial_x v-\partial_y u$ "
            r"(s$^{-1}$), shared across all cases"
        ),
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        "Water vorticity during fixed-ice melting\n"
        f"T_inf={config.thermal.buoyancy_reference_temperature_c:g} degC, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return frame_observed_abs_max


class VorticitySequenceStream:
    """Write vorticity PNG/GIF frames incrementally with constant frame memory."""

    def __init__(
        self,
        output_dir: Path,
        config: IceFlowConfig,
        scales: LatticeScales,
        *,
        display_abs_max_s_1: float,
        velocity_display_max_m_s: float,
    ) -> None:
        self.config = config
        self.scales = scales
        self.color_abs_max_s_1 = _positive(
            "display_abs_max_s_1",
            display_abs_max_s_1,
        )
        self.quiver_speed_max_m_s = _positive(
            "velocity_display_max_m_s",
            velocity_display_max_m_s,
        )
        self.geometry = _flow_plot_geometry(config)
        self.frame_directory = output_dir / "vorticity_frames"
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frame_directory.glob("frame_*.png"):
            stale_frame.unlink()
        self.animation_path = output_dir / "vorticity_field.gif"
        self._gif = _StreamingGif(self.animation_path)
        self.frame_count = 0
        self.observed_abs_max_s_1 = 0.0
        self._closed = False
        self._summary: VorticitySequenceSummary | None = None

    def append(self, snapshot: CoupledSnapshot) -> Path:
        if self._closed:
            raise RuntimeError("cannot append to a closed vorticity sequence")
        frame_path = self.frame_directory / f"frame_{self.frame_count:05d}.png"
        frame_observed_abs_max = _render_vorticity_frame(
            frame_path,
            self.config,
            self.scales,
            snapshot,
            frame_index=self.frame_count,
            color_abs_max_s_1=self.color_abs_max_s_1,
            quiver_speed_max_m_s=self.quiver_speed_max_m_s,
            geometry=self.geometry,
        )
        self._gif.append_png(frame_path)
        self.observed_abs_max_s_1 = max(
            self.observed_abs_max_s_1,
            frame_observed_abs_max,
        )
        self.frame_count += 1
        return frame_path

    def close(self) -> VorticitySequenceSummary:
        if not self._closed:
            self._gif.close()
            self._summary = VorticitySequenceSummary(
                frame_directory=self.frame_directory.name,
                frame_pattern="frame_*.png",
                frame_count=self.frame_count,
                animation=self.animation_path.name,
                observed_abs_max_s_1=self.observed_abs_max_s_1,
                color_abs_max_s_1=self.color_abs_max_s_1,
                quiver_stride_cells=self.geometry.stride_cells,
            )
            self._closed = True
        assert self._summary is not None
        return self._summary


def write_vorticity_sequence(
    output_dir: Path,
    config: IceFlowConfig,
    scales: LatticeScales,
    snapshots: list[CoupledSnapshot],
    *,
    display_abs_max_s_1: float | None = None,
    velocity_display_max_m_s: float | None = None,
) -> VorticitySequenceSummary:
    """Compatibility wrapper that streams an existing snapshot iterable."""

    if not snapshots:
        raise ValueError("at least one snapshot is required for vorticity plotting")
    if display_abs_max_s_1 is None:
        _, display_abs_max_s_1 = vorticity_sequence_limits_s_1(
            config,
            scales,
            snapshots,
        )
    if velocity_display_max_m_s is None:
        _, velocity_display_max_m_s = velocity_sequence_limits_m_s(
            config,
            scales,
            snapshots,
        )
    stream = VorticitySequenceStream(
        output_dir,
        config,
        scales,
        display_abs_max_s_1=display_abs_max_s_1,
        velocity_display_max_m_s=velocity_display_max_m_s,
    )
    try:
        for snapshot in snapshots:
            stream.append(snapshot)
    except BaseException:
        stream.close()
        raise
    return stream.close()


def _temperature_field_for_plot(
    config: IceFlowConfig,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Return temperature masked to the water/ice thermal domain."""

    expected_shape = (config.ny, config.nx)
    temperature_c = np.asarray(snapshot.temperature_c, dtype=np.float64)
    water_phase = np.asarray(snapshot.water_phase)
    phase_change_material = np.asarray(snapshot.phase_change_material)
    if (
        temperature_c.shape != expected_shape
        or water_phase.shape != expected_shape
        or phase_change_material.shape != expected_shape
    ):
        raise ValueError(
            f"temperature frame {frame_index} fields must have shape {expected_shape}"
        )

    visible = (~_static_wall_mask(config)) & (
        (water_phase >= 0.5) | (phase_change_material != 0)
    )
    if not np.any(visible):
        raise ValueError(f"temperature frame {frame_index} has no visible cells")
    if not np.isfinite(temperature_c[visible]).all():
        raise ValueError(
            f"temperature frame {frame_index} contains NaN or infinity in "
            "the water/ice domain"
        )
    return np.ma.masked_where(~visible, temperature_c), visible


def _render_temperature_frame(
    path: Path,
    config: IceFlowConfig,
    snapshot: CoupledSnapshot,
    *,
    frame_index: int,
    color_min_c: float,
    color_max_c: float,
) -> tuple[float, float]:
    """Render one temperature frame and return its observed range."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    plotted_temperature, visible = _temperature_field_for_plot(
        config,
        snapshot,
        frame_index=frame_index,
    )
    visible_temperature = np.asarray(snapshot.temperature_c, dtype=np.float64)[visible]
    observed_min_c = float(np.min(visible_temperature))
    observed_max_c = float(np.max(visible_temperature))
    extent_m = (
        0.0,
        config.nx * config.dx,
        0.0,
        config.ny * config.dx,
    )
    colormap = plt.get_cmap("inferno").copy()
    colormap.set_bad("#aeb4ba")

    figure, axis = plt.subplots(figsize=(6.4, 8.0), constrained_layout=True)
    temperature_image = axis.imshow(
        plotted_temperature,
        origin="lower",
        extent=extent_m,
        cmap=colormap,
        norm=Normalize(vmin=color_min_c, vmax=color_max_c),
        interpolation="nearest",
        aspect="equal",
    )
    _draw_flow_boundaries(
        axis,
        snapshot,
        extent_m,
        water_surface_color="white",
    )
    figure.colorbar(
        temperature_image,
        ax=axis,
        label="temperature (°C), shared across all cases",
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        "Temperature during fixed-ice melting\n"
        f"T_inf={config.thermal.buoyancy_reference_temperature_c:g} degC, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return observed_min_c, observed_max_c


class TemperatureSequenceStream:
    """Write temperature PNG/GIF frames incrementally with fixed limits."""

    def __init__(
        self,
        output_dir: Path,
        config: IceFlowConfig,
        *,
        temperature_min_c: float,
        temperature_max_c: float,
    ) -> None:
        color_min_c = float(temperature_min_c)
        color_max_c = float(temperature_max_c)
        if not math.isfinite(color_min_c) or not math.isfinite(color_max_c):
            raise ValueError("temperature visualization limits must be finite")
        if color_max_c < color_min_c:
            raise ValueError("temperature_max_c must not be below temperature_min_c")
        self.config = config
        self.color_min_c = color_min_c
        self.color_max_c = color_max_c
        self.frame_directory = output_dir / "temperature_frames"
        self.frame_directory.mkdir(parents=True, exist_ok=True)
        for stale_frame in self.frame_directory.glob("frame_*.png"):
            stale_frame.unlink()
        self.animation_path = output_dir / "temperature_field.gif"
        self._gif = _StreamingGif(self.animation_path)
        self.frame_count = 0
        self.observed_min_c = math.inf
        self.observed_max_c = -math.inf
        self._closed = False
        self._summary: TemperatureSequenceSummary | None = None

    def append(self, snapshot: CoupledSnapshot) -> Path:
        if self._closed:
            raise RuntimeError("cannot append to a closed temperature sequence")
        frame_path = self.frame_directory / f"frame_{self.frame_count:05d}.png"
        observed_min_c, observed_max_c = _render_temperature_frame(
            frame_path,
            self.config,
            snapshot,
            frame_index=self.frame_count,
            color_min_c=self.color_min_c,
            color_max_c=self.color_max_c,
        )
        self._gif.append_png(frame_path)
        self.observed_min_c = min(self.observed_min_c, observed_min_c)
        self.observed_max_c = max(self.observed_max_c, observed_max_c)
        self.frame_count += 1
        return frame_path

    def close(self) -> TemperatureSequenceSummary:
        if not self._closed:
            self._gif.close()
            self._summary = TemperatureSequenceSummary(
                frame_directory=self.frame_directory.name,
                frame_pattern="frame_*.png",
                frame_count=self.frame_count,
                animation=self.animation_path.name,
                observed_min_c=self.observed_min_c,
                observed_max_c=self.observed_max_c,
                color_min_c=self.color_min_c,
                color_max_c=self.color_max_c,
            )
            self._closed = True
        assert self._summary is not None
        return self._summary


def write_temperature_sequence(
    output_dir: Path,
    config: IceFlowConfig,
    snapshots: list[CoupledSnapshot],
    *,
    temperature_min_c: float | None = None,
    temperature_max_c: float | None = None,
) -> TemperatureSequenceSummary:
    """Compatibility wrapper that streams an existing snapshot iterable."""

    if not snapshots:
        raise ValueError("at least one snapshot is required for temperature plotting")
    if temperature_min_c is None:
        temperature_min_c = float(config.thermal.properties.melting_temperature_c)
    if temperature_max_c is None:
        temperature_max_c = float(config.thermal.initial_water_temperature_c)
    stream = TemperatureSequenceStream(
        output_dir,
        config,
        temperature_min_c=temperature_min_c,
        temperature_max_c=temperature_max_c,
    )
    try:
        for snapshot in snapshots:
            stream.append(snapshot)
    except BaseException:
        stream.close()
        raise
    return stream.close()


def write_final_plot(
    path: Path,
    config: IceFlowConfig,
    snapshot: CoupledSnapshot,
    *,
    temperature_max_c: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent_m = (
        0.0,
        config.nx * config.dx,
        0.0,
        config.ny * config.dx,
    )
    wall = _static_wall_mask(config)

    temperature, _ = _temperature_field_for_plot(
        config,
        snapshot,
        frame_index=0,
    )
    phase = np.ma.masked_where(
        (wall != 0) | (snapshot.phase_change_material == 0),
        snapshot.liquid_fraction,
    )
    temperature_min_c = float(config.thermal.properties.melting_temperature_c)
    if temperature_max_c is None:
        temperature_max_c = float(config.thermal.initial_water_temperature_c)
    temperature_max_c = float(temperature_max_c)
    if not math.isfinite(temperature_max_c):
        raise ValueError("temperature_max_c must be finite")
    if temperature_max_c < temperature_min_c:
        raise ValueError("temperature_max_c must not be below the melting temperature")

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 6.0), constrained_layout=True)
    temperature_image = axes[0].imshow(
        temperature,
        origin="lower",
        extent=extent_m,
        cmap="inferno",
        vmin=temperature_min_c,
        vmax=temperature_max_c,
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].contour(
        phase,
        levels=(config.thermal.solid_liquid_threshold,),
        colors="cyan",
        linewidths=1.1,
        origin="lower",
        extent=extent_m,
    )
    axes[0].contour(
        snapshot.water_phase,
        levels=(0.5,),
        colors="white",
        linestyles="--",
        linewidths=0.9,
        origin="lower",
        extent=extent_m,
    )
    axes[0].set_title("Final temperature")
    figure.colorbar(temperature_image, ax=axes[0], label="temperature (°C)")

    phase_image = axes[1].imshow(
        phase,
        origin="lower",
        extent=extent_m,
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
        extent=extent_m,
    )
    axes[1].contour(
        snapshot.water_phase,
        levels=(0.5,),
        colors="black",
        linestyles="--",
        linewidths=0.9,
        origin="lower",
        extent=extent_m,
    )
    axes[1].set_title("Final liquid fraction")
    figure.colorbar(phase_image, ax=axes[1], label="liquid fraction")

    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
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
    initial_snapshot: CoupledSnapshot,
    final_snapshot: CoupledSnapshot,
    snapshot_count: int,
    velocity_sequence: VelocitySequenceSummary | None,
    vorticity_sequence: VorticitySequenceSummary | None,
    temperature_sequence: TemperatureSequenceSummary | None,
    visualization_limits: VisualizationLimits,
) -> None:
    initial = initial_snapshot
    final = final_snapshot
    if snapshot_count <= 0:
        raise ValueError("snapshot_count must be positive")
    density_ratio = config.rho_ice / config.rho_water
    if config.gravity == (0.0, 0.0):
        gravity_assumption = (
            "mechanical gravity disabled for a controlled coupling check"
        )
    elif config.well_balanced_hydrostatics:
        gravity_assumption = (
            "vertical gravity with a frozen well-balanced hydrostatic reference"
        )
    else:
        gravity_assumption = "vertical gravity without well-balanced hydrostatics"
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
        "freshwater_buoyancy": {
            "model": config.thermal.water_buoyancy_model,
            "far_field_temperature_c": (
                config.thermal.buoyancy_reference_temperature_c
            ),
            "density_max_temperature_c": (
                config.thermal.freshwater_density_max_temperature_c
            ),
            "quadratic_coefficient_1_k2": (
                config.thermal.freshwater_density_quadratic_coefficient_1_k2
            ),
            "gravity_m_s2": abs(float(config.gravity[1])),
            "gravity_vector_m_s2": list(config.gravity),
            "well_balanced_hydrostatics": config.well_balanced_hydrostatics,
            "water_side_phase_weight": "clamp(2*phi-1, 0, 1)",
        },
        "assumptions": [
            "one fixed connected ice body",
            "real ice/water density ratio with free-surface volume accommodation",
            gravity_assumption,
            "zero surface tension to suppress unrelated capillary transients",
            "air is hydrodynamic but excluded from heat conduction",
            "two-dimensional unit out-of-plane depth",
        ],
        "requested": {
            "end_time_s": float(args.end_time_s),
            "output_interval_s": float(args.output_interval_s),
            "water_temperature_c": float(args.water_temperature_c),
            "save_npz": bool(args.save_npz),
            "velocity_visualization_max_m_s": (visualization_limits.velocity_max_m_s),
            "vorticity_visualization_max_s_1": (
                visualization_limits.vorticity_abs_max_s_1
            ),
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
            "reference_lattice_velocity": scales.reference_lattice_velocity,
            "reference_velocity_m_s": scales.reference_velocity_m_s,
            "scheduled_snapshot_steps": targets,
        },
        "velocity_visualization": (
            None
            if velocity_sequence is None
            else {
                "field_definition": "u + 0.5 * fluid_force",
                "source_field": "physical_velocity_lattice",
                "visible_region": "liquid water cells with water_phase >= 0.5",
                "magnitude_units": "m/s",
                "normalization": (
                    "shared across every frame and temperature case in this invocation"
                ),
                "magnitude_range_m_s": [0.0, velocity_sequence.color_max_m_s],
                "observed_max_m_s": velocity_sequence.observed_max_m_s,
                "quiver_scale": (
                    "shared across every frame and temperature case in this invocation"
                ),
                "quiver_reference_speed_m_s": velocity_sequence.color_max_m_s,
                "quiver_stride_cells": velocity_sequence.quiver_stride_cells,
            }
        ),
        "vorticity_visualization": (
            None
            if vorticity_sequence is None
            else {
                "field_definition": "omega_z = d(v)/dx - d(u)/dy",
                "source_field": "physical_velocity_lattice",
                "units": "s^-1",
                "sign_convention": (
                    "positive is counterclockwise for x right and y up"
                ),
                "spatial_discretization": (
                    "second-order centered differences with spacing dx"
                ),
                "visible_region": (
                    "liquid water cells whose four axial neighbors are also "
                    "liquid water"
                ),
                "normalization": (
                    "symmetric and shared across every frame and temperature "
                    "case in this invocation"
                ),
                "range_s_1": [
                    -vorticity_sequence.color_abs_max_s_1,
                    vorticity_sequence.color_abs_max_s_1,
                ],
                "observed_abs_max_s_1": (vorticity_sequence.observed_abs_max_s_1),
                "quiver_reference_speed_m_s": (visualization_limits.velocity_max_m_s),
                "quiver_stride_cells": vorticity_sequence.quiver_stride_cells,
            }
        ),
        "temperature_visualization": (
            None
            if temperature_sequence is None
            else {
                "source_field": "temperature_c",
                "units": "degC",
                "visible_region": (
                    "water cells with water_phase >= 0.5 and phase-change "
                    "material cells; air and container walls are masked"
                ),
                "normalization": (
                    "shared across every frame and temperature case in this invocation"
                ),
                "range_c": [
                    temperature_sequence.color_min_c,
                    temperature_sequence.color_max_c,
                ],
                "observed_range_c": [
                    temperature_sequence.observed_min_c,
                    temperature_sequence.observed_max_c,
                ],
            }
        ),
        "density_ratio_ice_to_water": density_ratio,
        "config": config.to_dict(),
        "results": {
            "snapshots": int(snapshot_count),
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
            "fields": "fields.npz" if args.save_npz else None,
            "plot": None if args.no_plot else "coupled_fixed_ice_melting_2d.png",
            "velocity_frames": (
                None
                if velocity_sequence is None
                else {
                    "directory": velocity_sequence.frame_directory,
                    "pattern": velocity_sequence.frame_pattern,
                    "count": velocity_sequence.frame_count,
                }
            ),
            "velocity_animation": (
                None if velocity_sequence is None else velocity_sequence.animation
            ),
            "vorticity_frames": (
                None
                if vorticity_sequence is None
                else {
                    "directory": vorticity_sequence.frame_directory,
                    "pattern": vorticity_sequence.frame_pattern,
                    "count": vorticity_sequence.frame_count,
                }
            ),
            "vorticity_animation": (
                None if vorticity_sequence is None else vorticity_sequence.animation
            ),
            "temperature_frames": (
                None
                if temperature_sequence is None
                else {
                    "directory": temperature_sequence.frame_directory,
                    "pattern": temperature_sequence.frame_pattern,
                    "count": temperature_sequence.frame_count,
                }
            ),
            "temperature_animation": (
                None if temperature_sequence is None else temperature_sequence.animation
            ),
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_case(
    args: argparse.Namespace,
    config: IceFlowConfig,
    targets: list[int],
    *,
    visualization_limits: VisualizationLimits,
) -> dict[str, object]:
    """Run and write one bath-temperature case."""

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
    initial_snapshot = _capture_snapshot(
        simulation,
        initial_total_enthalpy_j_m=initial_total_enthalpy,
        initial_solid_volume_cells=initial_solid_volume,
    )
    final_snapshot = initial_snapshot
    snapshot_count = 0
    retained_snapshots: list[CoupledSnapshot] | None = [] if args.save_npz else None

    velocity_stream = None
    vorticity_stream = None
    temperature_stream = None
    velocity_sequence = None
    vorticity_sequence = None
    temperature_sequence = None
    try:
        if not args.no_plot:
            velocity_stream = VelocitySequenceStream(
                output_dir,
                config,
                scales,
                display_max_m_s=visualization_limits.velocity_max_m_s,
            )
            vorticity_stream = VorticitySequenceStream(
                output_dir,
                config,
                scales,
                display_abs_max_s_1=visualization_limits.vorticity_abs_max_s_1,
                velocity_display_max_m_s=visualization_limits.velocity_max_m_s,
            )
            temperature_stream = TemperatureSequenceStream(
                output_dir,
                config,
                temperature_min_c=visualization_limits.temperature_min_c,
                temperature_max_c=visualization_limits.temperature_max_c,
            )

        with (output_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as history_stream:
            history_writer = csv.writer(history_stream)
            history_writer.writerow(HISTORY_COLUMNS)

            for frame_index, target in enumerate(targets):
                if frame_index == 0:
                    snapshot = initial_snapshot
                else:
                    simulation.step(target - simulation.steps)
                    snapshot = _capture_snapshot(
                        simulation,
                        initial_total_enthalpy_j_m=initial_total_enthalpy,
                        initial_solid_volume_cells=initial_solid_volume,
                    )

                history_writer.writerow(_history_row(snapshot))
                history_stream.flush()
                if retained_snapshots is not None:
                    retained_snapshots.append(snapshot)
                if velocity_stream is not None:
                    velocity_stream.append(snapshot)
                if vorticity_stream is not None:
                    vorticity_stream.append(snapshot)
                if temperature_stream is not None:
                    temperature_stream.append(snapshot)

                snapshot_count += 1
                final_snapshot = snapshot
                if frame_index > 0 and not args.quiet:
                    print(
                        f"t={snapshot.physical_time_s:.4f} s | "
                        f"steps={snapshot.lbm_steps} | "
                        f"melted={100.0 * snapshot.melted_fraction:6.2f}% | "
                        "water residual="
                        f"{snapshot.water_volume_residual_cells:+.3e} cells",
                        flush=True,
                    )
    finally:
        try:
            if velocity_stream is not None:
                velocity_sequence = velocity_stream.close()
        finally:
            try:
                if vorticity_stream is not None:
                    vorticity_sequence = vorticity_stream.close()
            finally:
                if temperature_stream is not None:
                    temperature_sequence = temperature_stream.close()

    if retained_snapshots is not None:
        write_fields_npz(output_dir / "fields.npz", config, retained_snapshots)

    if not args.no_plot:
        write_final_plot(
            output_dir / "coupled_fixed_ice_melting_2d.png",
            config,
            final_snapshot,
            temperature_max_c=visualization_limits.temperature_max_c,
        )
    write_metadata(
        output_dir / "metadata.json",
        args=args,
        config=config,
        scales=scales,
        targets=targets,
        initial_snapshot=initial_snapshot,
        final_snapshot=final_snapshot,
        snapshot_count=snapshot_count,
        velocity_sequence=velocity_sequence,
        vorticity_sequence=vorticity_sequence,
        temperature_sequence=temperature_sequence,
        visualization_limits=visualization_limits,
    )

    if not args.quiet:
        final = final_snapshot
        print(
            "Coupled fixed-ice melting example complete "
            f"(T_inf={args.water_temperature_c:g} degC)"
        )
        print(
            f"  domain: {config.nx * config.dx:.6f} x "
            f"{config.ny * config.dx:.6f} m, "
            f"grid={config.nx} x {config.ny}"
        )
        print(
            f"  simulated: {final.physical_time_s:.6f} s in {final.lbm_steps} LBM steps"
        )
        print(
            f"  ice melted: {100.0 * final.melted_fraction:.3f}%, "
            "equivalent melt depth="
            f"{final.equivalent_uniform_melt_depth_m:.9f} m"
        )
        if velocity_sequence is not None:
            print(
                "  velocity sequence: "
                f"{velocity_sequence.frame_count} frames, fixed |u| range "
                f"[0, {velocity_sequence.color_max_m_s:.6e}] m/s"
            )
        if vorticity_sequence is not None:
            print(
                "  vorticity sequence: "
                f"{vorticity_sequence.frame_count} frames, fixed omega_z range "
                f"[{-vorticity_sequence.color_abs_max_s_1:.6e}, "
                f"{vorticity_sequence.color_abs_max_s_1:.6e}] s^-1"
            )
        if temperature_sequence is not None:
            print(
                "  temperature sequence: "
                f"{temperature_sequence.frame_count} frames, fixed range "
                f"[{temperature_sequence.color_min_c:g}, "
                f"{temperature_sequence.color_max_c:g}] degC"
            )
        print(f"  output: {output_dir}")

    final = final_snapshot
    return {
        "water_temperature_c": float(args.water_temperature_c),
        "output_directory": str(output_dir),
        "snapshots": snapshot_count,
        "lbm_steps": final.lbm_steps,
        "physical_time_s": final.physical_time_s,
        "melted_fraction": final.melted_fraction,
        "observed_max_velocity_m_s": (
            None if velocity_sequence is None else velocity_sequence.observed_max_m_s
        ),
        "velocity_color_max_m_s": (
            None if velocity_sequence is None else velocity_sequence.color_max_m_s
        ),
        "observed_max_abs_vorticity_s_1": (
            None
            if vorticity_sequence is None
            else vorticity_sequence.observed_abs_max_s_1
        ),
        "vorticity_color_max_abs_s_1": (
            None if vorticity_sequence is None else vorticity_sequence.color_abs_max_s_1
        ),
        "temperature_color_min_c": (
            None if args.no_plot else visualization_limits.temperature_min_c
        ),
        "temperature_color_max_c": (
            None if args.no_plot else visualization_limits.temperature_max_c
        ),
        "observed_min_temperature_c": (
            None
            if temperature_sequence is None
            else temperature_sequence.observed_min_c
        ),
        "observed_max_temperature_c": (
            None
            if temperature_sequence is None
            else temperature_sequence.observed_max_c
        ),
        "velocity_frames": (
            0 if velocity_sequence is None else velocity_sequence.frame_count
        ),
        "velocity_animation": (
            None if velocity_sequence is None else velocity_sequence.animation
        ),
        "vorticity_frames": (
            0 if vorticity_sequence is None else vorticity_sequence.frame_count
        ),
        "vorticity_animation": (
            None if vorticity_sequence is None else vorticity_sequence.animation
        ),
        "temperature_frames": (
            0 if temperature_sequence is None else temperature_sequence.frame_count
        ),
        "temperature_animation": (
            None if temperature_sequence is None else temperature_sequence.animation
        ),
    }


def _temperature_directory_name(temperature_c: float) -> str:
    """Return a stable filesystem label such as ``T_5p6C``."""

    value = float(temperature_c)
    if not math.isfinite(value):
        raise ValueError("water temperature must be finite")
    label = f"{value:.12g}".replace("-", "m").replace("+", "").replace(".", "p")
    return f"T_{label}C"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    sweep_temperatures = args.water_temperatures_c
    sweep_root = Path(args.output_dir) if sweep_temperatures is not None else None
    case_specs: list[tuple[argparse.Namespace, IceFlowConfig, list[int]]] = []
    try:
        if sweep_temperatures is None:
            case_args = args
            config = create_config(case_args)
            targets = _snapshot_step_targets(
                config, case_args.end_time_s, case_args.output_interval_s
            )
            case_specs.append((case_args, config, targets))
        else:
            directory_names = [
                _temperature_directory_name(value) for value in sweep_temperatures
            ]
            if len(set(directory_names)) != len(directory_names):
                raise ValueError("water temperature sweep contains duplicate cases")
            for temperature_c, directory_name in zip(
                sweep_temperatures, directory_names
            ):
                case_args = argparse.Namespace(**vars(args))
                case_args.water_temperatures_c = None
                case_args.water_temperature_c = float(temperature_c)
                case_args.output_dir = str(sweep_root / directory_name)
                config = create_config(case_args)
                targets = _snapshot_step_targets(
                    config, case_args.end_time_s, case_args.output_interval_s
                )
                case_specs.append((case_args, config, targets))
        visualization_limits = _shared_visualization_limits(
            args,
            [float(case[0].water_temperature_c) for case in case_specs],
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    summaries = [
        _run_case(
            case_args,
            config,
            targets,
            visualization_limits=visualization_limits,
        )
        for case_args, config, targets in case_specs
    ]
    if sweep_root is not None:
        sweep_root.mkdir(parents=True, exist_ok=True)
        sweep_data = {
            "model": "fixed-ice freshwater-temperature sweep",
            "temperatures_c": [
                float(case[0].water_temperature_c) for case in case_specs
            ],
            "visualization_scales": {
                "velocity_magnitude_range_m_s": [
                    0.0,
                    visualization_limits.velocity_max_m_s,
                ],
                "velocity_quiver_reference_speed_m_s": (
                    visualization_limits.velocity_max_m_s
                ),
                "vorticity_range_s_1": [
                    -visualization_limits.vorticity_abs_max_s_1,
                    visualization_limits.vorticity_abs_max_s_1,
                ],
                "temperature_range_c": [
                    visualization_limits.temperature_min_c,
                    visualization_limits.temperature_max_c,
                ],
                "normalization": "shared across every temperature case",
            },
            "velocity_normalization": (
                "fixed across every frame and temperature case; "
                "observed_max_velocity_m_s remains an unclipped diagnostic"
            ),
            "vorticity_normalization": (
                "symmetric and fixed across every frame and temperature case; "
                "observed_max_abs_vorticity_s_1 remains an unclipped diagnostic"
            ),
            "temperature_normalization": (
                "fixed across every frame and temperature case; observed "
                "temperature extrema remain unclipped diagnostics"
            ),
            "cases": summaries,
        }
        (sweep_root / "temperature_sweep.json").write_text(
            json.dumps(sweep_data, indent=2), encoding="utf-8"
        )
        if not args.quiet:
            print(f"Temperature sweep complete: {sweep_root}")


if __name__ == "__main__":
    main()
