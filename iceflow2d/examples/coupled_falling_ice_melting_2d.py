"""Fully coupled example of an ice block falling into a 90 degC water bath.

All four container walls are adiabatic. The ice starts in the insulated air
cap, falls through the free surface, and exchanges heat, mass, and momentum
with the hot water while it melts. Water remains a liquid at 90 degC;
evaporation, boiling, and vapour transport are outside this model.

Configuration and scheduling helpers do not import the CUDA simulator.  The
``IceFlow2D`` import is deliberately confined to :func:`_run_case`, keeping the
module importable for pure-CPU geometry and configuration contract tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iceflow2d.config import IceFlowConfig, create_iceflow_config  # noqa: E402
from iceflow2d import reporting  # noqa: E402
from iceflow2d.config import (  # noqa: E402
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundarySet,
    ThermalConfig,
)


DEFAULT_OUTPUT_DIR = "outputs/iceflow2d/coupled_falling_ice_melting_2d"
DEFAULT_BOUNDARY_CELLS = 3
# Representing 4 m/s by 0.1 LU keeps the first-contact transient in the
# low-Mach lattice envelope at the default 100 x 200 resolution.
DEFAULT_REFERENCE_VELOCITY_M_S = 4.0
DEFAULT_THERMAL_UPDATE_INTERVAL = 8
DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S = 0.1
DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1 = 250.0
SCENARIO_LABEL = reporting.DEFAULT_SCENARIO_LABEL


@dataclass(slots=True)
class FallingMeltingSnapshot:
    """One coupled field snapshot augmented with moving-body diagnostics."""

    coupled: reporting.CoupledSnapshot
    body_active: bool
    body_center_x_m: float
    body_center_y_m: float
    body_velocity_x_m_s: float
    body_velocity_y_m_s: float
    body_angle_rad: float
    body_angular_velocity_rad_s: float
    body_mass_lattice: float
    body_mass_kg_m: float
    body_inertia_lattice: float
    body_inertia_kg_m: float
    body_solid_center_local_x_cells: float
    body_solid_center_local_y_cells: float
    cumulative_melted_mass_lattice: float
    cumulative_melted_momentum_x_lattice: float
    cumulative_melted_momentum_y_lattice: float
    cumulative_fluid_melt_momentum_x_lattice: float
    cumulative_fluid_melt_momentum_y_lattice: float
    melt_momentum_residual_x_lattice: float
    melt_momentum_residual_y_lattice: float
    cumulative_melted_angular_momentum_lattice: float
    cumulative_fluid_melt_angular_momentum_lattice: float
    melt_angular_momentum_residual_lattice: float
    ale_water_residual_cells: float
    ale_energy_residual_j_m: float
    phase_aperture_water_residual_cells: float
    phase_aperture_energy_residual_j_m: float
    phase_aperture_capacity_margin_cells: float
    melt_injection_mass_residual_kg_m: float
    thermal_initial_total_mass_kg_m: float
    thermal_total_mass_kg_m: float
    thermal_total_mass_residual_kg_m: float
    water_mean_temperature_c: float
    aperture_energy_correction_abs_j_m: float

    def __getattr__(self, name: str) -> Any:
        # The common visualization writers consume CoupledSnapshot fields by
        # attribute, so composition keeps the reusable format without copying
        # the large arrays a second time.
        return getattr(self.coupled, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Taichi CUDA air-water LBM with a freely falling and melting ice "
            "block in a hot-water bath"
        )
    )
    parser.add_argument("--domain-width-m", type=float, default=0.025)
    parser.add_argument("--domain-height-m", type=float, default=0.050)
    parser.add_argument("--resolution-x", type=int, default=100)
    parser.add_argument("--resolution-y", type=int, default=200)
    parser.add_argument("--water-level-m", type=float, default=0.030)
    parser.add_argument("--ice-size-m", type=float, default=0.008)
    parser.add_argument("--ice-angle-degrees", type=float, default=5.0)
    parser.add_argument(
        "--drop-height-m",
        type=float,
        default=0.0015,
        help="air gap between the water surface and the lowest rotated ice corner",
    )
    parser.add_argument("--boundary-cells", type=int, default=DEFAULT_BOUNDARY_CELLS)
    parser.add_argument(
        "--initial-horizontal-speed-lattice",
        type=float,
        default=0.0,
        help="initial horizontal rigid-body speed in cells per LBM step",
    )
    parser.add_argument(
        "--initial-vertical-speed-lattice",
        type=float,
        default=0.0,
        help="initial vertical rigid-body speed in cells per LBM step",
    )
    parser.add_argument(
        "--reference-velocity-m-s",
        type=float,
        default=DEFAULT_REFERENCE_VELOCITY_M_S,
        help="physical speed represented by the fixed reference speed 0.1 LU",
    )

    parser.add_argument("--end-time-s", type=float, default=3.0)
    parser.add_argument(
        "--output-interval-s",
        type=float,
        default=0.01,
        help="physical-time spacing of history rows and field-sequence frames",
    )
    parser.add_argument(
        "--velocity-visualization-max-m-s",
        type=float,
        default=DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S,
    )
    parser.add_argument(
        "--vorticity-visualization-max-s-1",
        type=float,
        default=DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
    )

    parser.add_argument("--water-temperature-c", type=float, default=90.0)
    parser.add_argument("--ice-temperature-c", type=float, default=0.0)
    parser.add_argument("--air-temperature-c", type=float, default=20.0)
    parser.add_argument("--melting-temperature-c", type=float, default=0.0)
    parser.add_argument("--gravity-m-s2", type=float, default=9.8)
    parser.add_argument(
        "--thermal-expansion-water-1-k",
        type=float,
        default=2.1e-4,
        help="constant expansion coefficient in the simplified linear Boussinesq law",
    )

    parser.add_argument("--density-water", type=float, default=1000.0)
    parser.add_argument("--density-ice", type=float, default=917.0)
    parser.add_argument("--density-air", type=float, default=1.25)
    parser.add_argument("--kinematic-viscosity-water", type=float, default=1.0e-6)
    parser.add_argument("--kinematic-viscosity-air", type=float, default=1.5e-5)
    parser.add_argument("--surface-tension", type=float, default=0.072)

    parser.add_argument("--water-specific-heat", type=float, default=4186.0)
    parser.add_argument("--ice-specific-heat", type=float, default=2100.0)
    parser.add_argument("--air-specific-heat", type=float, default=1005.0)
    parser.add_argument("--water-conductivity", type=float, default=0.60)
    parser.add_argument("--ice-conductivity", type=float, default=2.20)
    parser.add_argument("--air-conductivity", type=float, default=0.026)
    parser.add_argument("--latent-heat", type=float, default=334000.0)

    parser.add_argument("--phase-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--thermal-update-interval",
        type=int,
        default=DEFAULT_THERMAL_UPDATE_INTERVAL,
        help=(
            "LBM steps per conduction/phase-change update; pose ALE and water "
            "advection remain coupled on every LBM step"
        ),
    )
    parser.add_argument("--interface-width", type=float, default=5.0)
    parser.add_argument("--mobility", type=float, default=0.1)
    parser.add_argument("--no-advection", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="disable final plot and velocity/vorticity/temperature sequences",
    )
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain sampled arrays and write fields.npz",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the default LBM-step progress bar",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress and run summaries",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.resolution_x <= 0 or args.resolution_y <= 0:
            raise ValueError("resolution components must be positive")
        if args.boundary_cells < 1:
            raise ValueError("boundary_cells must be positive")
        if args.phase_warmup_steps < 0:
            raise ValueError("phase_warmup_steps must be non-negative")
        if args.thermal_update_interval < 1:
            raise ValueError("thermal_update_interval must be positive")
        reporting._positive("end_time_s", args.end_time_s)
        reporting._positive("output_interval_s", args.output_interval_s)
        reporting._positive(
            "velocity_visualization_max_m_s",
            args.velocity_visualization_max_m_s,
        )
        reporting._positive(
            "vorticity_visualization_max_s_1",
            args.vorticity_visualization_max_s_1,
        )
        derive_geometry(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _aligned_cell_count(name: str, length_m: float, spacing_m: float) -> int:
    cells = reporting._positive(name, length_m) / spacing_m
    rounded = round(cells)
    if not math.isclose(cells, rounded, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must align with lattice-cell faces")
    return int(rounded)


def _integer_bin_fraction(count: int, total: int) -> float:
    return (int(count) + 0.5) / int(total)


def derive_geometry(args: argparse.Namespace) -> dict[str, float | int]:
    """Derive a full-width pool and the initial rotated body placement."""

    nx = int(args.resolution_x)
    ny = int(args.resolution_y)
    boundary = int(args.boundary_cells)
    if nx <= 2 * boundary + 4 or ny <= 2 * boundary + 4:
        raise ValueError("resolution is too small for the requested boundary layers")

    width_m = reporting._positive("domain_width_m", args.domain_width_m)
    height_m = reporting._positive("domain_height_m", args.domain_height_m)
    dx = width_m / nx
    dy = height_m / ny
    if not math.isclose(dx, dy, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("the falling/melting example requires square cells")

    water_height = _aligned_cell_count("water_level_m", args.water_level_m, dx)
    ice_width = _aligned_cell_count("ice_size_m", args.ice_size_m, dx)
    ice_height = _aligned_cell_count("ice_size_m", args.ice_size_m, dy)
    drop_height_cells = (
        reporting._non_negative("drop_height_m", args.drop_height_m) / dx
    )
    angle = math.radians(float(args.ice_angle_degrees))
    if not math.isfinite(angle):
        raise ValueError("ice_angle_degrees must be finite")
    if ice_width < 4 or ice_height < 4:
        raise ValueError("ice must span at least four cells in each direction")
    if water_height <= boundary + 1 or water_height >= ny - boundary:
        raise ValueError("water level must leave both an active pool and an air cap")

    interface_clearance = max(2, int(math.ceil(float(args.interface_width))))
    if drop_height_cells < interface_clearance:
        raise ValueError(
            "drop height must keep the ice outside the diffuse water interface "
            f"(at least {interface_clearance} cells)"
        )

    half_width = 0.5 * ice_width
    half_height = 0.5 * ice_height
    extent_x = abs(math.cos(angle)) * half_width + abs(math.sin(angle)) * half_height
    extent_y = abs(math.sin(angle)) * half_width + abs(math.cos(angle)) * half_height
    left_cell = (nx - ice_width) // 2
    center_x = left_cell + half_width
    lowest_y = water_height + drop_height_cells
    center_y = lowest_y + extent_y
    ice_base_y = center_y - half_height
    contact_boundary = float(boundary)
    if (
        center_x - extent_x < contact_boundary
        or center_x + extent_x > nx - contact_boundary
    ):
        raise ValueError("rotated ice overlaps a side wall")
    if center_y + extent_y > ny - contact_boundary:
        raise ValueError("ice and drop height overlap the top wall")

    velocity = (
        float(args.initial_horizontal_speed_lattice),
        float(args.initial_vertical_speed_lattice),
    )
    if not all(math.isfinite(value) for value in velocity):
        raise ValueError("initial ice velocity must be finite")

    return {
        "dx_m": dx,
        "water_width": nx - boundary,
        "water_height": water_height,
        "water_width_fraction": _integer_bin_fraction(nx - boundary, nx),
        "water_height_fraction": _integer_bin_fraction(water_height, ny),
        "ice_width": ice_width,
        "ice_height": ice_height,
        "ice_width_fraction": _integer_bin_fraction(ice_width, nx),
        "ice_height_fraction": _integer_bin_fraction(ice_height, ny),
        "ice_angle_rad": angle,
        "ice_base_y_cells": ice_base_y,
        "ice_center_x_cells": center_x,
        "ice_center_y_cells": center_y,
        "ice_lowest_y_cells": lowest_y,
        "ice_highest_y_cells": center_y + extent_y,
        "drop_height_cells": drop_height_cells,
        "interface_clearance_cells": interface_clearance,
    }


def create_config(args: argparse.Namespace | None = None) -> IceFlowConfig:
    """Build the moving ``body_ale`` thermal configuration without CUDA."""

    if args is None:
        args = parse_args([])
    geometry = derive_geometry(args)
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
    thermal = ThermalConfig(
        properties=properties,
        boundaries=ThermalBoundarySet(),
        initial_water_temperature_c=args.water_temperature_c,
        initial_ice_temperature_c=args.ice_temperature_c,
        initial_air_temperature_c=args.air_temperature_c,
        advection_enabled=not args.no_advection,
        water_air_interface_adiabatic=True,
        update_interval_lbm_steps=int(args.thermal_update_interval),
        solid_liquid_threshold=0.5,
        max_fourier_number=0.15,
        max_courant_number=0.50,
        water_buoyancy_model="linear",
        thermal_expansion_water_1_k=args.thermal_expansion_water_1_k,
        buoyancy_reference_temperature_c=args.water_temperature_c,
        moving_body_scheme="body_ale",
    )
    gravity = reporting._non_negative("gravity_m_s2", args.gravity_m_s2)
    config = create_iceflow_config(
        resolution=(int(args.resolution_x), int(args.resolution_y)),
        dx=float(geometry["dx_m"]),
        reference_velocity=reporting._positive(
            "reference_velocity_m_s", args.reference_velocity_m_s
        ),
        rho_water=args.density_water,
        rho_air=args.density_air,
        rho_ice=args.density_ice,
        viscosity_water=args.kinematic_viscosity_water,
        viscosity_air=args.kinematic_viscosity_air,
        sigma=args.surface_tension,
        gravity=(0.0, -gravity),
        interface_width=args.interface_width,
        mobility=args.mobility,
        phase_warmup_steps=args.phase_warmup_steps,
        boundary_cells=int(args.boundary_cells),
        thermal=thermal,
        well_balanced_hydrostatics=True,
        water_width_fraction=float(geometry["water_width_fraction"]),
        water_height_fraction=float(geometry["water_height_fraction"]),
        ice_width_fraction=float(geometry["ice_width_fraction"]),
        ice_height_fraction=float(geometry["ice_height_fraction"]),
        ice_base_y_cells=float(geometry["ice_base_y_cells"]),
        ice_initial_velocity=(
            args.initial_horizontal_speed_lattice,
            args.initial_vertical_speed_lattice,
        ),
        ice_initial_angle=float(geometry["ice_angle_rad"]),
        ice_fixed=False,
        volume_projection_tolerance=1.0e-8,
        volume_projection_max_shift=0.50,
        air_interface_relaxation_time=0.8,
        output_dir=str(args.output_dir),
        show_gui=False,
        save_npz=False,
    )
    if (
        config.water_width != geometry["water_width"]
        or config.water_height != geometry["water_height"]
        or config.ice_width != geometry["ice_width"]
        or config.ice_height != geometry["ice_height"]
    ):
        raise RuntimeError("derived geometry disagrees with config discretization")
    return config


def _read_value(simulation: Any, *names: str, default: Any) -> Any:
    """Read a Python value, method, or zero-dimensional Taichi field."""

    sources = (simulation, getattr(simulation, "thermal", None))
    for source in sources:
        if source is None:
            continue
        for name in names:
            if not hasattr(source, name):
                continue
            value = getattr(source, name)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            try:
                value = value[None]
            except (KeyError, TypeError, IndexError, AttributeError):
                pass
            return value
    return default


def _scalar_diagnostic(simulation: Any, *names: str, default: float) -> float:
    value = _read_value(simulation, *names, default=default)
    array = np.asarray(value)
    if array.size != 1:
        return float(default)
    return float(array.reshape(-1)[0])


def _vector_diagnostic(
    simulation: Any,
    *names: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    value = _read_value(simulation, *names, default=default)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2:
        return (float(default[0]), float(default[1]))
    return (float(array[0]), float(array[1]))


def _thermal_total_mass_kg_m(simulation: Any, *, default: float) -> float:
    """Reduce the actual moving thermal state, with a lightweight-test fallback."""

    thermal = getattr(simulation, "thermal", None)
    for name in ("mass_energy_totals", "totals"):
        reducer = getattr(thermal, name, None)
        if not callable(reducer):
            continue
        totals = reducer()
        value = getattr(totals, "total_mass_kg_m", None)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return _scalar_diagnostic(
        simulation,
        "thermal_total_mass_kg_m",
        "total_mass_kg_m",
        default=default,
    )


def _capture_snapshot(
    simulation: Any,
    *,
    initial_total_enthalpy_j_m: float,
    initial_solid_volume_cells: float,
    initial_total_mass_kg_m: float | None = None,
) -> FallingMeltingSnapshot:
    synchronize = getattr(simulation, "synchronize_diagnostics", None)
    if callable(synchronize):
        synchronize()
    reducer = getattr(simulation.thermal, "mass_energy_totals", None)
    thermal_totals = reducer() if callable(reducer) else None
    coupled = reporting._capture_snapshot(
        simulation,
        initial_total_enthalpy_j_m=initial_total_enthalpy_j_m,
        initial_solid_volume_cells=initial_solid_volume_cells,
        thermal_totals=thermal_totals,
    )
    scales = LatticeScales.from_iceflow_config(simulation.config)
    center = _vector_diagnostic(
        simulation,
        "body_center",
        default=tuple(float(value) for value in simulation.config.ice_initial_center),
    )
    velocity = _vector_diagnostic(
        simulation,
        "body_velocity",
        default=tuple(float(value) for value in simulation.config.ice_initial_velocity),
    )
    solid_center_local = _vector_diagnostic(
        simulation,
        "body_solid_center_local",
        "body_local_center_of_mass",
        "body_center_local",
        "solid_center_local",
        default=(0.0, 0.0),
    )
    melted_momentum = _vector_diagnostic(
        simulation,
        "cumulative_melted_momentum_lattice",
        "melted_momentum_lattice",
        default=(0.0, 0.0),
    )
    fluid_melt_momentum = _vector_diagnostic(
        simulation,
        "cumulative_fluid_melt_momentum_lattice",
        default=(0.0, 0.0),
    )
    melt_momentum_residual = _vector_diagnostic(
        simulation,
        "melt_momentum_residual_lattice",
        default=(0.0, 0.0),
    )
    fallback_mass = (
        coupled.solid_volume_cells
        * simulation.config.rho_ice
        / simulation.config.rho_water
    )
    body_mass_lattice = _scalar_diagnostic(
        simulation,
        "body_mass_lattice",
        "body_mass",
        default=fallback_mass,
    )
    initial_mass = float(simulation.config.ice_mass_lattice)
    body_inertia_lattice = _scalar_diagnostic(
        simulation,
        "body_inertia_lattice",
        "body_inertia",
        default=(
            0.0
            if initial_mass <= 0.0
            else simulation.config.ice_inertia_lattice
            * max(0.0, body_mass_lattice)
            / initial_mass
        ),
    )
    cumulative_melted_mass = _scalar_diagnostic(
        simulation,
        "cumulative_melted_mass_lattice",
        "melted_mass_lattice",
        default=max(0.0, initial_mass - body_mass_lattice),
    )
    body_active = bool(
        _scalar_diagnostic(
            simulation,
            "body_active",
            default=1.0 if body_mass_lattice > 1.0e-12 else 0.0,
        )
    )
    angle = _scalar_diagnostic(
        simulation,
        "body_angle",
        default=float(simulation.config.ice_initial_angle),
    )
    angular_velocity_lattice = _scalar_diagnostic(
        simulation,
        "body_angular_velocity",
        default=float(simulation.config.ice_initial_angular_velocity),
    )
    dx = float(simulation.config.dx)
    rho_water = float(simulation.config.rho_water)
    body_mass_kg_m = body_mass_lattice * rho_water * dx * dx
    fallback_total_mass = body_mass_kg_m + (
        max(0.0, coupled.water_volume_current_cells) * rho_water * dx * dx
    )
    thermal_total_mass = (
        float(thermal_totals.total_mass_kg_m)
        if thermal_totals is not None
        else _thermal_total_mass_kg_m(simulation, default=fallback_total_mass)
    )
    thermal_initial_total_mass = (
        thermal_total_mass
        if initial_total_mass_kg_m is None
        else float(initial_total_mass_kg_m)
    )
    mean_temperature = simulation.config.thermal.initial_water_temperature_c
    if thermal_totals is not None and hasattr(
        thermal_totals, "water_sensible_energy_j_m"
    ):
        mean_temperature = simulation.config.thermal.properties.melting_temperature_c
        if thermal_totals.water_mass_kg_m > 0.0:
            mean_temperature += thermal_totals.water_sensible_energy_j_m / (
                thermal_totals.water_mass_kg_m
                * simulation.config.thermal.properties.specific_heat_water_j_kg_k
            )
    return FallingMeltingSnapshot(
        coupled=coupled,
        body_active=body_active,
        body_center_x_m=center[0] * dx,
        body_center_y_m=center[1] * dx,
        body_velocity_x_m_s=velocity[0] * scales.velocity_scale_m_s,
        body_velocity_y_m_s=velocity[1] * scales.velocity_scale_m_s,
        body_angle_rad=angle,
        body_angular_velocity_rad_s=angular_velocity_lattice / scales.dt_s,
        body_mass_lattice=body_mass_lattice,
        body_mass_kg_m=body_mass_kg_m,
        body_inertia_lattice=body_inertia_lattice,
        body_inertia_kg_m=body_inertia_lattice * rho_water * dx**4,
        body_solid_center_local_x_cells=solid_center_local[0],
        body_solid_center_local_y_cells=solid_center_local[1],
        cumulative_melted_mass_lattice=cumulative_melted_mass,
        cumulative_melted_momentum_x_lattice=melted_momentum[0],
        cumulative_melted_momentum_y_lattice=melted_momentum[1],
        cumulative_fluid_melt_momentum_x_lattice=fluid_melt_momentum[0],
        cumulative_fluid_melt_momentum_y_lattice=fluid_melt_momentum[1],
        melt_momentum_residual_x_lattice=melt_momentum_residual[0],
        melt_momentum_residual_y_lattice=melt_momentum_residual[1],
        cumulative_melted_angular_momentum_lattice=_scalar_diagnostic(
            simulation,
            "cumulative_melted_angular_momentum_lattice",
            default=0.0,
        ),
        cumulative_fluid_melt_angular_momentum_lattice=_scalar_diagnostic(
            simulation,
            "cumulative_fluid_melt_angular_momentum_lattice",
            default=0.0,
        ),
        melt_angular_momentum_residual_lattice=_scalar_diagnostic(
            simulation,
            "melt_angular_momentum_residual_lattice",
            default=0.0,
        ),
        ale_water_residual_cells=_scalar_diagnostic(
            simulation,
            "ale_water_residual_cells",
            default=0.0,
        ),
        ale_energy_residual_j_m=_scalar_diagnostic(
            simulation,
            "ale_energy_residual_j_m",
            default=0.0,
        ),
        phase_aperture_water_residual_cells=_scalar_diagnostic(
            simulation,
            "phase_aperture_water_residual_cells",
            default=0.0,
        ),
        phase_aperture_energy_residual_j_m=_scalar_diagnostic(
            simulation,
            "phase_aperture_energy_residual_j_m",
            default=0.0,
        ),
        phase_aperture_capacity_margin_cells=_scalar_diagnostic(
            simulation,
            "phase_aperture_capacity_margin_cells",
            default=0.0,
        ),
        melt_injection_mass_residual_kg_m=_scalar_diagnostic(
            simulation,
            "melt_injection_mass_residual_kg_m",
            default=0.0,
        ),
        thermal_initial_total_mass_kg_m=thermal_initial_total_mass,
        thermal_total_mass_kg_m=thermal_total_mass,
        thermal_total_mass_residual_kg_m=(
            thermal_total_mass - thermal_initial_total_mass
        ),
        water_mean_temperature_c=mean_temperature,
        aperture_energy_correction_abs_j_m=_scalar_diagnostic(
            simulation, "aperture_energy_correction_abs_j_m", default=0.0
        ),
    )


MOTION_HISTORY_COLUMNS = (
    "body_active",
    "body_center_x_m",
    "body_center_y_m",
    "body_velocity_x_m_s",
    "body_velocity_y_m_s",
    "body_angle_rad",
    "body_angular_velocity_rad_s",
    "body_mass_lattice",
    "body_mass_kg_m",
    "body_inertia_lattice",
    "body_inertia_kg_m",
    "body_solid_center_local_x_cells",
    "body_solid_center_local_y_cells",
    "cumulative_melted_mass_lattice",
    "cumulative_melted_momentum_x_lattice",
    "cumulative_melted_momentum_y_lattice",
    "cumulative_fluid_melt_momentum_x_lattice",
    "cumulative_fluid_melt_momentum_y_lattice",
    "melt_momentum_residual_x_lattice",
    "melt_momentum_residual_y_lattice",
    "cumulative_melted_angular_momentum_lattice",
    "cumulative_fluid_melt_angular_momentum_lattice",
    "melt_angular_momentum_residual_lattice",
    "ale_water_residual_cells",
    "ale_energy_residual_j_m",
    "phase_aperture_water_residual_cells",
    "phase_aperture_energy_residual_j_m",
    "phase_aperture_capacity_margin_cells",
    "melt_injection_mass_residual_kg_m",
    "thermal_initial_total_mass_kg_m",
    "thermal_total_mass_kg_m",
    "thermal_total_mass_residual_kg_m",
    "water_mean_temperature_c",
    "aperture_energy_correction_abs_j_m",
)
HISTORY_COLUMNS = reporting.HISTORY_COLUMNS + MOTION_HISTORY_COLUMNS


def _history_row(snapshot: FallingMeltingSnapshot) -> tuple[object, ...]:
    return reporting._history_row(snapshot) + (
        int(snapshot.body_active),
        f"{snapshot.body_center_x_m:.17g}",
        f"{snapshot.body_center_y_m:.17g}",
        f"{snapshot.body_velocity_x_m_s:.17g}",
        f"{snapshot.body_velocity_y_m_s:.17g}",
        f"{snapshot.body_angle_rad:.17g}",
        f"{snapshot.body_angular_velocity_rad_s:.17g}",
        f"{snapshot.body_mass_lattice:.17g}",
        f"{snapshot.body_mass_kg_m:.17g}",
        f"{snapshot.body_inertia_lattice:.17g}",
        f"{snapshot.body_inertia_kg_m:.17g}",
        f"{snapshot.body_solid_center_local_x_cells:.17g}",
        f"{snapshot.body_solid_center_local_y_cells:.17g}",
        f"{snapshot.cumulative_melted_mass_lattice:.17g}",
        f"{snapshot.cumulative_melted_momentum_x_lattice:.17g}",
        f"{snapshot.cumulative_melted_momentum_y_lattice:.17g}",
        f"{snapshot.cumulative_fluid_melt_momentum_x_lattice:.17g}",
        f"{snapshot.cumulative_fluid_melt_momentum_y_lattice:.17g}",
        f"{snapshot.melt_momentum_residual_x_lattice:.17g}",
        f"{snapshot.melt_momentum_residual_y_lattice:.17g}",
        f"{snapshot.cumulative_melted_angular_momentum_lattice:.17g}",
        f"{snapshot.cumulative_fluid_melt_angular_momentum_lattice:.17g}",
        f"{snapshot.melt_angular_momentum_residual_lattice:.17g}",
        f"{snapshot.ale_water_residual_cells:.17g}",
        f"{snapshot.ale_energy_residual_j_m:.17g}",
        f"{snapshot.phase_aperture_water_residual_cells:.17g}",
        f"{snapshot.phase_aperture_energy_residual_j_m:.17g}",
        f"{snapshot.phase_aperture_capacity_margin_cells:.17g}",
        f"{snapshot.melt_injection_mass_residual_kg_m:.17g}",
        f"{snapshot.thermal_initial_total_mass_kg_m:.17g}",
        f"{snapshot.thermal_total_mass_kg_m:.17g}",
        f"{snapshot.thermal_total_mass_residual_kg_m:.17g}",
        f"{snapshot.water_mean_temperature_c:.17g}",
        f"{snapshot.aperture_energy_correction_abs_j_m:.17g}",
    )


def write_history_csv(path: Path, snapshots: list[FallingMeltingSnapshot]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HISTORY_COLUMNS)
        writer.writerows(_history_row(snapshot) for snapshot in snapshots)


def write_fields_npz(
    path: Path,
    config: IceFlowConfig,
    snapshots: list[FallingMeltingSnapshot],
) -> None:
    """Write the common fields plus synchronized moving-body time series."""

    arrays = reporting.snapshot_arrays(config, snapshots)
    arrays.update(
        body_active=np.asarray([item.body_active for item in snapshots], dtype=np.int8),
        body_center_m=np.asarray(
            [[item.body_center_x_m, item.body_center_y_m] for item in snapshots],
            dtype=np.float64,
        ),
        body_velocity_m_s=np.asarray(
            [
                [item.body_velocity_x_m_s, item.body_velocity_y_m_s]
                for item in snapshots
            ],
            dtype=np.float64,
        ),
        body_angle_rad=np.asarray(
            [item.body_angle_rad for item in snapshots], dtype=np.float64
        ),
        body_angular_velocity_rad_s=np.asarray(
            [item.body_angular_velocity_rad_s for item in snapshots],
            dtype=np.float64,
        ),
        body_mass_lattice=np.asarray(
            [item.body_mass_lattice for item in snapshots], dtype=np.float64
        ),
        body_mass_kg_m=np.asarray(
            [item.body_mass_kg_m for item in snapshots], dtype=np.float64
        ),
        body_inertia_lattice=np.asarray(
            [item.body_inertia_lattice for item in snapshots], dtype=np.float64
        ),
        body_inertia_kg_m=np.asarray(
            [item.body_inertia_kg_m for item in snapshots], dtype=np.float64
        ),
        body_solid_center_local_cells=np.asarray(
            [
                [
                    item.body_solid_center_local_x_cells,
                    item.body_solid_center_local_y_cells,
                ]
                for item in snapshots
            ],
            dtype=np.float64,
        ),
        water_volume_residual_cells=np.asarray(
            [item.water_volume_residual_cells for item in snapshots],
            dtype=np.float64,
        ),
        cumulative_melted_mass_lattice=np.asarray(
            [item.cumulative_melted_mass_lattice for item in snapshots],
            dtype=np.float64,
        ),
        cumulative_melted_momentum_lattice=np.asarray(
            [
                [
                    item.cumulative_melted_momentum_x_lattice,
                    item.cumulative_melted_momentum_y_lattice,
                ]
                for item in snapshots
            ],
            dtype=np.float64,
        ),
        cumulative_fluid_melt_momentum_lattice=np.asarray(
            [
                [
                    item.cumulative_fluid_melt_momentum_x_lattice,
                    item.cumulative_fluid_melt_momentum_y_lattice,
                ]
                for item in snapshots
            ],
            dtype=np.float64,
        ),
        melt_momentum_residual_lattice=np.asarray(
            [
                [
                    item.melt_momentum_residual_x_lattice,
                    item.melt_momentum_residual_y_lattice,
                ]
                for item in snapshots
            ],
            dtype=np.float64,
        ),
        cumulative_melted_angular_momentum_lattice=np.asarray(
            [item.cumulative_melted_angular_momentum_lattice for item in snapshots],
            dtype=np.float64,
        ),
        cumulative_fluid_melt_angular_momentum_lattice=np.asarray(
            [item.cumulative_fluid_melt_angular_momentum_lattice for item in snapshots],
            dtype=np.float64,
        ),
        melt_angular_momentum_residual_lattice=np.asarray(
            [item.melt_angular_momentum_residual_lattice for item in snapshots],
            dtype=np.float64,
        ),
        ale_water_residual_cells=np.asarray(
            [item.ale_water_residual_cells for item in snapshots], dtype=np.float64
        ),
        ale_energy_residual_j_m=np.asarray(
            [item.ale_energy_residual_j_m for item in snapshots], dtype=np.float64
        ),
        phase_aperture_water_residual_cells=np.asarray(
            [item.phase_aperture_water_residual_cells for item in snapshots],
            dtype=np.float64,
        ),
        phase_aperture_energy_residual_j_m=np.asarray(
            [item.phase_aperture_energy_residual_j_m for item in snapshots],
            dtype=np.float64,
        ),
        phase_aperture_capacity_margin_cells=np.asarray(
            [item.phase_aperture_capacity_margin_cells for item in snapshots],
            dtype=np.float64,
        ),
        melt_injection_mass_residual_kg_m=np.asarray(
            [item.melt_injection_mass_residual_kg_m for item in snapshots],
            dtype=np.float64,
        ),
        thermal_initial_total_mass_kg_m=np.asarray(
            [item.thermal_initial_total_mass_kg_m for item in snapshots],
            dtype=np.float64,
        ),
        thermal_total_mass_kg_m=np.asarray(
            [item.thermal_total_mass_kg_m for item in snapshots], dtype=np.float64
        ),
        thermal_total_mass_residual_kg_m=np.asarray(
            [item.thermal_total_mass_residual_kg_m for item in snapshots],
            dtype=np.float64,
        ),
        water_mean_temperature_c=np.asarray(
            [item.water_mean_temperature_c for item in snapshots], dtype=np.float64
        ),
        aperture_energy_correction_abs_j_m=np.asarray(
            [item.aperture_energy_correction_abs_j_m for item in snapshots],
            dtype=np.float64,
        ),
    )
    np.savez_compressed(path, **arrays)


def write_final_plot(
    path: Path,
    config: IceFlowConfig,
    snapshot: FallingMeltingSnapshot,
    *,
    temperature_max_c: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (0.0, config.nx * config.dx, 0.0, config.ny * config.dx)
    temperature, _ = reporting._temperature_field_for_plot(
        config, snapshot, frame_index=0
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 8.0), constrained_layout=True)
    image = axes[0].imshow(
        temperature,
        origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=config.thermal.properties.melting_temperature_c,
        vmax=temperature_max_c,
        interpolation="nearest",
        aspect="equal",
    )
    reporting._draw_flow_boundaries(
        axes[0], snapshot, extent, water_surface_color="white"
    )
    figure.colorbar(image, ax=axes[0], label="temperature (degC)")
    axes[0].set_title("Final temperature")

    fraction = np.ma.masked_where(
        np.asarray(snapshot.phase_change_material) == 0,
        np.asarray(snapshot.liquid_fraction),
    )
    phase_image = axes[1].imshow(
        fraction,
        origin="lower",
        extent=extent,
        cmap="Blues_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    reporting._draw_flow_boundaries(
        axes[1], snapshot, extent, water_surface_color="black"
    )
    figure.colorbar(phase_image, ax=axes[1], label="liquid fraction")
    axes[1].set_title("Final phase state")
    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.suptitle(
        "Coupled falling-ice melting, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%, "
        f"center=({snapshot.body_center_x_m:.4f}, {snapshot.body_center_y_m:.4f}) m"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _sequence_metadata(
    velocity: reporting.VelocitySequenceSummary | None,
    vorticity: reporting.VorticitySequenceSummary | None,
    temperature: reporting.TemperatureSequenceSummary | None,
) -> dict[str, Any]:
    return {
        "velocity": (
            None
            if velocity is None
            else {
                "directory": velocity.frame_directory,
                "pattern": velocity.frame_pattern,
                "frames": velocity.frame_count,
                "animation": velocity.animation,
                "units": "m/s",
                "observed_max": velocity.observed_max_m_s,
                "color_max": velocity.color_max_m_s,
            }
        ),
        "vorticity": (
            None
            if vorticity is None
            else {
                "directory": vorticity.frame_directory,
                "pattern": vorticity.frame_pattern,
                "frames": vorticity.frame_count,
                "animation": vorticity.animation,
                "units": "s^-1",
                "observed_abs_max": vorticity.observed_abs_max_s_1,
                "color_abs_max": vorticity.color_abs_max_s_1,
            }
        ),
        "temperature": (
            None
            if temperature is None
            else {
                "directory": temperature.frame_directory,
                "pattern": temperature.frame_pattern,
                "frames": temperature.frame_count,
                "animation": temperature.animation,
                "units": "degC",
                "observed_range": [
                    temperature.observed_min_c,
                    temperature.observed_max_c,
                ],
                "color_range": [temperature.color_min_c, temperature.color_max_c],
            }
        ),
    }


def write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    config: IceFlowConfig,
    scales: LatticeScales,
    targets: list[int],
    initial: FallingMeltingSnapshot,
    final: FallingMeltingSnapshot,
    snapshot_count: int,
    velocity_sequence: reporting.VelocitySequenceSummary | None,
    vorticity_sequence: reporting.VorticitySequenceSummary | None,
    temperature_sequence: reporting.TemperatureSequenceSummary | None,
) -> None:
    sequences = _sequence_metadata(
        velocity_sequence, vorticity_sequence, temperature_sequence
    )
    props = config.thermal.properties
    energy_after_full_melt = (
        initial.total_enthalpy_j_m - initial.body_mass_kg_m * props.latent_heat_j_kg
    )
    equilibrium_temperature = None
    if energy_after_full_melt >= 0.0 and initial.thermal_total_mass_kg_m > 0.0:
        equilibrium_temperature = (
            props.melting_temperature_c
            + energy_after_full_melt
            / (initial.thermal_total_mass_kg_m * props.specific_heat_water_j_kg_k)
        )
    data = {
        "model": "coupled two-dimensional falling-ice melting",
        "backend": "taichi-cuda",
        "hydrodynamics": "air-water phase-field D2Q9 LBM",
        "thermal_solver": "conservative finite-volume enthalpy in water and body coordinates",
        "mechanics": "freely translating and rotating, thermally eroding rigid ice body",
        "moving_body_scheme": config.thermal.moving_body_scheme,
        "phase_change_mass_conversion": (
            "delta_V_water=(rho_ice/rho_water)*(V_solid_initial-V_solid_current)"
        ),
        "closed_bath_reference": {
            "all_walls_adiabatic": all(
                getattr(config.thermal.boundaries, side).kind == "adiabatic"
                for side in ("left", "right", "bottom", "top")
            ),
            "initial_energy_j_m": initial.total_enthalpy_j_m,
            "full_melt_added_water_area_m2": initial.body_mass_kg_m / config.rho_water,
            "melt_water_to_ice_volume_ratio": config.rho_ice / config.rho_water,
            "enough_heat_for_full_melt": equilibrium_temperature is not None,
            "fully_melted_equilibrium_temperature_c": equilibrium_temperature,
        },
        "assumptions": [
            "liquid water remains liquid at 90 degC",
            "no evaporation, boiling, or water-vapour transport",
            "water-air thermal interface is adiabatic",
            "local temperature reconstruction with bounded global sensible-energy correction",
            "linear Boussinesq water buoyancy over the 0--90 degC range",
            "two-dimensional quantities are reported per unit out-of-plane depth",
        ],
        "requested": {
            "end_time_s": float(args.end_time_s),
            "output_interval_s": float(args.output_interval_s),
            "water_temperature_c": float(args.water_temperature_c),
            "save_npz": bool(args.save_npz),
            "progress_enabled": bool(args.progress and not args.quiet),
        },
        "geometry": {
            "domain_width_m": config.nx * config.dx,
            "domain_height_m": config.ny * config.dx,
            "full_width_pool": config.water_width == config.nx - config.boundary_cells,
            "water_level_m": config.water_height * config.dx,
            "ice_width_m": config.ice_width * config.dx,
            "ice_height_m": config.ice_height * config.dx,
            "drop_height_m": float(args.drop_height_m),
            "ice_initial_angle_rad": config.ice_initial_angle,
        },
        "lattice_scaling": {
            "dx_m": scales.dx_m,
            "dt_s": scales.dt_s,
            "velocity_scale_m_s": scales.velocity_scale_m_s,
            "scheduled_snapshot_steps": targets,
            "thermal_update_interval_lbm_steps": (
                config.thermal.update_interval_lbm_steps
            ),
        },
        "config": config.to_dict(),
        "results": {
            "snapshots": snapshot_count,
            "lbm_steps": final.lbm_steps,
            "physical_time_s": final.physical_time_s,
            "melted_fraction": final.melted_fraction,
            "initial_body_mass_lattice": initial.body_mass_lattice,
            "final_body_mass_lattice": final.body_mass_lattice,
            "final_body_active": final.body_active,
            "final_body_center_m": [final.body_center_x_m, final.body_center_y_m],
            "final_body_velocity_m_s": [
                final.body_velocity_x_m_s,
                final.body_velocity_y_m_s,
            ],
            "final_body_angle_rad": final.body_angle_rad,
            "body_solid_center_local_cells": [
                final.body_solid_center_local_x_cells,
                final.body_solid_center_local_y_cells,
            ],
            "cumulative_melted_mass_lattice": (final.cumulative_melted_mass_lattice),
            "cumulative_melted_momentum_lattice": [
                final.cumulative_melted_momentum_x_lattice,
                final.cumulative_melted_momentum_y_lattice,
            ],
            "cumulative_fluid_melt_momentum_lattice": [
                final.cumulative_fluid_melt_momentum_x_lattice,
                final.cumulative_fluid_melt_momentum_y_lattice,
            ],
            "melt_momentum_residual_lattice": [
                final.melt_momentum_residual_x_lattice,
                final.melt_momentum_residual_y_lattice,
            ],
            "cumulative_melted_angular_momentum_lattice": (
                final.cumulative_melted_angular_momentum_lattice
            ),
            "cumulative_fluid_melt_angular_momentum_lattice": (
                final.cumulative_fluid_melt_angular_momentum_lattice
            ),
            "melt_angular_momentum_residual_lattice": (
                final.melt_angular_momentum_residual_lattice
            ),
            "water_volume_residual_cells": final.water_volume_residual_cells,
            "energy_residual_j_m": final.energy_residual_j_m,
            "ale_water_residual_cells": final.ale_water_residual_cells,
            "ale_energy_residual_j_m": final.ale_energy_residual_j_m,
            "phase_aperture_water_residual_cells": (
                final.phase_aperture_water_residual_cells
            ),
            "phase_aperture_energy_residual_j_m": (
                final.phase_aperture_energy_residual_j_m
            ),
            "phase_aperture_capacity_margin_cells": (
                final.phase_aperture_capacity_margin_cells
            ),
            "melt_injection_mass_residual_kg_m": (
                final.melt_injection_mass_residual_kg_m
            ),
            "thermal_initial_total_mass_kg_m": (final.thermal_initial_total_mass_kg_m),
            "thermal_total_mass_kg_m": final.thermal_total_mass_kg_m,
            "thermal_total_mass_residual_kg_m": (
                final.thermal_total_mass_residual_kg_m
            ),
            "water_mean_temperature_c": final.water_mean_temperature_c,
            "aperture_energy_correction_abs_j_m": final.aperture_energy_correction_abs_j_m,
        },
        "field_sequences": sequences,
        "field_output": {
            "write_mode": ("disabled" if args.no_plot else "streaming_png_gif"),
            "raw_fields_retained": bool(args.save_npz),
            "snapshot_arrays_retained_in_memory": bool(args.save_npz),
        },
        "outputs": {
            "history": "history.csv",
            "fields": "fields.npz" if args.save_npz else None,
            "plot": None if args.no_plot else "coupled_falling_ice_melting_2d.png",
            "velocity_frames": (
                None if velocity_sequence is None else velocity_sequence.frame_directory
            ),
            "velocity_animation": (
                None if velocity_sequence is None else velocity_sequence.animation
            ),
            "vorticity_frames": (
                None
                if vorticity_sequence is None
                else vorticity_sequence.frame_directory
            ),
            "vorticity_animation": (
                None if vorticity_sequence is None else vorticity_sequence.animation
            ),
            "temperature_frames": (
                None
                if temperature_sequence is None
                else temperature_sequence.frame_directory
            ),
            "temperature_animation": (
                None if temperature_sequence is None else temperature_sequence.animation
            ),
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _snapshot_step_targets(
    config: IceFlowConfig,
    end_time_s: float,
    output_interval_s: float,
) -> list[int]:
    return reporting._snapshot_step_targets(config, end_time_s, output_interval_s)


def _progress_chunk_steps(total_lbm_steps: int) -> int:
    """Choose about two thousand progress updates for a complete run."""

    return max(1, math.ceil(max(0, int(total_lbm_steps)) / 2000))


def _advance_simulation_to_target(
    simulation: Any,
    target_step: int,
    progress: Any,
    *,
    chunk_steps: int,
) -> None:
    """Advance in bounded chunks so the default progress bar stays live."""

    target = int(target_step)
    current = int(simulation.steps)
    if target < current:
        raise ValueError("target_step must not precede the current simulation step")
    chunk_limit = max(1, int(chunk_steps))
    while current < target:
        count = min(chunk_limit, target - current)
        simulation.step(count)
        updated = int(simulation.steps)
        if updated != current + count:
            raise RuntimeError("simulation step counter did not advance as requested")
        progress.advance(count)
        current = updated


def _run_case(
    args: argparse.Namespace,
    config: IceFlowConfig,
    targets: list[int],
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is the first import that loads simulator.py and its Taichi kernels.
    from iceflow2d import IceFlow2D

    show_progress = bool(args.progress and not args.quiet)
    if not args.quiet:
        print(
            "Initializing Taichi CUDA fields and compiling coupled kernels...",
            file=sys.stderr,
            flush=True,
        )
    simulation = IceFlow2D(config)
    scales = LatticeScales.from_iceflow_config(config)
    initial_total_enthalpy = float(
        simulation.thermal.total_enthalpy_j_m(simulation.wall_mask)
    )
    initial_solid_volume = float(simulation.phase_change_solid_volume_cells())
    initial = _capture_snapshot(
        simulation,
        initial_total_enthalpy_j_m=initial_total_enthalpy,
        initial_solid_volume_cells=initial_solid_volume,
    )
    final = initial
    snapshot_count = 0
    retained: list[FallingMeltingSnapshot] | None = [] if args.save_npz else None
    limits = reporting._shared_visualization_limits(args, [args.water_temperature_c])

    velocity_stream = None
    vorticity_stream = None
    temperature_stream = None
    velocity_sequence = None
    vorticity_sequence = None
    temperature_sequence = None
    total_lbm_steps = targets[-1] if targets else 0
    progress_chunk = _progress_chunk_steps(total_lbm_steps)
    try:
        if not args.no_plot:
            velocity_stream = reporting.VelocitySequenceStream(
                output_dir,
                config,
                scales,
                display_max_m_s=limits.velocity_max_m_s,
                scenario_label=SCENARIO_LABEL,
            )
            vorticity_stream = reporting.VorticitySequenceStream(
                output_dir,
                config,
                scales,
                display_abs_max_s_1=limits.vorticity_abs_max_s_1,
                velocity_display_max_m_s=limits.velocity_max_m_s,
                scenario_label=SCENARIO_LABEL,
            )
            temperature_stream = reporting.TemperatureSequenceStream(
                output_dir,
                config,
                temperature_min_c=limits.temperature_min_c,
                temperature_max_c=limits.temperature_max_c,
                scenario_label=SCENARIO_LABEL,
            )

        with (
            (output_dir / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream,
            reporting._TerminalProgress(
                total_lbm_steps,
                enabled=show_progress,
                label="Coupled falling-ice melting",
                unit="LBM steps",
            ) as progress,
        ):
            writer = csv.writer(stream)
            writer.writerow(HISTORY_COLUMNS)
            for frame_index, target in enumerate(targets):
                if frame_index == 0:
                    snapshot = initial
                else:
                    _advance_simulation_to_target(
                        simulation,
                        target,
                        progress,
                        chunk_steps=progress_chunk,
                    )
                    snapshot = _capture_snapshot(
                        simulation,
                        initial_total_enthalpy_j_m=initial_total_enthalpy,
                        initial_solid_volume_cells=initial_solid_volume,
                        initial_total_mass_kg_m=(
                            initial.thermal_initial_total_mass_kg_m
                        ),
                    )
                writer.writerow(_history_row(snapshot))
                stream.flush()
                if retained is not None:
                    retained.append(snapshot)
                if not args.no_plot:
                    progress.set_status(
                        f"writing field frame {frame_index + 1}/{len(targets)}"
                    )
                if velocity_stream is not None:
                    velocity_stream.append(snapshot)
                if vorticity_stream is not None:
                    vorticity_stream.append(snapshot)
                if temperature_stream is not None:
                    temperature_stream.append(snapshot)
                snapshot_count += 1
                final = snapshot
                output_status = (
                    f"snapshot {snapshot_count}/{len(targets)} recorded"
                    if args.no_plot
                    else f"fields {snapshot_count}/{len(targets)} written"
                )
                progress.set_status(
                    f"{output_status} | t={snapshot.physical_time_s:.4f} s | "
                    f"melted={100.0 * snapshot.melted_fraction:.2f}%"
                )
                if frame_index > 0 and not args.quiet and not show_progress:
                    print(
                        f"t={snapshot.physical_time_s:.4f} s | "
                        f"steps={snapshot.lbm_steps} | "
                        f"y_body={snapshot.body_center_y_m:.5f} m | "
                        f"melted={100.0 * snapshot.melted_fraction:6.2f}%",
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

    if retained is not None:
        write_fields_npz(output_dir / "fields.npz", config, retained)
    if not args.no_plot:
        write_final_plot(
            output_dir / "coupled_falling_ice_melting_2d.png",
            config,
            final,
            temperature_max_c=limits.temperature_max_c,
        )
    write_metadata(
        output_dir / "metadata.json",
        args=args,
        config=config,
        scales=scales,
        targets=targets,
        initial=initial,
        final=final,
        snapshot_count=snapshot_count,
        velocity_sequence=velocity_sequence,
        vorticity_sequence=vorticity_sequence,
        temperature_sequence=temperature_sequence,
    )
    if not args.quiet:
        print(
            "Coupled falling-ice melting example complete "
            f"(T_inf={args.water_temperature_c:g} degC)"
        )
        print(
            f"  simulated: {final.physical_time_s:.6f} s in {final.lbm_steps} LBM steps"
        )
        print(f"  ice melted: {100.0 * final.melted_fraction:.3f}%")
        print(f"  output: {output_dir}")
    return {
        "output_directory": str(output_dir),
        "snapshots": snapshot_count,
        "lbm_steps": final.lbm_steps,
        "physical_time_s": final.physical_time_s,
        "melted_fraction": final.melted_fraction,
        "velocity_frames": (
            0 if velocity_sequence is None else velocity_sequence.frame_count
        ),
        "vorticity_frames": (
            0 if vorticity_sequence is None else vorticity_sequence.frame_count
        ),
        "temperature_frames": (
            0 if temperature_sequence is None else temperature_sequence.frame_count
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = create_config(args)
    targets = _snapshot_step_targets(config, args.end_time_s, args.output_interval_s)
    _run_case(args, config, targets)


if __name__ == "__main__":
    main()
