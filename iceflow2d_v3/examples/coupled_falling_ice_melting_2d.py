"""Fully coupled example of an ice block falling into a 90 degC water bath.

All four container walls are adiabatic. The ice starts in the insulated air
cap, falls through the free surface, and exchanges heat and boundary forces
with the hot water. Melted ice disappears without adding liquid. Water remains a liquid at 90 degC;
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

from iceflow2d_v3.config import IceFlowConfig, create_iceflow_config  # noqa: E402
from iceflow2d_v3 import reporting  # noqa: E402
from iceflow2d_v3.config import (  # noqa: E402
    DEFAULT_REFERENCE_VELOCITY_M_S,
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundarySet,
    ThermalConfig,
)


DEFAULT_OUTPUT_DIR = "iceflow2d_v3/outputs/iceflow2d_v3/coupled_falling_ice_melting_2d"
DEFAULT_BOUNDARY_CELLS = 3
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
    body_melted_fraction: float = 0.0
    bodies: tuple[FallingMeltingSnapshot, ...] = ()

    def __getattr__(self, name: str) -> Any:
        # The common visualization writers consume CoupledSnapshot fields by
        # attribute, so composition keeps the reusable format without copying
        # the large arrays a second time.
        return getattr(self.coupled, name)


def build_parser(*, falling_geometry: bool = True) -> argparse.ArgumentParser:
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
    if falling_geometry:
        parser.add_argument("--water-level-m", type=float, default=0.030)
    parser.add_argument("--ice-size-m", type=float, default=0.008)
    parser.add_argument("--ice-angle-degrees", type=float, default=5.0)
    parser.add_argument(
        "--ice-centers-x-m",
        type=float,
        nargs="+",
        default=None,
        help="initial horizontal centers of independent ice blocks, in metres",
    )
    if falling_geometry:
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
    parser.add_argument(
        "--momentum-collision-model",
        choices=("rm_mrt", "cm_mrt", "cm_hrr"),
        default="cm_mrt",
        help="pressure momentum collision: cm_mrt (default), rm_mrt, or experimental cm_hrr",
    )

    parser.add_argument(
        "--hrr-kinetic-blend",
        type=float,
        default=0.0,
        help="cm_hrr kinetic stress fraction in [0,1]; 0 reconstructs from velocity gradients",
    )
    parser.add_argument(
        "--phase-momentum-coupling",
        choices=("legacy", "consistent_flux"),
        default="consistent_flux",
        help="experimental conservative phase mass-flux momentum correction",
    )
    parser.add_argument(
        "--exposed-node-refill",
        choices=("equilibrium", "non_equilibrium"),
        default="equilibrium",
        help="extrapolate non-equilibrium stress into newly exposed fluid",
    )
    parser.add_argument(
        "--density-gradient-source-limit",
        type=float,
        default=0.0,
        help="local density-source budget; zero disables limiting",
    )
    parser.add_argument(
        "--phase-flux-acceleration-limit",
        type=float,
        default=0.0,
        help="symmetric numerical flux-force budget; zero disables limiting",
    )
    parser.add_argument(
        "--refill-moment-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="match exposed-node pressure and half-force momentum",
    )
    parser.add_argument(
        "--gas-state-repair", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--phase-gradient-scheme",
        choices=("legacy", "connected_ls"),
        default="legacy",
        help="connected_ls reconstructs boundary gradients from locally reachable fluid",
    )
    parser.add_argument(
        "--cm-acoustic-relaxation-rate",
        type=float,
        default=None,
        help=(
            "CM-MRT trace/22 or CM-HRR trace/pressure-mode relaxation rate in (0, 2); "
            "default 0.25 for cm_mrt adds acoustic damping without changing "
            "shear viscosity; 1 restores the original collision and is used for rm_mrt"
        ),
    )

    parser.add_argument("--end-time-s", type=float, default=5.0)
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
        "--velocity-include-air",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show both air and water in velocity frames with separate color scales",
    )
    parser.add_argument(
        "--vorticity-visualization-max-s-1",
        type=float,
        default=DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
    )
    parser.add_argument("--air-velocity-visualization-max-m-s", type=float, default=4.0)
    parser.add_argument(
        "--air-vorticity-visualization-max-s-1", type=float, default=5000.0
    )
    parser.add_argument("--air-velocity-colormap", default="Oranges")
    parser.add_argument("--air-vorticity-colormap", default="PuOr")
    parser.add_argument(
        "--vorticity-include-air", action=argparse.BooleanOptionalAction, default=False
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
    parser.add_argument(
        "--kinematic-viscosity-water",
        type=float,
        default=None,
        help="constant water viscosity in m^2/s; default uses local-temperature IAPWS mu(T)/rho",
    )
    parser.add_argument(
        "--kinematic-viscosity-air",
        type=float,
        default=None,
        help="constant air viscosity in m^2/s; default uses Sutherland mu(T_air)/rho",
    )
    parser.add_argument("--surface-tension", type=float, default=0.0)

    parser.add_argument("--water-specific-heat", type=float, default=4186.0)
    parser.add_argument("--ice-specific-heat", type=float, default=2100.0)
    parser.add_argument("--water-conductivity", type=float, default=0.60)
    parser.add_argument("--ice-conductivity", type=float, default=2.20)
    parser.add_argument("--latent-heat", type=float, default=334000.0)

    parser.add_argument("--phase-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--thermal-scheme",
        choices=("enthalpy_fv", "enthalpy_d2q5"),
        default="enthalpy_fv",
        help="bulk heat solver: finite volume (default) or D2Q5 distributions",
    )
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
    parser.add_argument(
        "--phase-normal-epsilon",
        type=float,
        default=1.0e-3,
        help=(
            "phase compression normal regularization in lattice-gradient units; "
            "default 1e-3 suppresses bulk phase spots; 0 restores the original algorithm"
        ),
    )
    parser.add_argument("--no-advection", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="disable final plot and velocity/vorticity/temperature/phase sequences",
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
        "--check-stability",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="check the full air/water LBM state at every LBM step",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress and run summaries",
    )
    return parser


def parse_args(
    argv: list[str] | None = None, *, parser=None, geometry_builder=None
) -> argparse.Namespace:
    parser = parser or build_parser()
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
        reporting._positive(
            "air_velocity_visualization_max_m_s",
            args.air_velocity_visualization_max_m_s,
        )
        reporting._positive(
            "air_vorticity_visualization_max_s_1",
            args.air_vorticity_visualization_max_s_1,
        )
        import matplotlib

        for name in (args.air_velocity_colormap, args.air_vorticity_colormap):
            if name not in matplotlib.colormaps:
                raise ValueError(f"unknown colormap: {name}")
        (geometry_builder or derive_geometry)(args)
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


def create_config(
    args: argparse.Namespace | None = None, *, geometry=None
) -> IceFlowConfig:
    """Build the moving ``body_ale`` thermal configuration without CUDA."""

    if args is None:
        args = parse_args([])
    geometry = derive_geometry(args) if geometry is None else geometry
    properties = PhaseChangeProperties(
        melting_temperature_c=args.melting_temperature_c,
        specific_heat_water_j_kg_k=args.water_specific_heat,
        specific_heat_ice_j_kg_k=args.ice_specific_heat,
        conductivity_water_w_m_k=args.water_conductivity,
        conductivity_ice_w_m_k=args.ice_conductivity,
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
        scheme=args.thermal_scheme,
    )
    gravity = reporting._non_negative("gravity_m_s2", args.gravity_m_s2)
    bodies = None
    if args.ice_centers_x_m is not None:
        from iceflow2d_v3.config import IceBodyConfig

        bodies = tuple(
            IceBodyConfig(
                size_cells=(int(geometry["ice_width"]), int(geometry["ice_height"])),
                center_cells=(
                    float(x) / float(geometry["dx_m"]),
                    float(geometry["ice_center_y_cells"]),
                ),
                angle_rad=float(geometry["ice_angle_rad"]),
                velocity_lattice=(
                    args.initial_horizontal_speed_lattice,
                    args.initial_vertical_speed_lattice,
                ),
                temperature_c=args.ice_temperature_c,
            )
            for x in args.ice_centers_x_m
        )
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
        momentum_collision_model=args.momentum_collision_model,
        cm_acoustic_relaxation_rate=args.cm_acoustic_relaxation_rate,
        hrr_kinetic_blend=args.hrr_kinetic_blend,
        phase_momentum_coupling=args.phase_momentum_coupling,
        exposed_node_refill=args.exposed_node_refill,
        density_gradient_source_limit=args.density_gradient_source_limit,
        phase_flux_acceleration_limit=args.phase_flux_acceleration_limit,
        refill_moment_projection=args.refill_moment_projection,
        gas_state_repair=args.gas_state_repair,
        phase_gradient_scheme=args.phase_gradient_scheme,
        sigma=args.surface_tension,
        gravity=(0.0, -gravity),
        interface_width=args.interface_width,
        mobility=args.mobility,
        phase_normal_epsilon=args.phase_normal_epsilon,
        phase_warmup_steps=args.phase_warmup_steps,
        boundary_cells=int(args.boundary_cells),
        thermal=thermal,
        well_balanced_hydrostatics=True,
        water_width_fraction=float(geometry["water_width_fraction"]),
        water_height_fraction=float(geometry["water_height_fraction"]),
        floor_water_depth_cells=int(geometry.get("floor_water_depth_cells", 0)),
        ice_width_fraction=float(geometry["ice_width_fraction"]),
        ice_height_fraction=float(geometry["ice_height_fraction"]),
        ice_base_y_cells=float(geometry["ice_base_y_cells"]),
        ice_initial_velocity=(
            args.initial_horizontal_speed_lattice,
            args.initial_vertical_speed_lattice,
        ),
        ice_initial_angle=float(geometry["ice_angle_rad"]),
        ice_fixed=False,
        ice_bodies=bodies,
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


def _capture_snapshot(
    simulation: Any,
    *,
    initial_solid_volume_cells: float,
) -> FallingMeltingSnapshot:
    coupled = reporting._capture_snapshot(
        simulation, initial_solid_volume_cells=initial_solid_volume_cells
    )
    scales = simulation.scales
    count = getattr(simulation, "body_count", 1)
    fields = {
        name: np.asarray(getattr(simulation, name).to_numpy()).reshape(count, -1)
        for name in (
            "body_center",
            "body_velocity",
            "body_local_center_of_mass",
            "body_mass_lattice",
            "body_inertia_lattice",
            "body_active",
            "body_angle",
            "body_angular_velocity",
            "cumulative_melted_mass_lattice",
        )
    }
    bodies = []
    for b in range(count):
        center = fields["body_center"][b]
        velocity = fields["body_velocity"][b]
        local_center = fields["body_local_center_of_mass"][b]
        mass = float(fields["body_mass_lattice"][b, 0])
        inertia = float(fields["body_inertia_lattice"][b, 0])
        melted = float(fields["cumulative_melted_mass_lattice"][b, 0])
        dx = simulation.config.dx
        rho_water = simulation.config.rho_water
        bodies.append(
            FallingMeltingSnapshot(
                coupled=coupled,
                body_melted_fraction=max(0.0, melted / (mass + melted))
                if mass + melted > 0
                else 0.0,
                body_active=bool(fields["body_active"][b, 0]),
                body_center_x_m=float(center[0]) * dx,
                body_center_y_m=float(center[1]) * dx,
                body_velocity_x_m_s=float(velocity[0]) * scales.velocity_scale_m_s,
                body_velocity_y_m_s=float(velocity[1]) * scales.velocity_scale_m_s,
                body_angle_rad=float(fields["body_angle"][b, 0]),
                body_angular_velocity_rad_s=float(fields["body_angular_velocity"][b, 0])
                / scales.dt_s,
                body_mass_lattice=mass,
                body_mass_kg_m=mass * rho_water * dx**2,
                body_inertia_lattice=inertia,
                body_inertia_kg_m=inertia * rho_water * dx**4,
                body_solid_center_local_x_cells=float(local_center[0]),
                body_solid_center_local_y_cells=float(local_center[1]),
                cumulative_melted_mass_lattice=float(
                    fields["cumulative_melted_mass_lattice"][b, 0]
                ),
            )
        )
    from dataclasses import replace

    return replace(bodies[0], bodies=tuple(bodies))


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
    "body_melted_fraction",
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
        f"{snapshot.body_melted_fraction:.17g}",
    )


def _history_rows(snapshot: FallingMeltingSnapshot):
    if len(snapshot.bodies) > 1:
        return [
            (body_id, *_history_row(body))
            for body_id, body in enumerate(snapshot.bodies)
        ]
    return [_history_row(snapshot)]


def _history_columns(snapshot: FallingMeltingSnapshot):
    return (
        (("body_id",) + HISTORY_COLUMNS)
        if len(snapshot.bodies) > 1
        else HISTORY_COLUMNS
    )


def write_history_csv(path: Path, snapshots: list[FallingMeltingSnapshot]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            _history_columns(snapshots[0]) if snapshots else HISTORY_COLUMNS
        )
        for snapshot in snapshots:
            writer.writerows(_history_rows(snapshot))


def write_fields_npz(
    path: Path,
    config: IceFlowConfig,
    snapshots: list[FallingMeltingSnapshot],
) -> None:
    """Write the common fields plus synchronized moving-body time series."""

    arrays = reporting.snapshot_arrays(config, snapshots)
    body_count = len(config.bodies)
    motion_shape = (len(snapshots), body_count)
    if body_count > 1:
        snapshots = [body for frame in snapshots for body in frame.bodies]
    motion = dict(
        body_melted_fraction=np.asarray(
            [item.body_melted_fraction for item in snapshots], dtype=np.float64
        ),
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
        cumulative_melted_mass_lattice=np.asarray(
            [item.cumulative_melted_mass_lattice for item in snapshots],
            dtype=np.float64,
        ),
    )
    if body_count > 1:
        motion = {
            key: values.reshape(*motion_shape, *values.shape[1:])
            for key, values in motion.items()
        }
    arrays.update(motion)
    arrays["body_id"] = np.arange(body_count, dtype=np.int32)
    np.savez_compressed(path, **arrays)


def write_final_plot(
    path: Path,
    config: IceFlowConfig,
    snapshot: FallingMeltingSnapshot,
    *,
    temperature_max_c: float,
    case_label: str = "Coupled falling-ice melting",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (0.0, config.nx * config.dx, 0.0, config.ny * config.dx)
    temperature, _ = reporting._temperature_field_for_plot(
        config, snapshot, frame_index=0
    )
    temperature_colormap = plt.get_cmap("inferno").copy()
    temperature_colormap.set_bad("#aeb4ba")
    rows, columns = (2, 1) if config.nx > config.ny else (1, 2)
    figure, axes = plt.subplots(
        rows, columns, figsize=(10.0, 8.0), constrained_layout=True
    )
    image = axes[0].imshow(
        temperature,
        origin="lower",
        extent=extent,
        cmap=temperature_colormap,
        vmin=config.thermal.properties.melting_temperature_c,
        vmax=temperature_max_c,
        interpolation="nearest",
        aspect="equal",
    )
    reporting._draw_flow_boundaries(
        axes[0], snapshot, extent, water_surface_color="white"
    )
    figure.colorbar(image, ax=axes[0], label="water temperature (degC)")
    axes[0].set_title("Final water temperature")

    # Use the same air/water field and mask as the phase animation. The thermal
    # wet-cell threshold would draw near-zero phase tails as sharp air patches.
    fraction = reporting._phase_field_for_plot(config, snapshot, frame_index=0)
    phase_colormap = plt.get_cmap("Blues").copy()
    phase_colormap.set_bad("#aeb4ba")
    phase_image = axes[1].imshow(
        fraction,
        origin="lower",
        extent=extent,
        cmap=phase_colormap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    reporting._draw_flow_boundaries(
        axes[1], snapshot, extent, water_surface_color="black"
    )
    figure.colorbar(phase_image, ax=axes[1], label="water phase (0: air, 1: water)")
    axes[1].set_title("Final air-water phase")
    for axis in axes:
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.suptitle(
        f"{case_label}, "
        f"t={snapshot.physical_time_s:.4f} s, "
        f"melted={100.0 * snapshot.melted_fraction:.2f}%, "
        + (
            f"{len(snapshot.bodies)} independent ice blocks"
            if len(snapshot.bodies) > 1
            else f"center=({snapshot.body_center_x_m:.4f}, {snapshot.body_center_y_m:.4f}) m"
        )
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _sequence_metadata(
    velocity: reporting.VelocitySequenceSummary | None,
    vorticity: reporting.VorticitySequenceSummary | None,
    temperature: reporting.TemperatureSequenceSummary | None,
    phase: reporting.PhaseSequenceSummary | None = None,
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
                "phases": ["water", "air"] if velocity.include_air else ["water"],
                "water_colormap": velocity.water_colormap,
                "air_colormap": velocity.air_colormap if velocity.include_air else None,
                "air_color_max": velocity.air_color_max_m_s
                if velocity.include_air
                else None,
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
                "phases": ["water", "air"] if vorticity.include_air else ["water"],
                "water_colormap": vorticity.water_colormap,
                "air_colormap": vorticity.air_colormap
                if vorticity.include_air
                else None,
                "air_color_abs_max": vorticity.air_color_abs_max_s_1
                if vorticity.include_air
                else None,
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
        "phase": (
            None
            if phase is None
            else {
                "directory": phase.frame_directory,
                "pattern": phase.frame_pattern,
                "frames": phase.frame_count,
                "animation": phase.animation,
                "field": "water_phase",
                "units": "dimensionless",
                "observed_range": [phase.observed_min, phase.observed_max],
                "color_range": [phase.color_min, phase.color_max],
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
    phase_sequence: reporting.PhaseSequenceSummary | None = None,
) -> None:
    sequences = _sequence_metadata(
        velocity_sequence, vorticity_sequence, temperature_sequence, phase_sequence
    )
    data = {
        "model": getattr(
            args, "model_label", "iceflow2d_v3: falling ice with disappearing melt"
        ),
        "backend": "taichi-cuda",
        "hydrodynamics": "air-water phase-field D2Q9 LBM",
        "thermal_solver": "water sensible heat and material-frame ice melting",
        "thermal_scheme": config.thermal.scheme,
        "viscosity": {
            "water_model": (
                "iapws_local_temperature"
                if config.viscosity_water is None
                else "constant"
            ),
            "air_model": (
                "sutherland_prescribed_temperature"
                if config.viscosity_air is None
                else "constant"
            ),
            "initial_water_m2_s": config.water_viscosity_at(
                config.thermal.initial_water_temperature_c
            ),
            "air_m2_s": config.air_viscosity_at(
                config.thermal.initial_air_temperature_c
            ),
            "interface_mixing": "linear kinematic viscosity in water phase fraction",
        },
        "mechanics": "freely translating and rotating, thermally eroding rigid ice body",
        "moving_body_scheme": config.thermal.moving_body_scheme,
        "phase_change": "melted ice disappears without adding liquid, heat, or momentum sources",
        "assumptions": [
            "liquid water remains liquid at 90 degC",
            "no evaporation, boiling, or water-vapour transport",
            "water-air thermal interface is adiabatic",
            "cells exposed by motion or erosion inherit the surrounding fluid state",
            "local temperature reconstruction with no global mass or energy correction",
            "linear Boussinesq water buoyancy over the 0--90 degC range",
            "two-dimensional quantities are reported per unit out-of-plane depth",
        ],
        "requested": {
            "end_time_s": float(args.end_time_s),
            "output_interval_s": float(args.output_interval_s),
            "water_temperature_c": float(args.water_temperature_c),
            "save_npz": bool(args.save_npz),
            "progress_enabled": bool(args.progress and not args.quiet),
            "stability_check_enabled": bool(args.check_stability),
        },
        "geometry": {
            "domain_width_m": config.nx * config.dx,
            "domain_height_m": config.ny * config.dx,
            "full_width_pool": config.water_width == config.nx - config.boundary_cells,
            "water_level_m": config.water_height * config.dx,
            "ice_width_m": config.ice_width * config.dx,
            "ice_height_m": config.ice_height * config.dx,
            "drop_height_m": getattr(args, "drop_height_m", None),
            "ice_initial_angle_rad": config.bodies[0].angle_rad,
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
            "water_mean_temperature_c": final.water_mean_temperature_c,
            "water_volume_current_cells": final.water_volume_current_cells,
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
            "plot": None
            if args.no_plot
            else f"{getattr(args, 'output_stem', 'coupled_falling_ice_melting_2d')}.png",
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
            "phase_frames": (
                None if phase_sequence is None else phase_sequence.frame_directory
            ),
            "phase_animation": (
                None if phase_sequence is None else phase_sequence.animation
            ),
        },
    }
    data["body_count"] = len(config.bodies)
    if hasattr(args, "water_column_width_m"):
        data["geometry"].update(
            water_column_width_m=float(args.water_column_width_m),
            water_column_height_m=float(args.water_column_height_m),
            floor_water_depth_m=config.floor_water_depth_cells * config.dx,
            floor_water_depth_cells=config.floor_water_depth_cells,
            ice_floor_clearance_cells=args.ice_floor_clearance_cells,
            water_column_bounds_m=[
                [config.boundary_cells * config.dx, config.water_width * config.dx],
                [config.boundary_cells * config.dx, config.water_height * config.dx],
            ],
            gate_removal="instantaneous at t=0; no gate remains in the domain",
            hydrostatic_reference=(
                "shallow pool with initial water-column pressure residual"
                if config.floor_water_depth_cells
                else "air atmosphere with initial water-column pressure residual"
            ),
        )
    data["geometry"]["ice_bodies"] = [
        {
            "body_id": b,
            "size_m": [n * config.dx for n in body.size_cells],
            "center_m": [x * config.dx for x in body.center_cells],
            "angle_rad": body.angle_rad,
        }
        for b, body in enumerate(config.bodies)
    ]
    if len(config.bodies) > 1:
        data["mechanics"] = (
            "independent eroding rigid bodies; frictionless inelastic raster contact"
        )
        data["contact"] = {
            "restitution": 0.0,
            "friction": 0.0,
            "geometry": "current sharp solid cells",
            "ice_ice_heat_transfer": "adiabatic",
        }
        data["results"] = {
            "snapshots": snapshot_count,
            "lbm_steps": final.lbm_steps,
            "physical_time_s": final.physical_time_s,
            "melted_fraction": final.melted_fraction,
            "water_mean_temperature_c": final.water_mean_temperature_c,
            "water_volume_current_cells": final.water_volume_current_cells,
            "bodies": [
                dict(
                    body_id=b,
                    **{name: getattr(body, name) for name in MOTION_HISTORY_COLUMNS},
                )
                for b, body in enumerate(final.bodies)
            ],
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
    scenario_label = getattr(args, "scenario_label", SCENARIO_LABEL)
    case_label = getattr(args, "case_label", "Coupled falling-ice melting")
    output_stem = getattr(args, "output_stem", "coupled_falling_ice_melting_2d")

    # This is the first import that loads simulator.py and its Taichi kernels.
    from iceflow2d_v3 import IceFlow2D

    show_progress = bool(args.progress and not args.quiet)
    if not args.quiet:
        print(
            "Initializing Taichi CUDA fields and compiling coupled kernels...",
            file=sys.stderr,
            flush=True,
        )
    simulation = IceFlow2D(config)
    simulation._thermal_stability_check_enabled = bool(args.check_stability)
    scales = LatticeScales.from_iceflow_config(config)
    initial_solid_volume = float(simulation.phase_change_solid_volume_cells())
    initial = _capture_snapshot(
        simulation,
        initial_solid_volume_cells=initial_solid_volume,
    )
    final = initial
    snapshot_count = 0
    retained: list[FallingMeltingSnapshot] | None = [] if args.save_npz else None
    limits = reporting._shared_visualization_limits(args, [args.water_temperature_c])

    velocity_stream = None
    vorticity_stream = None
    temperature_stream = None
    phase_stream = None
    velocity_sequence = None
    vorticity_sequence = None
    temperature_sequence = None
    phase_sequence = None
    total_lbm_steps = targets[-1] if targets else 0
    progress_chunk = _progress_chunk_steps(total_lbm_steps)
    try:
        if not args.no_plot:
            velocity_stream = reporting.VelocitySequenceStream(
                output_dir,
                config,
                scales,
                display_max_m_s=limits.velocity_max_m_s,
                scenario_label=scenario_label,
                include_air=args.velocity_include_air,
                air_display_max_m_s=args.air_velocity_visualization_max_m_s,
                air_colormap=args.air_velocity_colormap,
            )
            vorticity_stream = reporting.VorticitySequenceStream(
                output_dir,
                config,
                scales,
                display_abs_max_s_1=limits.vorticity_abs_max_s_1,
                velocity_display_max_m_s=limits.velocity_max_m_s,
                scenario_label=scenario_label,
                include_air=args.vorticity_include_air,
                air_display_abs_max_s_1=args.air_vorticity_visualization_max_s_1,
                air_velocity_display_max_m_s=args.air_velocity_visualization_max_m_s,
                air_colormap=args.air_vorticity_colormap,
            )
            temperature_stream = reporting.TemperatureSequenceStream(
                output_dir,
                config,
                temperature_min_c=limits.temperature_min_c,
                temperature_max_c=limits.temperature_max_c,
                scenario_label=scenario_label,
            )
            phase_stream = reporting.PhaseSequenceStream(
                output_dir,
                config,
                scenario_label=scenario_label,
            )

        with (
            (output_dir / "history.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream,
            reporting._TerminalProgress(
                total_lbm_steps,
                enabled=show_progress,
                label=case_label,
                unit="LBM steps",
            ) as progress,
        ):
            writer = csv.writer(stream)
            writer.writerow(_history_columns(initial))
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
                        initial_solid_volume_cells=initial_solid_volume,
                    )
                writer.writerows(_history_rows(snapshot))
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
                if phase_stream is not None:
                    phase_stream.append(snapshot)
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
                        f"bodies={len(config.bodies)} | "
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
                try:
                    if temperature_stream is not None:
                        temperature_sequence = temperature_stream.close()
                finally:
                    if phase_stream is not None:
                        phase_sequence = phase_stream.close()

    if retained is not None:
        write_fields_npz(output_dir / "fields.npz", config, retained)
    if not args.no_plot:
        write_final_plot(
            output_dir / f"{output_stem}.png",
            config,
            final,
            temperature_max_c=limits.temperature_max_c,
            case_label=case_label,
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
        phase_sequence=phase_sequence,
    )
    if not args.quiet:
        print(
            f"{case_label} example complete (T_inf={args.water_temperature_c:g} degC)"
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
        "phase_frames": 0 if phase_sequence is None else phase_sequence.frame_count,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = create_config(args)
    targets = _snapshot_step_targets(config, args.end_time_s, args.output_interval_s)
    _run_case(args, config, targets)


if __name__ == "__main__":
    main()
