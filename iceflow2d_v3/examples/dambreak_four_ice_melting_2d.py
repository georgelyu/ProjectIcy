"""A released 65 x 45 mm hot-water column hitting four ice blocks in shallow water.

Run ``python -m iceflow2d_v3.examples.dambreak_four_ice_melting_2d``.
The default 8 s run streams velocity, vorticity, temperature and phase PNG/GIF
outputs without saving raw fields. Molecular viscosities and heat coupling are
the same as in ``four_falling_ice_melting_2d``.
The floor is initially wet, with three resolved water cells below the ice.
"""

from pathlib import Path
import json
import math
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as common


def build_parser():
    parser = common.build_parser(falling_geometry=False)
    parser.description = (
        "Hot-water dam break with four freely moving and melting ice blocks"
    )
    parser.add_argument(
        "--water-column-width-m",
        type=float,
        default=0.065,
        help="water-column width measured from the inner left wall",
    )
    parser.add_argument(
        "--water-column-height-m",
        type=float,
        default=0.045,
        help="water-column height measured from the inner floor",
    )
    parser.add_argument(
        "--ice-floor-clearance-cells",
        type=int,
        default=3,
        help="initial gap from the inner floor to the lowest ice corner, in cells",
    )
    parser.add_argument(
        "--floor-water-depth-m",
        type=float,
        default=None,
        help=(
            "full-width water depth above the inner floor; by default, estimate "
            "the ice draft plus floor clearance and round up to a whole cell; "
            "use 0 with zero ice clearance to restore a dry floor"
        ),
    )
    parser.set_defaults(
        domain_width_m=0.15,
        domain_height_m=0.05,
        reference_velocity_m_s=4.0,
        resolution_x=450,
        resolution_y=150,
        ice_size_m=0.01,
        ice_angle_degrees=0.0,
        ice_centers_x_m=[0.085, 0.103, 0.121, 0.139],
        end_time_s=8.0,
        interface_width=5.0,
        phase_gradient_scheme="connected_ls",
        phase_normal_epsilon=0.01,
        phase_warmup_steps=2000,
        density_gradient_source_limit=0.05,
        phase_flux_acceleration_limit=0.01,
        refill_moment_projection=True,
        gas_state_repair=True,
        check_stability=True,
        save_npz=False,
        no_plot=False,
        velocity_visualization_max_m_s=0.8,
        velocity_include_air=True,
        vorticity_include_air=True,
        air_velocity_visualization_max_m_s=4.0,
        air_vorticity_visualization_max_s_1=5000.0,
        air_velocity_colormap="Oranges",
        air_vorticity_colormap="PuOr",
        scenario_label="hot-water dam break with four melting ice blocks",
        case_label="Dam-break four-ice melting",
        model_label="iceflow2d_v3: dam break with disappearing ice melt",
        output_stem="dambreak_four_ice_melting_2d",
        output_dir=Path("iceflow2d_v3/outputs/dambreak_four_ice_melting_2d"),
    )
    return parser


def derive_geometry(args):
    """Place the water column, shallow pool and ice above the inner floor."""
    nx, ny = int(args.resolution_x), int(args.resolution_y)
    boundary = int(args.boundary_cells)
    if nx <= 2 * boundary + 4 or ny <= 2 * boundary + 4:
        raise ValueError("resolution is too small for the requested boundary layers")
    dx = common.reporting._positive("domain_width_m", args.domain_width_m) / nx
    dy = common.reporting._positive("domain_height_m", args.domain_height_m) / ny
    if not math.isclose(dx, dy, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("the dam-break example requires square cells")
    water_width = boundary + common._aligned_cell_count(
        "water_column_width_m", args.water_column_width_m, dx
    )
    water_height = boundary + common._aligned_cell_count(
        "water_column_height_m", args.water_column_height_m, dx
    )
    if not boundary + 1 < water_width < nx - boundary:
        raise ValueError(
            "water-column width must leave a downstream region inside the walls"
        )
    if not boundary + 1 < water_height < ny - boundary:
        raise ValueError("water-column height must leave an air cap inside the walls")
    ice_size = common._aligned_cell_count("ice_size_m", args.ice_size_m, dx)
    if ice_size < 4:
        raise ValueError("ice must span at least four cells in each direction")
    angle = math.radians(float(args.ice_angle_degrees))
    if not math.isfinite(angle):
        raise ValueError("ice_angle_degrees must be finite")
    extent = 0.5 * ice_size * (abs(math.cos(angle)) + abs(math.sin(angle)))
    clearance = common.reporting._non_negative(
        "ice_floor_clearance_cells", args.ice_floor_clearance_cells
    )
    if int(clearance) != clearance:
        raise ValueError("ice_floor_clearance_cells must be an integer")
    center_y = boundary + extent + clearance
    if center_y + extent > ny - boundary:
        raise ValueError("ice overlaps the top wall")
    if args.floor_water_depth_m is None:
        rho_water = common.reporting._positive("density_water", args.density_water)
        rho_ice = common.reporting._positive("density_ice", args.density_ice)
        rho_air = common.reporting._positive("density_air", args.density_air)
        if not rho_air < rho_ice < rho_water:
            raise ValueError(
                "automatic floor water depth requires air < ice < water density"
            )
        # Exact draft for an axis-aligned square, conservative for a rotated
        # square at the usual ice/water density ratio. Round up so the pool
        # supports the ice with at least the requested floor clearance.
        draft = 2.0 * extent * (rho_ice - rho_air) / (rho_water - rho_air)
        floor_depth = math.ceil(draft + clearance)
    else:
        depth_m = common.reporting._non_negative(
            "floor_water_depth_m", args.floor_water_depth_m
        )
        floor_depth = (
            common._aligned_cell_count("floor_water_depth_m", depth_m, dx)
            if depth_m > 0.0
            else 0
        )
    if boundary + floor_depth >= water_height:
        raise ValueError("floor water depth must be lower than the water column")
    if floor_depth and floor_depth <= clearance:
        raise ValueError("floor water must reach the ice above its floor clearance")
    if not args.ice_centers_x_m:
        raise ValueError("at least one ice center is required")
    for x_m in args.ice_centers_x_m:
        center_x = common.reporting._positive("ice_centers_x_m", x_m) / dx
        if center_x - extent < water_width:
            raise ValueError("ice must start to the right of the water column")
        if center_x + extent > nx - boundary:
            raise ValueError("ice overlaps the right wall")
    return {
        "dx_m": dx,
        "water_width": water_width,
        "water_height": water_height,
        "floor_water_depth_cells": floor_depth,
        "water_width_fraction": common._integer_bin_fraction(water_width, nx),
        "water_height_fraction": common._integer_bin_fraction(water_height, ny),
        "ice_width": ice_size,
        "ice_height": ice_size,
        "ice_width_fraction": common._integer_bin_fraction(ice_size, nx),
        "ice_height_fraction": common._integer_bin_fraction(ice_size, ny),
        "ice_angle_rad": angle,
        "ice_base_y_cells": center_y - 0.5 * ice_size,
        "ice_center_x_cells": args.ice_centers_x_m[0] / dx,
        "ice_center_y_cells": center_y,
    }


def parse_args(argv=None):
    return common.parse_args(
        argv, parser=build_parser(), geometry_builder=derive_geometry
    )


def create_config(args=None):
    args = parse_args([]) if args is None else args
    return common.create_config(args, geometry=derive_geometry(args))


def main(argv=None):
    return run_case(parse_args(argv))


def run_case(args):
    """Run a validated dam-break configuration, retaining failure diagnostics."""
    config = create_config(args)
    targets = common._snapshot_step_targets(
        config, args.end_time_s, args.output_interval_s
    )
    try:
        return common._run_case(args, config, targets)
    except Exception as exc:
        # Keep a failed attempt distinguishable from a completed simulation;
        # already-streamed images and closed GIFs remain usable, without arrays.
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        step_match = re.search(r"at step (\d+)", str(exc))
        attempted_step = int(step_match[1]) if step_match else None
        scales = common.LatticeScales.from_iceflow_config(config)
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "requested_time_s": args.end_time_s,
            "attempted_failure_step": attempted_step,
            "attempted_failure_time_s": None
            if attempted_step is None
            else attempted_step * scales.dt_s,
            "configuration": config.to_dict(),
            "velocity_include_air": args.velocity_include_air,
            "velocity_color_max_m_s": args.velocity_visualization_max_m_s,
            "visualization": {
                "water_velocity": {
                    "colormap": "viridis",
                    "range_m_s": [0, args.velocity_visualization_max_m_s],
                },
                "air_velocity": {
                    "colormap": args.air_velocity_colormap,
                    "range_m_s": [0, args.air_velocity_visualization_max_m_s],
                },
                "water_vorticity": {
                    "colormap": "RdBu_r",
                    "range_s_1": [
                        -args.vorticity_visualization_max_s_1,
                        args.vorticity_visualization_max_s_1,
                    ],
                },
                "air_vorticity": {
                    "colormap": args.air_vorticity_colormap,
                    "range_s_1": [
                        -args.air_vorticity_visualization_max_s_1,
                        args.air_vorticity_visualization_max_s_1,
                    ],
                },
                "vorticity_include_air": args.vorticity_include_air,
            },
            "saved_png_frames": {
                kind: len(list((output_dir / f"{kind}_frames").glob("frame_*.png")))
                for kind in ("velocity", "vorticity", "temperature", "phase")
            },
            "raw_fields_saved": bool(list(output_dir.glob("*.npz"))),
        }
        (output_dir / "run_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
