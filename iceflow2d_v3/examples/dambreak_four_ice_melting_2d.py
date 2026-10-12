"""A 90 C, 65 x 45 mm released water column pushes and melts four floor ice blocks.

Run ``python -m iceflow2d_v3.examples.dambreak_four_ice_melting_2d``.
The 150 x 50 mm / 450 x 150 domain includes three wall cells on each side.
Water-column dimensions are measured from the inside of the left/bottom wall.
The gate is absent at t=0; ice moves freely and melting transfers mass to water.
"""

from pathlib import Path
import math
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as common


def build_parser():
    parser = common.build_parser(include_falling_geometry=False)
    parser.description = "Hot-water dam break pushing and melting four freely moving ice blocks"
    parser.add_argument("--water-column-width-m", type=float, default=0.065)
    parser.add_argument("--water-column-height-m", type=float, default=0.045)
    parser.add_argument("--close-subgrid-wall-gaps", action=common.argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subgrid-wall-gap-cells", type=int, default=1)
    parser.add_argument("--preserve-contact-fluid-gap", action=common.argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-when-all-ice-melted", action=common.argparse.BooleanOptionalAction, default=True,
                        help="after all ice mass is zero, finish at the next integer second and save all outputs")
    parser.set_defaults(
        domain_width_m=0.15, domain_height_m=0.05,
        resolution_x=450, resolution_y=150, reference_velocity_m_s=4.0,
        ice_size_m=0.01, ice_angle_degrees=0.0,
        ice_centers_x_m=[0.08, 0.10, 0.12, 0.14],
        water_level_m=0.046, drop_height_m=0.0,
        water_temperature_c=90.0, ice_temperature_c=0.0, air_temperature_c=20.0,
        gravity_m_s2=9.81, end_time_s=10.0, output_interval_s=0.01,
        interface_width=5.0, phase_warmup_steps=2000,
        initial_water_geometry="dam_break",
        phase_momentum_coupling="consistent_flux",
        phase_momentum_flux_model="accepted_links",
        momentum_phase_gradient="open_links_air",
        phase_momentum_water_cutoff=0.001,
        air_viscosity_floor_lattice=0.1,
        linear_damping=1.0, angular_damping=1.0,
        check_stability=True, save_npz=False,
        velocity_visualization_max_m_s=1.5,
        output_dir=Path("iceflow2d_v3/outputs/dambreak_four_ice_melting_2d"),
    )
    return parser


def derive_geometry(args):
    nx, ny, boundary = int(args.resolution_x), int(args.resolution_y), int(args.boundary_cells)
    if nx <= 2 * boundary + 4 or ny <= 2 * boundary + 4:
        raise ValueError("resolution is too small for the requested boundary layers")
    width = common.reporting._positive("domain_width_m", args.domain_width_m)
    height = common.reporting._positive("domain_height_m", args.domain_height_m)
    dx, dy = width / nx, height / ny
    if not math.isclose(dx, dy, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("dam-break example requires square cells")
    column_width = common._aligned_cell_count("water_column_width_m", args.water_column_width_m, dx)
    column_height = common._aligned_cell_count("water_column_height_m", args.water_column_height_m, dx)
    water_width, water_height = boundary + column_width, boundary + column_height
    if column_width < 2 or column_height < 2:
        raise ValueError("water column must span at least two cells in each direction")
    if water_width >= nx - boundary or water_height >= ny - boundary:
        raise ValueError("water column must leave downstream air and an air cap")
    ice_width = common._aligned_cell_count("ice_size_m", args.ice_size_m, dx)
    if ice_width < 4:
        raise ValueError("ice must span at least four cells in each direction")
    if not args.ice_centers_x_m:
        raise ValueError("ice_centers_x_m must specify at least one ice block")
    angle = math.radians(float(args.ice_angle_degrees))
    if not math.isfinite(angle):
        raise ValueError("ice_angle_degrees must be finite")
    extent = (abs(math.cos(angle)) + abs(math.sin(angle))) * ice_width / 2
    center_y = boundary + extent
    if center_y + extent > ny - boundary:
        raise ValueError("ice overlaps the top wall")
    for x in args.ice_centers_x_m:
        cx = common.reporting._positive("ice center x", x) / dx
        if cx - extent <= water_width or cx + extent > nx - boundary:
            raise ValueError("ice must start downstream of the column and inside the right wall")
    return dict(
        dx_m=dx, water_width=water_width, water_height=water_height,
        water_width_fraction=common._integer_bin_fraction(water_width, nx),
        water_height_fraction=common._integer_bin_fraction(water_height, ny),
        ice_width=ice_width, ice_height=ice_width,
        ice_width_fraction=common._integer_bin_fraction(ice_width, nx),
        ice_height_fraction=common._integer_bin_fraction(ice_width, ny),
        ice_angle_rad=angle, ice_base_y_cells=float(boundary),
        ice_center_x_cells=float(args.ice_centers_x_m[0]) / dx,
        ice_center_y_cells=center_y, ice_lowest_y_cells=float(boundary),
        ice_highest_y_cells=center_y + extent, drop_height_cells=0.0,
        interface_clearance_cells=0,
    )


def parse_args(argv=None):
    return common.parse_args(argv, parser=build_parser(), geometry_builder=derive_geometry)


def create_config(args=None):
    return common.create_config(parse_args([]) if args is None else args,
                                geometry_builder=derive_geometry)


def main(argv=None):
    args = parse_args(argv)
    config = create_config(args)
    targets = common._snapshot_step_targets(config, args.end_time_s, args.output_interval_s)
    return common._run_case(args, config, targets,
                            scenario_label="hot-water dam break / four melting ice blocks",
                            final_plot_name="dambreak_four_ice_melting_2d.png",
                            stop_when_all_ice_melted=args.stop_when_all_ice_melted)


if __name__ == "__main__":
    main()
