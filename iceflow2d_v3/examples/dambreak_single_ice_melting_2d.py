"""A released 65 x 45 mm hot-water column hitting one free melting ice block.

Run ``python -m iceflow2d_v3.examples.dambreak_single_ice_melting_2d``.
Physical, numerical and visualization defaults match the four-ice case.
This case retains its dry floor and initially resting single ice block.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as dambreak


def build_parser():
    parser = dambreak.build_parser()
    parser.description = (
        "Hot-water dam break with one freely moving and melting ice block"
    )
    parser.set_defaults(
        ice_centers_x_m=parser.get_default("ice_centers_x_m")[:1],
        floor_water_depth_m=0.0,
        ice_floor_clearance_cells=0,
        scenario_label="hot-water dam break with one melting ice block",
        case_label="Dam-break single-ice melting",
        model_label="iceflow2d_v3: dam break with one disappearing ice melt",
        output_stem="dambreak_single_ice_melting_2d",
        output_dir=Path("iceflow2d_v3/outputs/dambreak_single_ice_melting_2d"),
    )
    return parser


def parse_args(argv=None):
    return dambreak.common.parse_args(
        argv, parser=build_parser(), geometry_builder=dambreak.derive_geometry
    )


def create_config(args=None):
    return dambreak.create_config(parse_args([]) if args is None else args)


def main(argv=None):
    return dambreak.run_case(parse_args(argv))


if __name__ == "__main__":
    main()
