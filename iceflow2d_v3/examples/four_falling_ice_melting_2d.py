"""Four 1 cm ice squares falling into a 15 cm by 5 cm hot-water bath.

Run ``python -m iceflow2d_v3.examples.four_falling_ice_melting_2d``.
All options and streaming outputs are shared with the single-ice example.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as common


def build_parser():
    parser = common.build_parser()
    parser.description = (
        "Four freely falling and melting ice blocks in a hot-water bath"
    )
    parser.set_defaults(
        domain_width_m=0.15,
        domain_height_m=0.05,
        reference_velocity_m_s=4.0,
        resolution_x=450,
        resolution_y=150,
        water_level_m=0.025,
        ice_size_m=0.01,
        ice_angle_degrees=0.0,
        drop_height_m=0.01,
        ice_centers_x_m=[0.03, 0.06, 0.09, 0.12],
        interface_width=5.0,
        phase_warmup_steps=2000,
        check_stability=True,
        output_dir=Path("iceflow2d_v3/outputs/four_falling_ice_melting_2d"),
    )
    return parser


def parse_args(argv=None):
    return common.parse_args(argv, parser=build_parser())


def create_config(args=None):
    return common.create_config(parse_args([]) if args is None else args)


def main(argv=None):
    args = parse_args(argv)
    config = create_config(args)
    targets = common._snapshot_step_targets(
        config, args.end_time_s, args.output_interval_s
    )
    return common._run_case(args, config, targets)


if __name__ == "__main__":
    main()
