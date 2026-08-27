from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INTERFACE_WIDTH = 5.0
DEFAULT_INTERFACE_CUTOFF = 1.0e-3


def derive_ice_fall_geometry(args: argparse.Namespace) -> dict[str, float | int]:
    """Derive a full-width pool and an OBB whose lowest point starts in air."""

    nx = int(args.resolution_x)
    ny = int(args.resolution_y)
    boundary = int(args.boundary_cells)
    if nx <= 2 * boundary + 4 or ny <= 2 * boundary + 4:
        raise ValueError("resolution is too small for the requested boundary thickness")

    fractions = {
        "water level": float(args.water_level_fraction),
        "ice width": float(args.ice_width_fraction),
        "ice height": float(args.ice_height_fraction),
    }
    for name, value in fractions.items():
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} fraction must be finite and in (0, 1]")

    water_height = int(ny * fractions["water level"])
    if water_height < boundary + 2:
        raise ValueError("water level must leave at least two active pool rows")
    if water_height > ny - boundary:
        raise ValueError("water level must leave room below the top wall")

    ice_width = int(nx * fractions["ice width"])
    ice_height = int(ny * fractions["ice height"])
    if ice_width < 2 or ice_height < 2:
        raise ValueError("ice fractions must discretize to at least 2 x 2 cells")
    angle_degrees = float(args.ice_angle_degrees)
    if not math.isfinite(angle_degrees):
        raise ValueError("ice angle must be finite")
    angle = math.radians(angle_degrees)

    if args.drop_height_cells is not None:
        drop_height = float(args.drop_height_cells)
    elif args.drop_height_fraction is not None:
        drop_height = float(args.drop_height_fraction) * ny
    else:
        drop_height = 0.20 * ny
    if not math.isfinite(drop_height) or drop_height < 0.0:
        raise ValueError("drop height must be finite and non-negative")
    interface_clearance = max(
        2,
        int(
            math.ceil(
                0.25
                * DEFAULT_INTERFACE_WIDTH
                * math.log((1.0 - DEFAULT_INTERFACE_CUTOFF) / DEFAULT_INTERFACE_CUTOFF)
            )
        ),
    )
    if interface_clearance % 2:
        interface_clearance += 1
    if drop_height < interface_clearance:
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
    lowest_y = float(water_height) + drop_height
    center_y = lowest_y + extent_y
    # IceFlowConfig defines center_y as ice_base_y_cells + ice_height/2.
    ice_base_y_cells = center_y - half_height

    if center_x - extent_x < boundary or center_x + extent_x > nx - boundary:
        raise ValueError("rotated ice block overlaps a side wall")
    if ice_base_y_cells < boundary:
        raise ValueError("derived ice base overlaps the bottom wall")
    if center_y + extent_y > ny - boundary:
        available = ny - boundary - water_height - 2.0 * extent_y
        raise ValueError(
            "ice block does not fit below the top wall; "
            f"maximum drop height for this geometry is {max(0.0, available):.3f} cells"
        )

    initial_velocity_x = float(args.initial_horizontal_speed)
    initial_velocity_y = float(args.initial_vertical_speed)
    if not math.isfinite(initial_velocity_x) or not math.isfinite(initial_velocity_y):
        raise ValueError("initial ice velocity must be finite")
    if math.hypot(initial_velocity_x, initial_velocity_y) > 0.08:
        raise ValueError("initial ice speed must not exceed 0.08 lattice cells/step")

    # IceFlowConfig computes water_width=int(nx*fraction).  Choosing the
    # midpoint of the desired integer bin fills every active column while
    # leaving the right wall untouched, without relying on float roundoff.
    water_width = nx - boundary
    water_width_fraction = (water_width + 0.5) / nx
    return {
        "water_width": water_width,
        "water_height": water_height,
        "water_width_fraction": water_width_fraction,
        "ice_width": ice_width,
        "ice_height": ice_height,
        "ice_angle": angle,
        "ice_base_y_cells": ice_base_y_cells,
        "ice_center_x": center_x,
        "ice_center_y": center_y,
        "ice_lowest_y": lowest_y,
        "ice_highest_y": center_y + extent_y,
        "drop_height": drop_height,
        "interface_clearance": interface_clearance,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Taichi CUDA 2D rigid ice block falling through air into a water pool"
    )
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--resolution-x", type=int, default=300)
    parser.add_argument("--resolution-y", type=int, default=600)
    parser.add_argument(
        "--reference-length-cells",
        type=int,
        default=None,
        help="physical reference length in cells (defaults to resolution-y)",
    )
    parser.add_argument(
        "--water-level-fraction",
        type=float,
        default=0.7,
        help="water-surface height divided by resolution-y (default: 0.70)",
    )
    parser.add_argument("--ice-width-fraction", type=float, default=0.10)
    parser.add_argument("--ice-height-fraction", type=float, default=0.05)
    parser.add_argument(
        "--ice-angle-degrees",
        type=float,
        default=5.0,
        help="initial counter-clockwise tilt in degrees",
    )
    drop_group = parser.add_mutually_exclusive_group()
    drop_group.add_argument(
        "--drop-height-cells",
        type=float,
        default=None,
        help="vertical air gap between the water surface and the lowest ice corner",
    )
    drop_group.add_argument(
        "--drop-height-fraction",
        type=float,
        default=0.03,
        help="air gap divided by resolution-y (default: 0.03)",
    )
    parser.add_argument("--initial-horizontal-speed", type=float, default=0.0)
    parser.add_argument("--initial-vertical-speed", type=float, default=0.0)
    parser.add_argument("--boundary-cells", type=int, default=3)
    parser.add_argument("--phase-warmup-steps", type=int, default=500)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--save-npz", action="store_true")
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        help="show frame progress even when stderr is not an interactive terminal",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="disable the frame progress bar",
    )
    parser.set_defaults(progress=None)

    args = parser.parse_args(argv)
    if args.frames < 0:
        parser.error("--frames must be non-negative")
    if args.steps_per_frame <= 0:
        parser.error("--steps-per-frame must be positive")
    if args.resolution_x <= 0 or args.resolution_y <= 0:
        parser.error("resolution components must be positive")
    if args.reference_length_cells is not None and args.reference_length_cells <= 0:
        parser.error("--reference-length-cells must be positive")
    if isinstance(args.boundary_cells, bool) or args.boundary_cells < 1:
        parser.error("--boundary-cells must be a positive integer")
    if args.phase_warmup_steps < 0:
        parser.error("--phase-warmup-steps must be non-negative")
    try:
        derive_ice_fall_geometry(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def create_ice_fall_config(args: argparse.Namespace, *, output_dir: str):
    """Create the regular IceFlowConfig for the derived falling-ice layout."""

    from iceflow2d import create_iceflow_config

    geometry = derive_ice_fall_geometry(args)
    config = create_iceflow_config(
        resolution=(args.resolution_x, args.resolution_y),
        reference_length_cells=(
            args.reference_length_cells
            if args.reference_length_cells is not None
            else args.resolution_y
        ),
        water_width_fraction=geometry["water_width_fraction"],
        water_height_fraction=args.water_level_fraction,
        ice_width_fraction=args.ice_width_fraction,
        ice_height_fraction=args.ice_height_fraction,
        ice_base_y_cells=geometry["ice_base_y_cells"],
        ice_initial_velocity=(
            args.initial_horizontal_speed,
            args.initial_vertical_speed,
        ),
        ice_initial_angle=geometry["ice_angle"],
        boundary_cells=args.boundary_cells,
        phase_warmup_steps=args.phase_warmup_steps,
        well_balanced_hydrostatics=True,
        output_dir=output_dir,
        show_gui=args.show_gui,
        save_npz=args.save_npz,
    )
    if (
        config.water_width != geometry["water_width"]
        or config.water_height != geometry["water_height"]
    ):
        raise RuntimeError(
            "derived pool dimensions disagree with IceFlowConfig discretization"
        )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or "outputs/iceflow2d/ice_fall"
    config = create_ice_fall_config(args, output_dir=output_dir)

    from iceflow2d import IceFlow2D

    simulation = IceFlow2D(config)
    show_progress = sys.stderr.isatty() if args.progress is None else args.progress
    simulation.run(
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        output_dir=Path(output_dir),
        show_progress=show_progress,
    )
    print(f"Wrote {args.frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
