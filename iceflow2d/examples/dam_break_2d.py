from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iceflow2d import IceFlow2D, create_iceflow_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi CUDA 2D rigid-ice/two-phase dam-break demo")
    parser.add_argument("--mode", choices=("coupled",), default="coupled")
    parser.add_argument("--material", choices=("ice",), default="ice")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--resolution-x", type=int, default=600)
    parser.add_argument("--resolution-y", type=int, default=300)
    parser.add_argument(
        "--reference-length-cells",
        type=int,
        default=None,
        help="physical reference length in cells (defaults to resolution-y)",
    )
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
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or "outputs/iceflow2d/coupled_ice"
    config = create_iceflow_config(
        mode=args.mode,
        material=args.material,
        resolution=(args.resolution_x, args.resolution_y),
        reference_length_cells=(
            args.reference_length_cells if args.reference_length_cells is not None else args.resolution_y
        ),
        output_dir=output_dir,
        show_gui=args.show_gui,
        save_npz=args.save_npz,
    )
    simulation = IceFlow2D(config)
    show_progress = sys.stderr.isatty() if args.progress is None else args.progress
    simulation.run(
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        output_dir=output_dir,
        show_progress=show_progress,
    )
    print(f"Wrote {args.frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
