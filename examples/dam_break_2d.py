from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mixture2d import Simulator2D, create_dambreak_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taichi CUDA 2D dam-break demo")
    parser.add_argument("--mode", choices=("fluid", "sand", "coupled"), default="fluid")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--resolution-x", type=int, default=600)
    parser.add_argument("--resolution-y", type=int, default=300)
    parser.add_argument("--particles-per-cell", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--save-npz", action="store_true")
    parser.add_argument("--water-retention", action="store_true")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_name = args.mode
    if args.mode == "coupled" and args.water_retention:
        default_name = "coupled_water_retention"
    output_dir = args.output_dir or f"outputs/dam_break_2d/{default_name}"
    cfg = create_dambreak_config(
        mode=args.mode,
        resolution=(args.resolution_x, args.resolution_y),
        particles_per_cell=args.particles_per_cell,
        output_dir=output_dir,
        show_gui=args.show_gui,
        save_npz=args.save_npz,
        water_retention=args.water_retention,
    )
    sim = Simulator2D(cfg)
    show_progress = sys.stderr.isatty() if args.progress is None else args.progress
    sim.run(
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        output_dir=output_dir,
        show_progress=show_progress,
    )
    print(f"Wrote {args.frames} frames to {output_dir}")


if __name__ == "__main__":
    main()
