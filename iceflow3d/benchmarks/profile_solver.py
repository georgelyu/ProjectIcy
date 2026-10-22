"""Profile the real coupled step after compilation, without changing its physics."""

import argparse
import cProfile
from pathlib import Path
import pstats
import sys
import time

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iceflow3d import IceFlow3D  # noqa: E402
from iceflow3d.examples.dambreak_four_ice_melting_3d import (  # noqa: E402
    create_config,
    parse_args,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-x", type=int, default=150)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, enable_fallback=False, kernel_profiler=True)
    sim = IceFlow3D(
        create_config(
            parse_args(
                [
                    "--resolution-x",
                    str(args.resolution_x),
                    "--phase-warmup-steps",
                    "0",
                ]
            )
        )
    )
    sim.step(16)
    ti.sync()
    ti.profiler.clear_kernel_profiler_info()
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    sim.step(args.steps)
    ti.sync()
    profiler.disable()
    print(f"Seconds per coupled step: {(time.perf_counter() - start) / args.steps:.6f}")
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(25)
    ti.profiler.print_kernel_profiler_info("count")


if __name__ == "__main__":
    main()
