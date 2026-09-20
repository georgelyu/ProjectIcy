"""Profile an unchanged frozen solver from a complete checkpoint."""
import argparse
import json
from pathlib import Path
import time

import taichi as ti

from iceflow2d_v2.checkpoint import Snapshot
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--diagnostic-migration', action='store_true',
                        help='Explicit source-change experiment only, never a production continuation.')
    parser.add_argument('--save-states', type=Path)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False, kernel_profiler=True)
    snapshots = [Snapshot.load(path) for path in args.checkpoints]
    sim = Simulation(Config(**snapshots[0].metadata['config']))
    sim.enable_graph()
    for path, snapshot in zip(args.checkpoints, snapshots):
        snapshot.restore(sim, check_source=not args.diagnostic_migration)
        sim.step(1)
        ti.profiler.clear_kernel_profiler_info()
        start = time.perf_counter()
        sim.step(args.steps)
        seconds = time.perf_counter()-start
        print(json.dumps({'checkpoint': str(path.resolve()), 'start_step': snapshot.metadata['steps'],
                          'profiled_steps': args.steps, 'wall_seconds': seconds,
                          'diagnostics': sim.diagnostics()}), flush=True)
        ti.profiler.print_kernel_profiler_info()
        if args.save_states:
            args.save_states.mkdir(parents=True, exist_ok=True)
            Snapshot(sim).save(args.save_states/f"step_{snapshot.metadata['steps']}_restart.npz")


if __name__ == '__main__':
    main()
