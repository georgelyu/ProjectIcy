"""Run via python -m iceflow2d_v2.examples.coupled_falling_ice_melting_2d."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation, SimulationError
from iceflow2d_v2.checkpoint import Snapshot
from iceflow2d_v2.examples._fields import FieldImageWriter, output_steps
from iceflow2d_v2.examples._progress import TerminalProgress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--end-time', type=float, default=10.0)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--sample-interval', type=float, default=0.01)
    parser.add_argument('--output-interval', '--output-interval-s', type=float, default=0.01,
                        help='Physical-time interval between temperature/velocity PNG frames (seconds).')
    parser.add_argument('--velocity-visualization-max-m-s', type=float, default=.1)
    parser.add_argument('--output', type=Path, default=Path('iceflow2d_v2/outputs/coupled'))
    parser.add_argument('--fixed-body', action='store_true')
    parser.add_argument('--no-melting', action='store_true')
    parser.add_argument('--no-advection', action='store_true')
    parser.add_argument('--reference-velocity', type=float, default=4.0)
    parser.add_argument('--pressure-iterations', type=int, default=240)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=True,
                        help='Show the LBM-step progress bar (default); --no-progress prints JSON diagnostics.')
    parser.add_argument('--graph', action='store_true')
    parser.add_argument('--native-graph', action='store_true', help='Optional CUDA 12.8/CUPTI launch adapter; numerical kernels stay Taichi.')
    parser.add_argument('--projection-solver', choices=['cg', 'multigrid'], default='cg')
    parser.add_argument('--thermal-block', type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument('--pressure-polish-iterations', type=int, default=24)
    parser.add_argument('--check-steps', type=int, default=1600,
                        help='Host validation interval; device guards still run every step.')
    parser.add_argument('--resume', type=Path,
                        help='Read an existing complete restart NPZ with matching configuration and sources; no NPZ is written.')
    args = parser.parse_args(argv)
    if args.native_graph and (args.arch != 'cuda' or args.profile):
        parser.error('--native-graph requires CUDA without --profile')
    config = Config(nx=args.nx, ny=2*args.nx, wall_cells=max(1, round(3*args.nx/100)),
                    fixed_body=args.fixed_body, melting=not args.no_melting, advection=not args.no_advection,
                    reference_velocity=args.reference_velocity, pressure_iterations=args.pressure_iterations,
                    projection_solver=args.projection_solver, coupling_steps=args.thermal_block,
                    pressure_polish_iterations=args.pressure_polish_iterations)
    if not math.isfinite(args.end_time) or args.end_time < 0:
        parser.error('end-time must be finite and nonnegative')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.sample_interval, args.check_steps, args.output_interval, args.velocity_visualization_max_m_s)):
        parser.error('sampling/output intervals, check steps and velocity display maximum must be finite and positive')
    count = args.steps if args.steps is not None else round(args.end_time/config.dt)
    if count < 0:
        parser.error('steps must be nonnegative')
    sample = max(1, round(args.sample_interval/config.dt))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.progress:
        print('Initializing simulation (Taichi compilation may take a while)...', file=sys.stderr, flush=True)
    ti.init(arch=ti.cuda if args.arch == 'cuda' else ti.cpu, default_fp=ti.f32,
            fast_math=False, kernel_profiler=args.profile)
    simulation = Simulation(config)
    if args.resume: Snapshot.load(args.resume).restore(simulation)
    if simulation.steps > count:
        parser.error('requested end step precedes the resumed state')
    if args.graph or args.native_graph:
        if args.progress:
            print('Compiling simulation graphs...', file=sys.stderr, flush=True)
        simulation.enable_graph(recovery=True)
    if args.native_graph:
        if args.progress:
            print('Building native CUDA graph...', file=sys.stderr, flush=True)
        from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
        simulation.schedule = CudaKernelSchedule(simulation)
    metadata = {**config.metadata(), 'requested_steps': count, 'requested_time_s': count*config.dt,
                'sample_interval_s': args.sample_interval, 'output_interval_s': args.output_interval,
                'field_outputs': {'temperature': 'temperature_frames/frame_*.png',
                                  'velocity': 'velocity_frames/frame_*.png', 'times': 'frames.csv',
                                  'velocity_color_max_m_s': args.velocity_visualization_max_m_s},
                'status': 'experimental; A-H validation incomplete', 'command': sys.argv,
                'execution': simulation.schedule.metadata if args.native_graph else 'taichi_graph' if args.graph else 'python'}
    root = Path(__file__).resolve().parents[1]
    metadata['source_sha256'] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted([*root.glob('*.py'), *root.glob('*.cpp'), *root.joinpath('examples').glob('*.py')])}
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    failure = None
    retries = 0
    with ((args.output/'diagnostics.csv').open('w', newline='', encoding='utf-8') as handle,
          FieldImageWriter(args.output, config, args.velocity_visualization_max_m_s) as images):
        diagnostics = simulation.diagnostics()
        writer = csv.DictWriter(handle, fieldnames=diagnostics)
        writer.writeheader(); writer.writerow(diagnostics); handle.flush()
        # Preserve the accepted starting state, including release at t=0.
        try:
            with TerminalProgress(count, dt=config.dt, initial=simulation.steps, enabled=args.progress) as progress:
                for target in output_steps(simulation.steps, count, config.dt, args.output_interval):
                    while simulation.steps < target:
                        chunk = min(args.check_steps, sample, target-simulation.steps)
                        diagnostics, retried = simulation.advance_checked(chunk)
                        retries += int(retried)
                        writer.writerow(diagnostics); handle.flush()
                        progress.update(simulation.steps)
                        if not args.progress:
                            print(json.dumps({k: diagnostics[k] for k in ('step', 'time_s', 'condensed_mass_relative_residual',
                                  'condensed_energy_relative_residual', 'capacity_residual_max', 'mach_max')}, ensure_ascii=False), flush=True)
                    images.append(simulation)
        except SimulationError as error:
            failure = str(error)
            diagnostics = simulation.diagnostics()
            writer.writerow(diagnostics); handle.flush()
        # A failed batch has already rolled back; preserve its last accepted state.
        images.append(simulation)
    result = {'completed_requested_steps': simulation.steps == count and failure is None,
              'validated_full_simulation': False, 'failure': failure, 'diagnostics': diagnostics,
              'rejected_diagnostics': simulation.last_failure_diagnostics,
              'capacity_solver_retries': retries, 'image_frames': images.frame_count}
    # Failed runs can contain NaN; JSON null preserves a readable failure report.
    result['diagnostics'] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                             for k, v in diagnostics.items()}
    if result['rejected_diagnostics']:
        result['rejected_diagnostics'] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                                        for k, v in result['rejected_diagnostics'].items()}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    if args.profile: ti.profiler.print_kernel_profiler_info()
    if args.native_graph: simulation.schedule.close()
    if failure:
        print(f'Validation failed: {failure}. Final images show the last accepted state; rejected diagnostics are in result.json. Full acceptance remains false.', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
