"""Accelerated production test of a moving body's final melting stage."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import taichi as ti

from iceflow2d_v2.checkpoint import Snapshot
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation, SimulationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--end-time', type=float, default=.2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    cfg = Config(nx=25, ny=50, wall_cells=1, water_level=.04, drop_height=-.025,
                 gravity=0, sigma=0, buoyancy=False, k_w=6000,
                 projection_solver='multigrid', pressure_iterations=1200,
                 transport_iterations=120)
    sim = Simulation(cfg)
    sim.enable_graph(recovery=True)
    from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
    sim.schedule = CudaKernelSchedule(sim)
    failure, retries = None, 0
    count = round(args.end_time/cfg.dt)
    initial = sim.diagnostics()
    all_melted_at = None
    with (args.output/'diagnostics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=initial)
        writer.writeheader(); writer.writerow(initial); handle.flush()
        try:
            while sim.steps < count:
                d, retried = sim.advance_checked(min(80, count-sim.steps))
                retries += int(retried)
                writer.writerow(d); handle.flush()
                if d['mass_ice_kg_m'] <= 0 and all_melted_at is None:
                    all_melted_at = d['time_s']
                print(json.dumps({k: d[k] for k in ('time_s', 'mass_ice_kg_m', 'mach_max',
                    'capacity_residual_max', 'geometry_courant')}), flush=True)
        except SimulationError as error:
            failure = str(error)
            if sim.last_failure_snapshot:
                sim.last_failure_snapshot.save(args.output/'rejected_restart.npz')
    d = sim.diagnostics()
    body_momentum = (np.asarray(sim.rigid.momentum[None], dtype=float)
                     -np.asarray(sim.rigid.momentum_comp[None], dtype=float))
    scale = np.array([cfg.mass_unit*cfg.dx/cfg.dt]*2+[cfg.mass_unit*cfg.dx**2/cfg.dt])
    result = {'config': cfg.metadata(), 'source_sha256': sim.source_sha256,
              'completed_requested_steps': sim.steps == count and failure is None,
              'failure': failure, 'rejected_diagnostics': sim.last_failure_diagnostics,
              'initial': initial, 'final': d, 'capacity_solver_retries': retries,
              'first_nonpositive_ice_mass_sample_s': all_melted_at,
              'remaining_geometry_mass_kg_m': float(sim.geometry.mass[None])*cfg.mass_unit,
              'remaining_body_momentum_SI': (body_momentum*scale).tolist(),
              'material_polygon_count': int(np.sum(sim.geometry.count.to_numpy())),
              'scope': 'Accelerated k_water=6000 W/(m K), zero gravity/surface tension, free body. Not the original full scene.'}
    Snapshot(sim).save(args.output/'restart.npz')
    sim.save(args.output/'final_state.npz')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    sim.schedule.close()
    print(json.dumps(result, indent=2))
    if failure:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
