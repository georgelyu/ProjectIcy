"""Production free-fall, uniform-fluid buoyancy and static-interface checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation
from iceflow2d_v2.checkpoint import Snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['freefall', 'buoyancy', 'float-equilibrium', 'static-interface'], required=True)
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--end-time', type=float, default=.01)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-graph', action='store_true')
    parser.add_argument('--check-steps', type=int, default=160)
    args = parser.parse_args()
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    changes = {
        'freefall': dict(water_level=-.1, drop_height=.132, sigma=0, angle_deg=0),
        'buoyancy': dict(water_level=.05, drop_height=-.02, sigma=0, angle_deg=0),
        'float-equilibrium': dict(water_level=.03, drop_height=-(917/1000)*.008, sigma=0, angle_deg=0),
        'static-interface': dict(water_level=.025, drop_height=.1, gravity=0, fixed_body=True),
    }[args.case]
    cfg = Config(nx=args.nx, ny=2*args.nx, wall_cells=max(1, round(3*args.nx/100)),
                 melting=False, projection_solver='multigrid', pressure_iterations=1200, **changes)
    sim = Simulation(cfg)
    origin = np.array(sim.geometry.origin[None])*cfg.dx
    phase0 = sim.flow.phi.to_numpy().astype(float)
    result = {'case': args.case, 'config': cfg.metadata(), 'source_sha256': sim.source_sha256}
    if args.case in ('buoyancy', 'float-equilibrium'):
        sim.flow.collide(); sim.flow.prepare_boundary(); sim.flow.reduce_boundary()
        measured = np.array(sim.flow.body_force[None])[:2]*cfg.mass_unit*cfg.dx/cfg.dt**2
        volume = sim.geometry.mass[None]/cfg.rho_s_lu*cfg.dx**2
        expected = cfg.rho_w*cfg.gravity*volume
        if args.case == 'float-equilibrium':
            expected = cfg.gravity*cfg.ice_size**2*cfg.rho_s
        result['initial_buoyancy_force_N_m'] = measured.tolist()
        result['archimedes_force_N_m'] = expected
        result['relative_buoyancy_error'] = float((measured[1]-expected)/expected)
    sim.enable_graph(recovery=True)
    if args.native_graph:
        from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
        sim.schedule = CudaKernelSchedule(sim)
    count = round(args.end_time/cfg.dt)
    history = []
    retries = 0
    args.output.mkdir(parents=True, exist_ok=True)
    while sim.steps < count:
        try:
            d, retried = sim.advance_checked(min(args.check_steps, count-sim.steps))
            retries += int(retried)
        except Exception:
            sim.save(args.output/'failure_state.npz')
            Snapshot(sim).save(args.output/'last_accepted_restart.npz')
            (sim.last_failure_snapshot or Snapshot(sim)).save(args.output/'failure_restart.npz')
            (args.output/'failure.json').write_text(json.dumps(sim.last_failure_diagnostics or sim.diagnostics(), indent=2))
            raise
        history.append(d)
        print(json.dumps({k:d[k] for k in ('step', 'time_s', 'mach_max', 'body_origin_y_m', 'capacity_residual_max')}), flush=True)
    result['history'] = history
    result['final'] = sim.diagnostics()
    result['capacity_solver_retries'] = retries
    result['execution'] = sim.schedule.metadata if args.native_graph else 'taichi_graph'
    result['maximum_vertical_displacement_m'] = max(abs(d['body_origin_y_m']-origin[1]) for d in history)
    result['maximum_angle_change_rad'] = max(abs(d['body_angle_rad']-np.radians(cfg.angle_deg)) for d in history)
    if args.case == 'freefall':
        t = sim.steps*cfg.dt
        expected = .5*cfg.gravity*(1-cfg.rho_a/cfg.rho_s)*t*t
        measured = origin[1]-sim.geometry.origin[None][1]*cfg.dx
        result['fall_distance_m'] = measured
        result['zero_drag_fall_distance_m'] = expected
        result['relative_freefall_error_including_gas_drag'] = (measured-expected)/expected
    if args.case == 'static-interface':
        result['phase_max_change'] = float(np.max(np.abs(sim.flow.phi.to_numpy()-phase0)))
        result['max_speed_m_s'] = float(np.max(np.linalg.norm(sim.flow.velocity.to_numpy(), axis=-1))*cfg.dx/cfg.dt)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    sim.save(args.output/'final_state.npz')
    if args.native_graph: sim.schedule.close()
    print(json.dumps({k:v for k,v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__': main()
