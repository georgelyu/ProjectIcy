"""Replay a saved failure and retain the first rejected device step.

--diagnostic-migration permits an explicit old-state/new-code bug experiment;
such an experiment is never accepted as a production continuation or 10 s run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import taichi as ti
from iceflow2d_v2.checkpoint import Snapshot, fields
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation, SimulationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=3200)
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--diagnostic-migration', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = Snapshot.load(args.checkpoint)
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    sim = Simulation(Config(**saved.metadata['config']))
    added_fields = []
    if args.diagnostic_migration:
        # Explicit bug experiment only. Newly introduced phase scratch and
        # its zero roundoff ledger have no old state to import. Every existing
        # array is copied unchanged; production Snapshot.restore stays strict.
        diagnostic = Snapshot(sim)
        added_fields = sorted(diagnostic.arrays.keys()-saved.arrays.keys())
        assert not (saved.arrays.keys()-diagnostic.arrays.keys())
        assert all(name.startswith(('rigid.phase.', 'rigid.subgrid.')) or name == 'coupling.torque_extra'
                   for name in added_fields), added_fields
        assert all(np.count_nonzero(diagnostic.arrays[name]) == 0 for name in added_fields)
        diagnostic.arrays.update(saved.arrays)
        for key in ('steps', 'wall_seconds', 'initial', 'config'):
            diagnostic.metadata[key] = saved.metadata[key]
        diagnostic.restore(sim)
    else:
        saved.restore(sim)
    if args.strict: sim.transport.polish_override = sim.cfg.pressure_iterations
    sim.enable_graph()
    from iceflow2d_v2.cuda_schedule import CudaKernelSchedule
    sim.schedule = CudaKernelSchedule(sim)
    target = sim.steps + args.steps
    history, failed = [], False
    while sim.steps < target:
        before = Snapshot(sim)
        count = min(16, target-sim.steps)
        sim.step(count)
        d = sim.diagnostics()
        try:
            sim.validate(d)
        except SimulationError:
            before.restore(sim)
            for _ in range(count):
                before = Snapshot(sim)
                sim.step(1)
                d = sim.diagnostics()
                try: sim.validate(d)
                except SimulationError:
                    failed = True
                    before.save(args.output/'last_accepted_restart.npz')
                    Snapshot(sim).save(args.output/'first_failure_restart.npz')
                    break
        history.append(d)
        print(json.dumps({k:d[k] for k in ('step','time_s','mach_max','capacity_residual_max','geometry_courant','batch_failure_flags')}), flush=True)
        if failed: break
    q = sim.state.q.to_numpy().astype(float)-sim.state.comp.to_numpy()
    velocity, capacity = sim.flow.velocity.to_numpy(), sim.geometry.capacity.to_numpy()
    speed = np.linalg.norm(velocity, axis=-1)
    index = np.unravel_index(np.argmax(speed), speed.shape)
    owners = sim.merge.owner.to_numpy()
    members = np.argwhere(np.all(owners == owners[index], axis=-1))
    result = {'failed': failed, 'diagnostic_only': True,
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'original_source_sha256': saved.metadata['source_sha256'],
              'current_source_sha256': sim.source_sha256,
              'diagnostic_zero_initialized_new_fields': added_fields,
              'strict': args.strict, 'history': history, 'final': d,
              'peak': {'index': list(map(int, index)), 'velocity_lattice': velocity[index].tolist(),
                       'capacity': float(capacity[index]), 'state': q[index].tolist(),
                       'owner': owners[index].tolist(), 'members': members.tolist(),
                       'patch_volume': float(capacity[tuple(members.T)].sum())}}
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    sim.save(args.output/'final_state.npz')
    Snapshot(sim).save(args.output/'final_restart.npz')
    sim.schedule.close()


if __name__ == '__main__': main()
