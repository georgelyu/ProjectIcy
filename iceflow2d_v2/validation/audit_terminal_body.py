"""Trace body angular momentum through an accepted production checkpoint.

Read-only diagnostic evolution; the strict production restart check is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import taichi as ti
from iceflow2d_v2.checkpoint import Snapshot
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation


def body(sim):
    g, r = sim.geometry, sim.rigid
    mass, inertia = float(g.mass[None]), float(g.inertia[None])
    origin = np.asarray(g.origin[None], float)
    first = np.asarray(g.first_moment[None], float)
    center = origin+first/mass if mass > 0 else origin
    p = np.asarray(r.momentum[None], float)-np.asarray(r.momentum_comp[None], float)
    spin = p[2]-(center[0]*p[1]-center[1]*p[0])
    return {'mass_lu': mass, 'inertia_lu': inertia, 'center_lu': center.tolist(),
            'world_momentum_lu': p.tolist(), 'spin_momentum_lu': float(spin),
            'omega_lu': float(spin/inertia) if inertia > 0 else 0.0,
            'stored_velocity_lu': np.asarray(g.velocity[None]).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    saved = Snapshot.load(args.checkpoint)
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    sim = Simulation(Config(**saved.metadata['config']))
    saved.restore(sim)
    records = []
    for obj, name in ((sim.flow, 'correct_boundary_pressure'),
                      (sim.rigid, 'commit_force'), (sim.rigid.phase, 'finish'),
                      (sim.rigid, 'advance_pose'), (sim.geometry, 'update_mass'),
                      (sim.rigid, 'contact')):
        original = getattr(obj, name)
        def traced(*a, _method=original, _name=type(obj).__name__+'.'+name, **kw):
            before = body(sim)
            if _name == 'Flow.correct_boundary_pressure':
                forces = sim.flow.boundary_impulse.to_numpy().astype(float)
            result = _method(*a, **kw)
            record = {'operator': _name, 'before': before, 'after': body(sim)}
            if _name == 'Flow.correct_boundary_pressure':
                delta = sim.flow.boundary_impulse.to_numpy().astype(float)-forces
                correction = delta.sum((0, 1))
                raw = forces.sum((0, 1))
                arm = np.asarray(sim.geometry.first_moment[None])/float(sim.geometry.mass[None])
                record['raw_boundary_force_lu'] = raw.tolist()
                record['pressure_correction_force_lu'] = correction.tolist()
                record['raw_torque_about_com_lu'] = float(raw[2]-arm[0]*raw[1]+arm[1]*raw[0])
                record['pressure_torque_about_com_lu'] = float(correction[2]-arm[0]*correction[1]+arm[1]*correction[0])
                record['drag_trace'] = float(np.trace(sim.flow.body_drag[None]))
            records.append(record)
            return result
        setattr(obj, name, traced)
    initial = sim.diagnostics()
    sim.step(args.steps)
    result = {'diagnostic_only': True, 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'source_sha256': sim.source_sha256, 'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'initial': initial, 'final': sim.diagnostics(), 'operators': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'steps': args.steps, 'body': body(sim)}), flush=True)


if __name__ == '__main__':
    main()
