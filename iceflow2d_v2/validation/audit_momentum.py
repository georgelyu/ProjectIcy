"""Read-only operator momentum audit of the production coupled step."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation
from iceflow2d_v2.checkpoint import Snapshot


def total(sim):
    q = sim.state.q.to_numpy().astype(float)-sim.state.comp.to_numpy()
    body = np.asarray(sim.rigid.momentum[None])-np.asarray(sim.rigid.momentum_comp[None])
    x, y = np.indices(q.shape[:2])
    return np.array([q[..., 4].sum(), q[..., 5].sum(),
                     ((x+.5)*q[..., 5]-(y+.5)*q[..., 4]).sum()])+body


def external(sim):
    pairs = ((sim.budgets.external, sim.budgets.external_comp),
             (sim.rigid.contact_impulse, sim.rigid.contact_impulse_comp),
             (sim.rigid.support_impulse, sim.rigid.support_impulse_comp))
    return sum((np.asarray(value[None], float)-np.asarray(comp[None], float)
                for value, comp in pairs), np.zeros(3))


def prepared_source(sim):
    source = sim.state.source.to_numpy().astype(float)
    x, y = np.indices(source.shape[:2])
    return np.array([source[..., 4].sum(), source[..., 5].sum(),
                     ((x+.5)*source[..., 5]-(y+.5)*source[..., 4]).sum()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', type=int, default=100)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=.072)
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    restart = Snapshot.load(args.restart) if args.restart else None
    cfg = (Config(**restart.metadata['config']) if restart else
           Config(nx=args.nx, ny=2*args.nx, wall_cells=max(1, round(3*args.nx/100)), sigma=args.sigma))
    sim = Simulation(cfg)
    if restart:
        restart.restore(sim)
    start_diagnostics = sim.diagnostics()
    start_total, start_external = total(sim), external(sim)
    scales = np.array([cfg.mass_unit*cfg.dx/cfg.dt]*2+[cfg.mass_unit*cfg.dx**2/cfg.dt])
    records = []
    # These are disjoint top-level operators in the current production step.
    # PhaseMomentum.finish changes the body; Coupling.finish prepares the
    # paired fluid source, which Transport.remap subsequently commits. Measure
    # that pending source separately to distinguish exchange from remapping.
    for obj, method in [(sim.rigid, 'commit_force'), (sim.budgets, 'accumulate'),
                        (sim.momentum, 'step'), (sim.transport, 'phase_step'),
                        (sim.coupling, 'prepare'), (sim.rigid.phase, 'finish'),
                        (sim.coupling, 'finish'), (sim.budgets, 'accumulate_numerical_air'),
                        (sim.rigid, 'contact'),
                        (sim.transport, 'remap'), (sim.merge, 'step')]:
        original = getattr(obj, method)
        def traced(*a, _original=original, _name=type(obj).__name__+'.'+method, **kw):
            before, force_before = total(sim), external(sim)
            result = _original(*a, **kw)
            delta = (total(sim)-before)*scales
            force_delta = (external(sim)-force_before)*scales
            entry = {'operator': _name, 'delta': delta.tolist(),
                     'external_impulse_delta': force_delta.tolist(),
                     'residual_delta': (delta-force_delta).tolist()}
            if _name == 'Coupling.finish':
                entry['prepared_source_momentum'] = (prepared_source(sim)*scales).tolist()
            records.append(entry)
            return result
        setattr(obj, method, traced)
    sim.step(args.steps)
    measured_delta = (total(sim)-start_total)*scales
    expected_delta = (external(sim)-start_external)*scales
    traced_delta = np.sum([item['delta'] for item in records], axis=0)
    traced_external = np.sum([item['external_impulse_delta'] for item in records], axis=0)
    groups = {
        'flow_and_forces': {'Rigid.commit_force', 'Budgets.accumulate', 'Momentum.step'},
        'phase_exchange_and_air_source': {'Coupling.prepare', 'PhaseMomentum.finish',
                                          'Coupling.finish', 'Budgets.accumulate_numerical_air'},
        'phase_transport': {'Transport.phase_step'},
        'capacity_remap': {'Transport.remap'},
        'cut_cell_merge': {'PatchMerge.step', 'PairMerge.step'},
        'contact': {'Rigid.contact'},
    }
    grouped = {name: np.sum([item['residual_delta'] for item in records
                            if item['operator'] in operators], axis=0).tolist()
               for name, operators in groups.items()}
    assert {item['operator'] for item in records} <= set().union(*groups.values())
    pending = np.sum([item['prepared_source_momentum'] for item in records
                      if item['operator'] == 'Coupling.finish'], axis=0)
    source_split = {
        **{key: value for key, value in grouped.items()
           if key not in ('phase_exchange_and_air_source', 'capacity_remap')},
        'prepared_phase_source_closure': (np.asarray(grouped['phase_exchange_and_air_source'])+pending).tolist(),
        'geometric_transport_excluding_sources': (np.asarray(grouped['capacity_remap'])-pending).tolist(),
    }
    g, f = sim.geometry, sim.flow
    reference = f.boundary_reference_impulse.to_numpy().astype(float)
    origin = np.array(g.origin[None])
    reference[..., 2] += origin[0]*reference[..., 1]-origin[1]*reference[..., 0]
    reference += f.wall_reference_impulse.to_numpy()
    mass = f.reference_density.to_numpy()*g.old_capacity.to_numpy()
    reference[..., 1] += mass*cfg.gravity_lu
    x, y = np.indices(mass.shape)
    reference[..., 2] += (x+.5)*mass*cfg.gravity_lu
    result = {'operators': records, 'reference_unbalance_last_step': (reference.sum((0, 1))*scales).tolist(),
              'total_momentum_change': measured_delta.tolist(),
              'external_impulse_change': expected_delta.tolist(),
              'balance_residual_change': (measured_delta-expected_delta).tolist(),
              'traced_momentum_closure_error': (measured_delta-traced_delta).tolist(),
              'traced_external_closure_error': (expected_delta-traced_external).tolist(),
              'grouped_residual_changes': grouped,
              'source_split_residual_changes': source_split,
              'source_split_scope': 'The pending fluid source is added to phase preparation and subtracted from the remap that commits it. This is diagnostic accounting only; no state is changed.',
              'initial_diagnostics': start_diagnostics, 'diagnostics': sim.diagnostics(),
              'source_sha256': sim.source_sha256,
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'restart_sha256': hashlib.sha256(args.restart.read_bytes()).hexdigest() if args.restart else None,
              'scope': 'Operator audit from a strict same-source checkpoint; diagnostic only, not a production continuation.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
