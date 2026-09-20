"""Complete restart snapshots, including distributions and compensated ledgers.

Snapshots are taken only at a synchronized sampling boundary. The smaller
visualization NPZ is deliberately not accepted as a restart checkpoint.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import taichi as ti


def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*root.glob('*.py'), *root.glob('*.cpp')])}


def fields(simulation):
    """Stable names for every distinct field; scratch state is included."""
    result, seen = {}, set()
    def visit(obj, path):
        if id(obj) in seen: return
        seen.add(id(obj))
        if isinstance(obj, ti.Field):
            result[path] = obj
        elif isinstance(obj, (list, tuple)):
            for i, child in enumerate(obj): visit(child, f'{path}.{i}')
        elif type(obj).__module__.startswith('iceflow2d_v2') and hasattr(obj, '__dict__'):
            for name, child in sorted(vars(obj).items()):
                if name not in ('cfg', 'geo', 'state', 'flow', 'transport', 'rigid', 'coupling'):
                    visit(child, f'{path}.{name}')
    for name in ('geometry', 'flow', 'state', 'thermal', 'coupling', 'transport',
                 'momentum', 'merge', 'rigid', 'budgets', 'health'):
        visit(getattr(simulation, name), name)
    return result


class Snapshot:
    def __init__(self, simulation):
        ti.sync()
        self.arrays = {name: field.to_numpy() for name, field in fields(simulation).items()}
        self.metadata = {'format': 2, 'config': asdict(simulation.cfg), 'steps': simulation.steps,
                         'wall_seconds': simulation.wall_seconds, 'initial': simulation.initial,
                         'source_sha256': simulation.source_sha256}

    def restore(self, simulation, check_source=True):
        if self.metadata['format'] != 2 or self.metadata['config'] != asdict(simulation.cfg):
            raise ValueError('Restart format/configuration does not match this simulation.')
        if check_source and self.metadata['source_sha256'] != simulation.source_sha256:
            raise ValueError('Restart source hashes differ; numerical continuation requires the same implementation.')
        destinations = fields(simulation)
        if self.arrays.keys() != destinations.keys():
            raise ValueError('Restart does not contain the complete field inventory.')
        for name, field in destinations.items():
            target = field.to_numpy()
            if target.shape != self.arrays[name].shape or target.dtype != self.arrays[name].dtype:
                raise ValueError(f'Restart layout mismatch: {name}')
        for name, field in destinations.items(): field.from_numpy(self.arrays[name])
        simulation.steps = self.metadata['steps']
        simulation.wall_seconds = self.metadata['wall_seconds']
        simulation.initial = self.metadata['initial']

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, metadata=json.dumps(self.metadata), **self.arrays)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        result = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as archive:
            if 'metadata' not in archive: raise ValueError('A visualization snapshot is not a complete restart.')
            result.metadata = json.loads(str(archive['metadata']))
            result.arrays = {key: archive[key] for key in archive.files if key != 'metadata'}
        return result
