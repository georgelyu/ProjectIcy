"""Compare optimized contact against a specified earlier frozen v2 kernel.

The reference is the independent v2 implementation, never legacy iceflow2d.
Both kernels see exactly the same geometry, momentum, and side order.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import taichi as ti

from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.rigid import Rigid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.reference.parent.name != 'iceflow2d_v2' or args.reference.name != 'rigid.py':
        raise ValueError('Only a frozen independent v2 rigid.py is a permitted reference.')
    name = 'iceflow2d_v2._frozen_contact_reference'
    spec = importlib.util.spec_from_file_location(name, args.reference)
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference_module
    spec.loader.exec_module(reference_module)
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    cfg = Config(nx=25, ny=50, wall_cells=1, gravity=0)
    geo = Geometry(cfg)
    reference, candidate = reference_module.Rigid(cfg, geo), Rigid(cfg, geo)
    rng = np.random.default_rng(97019)
    records = []
    for shape in ['full', 'eroded', 'single_cell']:
        solid = np.ones((cfg.nb, cfg.nb), np.float32)
        if shape == 'eroded':
            solid[:] = 0
            solid[2:-2, 2:-2] = rng.uniform(.15, 1, (cfg.nb-4, cfg.nb-4))
        elif shape == 'single_cell':
            solid[:] = 0
            solid[-2, cfg.nb//2] = .23
        geo.solid.from_numpy(solid); geo.solid_comp.fill(0)
        for angle in [0., 5., 45., 89.999, 137., 270.]:
            geo.angle[None] = np.radians(angle)
            geo.update_rotation(); geo.reconstruct()
            for position in [(3.2, 25), (21.8, 25), (12.5, 3.2), (12.5, 46.8),
                             (3.2, 3.2), (21.8, 46.8), (6.1, 25), (12.5, 25)]:
                geo.origin[None] = position
                geo.rasterize()
                velocity = rng.uniform(-.03, .03, 3)
                velocity[2] *= .03
                local = np.asarray(geo.mass_matrix[None])@velocity
                world = local.copy(); world[2] += position[0]*local[1]-position[1]*local[0]
                high = world.astype(np.float32)
                low = (high.astype(float)-world).astype(np.float32)
                outputs = []
                for solver in [reference, candidate]:
                    geo.origin[None] = position
                    geo.velocity[None] = velocity
                    solver.momentum[None] = high
                    solver.momentum_comp[None] = low
                    solver.contact_impulse[None] = [0, 0, 0]
                    solver.contact_impulse_comp[None] = [0, 0, 0]
                    solver.contact()
                    outputs.append({
                        'origin': np.asarray(geo.origin[None]),
                        'momentum': np.asarray(solver.momentum[None]),
                        'momentum_comp': np.asarray(solver.momentum_comp[None]),
                        'contact_impulse': np.asarray(solver.contact_impulse[None]),
                        'contact_impulse_comp': np.asarray(solver.contact_impulse_comp[None]),
                    })
                differences = {key: float(np.max(np.abs(outputs[0][key]-outputs[1][key])))
                               for key in outputs[0] if not np.array_equal(outputs[0][key], outputs[1][key])}
                records.append({'shape': shape, 'angle_deg': angle, 'origin': position,
                                'bitwise_equal': not differences, 'differences': differences})
    result = {'cases': len(records), 'all_bitwise_equal': all(r['bitwise_equal'] for r in records),
              'records': records, 'reference': str(args.reference.resolve()),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256(Path(sys.modules[Rigid.__module__].__file__).read_bytes()).hexdigest(),
              'scope': 'v2 contact pose, momentum and compensated impulse equivalence, including wall/corner impacts and f32 extremum ties.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))
    if not result['all_bitwise_equal']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
