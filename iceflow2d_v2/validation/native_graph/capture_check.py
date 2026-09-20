"""Optional CUDA stream-capture experiment using only documented driver APIs."""
import ctypes
import argparse
import json
from pathlib import Path
import time
import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation
from iceflow2d_v2.checkpoint import Snapshot
from iceflow2d_v2.cuda_schedule import load_library

parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=25)
parser.add_argument('--steps', type=int, default=128)
args = parser.parse_args()
ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
sim = Simulation(Config(nx=args.nx, ny=2*args.nx, wall_cells=max(1, round(.03*args.nx)), warmup_steps=0,
                        projection_solver='multigrid'))
sim.enable_graph()
sim.step(2)
library, _ = load_library()
assert library.begin_probe() == 0
sim.schedule.run(1); ti.sync()
assert library.end_probe() == 0
before = Snapshot(sim)
assert library.record_begin() == 0
sim.schedule.run(1); ti.sync()
assert library.record_end(1) == 0
reference = Snapshot(sim)
before.restore(sim)
assert library.launch_graph(1) == 0
actual = Snapshot(sim)
differences = []
for k, a in reference.arrays.items():
    b = actual.arrays[k]
    if not np.array_equal(a, b):
        differences.append((k, float(np.nanmax(np.abs(a.astype(float)-b)))))
print('DIFFERENCES', json.dumps(differences), flush=True)
outputs = {}
for name, run in [('taichi', lambda: sim.schedule.run(args.steps)), ('cuda', lambda: library.launch_graph(args.steps))]:
    before.restore(sim)
    start = time.perf_counter()
    run(); ti.sync()
    print('TIMING', name, time.perf_counter()-start, flush=True)
    outputs[name] = Snapshot(sim)
for key, a in outputs['taichi'].arrays.items():
    if key.endswith('.parent') or key == 'merge.members': continue
    b = outputs['cuda'].arrays[key]
    np.testing.assert_allclose(b, a, rtol=2e-5, atol=2e-6, err_msg=key)
print('MULTISTEP FIELDS AGREE', args.steps, flush=True)
