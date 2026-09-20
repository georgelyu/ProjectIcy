"""Observe one warmed Taichi step through the public CUDA callback API."""
import ctypes
from pathlib import Path
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation

ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
sim = Simulation(Config(nx=25, ny=50, warmup_steps=0, projection_solver='multigrid'))
sim.enable_graph(); sim.step(2)
from iceflow2d_v2.cuda_schedule import load_library
probe, _ = load_library()
status = probe.begin_probe()
if status: raise RuntimeError(f'CUPTI subscribe failed: {status}')
try: sim.schedule.run(1); ti.sync()
finally:
    status = probe.end_probe()
    if status: raise RuntimeError(f'CUPTI unsubscribe failed: {status}')
