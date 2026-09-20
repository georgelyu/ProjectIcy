"""Warm-kernel microbenchmark, independent of initialization and graph capture."""
import time
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.state import State
from iceflow2d_v2.transport import Transport

ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False, kernel_profiler=True)
cfg = Config(nx=200, ny=400, wall_cells=6)
geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
operator = Transport(cfg, geo, state, flow)
operator.build_predictor()
operator.r.fill(.01); operator.z.fill(.02); operator.d.fill(.03)
operator.dot_columns(); operator.dot_finish(); ti.sync()
ti.profiler.clear_kernel_profiler_info()
start = time.perf_counter()
for _ in range(100): operator.dot_columns(); operator.dot_finish()
ti.sync()
print('100 REDUCTIONS WALL SECONDS', time.perf_counter()-start)
ti.profiler.print_kernel_profiler_info()
