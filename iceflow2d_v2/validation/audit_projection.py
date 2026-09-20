"""Report original-operator residuals before/after multigrid and CG."""
import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.state import State
from iceflow2d_v2.transport import Transport


def main():
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    cfg = Config(nx=25, ny=50, advection=False, projection_solver='multigrid')
    g = Geometry(cfg); f = Flow(cfg, g); s = State(cfg, g, f); tr = Transport(cfg, g, s, f)
    g.remember_geometry(); g.origin[None] = np.asarray(g.origin[None])+[.04, -.06]; g.angle[None] += .007
    g.rasterize(); tr.build_predictor(); tr.initialize_projection()
    tr.seed_components(); tr.components.seed(tr.diag, 1e-15); tr.components.connect(tr.k, True)
    tr.components.labels(tr.component); tr.reduce_small_components(); tr.nullspace_component(); tr.project_nullspace()
    print('initial', np.max(np.abs(tr.r.to_numpy())), flush=True)
    tr.multigrid.solve(tr.k, tr.r); tr.accept_multigrid(tr.multigrid.levels[0])
    print('after MG', np.max(np.abs(tr.r.to_numpy())), flush=True)
    tr.dot_columns(); tr.dot_finish(); tr.start_cg()
    for iteration in range(180):
        tr.matrix_product(); tr.dot_columns(); tr.dot_finish()
        tr.cg_update(); tr.remember_rr(); tr.dot_columns(); tr.dot_finish(); tr.cg_direction()
        if (iteration+1) % 12 == 0:
            print(iteration+1, np.max(np.abs(tr.r.to_numpy())), tr.rr[None], flush=True)


if __name__ == '__main__': main()
