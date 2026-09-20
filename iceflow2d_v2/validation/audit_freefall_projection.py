"""Compare a failing aligned-body projection with an independent sparse solve."""
from pathlib import Path
import numpy as np
import taichi as ti
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from iceflow2d_v2.config import Config
from iceflow2d_v2.simulation import Simulation


def main():
    ti.init(arch=ti.cuda, default_fp=ti.f32, fast_math=False)
    cfg = Config(nx=100, ny=200, melting=False, sigma=0, angle_deg=0,
                 water_level=-.1, drop_height=.132, projection_solver='multigrid')
    sim = Simulation(cfg); tr = sim.transport
    trace = {}
    for name in ('initialize_projection', 'project_nullspace', 'accept_multigrid', 'volume_fluxes', 'commit_transport'):
        original = getattr(tr, name)
        def wrapped(*args, _original=original, _name=name):
            _original(*args)
            trace[_name] = {k:getattr(tr, k).to_numpy() for k in ('k', 'r', 'x', 'diag', 'component')}
            print(_name, 'maxr', np.max(abs(tr.r.to_numpy())), 'maxx', np.max(abs(tr.x.to_numpy())), flush=True)
        setattr(tr, name, wrapped)
    sim.step(1)
    weights = trace['initialize_projection']['k'].astype(float)
    nx, ny, _ = weights.shape
    rows, cols, vals = [], [], []
    for i in range(nx):
        for j in range(ny):
            for k, (dx, dy) in enumerate(((1,0),(0,1),(1,1),(-1,1))):
                ni, nj = i+dx, j+dy
                if 0 <= ni < nx and 0 <= nj < ny and weights[i,j,k] > 0:
                    a, b, w = i*ny+j, ni*ny+nj, weights[i,j,k]
                    rows += [a,b,a,b]; cols += [a,b,b,a]; vals += [w,w,-w,-w]
    a = coo_matrix((vals,(rows,cols)),shape=(nx*ny,nx*ny)).tocsr()
    active = a.diagonal() > 1e-15
    _, labels = connected_components(a)
    pins = [np.flatnonzero((labels==label) & active)[0] for label in np.unique(labels[active])]
    active[pins] = False
    rhs = trace['project_nullspace']['r'].astype(float).ravel()
    exact = np.zeros(nx*ny); exact[active] = spsolve(a[active][:,active], rhs[active])
    candidate = tr.x.to_numpy().ravel().astype(float)
    for name, x in [('direct',exact),('production',candidate)]:
        residual = rhs-a@x
        print(name, 'maxx',max(abs(x)), 'maxresidual',max(abs(residual)),
              'worst',np.unravel_index(np.argmax(abs(residual)),(nx,ny)), flush=True)
    print('rhs sum',rhs.sum(),'minimum positive diagonal',a.diagonal()[a.diagonal()>0].min(), flush=True)
    print(sim.diagnostics(),flush=True)
    root=Path('iceflow2d_v2/outputs/freefall_projection');root.mkdir(exist_ok=True)
    np.savez_compressed(root/'audit.npz', **{f'{stage}_{key}':value for stage,fields in trace.items() for key,value in fields.items()}, exact=exact.reshape(nx,ny))
    sim.save(root/'state.npz')


if __name__ == '__main__': main()
