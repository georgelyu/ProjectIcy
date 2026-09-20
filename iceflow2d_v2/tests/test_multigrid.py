"""Independent matrix checks for the optional capacity-solve accelerator."""
import numpy as np
import taichi as ti
from iceflow2d_v2.multigrid import Level


EDGES = ((1, 0), (0, 1), (1, 1), (-1, 1))


def laplacian(weights):
    nx, ny, _ = weights.shape
    matrix = np.zeros((nx*ny, nx*ny))
    for i in range(nx):
        for j in range(ny):
            for k, (di, dj) in enumerate(EDGES):
                ni, nj = i+di, j+dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    a, b, w = i*ny+j, ni*ny+nj, weights[i, j, k]
                    matrix[a, a] += w; matrix[b, b] += w
                    matrix[a, b] -= w; matrix[b, a] -= w
    return matrix


def test_coarse_weights_are_galerkin_including_odd_edges_and_cuts():
    nx, ny = 17, 23
    rng = np.random.default_rng(4301)
    weights = rng.uniform(.01, 1, (nx, ny, 4)).astype(np.float32)
    weights[rng.random(weights.shape) < .3] = 0
    for i in range(nx):
        for j in range(ny):
            for k, (di, dj) in enumerate(EDGES):
                if not (0 <= i+di < nx and 0 <= j+dj < ny): weights[i, j, k] = 0
    fine, coarse = Level(nx, ny), Level((nx+1)//2, (ny+1)//2)
    fine.k.from_numpy(weights); coarse.coarsen(fine)
    interpolation = np.zeros((nx*ny, coarse.nx*coarse.ny))
    for i in range(nx):
        for j in range(ny): interpolation[i*ny+j, (i//2)*coarse.ny+j//2] = 1
    expected = interpolation.T@laplacian(weights)@interpolation
    np.testing.assert_allclose(laplacian(coarse.k.to_numpy()), expected, rtol=2e-7, atol=5e-7)


def test_multigrid_and_cg_close_the_same_moving_capacity():
    from iceflow2d_v2.config import Config
    from iceflow2d_v2.geometry import Geometry
    from iceflow2d_v2.flow import Flow
    from iceflow2d_v2.state import State
    from iceflow2d_v2.transport import Transport
    answers = []
    for solver in ('cg', 'multigrid'):
        cfg = Config(nx=25, ny=50, projection_solver=solver, advection=False, pressure_iterations=300,
                     pressure_polish_iterations=128)
        geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
        transport = Transport(cfg, geo, state, flow)
        geo.remember_geometry()
        geo.origin[None] = np.asarray(geo.origin[None])+[.04, -.06]
        geo.angle[None] += .007
        geo.rasterize(); transport.remap()
        q = state.q.to_numpy().astype(float)-state.comp.to_numpy()
        assert np.max(np.abs(q[..., 0]+q[..., 1]-geo.capacity.to_numpy())) < 2e-6
        answers.append(q)
    np.testing.assert_allclose(answers[0], answers[1], atol=3e-6, rtol=3e-5)


def test_projection_flux_is_independent_of_a_large_potential_gauge():
    from iceflow2d_v2.config import Config
    from iceflow2d_v2.geometry import Geometry
    from iceflow2d_v2.flow import Flow
    from iceflow2d_v2.state import State
    from iceflow2d_v2.transport import Transport
    answers = []
    for gauge in (0., 10000.):
        cfg = Config(nx=25, ny=50, advection=False, pressure_iterations=300)
        geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
        transport = Transport(cfg, geo, state, flow)
        transport.x.fill(gauge)
        geo.remember_geometry()
        geo.origin[None] = np.asarray(geo.origin[None])+[.04, -.06]
        geo.angle[None] += .007
        geo.rasterize(); transport.remap()
        q = state.q.to_numpy().astype(float)-state.comp.to_numpy()
        assert np.max(np.abs(q[..., 0]+q[..., 1]-geo.capacity.to_numpy())) < 2e-6
        answers.append(transport.flux.to_numpy())
    np.testing.assert_allclose(answers[0], answers[1], rtol=1e-6, atol=2e-9)


def test_tiny_aperture_does_not_relax_the_absolute_volume_constraint():
    from iceflow2d_v2.config import Config
    from iceflow2d_v2.geometry import Geometry
    from iceflow2d_v2.flow import Flow
    from iceflow2d_v2.state import State
    from iceflow2d_v2.transport import Transport
    cfg = Config(nx=25, ny=50, advection=False, pressure_iterations=400, transport_iterations=80)
    geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
    geo.solid.fill(0); geo.reconstruct(); geo.rasterize()
    capacity = geo.capacity.to_numpy(); capacity[10, 15] = 0
    geo.capacity.from_numpy(capacity); geo.remember_geometry(); state.initialize()
    capacity[10, 15] = .001; geo.capacity.from_numpy(capacity)
    faces = geo.face_open.to_numpy()
    faces[9, 15, 0] = faces[10, 15, 0] = 5e-14
    faces[10, 14, 1] = faces[10, 15, 1] = 5e-14
    geo.face_open.from_numpy(faces); geo.old_face_open.from_numpy(faces)
    state.source[4, 5] = [.001, 0, 0, 0, 0, 0]
    transport = Transport(cfg, geo, state, flow); transport.remap()
    q = state.q.to_numpy().astype(float)-state.comp.to_numpy()
    assert transport.initial_rr[None] > 1e5
    assert np.max(np.abs(q[..., 0]+q[..., 1]-capacity)) < 2e-6


def test_parallel_cg_reduction_matches_independent_positive_edge_energy():
    import math
    from iceflow2d_v2.config import Config
    from iceflow2d_v2.geometry import Geometry
    from iceflow2d_v2.flow import Flow
    from iceflow2d_v2.state import State
    from iceflow2d_v2.transport import Transport
    cfg = Config(nx=25, ny=50)
    geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
    transport = Transport(cfg, geo, state, flow)
    rng = np.random.default_rng(139)
    weights = rng.uniform(1e-5, 1, (cfg.nx, cfg.ny, 4)).astype(np.float32)
    d = 1e8+rng.normal(size=(cfg.nx, cfg.ny))
    r = rng.normal(size=d.shape); z = r/rng.uniform(.01, 5, d.shape)
    terms = []
    for i in range(cfg.nx):
        for j in range(cfg.ny):
            for k, (di, dj) in enumerate(EDGES):
                ni, nj = i+di, j+dj
                if 0 <= ni < cfg.nx and 0 <= nj < cfg.ny:
                    terms.append(float(weights[i, j, k])*(d[i, j]-d[ni, nj])**2)
    transport.k.from_numpy(weights); transport.d.from_numpy(d)
    transport.r.from_numpy(r); transport.z.from_numpy(z)
    transport.dot_columns(); transport.dot_finish()
    np.testing.assert_allclose(transport.rr[None], math.fsum((r*z).ravel()), rtol=2e-13)
    np.testing.assert_allclose(transport.dad[None], math.fsum(terms), rtol=2e-13)
