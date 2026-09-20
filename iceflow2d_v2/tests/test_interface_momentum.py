import numpy as np
import taichi as ti
from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.state import State
from iceflow2d_v2.coupling import Coupling
from iceflow2d_v2.lattice import E


@ti.kernel
def stress_populations(flow: ti.template(), out: ti.types.ndarray(dtype=ti.f32, ndim=1)):
    for q in range(9): out[q] = flow.capillary_population(q, 5, 5)


def test_capillary_population_has_the_prescribed_symmetric_tensor():
    cfg = Config(nx=25, ny=50)
    geo = Geometry(cfg); flow = Flow(cfg, geo)
    tensor = np.array([[.012, -.009], [-.009, -.017]])
    flow.capillary_stress[5, 5] = [.012, -.009, -.017]
    out = np.empty(9, np.float32); stress_populations(flow, out)
    e = np.asarray(E)
    np.testing.assert_allclose(out.sum(), 0, atol=3e-9)
    np.testing.assert_allclose(e.T@out, 0, atol=3e-9)
    np.testing.assert_allclose(np.einsum('qi,qj,q->ij', e, e, out), tensor, atol=4e-9)


def test_melt_momentum_and_torque_use_the_actual_interface_point():
    cfg = Config(nx=25, ny=50, drop_height=-.006)
    geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
    coupling = Coupling(cfg, geo, state, flow)
    geo.velocity[None] = [.01, -.006, .003]
    coupling.step()
    assert coupling.failure[None] == 0
    records, points = coupling.exchange.to_numpy().astype(float), coupling.point.to_numpy().astype(float)
    source = state.source.to_numpy().astype(float)
    x, y = np.indices(source.shape[:2])
    expected = np.array([records[..., 4].sum(), records[..., 5].sum(),
                         (points[..., 0]*records[..., 5]-points[..., 1]*records[..., 4]).sum()])
    measured = np.array([source[..., 4].sum(), source[..., 5].sum(),
                         ((x+.5)*source[..., 5]-(y+.5)*source[..., 4]).sum()])
    assert np.linalg.norm(expected) > 1e-8
    np.testing.assert_allclose(measured, expected, rtol=5e-7, atol=1e-10)
