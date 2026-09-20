import numpy as np
import pytest

from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.state import State
from iceflow2d_v2.coupling import Coupling
from iceflow2d_v2.rigid import Rigid
from iceflow2d_v2.phase_momentum import PhaseMomentum


@pytest.mark.parametrize('remaining', [.5, .001, 0.0])
def test_geometric_phase_transfer_preserves_twist_and_releases_all_momentum(remaining):
    # Synthetic zero normal/relative flux isolates material removal from
    # Stefan recoil; the production reference densities remain unchanged.
    cfg = Config(nx=25, ny=50, gravity=0, sigma=0, buoyancy=False)
    geo = Geometry(cfg)
    fraction = np.zeros((cfg.nb, cfg.nb), np.float32)
    a, b = cfg.nb-2, cfg.nb-2
    fraction[a, b] = .08
    geo.solid.from_numpy(fraction); geo.reconstruct(); geo.rasterize()
    flow = Flow(cfg, geo); state = State(cfg, geo, flow)
    coupling = Coupling(cfg, geo, state, flow)
    rigid = Rigid(cfg, geo); phase = PhaseMomentum(cfg, geo, rigid, coupling)
    twist = np.array([.015, -.007, .2])
    local = np.asarray(geo.mass_matrix[None])@twist
    origin = np.asarray(geo.origin[None])
    world = local.copy(); world[2] += origin[0]*local[1]-origin[1]*local[0]
    high = world.astype(np.float32)
    rigid.momentum[None] = high
    rigid.momentum_comp[None] = (high.astype(float)-world).astype(np.float32)
    phase.capture()
    old_area = float(geo.integrals.to_numpy()[a, b, 0])
    fraction[a, b] *= remaining
    geo.solid.from_numpy(fraction); geo.reconstruct(); geo.update_mass()
    dm = (old_area-float(geo.integrals.to_numpy()[a, b, 0]))*cfg.rho_s_lu
    mass = np.zeros((cfg.nb, cfg.nb, 12), np.float32); mass[a, b, 0] = dm
    coupling.mass.from_numpy(mass)
    coupling.melt_step[None] = dm
    points = coupling.point.to_numpy(); points[a, b, 0] = origin+[2, 2]
    coupling.point.from_numpy(points)
    lengths = geo.surface_length.to_numpy(); lengths[a, b, 0] = 1
    geo.surface_length.from_numpy(lengths)
    phase.finish()
    np.testing.assert_allclose(np.asarray(phase.twist[None]), twist, atol=2e-10, rtol=0)
    target = np.asarray(geo.mass_matrix[None])@twist
    target[2] += origin[0]*target[1]-origin[1]*target[0]
    actual = np.asarray(rigid.momentum[None], float)-np.asarray(rigid.momentum_comp[None], float)
    np.testing.assert_allclose(actual, target, atol=2e-13, rtol=2e-10)
    records = coupling.exchange.to_numpy().astype(float)
    points = coupling.point.to_numpy().astype(float)
    extra = coupling.torque_extra.to_numpy()
    outgoing = np.array([records[..., 4].sum(), records[..., 5].sum(),
        (points[..., 0]*records[..., 5]-points[..., 1]*records[..., 4]+extra).sum()])
    np.testing.assert_allclose(actual+outgoing, world, atol=1e-9, rtol=3e-8)
    np.testing.assert_allclose(np.asarray(phase.roundoff_total[None]), actual+outgoing-world, atol=2e-13)
    if remaining == 0:
        np.testing.assert_array_equal(actual, 0)
        assert float(geo.mass[None]) == 0 and np.count_nonzero(geo.count.to_numpy()) == 0


def test_accepted_heat_transfers_body_momentum_and_torque_into_fluid_cells():
    cfg = Config(nx=25, ny=50, drop_height=-.006, gravity=0, sigma=0, buoyancy=False)
    geo = Geometry(cfg); flow = Flow(cfg, geo); state = State(cfg, geo, flow)
    coupling = Coupling(cfg, geo, state, flow); rigid = Rigid(cfg, geo)
    phase = PhaseMomentum(cfg, geo, rigid, coupling)
    twist = np.array([.01, -.006, .003])
    before = np.asarray(geo.mass_matrix[None])@twist
    origin = np.asarray(geo.origin[None])
    before[2] += origin[0]*before[1]-origin[1]*before[0]
    high = before.astype(np.float32)
    rigid.momentum[None] = high
    rigid.momentum_comp[None] = (high.astype(float)-before).astype(np.float32)
    rigid.recover_velocity()
    phase.capture(); coupling.prepare()
    geo.erode(coupling.mass); geo.update_mass()
    phase.finish(); coupling.finish()
    assert coupling.failure[None] == 0 and coupling.melt_step[None] > 0
    after = np.asarray(rigid.momentum[None], float)-np.asarray(rigid.momentum_comp[None], float)
    source = state.source.to_numpy().astype(float)
    x, y = np.indices(source.shape[:2])
    received = np.array([source[..., 4].sum(), source[..., 5].sum(),
                        ((x+.5)*source[..., 5]-(y+.5)*source[..., 4]).sum()])
    # Air source carries zero momentum because the fluid is initially at rest.
    np.testing.assert_allclose(after+received, before, atol=2e-10, rtol=0)
