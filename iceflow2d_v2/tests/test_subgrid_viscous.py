import numpy as np
import pytest

from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.state import State
from iceflow2d_v2.rigid import Rigid
from iceflow2d_v2.agglomeration import PatchMerge
from iceflow2d_v2.subgrid_viscous import SubgridViscous


def setup_body(fraction=.005, fixed=False):
    cfg = Config(nx=25, ny=50, drop_height=-.012, gravity=0, sigma=0,
                 buoyancy=False, fixed_body=fixed, warmup_steps=0)
    geo = Geometry(cfg)
    if fraction != 1:
        fractions = np.zeros((cfg.nb, cfg.nb), np.float32)
        a, b = cfg.nb//2, cfg.nb//2
        fractions[a, b] = fraction
        geo.solid.from_numpy(fractions)
        normal = geo.normal.to_numpy(); normal[a, b] = [.6, .8]
        geo.normal.from_numpy(normal)
        geo.reconstruct(); geo.rasterize(); geo.remember_geometry()
    flow = Flow(cfg, geo); state = State(cfg, geo, flow)
    patches = PatchMerge(cfg, geo, state, flow); patches.step()
    rigid = Rigid(cfg, geo)
    viscous = SubgridViscous(cfg, geo, state, flow, rigid, patches)
    flow.tau_shear.fill(.8)
    return cfg, geo, flow, state, rigid, viscous


def set_twist(geo, rigid, velocity):
    center = np.asarray(geo.origin[None])+np.asarray(geo.first_moment[None])/geo.mass[None]
    p = np.asarray(velocity)*[geo.mass[None], geo.mass[None], geo.inertia[None]]
    p[2] += center[0]*p[1]-center[1]*p[0]
    high = p.astype(np.float32)
    rigid.momentum[None] = high
    rigid.momentum_comp[None] = (high.astype(float)-p).astype(np.float32)
    rigid.recover_velocity()
    return center


def set_fluid_twist(cfg, state, center, velocity):
    q = state.q.to_numpy().astype(float)-state.comp.to_numpy()
    mass = q[..., 0]+cfg.rho_a_lu*q[..., 1]
    x, y = np.indices(mass.shape)
    q[..., 4] = mass*(velocity[0]-velocity[2]*(y+.5-center[1]))
    q[..., 5] = mass*(velocity[1]+velocity[2]*(x+.5-center[0]))
    high = q.astype(np.float32)
    state.q.from_numpy(high); state.comp.from_numpy((high.astype(float)-q).astype(np.float32))


def inventories(cfg, geo, state, rigid):
    q = state.q.to_numpy().astype(float)-state.comp.to_numpy()
    mass = q[..., 0]+cfg.rho_a_lu*q[..., 1]
    x, y = np.indices(mass.shape)
    p = np.asarray(rigid.momentum[None], float)-np.asarray(rigid.momentum_comp[None], float)
    total = p+np.array([q[..., 4].sum(), q[..., 5].sum(),
                       ((x+.5)*q[..., 5]-(y+.5)*q[..., 4]).sum()])
    kinetic = np.sum((q[..., 4][mass>0]**2+q[..., 5][mass>0]**2)/(2*mass[mass>0]))
    if geo.mass[None] > 0:
        center = np.asarray(geo.origin[None])+np.asarray(geo.first_moment[None])/geo.mass[None]
        spin = p[2]-(center[0]*p[1]-center[1]*p[0])
        kinetic += (p[0]**2+p[1]**2)/(2*geo.mass[None])+spin**2/(2*geo.inertia[None])
    return total, kinetic, q[..., :4].copy()


@pytest.mark.parametrize('fraction', [.005, 1e-8])
def test_unresolved_body_viscosity_conserves_momentum_and_dissipates(fraction):
    cfg, geo, flow, state, rigid, viscous = setup_body(fraction)
    assert not geo.mask.to_numpy().any()
    set_twist(geo, rigid, [.003, -.002, .8])
    before, energy, scalar = inventories(cfg, geo, state, rigid)
    initial_energy = energy
    for _ in range(5):
        viscous.step()
        after, next_energy, current_scalar = inventories(cfg, geo, state, rigid)
        assert viscous.enabled[None] == 1 and viscous.failure[None] == 0
        np.testing.assert_allclose(after, before, atol=2e-11, rtol=1e-7)
        np.testing.assert_array_equal(current_scalar, scalar)
        assert next_energy <= energy+1e-13*initial_energy
        energy = next_energy
    assert energy < .9*initial_energy
    np.testing.assert_allclose(viscous.dissipated_total[None], initial_energy-energy,
                               atol=1e-14, rtol=2e-5)


def test_common_rigid_motion_has_no_viscous_exchange():
    cfg, geo, flow, state, rigid, viscous = setup_body()
    velocity = [.001, -.002, .0002]
    center = set_twist(geo, rigid, velocity)
    set_fluid_twist(cfg, state, center, velocity)
    before = state.q.to_numpy().astype(float)-state.comp.to_numpy()
    body_before = np.asarray(rigid.momentum[None], float)-np.asarray(rigid.momentum_comp[None], float)
    viscous.step()
    after = state.q.to_numpy().astype(float)-state.comp.to_numpy()
    body_after = np.asarray(rigid.momentum[None], float)-np.asarray(rigid.momentum_comp[None], float)
    np.testing.assert_allclose(after, before, atol=1e-13, rtol=0)
    np.testing.assert_allclose(body_after, body_before, atol=1e-13, rtol=0)
    assert viscous.failure[None] == 0


def test_fixed_unresolved_body_records_support_and_dissipation():
    cfg, geo, flow, state, rigid, viscous = setup_body(fixed=True)
    center = set_twist(geo, rigid, [0, 0, 0])
    set_fluid_twist(cfg, state, center, [.001, -.002, .0002])
    before, energy, scalar = inventories(cfg, geo, state, rigid)
    viscous.step()
    after, final_energy, final_scalar = inventories(cfg, geo, state, rigid)
    support = np.asarray(rigid.support_impulse[None], float)-np.asarray(rigid.support_impulse_comp[None], float)
    np.testing.assert_allclose(after-before, support, atol=1e-11, rtol=1e-7)
    np.testing.assert_array_equal(rigid.momentum[None], 0)
    np.testing.assert_array_equal(final_scalar, scalar)
    assert final_energy < energy and viscous.dissipated_total[None] > 0
    assert viscous.failure[None] == 0


@pytest.mark.parametrize('fraction', [0, 1])
def test_empty_or_snibb_resolved_body_is_unchanged(fraction):
    cfg, geo, flow, state, rigid, viscous = setup_body(fraction)
    if fraction:
        set_twist(geo, rigid, [.001, -.002, .0002])
        flow.collide(); flow.prepare_boundary(); flow.reduce_boundary()
        assert np.trace(flow.body_drag[None]) > 0
    before = state.q.to_numpy().copy()
    body_before = np.asarray(rigid.momentum[None]).copy()
    viscous.step()
    assert viscous.enabled[None] == 0 and viscous.failure[None] == 0
    np.testing.assert_array_equal(state.q.to_numpy(), before)
    np.testing.assert_array_equal(rigid.momentum[None], body_before)
