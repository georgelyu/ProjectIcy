from dataclasses import replace
import numpy as np
import pytest
from iceflow2d_v2.config import Config
from iceflow2d_v2.geometry import Geometry
from iceflow2d_v2.flow import Flow
from iceflow2d_v2.state import State
from iceflow2d_v2.thermal import Thermal
from iceflow2d_v2.coupling import Coupling
from iceflow2d_v2.transport import Transport
from iceflow2d_v2.simulation import Simulation


def setup(config=None):
    c = config or Config(nx=25, ny=50, fixed_body=True, advection=False, pressure_iterations=160)
    g = Geometry(c); f = Flow(c, g); s = State(c, g, f)
    return c, g, f, s


def totals(s, g, c):
    q = s.q.to_numpy().astype(float)-s.comp.to_numpy()
    ms, es = g.totals()
    return np.array([q[..., 0].sum()+ms, q[..., 2].sum()+es+c.latent_lu*q[..., 0].sum(), q[..., 1].sum(), q[..., 3].sum()])


def test_water_air_temperature_jump_is_adiabatic_at_actual_capacity_ratio():
    c, g, f, s = setup()
    t = Thermal(c, g, s)
    before = totals(s, g, c)
    for _ in range(32): t.step()
    after = totals(s, g, c)
    np.testing.assert_allclose(after, before, rtol=1e-8, atol=1e-9)
    q = s.q.to_numpy()
    np.testing.assert_allclose(s.temperature_w.to_numpy()[q[..., 0] > 1e-6], 1, atol=5e-7)
    np.testing.assert_allclose(s.temperature_a.to_numpy()[q[..., 1] > 1e-6], 20/90, atol=5e-7)
    np.testing.assert_allclose(c.rho_w/c.rho_a, 800)
    np.testing.assert_allclose(1/c.air_capacity_lu, 3332.139303482587, rtol=1e-10)


def test_initial_compensated_phase_volumes_exactly_fill_closed_geometry():
    c, g, f, s = setup()
    q = s.q.to_numpy().astype(float)-s.comp.to_numpy().astype(float)
    residual = q[..., 0]+q[..., 1]-g.capacity.to_numpy()
    assert np.max(np.abs(residual)) < 4e-15
    assert abs(residual.sum()) < 2e-13


def test_thermal_internal_flux_conserves_energy_and_limits_tiny_capacity():
    c, g, f, s = setup()
    q = s.q.to_numpy()
    x = np.arange(c.nx)[:, None]
    q[..., 2] = q[..., 0]*(.5+.4*np.sin(2*np.pi*x/c.nx))
    s.q.from_numpy(q); s.temperatures()
    t = Thermal(c, g, s)
    before = totals(s, g, c)
    for _ in range(50): t.step()
    after = totals(s, g, c)
    np.testing.assert_allclose(after[:2], before[:2], rtol=2e-8, atol=1e-9)
    water = s.q.to_numpy()[..., 0] > 1e-7
    temp = s.temperature_w.to_numpy()[water]
    assert temp.min() > .09 and temp.max() < .91


def test_single_accepted_melt_mass_energy_and_stefan_jump():
    c, g, f, s = setup(replace(Config(nx=25, ny=50), fixed_body=True, advection=False,
                              drop_height=-.006, pressure_iterations=160))
    cp = Coupling(c, g, s, f)
    before = totals(s, g, c)
    cp.step()
    dm = cp.mass.to_numpy().astype(float)
    ql, qs = cp.heat.to_numpy().astype(float), cp.q_s.to_numpy().astype(float)
    assert dm.sum() > 0
    np.testing.assert_allclose(ql-qs, dm*c.latent_lu, rtol=2e-6, atol=2e-10)
    src = s.source.to_numpy().astype(float).sum((0, 1))
    after = totals(s, g, c)
    np.testing.assert_allclose(after[0]+src[0], before[0], rtol=1e-9)
    np.testing.assert_allclose(after[1]+src[2]+src[0]*c.latent_lu, before[1], rtol=1e-9)
    np.testing.assert_allclose(src[1], dm.sum()*(1/c.rho_s_lu-1), rtol=1e-6)
    difference = cp.liquid_velocity.to_numpy()-cp.material_velocity.to_numpy()
    length = g.surface_length.to_numpy()
    expected = np.zeros_like(difference)
    active = length > 0
    expected[active] = (dm[active]/length[active]*(1-1/c.rho_s_lu))[:, None]*cp.normal.to_numpy()[active]
    np.testing.assert_allclose(difference, expected, atol=2e-11, rtol=3e-5)
    source = s.source.to_numpy()
    connected = cp.air_connected.to_numpy().astype(bool)
    assert np.all(source[..., 1][~connected] == 0), 'numerical air appeared outside main gas component'
    np.testing.assert_array_equal(source[..., 3], 0)
    np.testing.assert_allclose(cp.air_heat_step.to_numpy().sum(dtype=float), source[..., 1].sum(dtype=float)*c.air_capacity_lu*20/90, rtol=1e-6)


def test_subcooled_ice_stores_sensible_heat_before_melting():
    c, g, f, s = setup(replace(Config(nx=25, ny=50), fixed_body=True, advection=False,
                              drop_height=-.006, ice_temperature=-10))
    cp = Coupling(c, g, s, f)
    old_mass, old_energy = g.totals()
    cp.step()
    mass, energy = g.totals()
    assert energy > old_energy
    assert mass == old_mass
    assert cp.mass.to_numpy().sum() == 0
    np.testing.assert_allclose(cp.heat.to_numpy(), cp.q_s.to_numpy(), atol=1e-9)


def test_no_gas_receiver_rejects_melting_without_deleting_ice():
    c, g, f, s = setup(replace(Config(nx=25, ny=50), fixed_body=True, advection=False, water_level=.05, drop_height=-.02))
    q = s.q.to_numpy(); q[..., 0] = g.capacity.to_numpy(); q[..., 1] = 0
    q[..., 2] = q[..., 0]; q[..., 3] = 0
    s.q.from_numpy(q); s.temperatures()
    cp = Coupling(c, g, s, f)
    before = totals(s, g, c)
    cp.step()
    assert cp.failure[None] == 1
    np.testing.assert_array_equal(totals(s, g, c), before)
    assert cp.mass.to_numpy().sum() == 0


def test_rigid_remap_preserves_mass_heat_and_constant_temperature():
    c, g, f, s = setup()
    tr = Transport(c, g, s, f)
    before = totals(s, g, c)
    origin = np.array(g.origin[None], float)
    g.remember_geometry()
    g.origin[None] = origin+[.02, -.03]
    g.angle[None] += .003
    g.rasterize()
    tr.remap(); s.temperatures()
    np.testing.assert_allclose(totals(s, g, c), before, rtol=1e-8, atol=1e-8)
    assert np.max(np.abs(tr.capacity_residual.to_numpy())) < 2e-5
    q = s.q.to_numpy()
    np.testing.assert_allclose(s.temperature_w.to_numpy()[q[..., 0] > 1e-6], 1, atol=2e-6)
    np.testing.assert_array_equal(s.temperature_a.to_numpy()[q[..., 1] > 1e-12], np.float32(20/90))
    assert q[..., :2].min() > -1e-7


def test_coupled_smoke_conserves_condensed_mass_and_enthalpy():
    sim = Simulation(Config(nx=25, ny=50, pressure_iterations=160))
    initial_state = sim.state.q.to_numpy().astype(float)-sim.state.comp.to_numpy()
    initial_wet = initial_state[..., 0] > 1e-8
    initial_h = sim.thermal.gw.to_numpy().astype(float).sum(axis=-1)
    np.testing.assert_allclose(initial_h[initial_wet], initial_state[..., 2][initial_wet]/initial_state[..., 0][initial_wet]+sim.cfg.latent_lu,
                               rtol=2e-7, atol=2e-7)
    sim.step(8)
    d = sim.validate()
    assert abs(d['condensed_mass_relative_residual']) < 1e-6
    assert abs(d['condensed_energy_relative_residual']) < 1e-6
    assert abs(d['air_unexpected_residual_kg_m']) < 1e-11
    assert d['capacity_residual_max'] < 2e-5
    state = sim.state.q.to_numpy().astype(float)-sim.state.comp.to_numpy()
    wet = state[..., 0] > 1e-8
    enthalpy = sim.thermal.gw.to_numpy().astype(float).sum(axis=-1)
    np.testing.assert_allclose(enthalpy[wet], state[..., 2][wet]/state[..., 0][wet]+sim.cfg.latent_lu,
                               rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize('block', (1, 2, 4, 8))
def test_regular_domain_fourier_mode_matches_discrete_d2q5_decay(block):
    c, g, f, s = setup(replace(Config(nx=25, ny=50), fixed_body=True, advection=False, k_w=100, coupling_steps=block))
    g.solid.fill(0); g.reconstruct(); g.rasterize(); g.remember_geometry()
    volume = g.capacity.to_numpy()
    x, y = np.indices(volume.shape)
    lx, ly = c.nx-2*c.wall_cells, c.ny-2*c.wall_cells
    mode = np.cos(np.pi*(x-c.wall_cells+.5)/lx)*np.cos(2*np.pi*(y-c.wall_cells+.5)/ly)
    q = np.zeros((*volume.shape, 6), np.float32)
    q[..., 0] = volume; q[..., 2] = volume*(.5+.2*mode)
    s.q.from_numpy(q); s.comp.fill(0); s.temperatures()
    thermal = Thermal(c, g, s)
    steps = 320
    for _ in range(steps): thermal.step()
    k = c.thermal_weight('water')/4
    decay = (1-4*k*(np.sin(np.pi/(2*lx))**2+np.sin(np.pi/ly)**2))**(steps//block)
    expected = .5+.2*mode*decay
    np.testing.assert_allclose(s.temperature_w.to_numpy()[volume > 0], expected[volume > 0], atol=4e-7, rtol=1e-6)


def test_last_ice_cell_exhaustion_keeps_all_mass_and_latent_energy():
    c, g, f, s = setup(replace(Config(nx=25, ny=50), fixed_body=True, advection=False, drop_height=-.012))
    remaining = np.zeros((c.nb, c.nb), np.float32); remaining[c.nb//2, c.nb//2] = 1e-6
    g.solid.from_numpy(remaining); g.reconstruct(); g.rasterize(); g.remember_geometry()
    s.initialize(); s.temperatures()
    cp = Coupling(c, g, s, f)
    before = totals(s, g, c)
    cp.step()
    source = s.source.to_numpy().astype(float).sum((0, 1))
    after = totals(s, g, c)
    assert g.totals()[0] < 1e-12
    assert source[0] > 0
    np.testing.assert_allclose(after[0]+source[0], before[0], rtol=1e-10)
    np.testing.assert_allclose(after[1]+source[2]+c.latent_lu*source[0], before[1], rtol=1e-10)


def test_frozen_hydrostatic_reference_does_not_force_stationary_cut_cells():
    config = Config(nx=25, ny=50, fixed_body=True, melting=False, advection=False,
                    sigma=0, pressure_iterations=160, water_level=-.1, drop_height=.132)
    sim = Simulation(config)
    sim.step(4)
    assert np.max(np.abs(sim.flow.velocity.to_numpy())) < 1e-9
    sim.validate()


def test_hydrostatic_reference_integrals_close_physical_momentum():
    sim = Simulation(Config(nx=25, ny=50, sigma=0, melting=False, pressure_iterations=160))
    sim.step(8)
    d = sim.diagnostics()
    # Absolute SI tolerances for a unit-depth column; gravity/support are much
    # larger and must cancel without counting a numerical reference source.
    assert abs(d['momentum_balance_residual_x']) < 1e-9
    assert abs(d['momentum_balance_residual_y']) < 2e-9
    assert abs(d['angular_balance_residual']) < 1e-9


def test_cold_water_buoyancy_in_masked_cut_volumes_closes_momentum():
    sim = Simulation(Config(nx=25, ny=50, sigma=0, melting=False, fixed_body=True,
                            drop_height=-.005, warmup_steps=0))
    q = sim.state.q.to_numpy(); q[..., 2] = .5*q[..., 0]
    sim.state.q.from_numpy(q); sim.state.temperatures()
    sim.initial = None; sim.initial = sim.diagnostics()
    sim.step(1)
    d = sim.validate()
    assert abs(d['momentum_balance_residual_y']) < 2e-9
    assert abs(d['angular_balance_residual']) < 2e-10


def test_constant_pressure_has_no_cut_cell_force_or_body_torque():
    from iceflow2d_v2.lattice import W
    sim = Simulation(Config(nx=25, ny=50, fixed_body=True, melting=False,
                            gravity=0, sigma=0, warmup_steps=0, water_level=-.1, drop_height=.132))
    pressure = 1e-4
    equilibrium = 3*pressure*np.array(W, np.float32)
    equilibrium[0] -= 3*pressure
    populations = np.broadcast_to(equilibrium[:, None, None], (9, 25, 50)).copy()
    sim.flow.pressure.fill(pressure)
    sim.flow.f.from_numpy(populations); sim.flow.fpost.from_numpy(populations)
    sim.flow.prepare_boundary(); sim.flow.reduce_boundary()
    sim.transport.build_predictor(); sim.momentum.step()
    q = sim.state.q.to_numpy().astype(float)-sim.state.comp.to_numpy()
    assert np.max(np.abs(q[..., 4:])) < 2e-10
    np.testing.assert_allclose(sim.flow.body_force[None], 0, atol=2e-9)


def test_local_cut_cell_merge_preserves_all_extensive_quantities_and_torque():
    from iceflow2d_v2.merge import CutCellMerge
    c, g, f, s = setup()
    q = s.q.to_numpy()
    rng = np.random.default_rng(343)
    mass = q[..., 0]+c.rho_a_lu*q[..., 1]
    q[..., 4:] = mass[..., None]*rng.normal(0, .01, (*mass.shape, 2))
    s.q.from_numpy(q)
    before = q.astype(float)
    merger = CutCellMerge(c, g, s, f); merger.step()
    after = s.q.to_numpy().astype(float)-s.comp.to_numpy()
    np.testing.assert_allclose(after.sum((0, 1)), before.sum((0, 1)), rtol=2e-8, atol=1e-9)
    x, y = np.indices(mass.shape)
    before_l = ((x+.5)*before[..., 5]-(y+.5)*before[..., 4]).sum()
    after_l = ((x+.5)*after[..., 5]-(y+.5)*after[..., 4]).sum()
    np.testing.assert_allclose(after_l, before_l, rtol=2e-7, atol=1e-8)
    assert after[..., :2].min() >= -1e-9


def test_reconstruction_obeys_authoritative_phase_and_half_force_momentum():
    from iceflow2d_v2.lattice import E
    c, g, f, s = setup()
    rng = np.random.default_rng(197)
    capacity = g.capacity.to_numpy()
    fraction = rng.uniform(.1, .9, capacity.shape).astype(np.float32)
    u = rng.uniform(-.02, .02, (*capacity.shape, 2)).astype(np.float32)
    acceleration = rng.uniform(-.001, .001, u.shape).astype(np.float32)
    q = s.q.to_numpy()
    q[..., 0] = capacity*fraction; q[..., 1] = capacity-q[..., 0]
    q[..., 2] = q[..., 0]; q[..., 3] = 0
    mass = q[..., 0]+c.rho_a_lu*q[..., 1]
    q[..., 4:] = mass[..., None]*u
    s.q.from_numpy(q); s.comp.fill(0); f.force.from_numpy(acceleration)
    s.prepare_new_nodes(); s.restore_populations()
    fluid = (capacity > 1e-8) & (g.mask.to_numpy() == 0)
    h = f.h.to_numpy().astype(float).sum(axis=0)
    raw = np.einsum('qi,qxy->xyi', np.array(E, float), f.f.to_numpy().astype(float))
    phi = q[..., 0][fluid]/capacity[fluid]
    rho = c.rho_a_lu+(1-c.rho_a_lu)*phi
    np.testing.assert_allclose(h[fluid], phi, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(raw[fluid], rho[:, None]*(u[fluid]-.5*acceleration[fluid]), rtol=5e-6, atol=2e-8)


def test_tiny_disconnected_corner_joins_neighbor_despite_large_tile_volume():
    from iceflow2d_v2.merge import CutCellMerge
    c, g, f, s = setup()
    volume = np.ones((c.nx, c.ny))
    volume[7, 8] = volume[8, 7] = 0
    volume[8, 8] = 2e-5
    edges = np.zeros((*volume.shape, 2))
    edges[:-1, :, 0] = np.minimum(volume[:-1], volume[1:])
    edges[:, :-1, 1] = np.minimum(volume[:, :-1], volume[:, 1:])
    g.capacity.from_numpy(volume); g.face_open.from_numpy(edges)
    q = np.zeros((*volume.shape, 6), np.float32)
    q[..., 0] = volume; q[..., 2] = volume
    q[8, 8, 2] = 0
    s.q.from_numpy(q); s.comp.fill(0)
    merger = CutCellMerge(c, g, s, f); merger.step()
    owner = merger.owner.to_numpy()[8, 8]
    assert tuple(owner) != (8, 8)
    same = np.all(merger.owner.to_numpy() == owner, axis=-1)
    assert volume[same].sum() >= 4
    after = s.q.to_numpy().astype(float)-s.comp.to_numpy()
    np.testing.assert_allclose(after.sum((0, 1)), q.astype(float).sum((0, 1)), atol=1e-9)
    assert after[8, 8, 2]/after[8, 8, 0] > .99


def test_connected_thin_film_merge_has_no_fixed_gather_radius():
    from iceflow2d_v2.merge import CutCellMerge
    c, g, f, s = setup()
    volume = np.zeros((c.nx, c.ny))
    # A volume valley separates two local maxima. An uphill-only attachment
    # stops at the first maximum and strands part of this connected film.
    volume[3:21, 25] = np.r_[np.linspace(.003, .01, 6), np.linspace(.008, 1e-7, 6), np.linspace(.001, .01, 6)]
    volume[21:24, 24:27] = 1
    edges = np.zeros((*volume.shape, 2))
    edges[:-1, :, 0] = np.minimum(volume[:-1], volume[1:])
    edges[:, :-1, 1] = np.minimum(volume[:, :-1], volume[:, 1:])
    g.capacity.from_numpy(volume); g.face_open.from_numpy(edges)
    q = np.zeros((*volume.shape, 6), np.float32)
    q[..., 0] = volume; q[..., 2] = .5*volume
    x, y = np.indices(volume.shape)
    q[..., 4] = volume*.001*(x-12); q[..., 5] = volume*.001*(y-20)
    s.q.from_numpy(q); s.comp.fill(0)
    merger = CutCellMerge(c, g, s, f); merger.step()
    owners = merger.owner.to_numpy()
    np.testing.assert_array_equal(owners[3, 25], owners[23, 25])
    after = s.q.to_numpy().astype(float)-s.comp.to_numpy()
    np.testing.assert_allclose(after.sum((0, 1)), q.astype(float).sum((0, 1)), rtol=2e-8, atol=1e-10)
    before_l = ((x+.5)*q[..., 5]-(y+.5)*q[..., 4]).sum()
    after_l = ((x+.5)*after[..., 5]-(y+.5)*after[..., 4]).sum()
    np.testing.assert_allclose(after_l, before_l, rtol=2e-7, atol=1e-8)


def test_empty_cell_roundoff_is_reported_without_fictitious_kinetic_energy():
    sim = Simulation(Config(nx=25, ny=50, warmup_steps=0))
    before = sim.diagnostics()
    q = sim.state.q.to_numpy()
    assert np.all(q[0, 0] == 0)
    q[0, 0, 4] = 1e-12
    sim.state.q.from_numpy(q)
    after = sim.diagnostics()
    momentum = float(q[0, 0, 4])*sim.cfg.mass_unit*sim.cfg.dx/sim.cfg.dt
    assert after['kinetic_energy_J_m'] == before['kinetic_energy_J_m']
    np.testing.assert_allclose(after['zero_mass_momentum_l1_kg_s'], momentum, rtol=1e-12)
    np.testing.assert_allclose(after['total_momentum_x'], momentum, rtol=1e-12)
    assert sim.state.q[0, 0][4] == q[0, 0, 4]


def test_long_cut_wall_keeps_nearby_full_receivers_for_each_patch():
    from iceflow2d_v2.merge import CutCellMerge
    c, g, f, s = setup()
    volume = np.zeros((c.nx, c.ny))
    volume[9:12, 9:42] = 1
    volume[8, 9:42] = .53
    edges = np.zeros((*volume.shape, 2))
    edges[:-1, :, 0] = np.minimum(volume[:-1], volume[1:])
    edges[:, :-1, 1] = np.minimum(volume[:, :-1], volume[:, 1:])
    g.capacity.from_numpy(volume); g.face_open.from_numpy(edges)
    q = np.zeros((*volume.shape, 6), np.float32); q[..., 0] = volume; q[..., 2] = volume
    s.q.from_numpy(q); s.comp.fill(0)
    merger = CutCellMerge(c, g, s, f); merger.step()
    owner = merger.owner.to_numpy()
    for j in range(9, 42):
        same = np.all(owner == owner[8, j], axis=-1)
        indices = np.argwhere(same)
        assert indices[:, 1].max()-indices[:, 1].min() <= 2
        assert volume[same].sum() >= 9


def test_cut_wall_corner_does_not_choose_a_collinear_full_receiver():
    """Reproduced growing transverse mode in a four-cell vertical wall strip."""
    from iceflow2d_v2.merge import CutCellMerge
    c, g, f, s = setup()
    volume = np.zeros((c.nx, c.ny))
    # Right boundary truncates the final 3x3 tile to a single column.
    volume[3:22, 12:45] = 1
    volume[3:22, 11] = .56
    edges = np.zeros((*volume.shape, 2))
    edges[:-1, :, 0] = np.minimum(volume[:-1], volume[1:])
    edges[:, :-1, 1] = np.minimum(volume[:, :-1], volume[:, 1:])
    g.capacity.from_numpy(volume); g.face_open.from_numpy(edges)
    q = np.zeros((*volume.shape, 6), np.float32)
    q[..., 0] = volume; q[..., 2] = volume
    s.q.from_numpy(q); s.comp.fill(0)
    merger = CutCellMerge(c, g, s, f); merger.step()
    owners = merger.owner.to_numpy()
    same = np.all(owners == owners[21, 11], axis=-1)
    members = np.argwhere(same)
    covariance = np.cov(members.T, aweights=volume[same], ddof=0)
    assert np.linalg.det(covariance) > .1, 'receiver still lies on one straight line'
    assert volume[same].sum() >= 4


def test_medium_cut_tiles_also_attach_to_a_fully_fluid_region():
    from iceflow2d_v2.merge import CutCellMerge
    c, g, f, s = setup()
    volume = np.zeros((c.nx, c.ny))
    volume[3:9, 9:18] = 1
    volume[9:15, 9:18] = .6
    edges = np.zeros((*volume.shape, 2))
    edges[:-1, :, 0] = np.minimum(volume[:-1], volume[1:])
    edges[:, :-1, 1] = np.minimum(volume[:, :-1], volume[:, 1:])
    g.capacity.from_numpy(volume); g.face_open.from_numpy(edges)
    q = np.zeros((*volume.shape, 6), np.float32)
    q[..., 0] = volume; q[..., 2] = volume
    s.q.from_numpy(q); s.comp.fill(0)
    merger = CutCellMerge(c, g, s, f); merger.step()
    owner = merger.owner.to_numpy()
    for i in range(9, 15):
        for j in range(9, 18):
            same = np.all(owner == owner[i, j], axis=-1)
            assert np.count_nonzero(volume[same] == 1) >= 4
