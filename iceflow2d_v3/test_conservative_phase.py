"""CUDA invariants for bounded dual-LBM conservation and geometry events."""

from dataclasses import replace
import math
import unittest

import numpy as np

from .simulator import IceFlow2D, ensure_taichi_cuda
from .test_iceflow import _falling_config


def config(**kwargs):
    values = dict(water_mass_conservation="conservative_lbm", phase_momentum_coupling="consistent_flux", sigma=0.0, phase_warmup_steps=10)
    values.update(kwargs)
    return _falling_config(**values)


def water(s):
    return float(s.phase_populations.to_numpy().sum(dtype=np.float64))


def ice(s):
    return float(s.thermal.body_solid_mass.to_numpy().sum(dtype=np.float64)) / (s.config.rho_water * s.config.dx**2)


class ConservativePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_taichi_cuda()

    def assert_state(self, s, total, atol=2e-9):
        self.assertAlmostEqual(water(s) + ice(s), total, delta=atol)
        phi = s.water_phase.to_numpy()
        self.assertTrue(np.isfinite(phi).all())
        active = s.conservative_phase.capacity.to_numpy() > 1e-14
        self.assertGreaterEqual(phi[active].min(), -2e-12)
        self.assertLessEqual(phi[active].max(), 1 + 2e-12)
        np.testing.assert_allclose(
            s.phase_populations.to_numpy().sum(axis=-1),
            s.thermal.water_volume_m2.to_numpy() / s.config.dx**2,
            atol=2e-12, rtol=2e-12,
        )

    def test_rotated_geometry_and_prescribed_motion(self):
        s = IceFlow2D(config())
        total = water(s) + ice(s)
        for k in range(8):
            s.body_reference_origin[None] += np.array([0.17, -0.29], np.float32)
            s.body_angle[None] = math.radians(5 + 7 * k)
            s._refresh_fluid_domain(resolve_contact=True)
            self.assert_state(s, total)
            expected = s.thermal.body_solid_mass.to_numpy().sum() / s.thermal._initial_body_cell_mass
            self.assertAlmostEqual(s.conservative_phase.ice_volume.to_numpy().sum(), expected, delta=2e-11)

    def test_initialization_clears_water_inside_solids(self):
        s = IceFlow2D(config(phase_warmup_steps=0))
        s.water_phase.fill(0.25)
        s._initialize_hydrostatic_equilibrium()
        closed = s.conservative_phase.capacity.to_numpy() == 0
        np.testing.assert_array_equal(s.water_phase.to_numpy()[closed], 0)
        np.testing.assert_array_equal(s.phase_populations.to_numpy()[closed], 0)

    def test_eroded_material_wall_support_keeps_mass_in_container(self):
        s = IceFlow2D(config(phase_warmup_steps=0))
        # Retain a subthreshold tile outside the resolved core. It still has
        # thermal mass and must not pass through a container wall.
        fractions = s.thermal.body_solid_fraction.to_numpy()
        fractions[fractions > 0] = 0.1
        fractions[fractions.shape[0] // 2, fractions.shape[1] // 2] = 1.0
        s.thermal.body_solid_mass.from_numpy(fractions * s.thermal._initial_body_cell_mass)
        s._update_moving_body_mass_properties()
        total = water(s) + ice(s)
        s.body_reference_origin[None] += np.array([-30.0, 0], np.float32)
        s._refresh_fluid_domain(resolve_contact=True)
        s.conservative_phase.project_melting(0.0)
        self.assert_state(s, total)
        self.assertAlmostEqual(s.conservative_phase.ice_volume.to_numpy().sum(), ice(s) * s.config.rho_water / s.config.rho_ice, delta=2e-10)

    def test_coupled_steps_preserve_total_and_energy_transport(self):
        s = IceFlow2D(config())
        total = water(s) + ice(s)
        for _ in range(8):
            s.step(16)
            self.assert_state(s, total)
        from .reporting import _capture_snapshot, snapshot_arrays
        snapshot = _capture_snapshot(s, initial_solid_volume_cells=64)
        arrays = snapshot_arrays(s.config, [snapshot])
        np.testing.assert_allclose(arrays["water_volume_cells"][0].T, s.phase_populations.to_numpy().sum(axis=-1), atol=2e-12)
        self.assertAlmostEqual(snapshot.water_volume_current_cells, water(s), delta=2e-10)
        self.assertAlmostEqual(arrays["water_sensible_energy_j_m"].sum(), s.thermal.water_sensible_energy.to_numpy().sum(), delta=1e-8)

    def test_pure_water_circulation_transports_heat(self):
        s = IceFlow2D(config(phase_warmup_steps=0, gravity=(0, 0)))
        velocity = np.zeros((s.nx, s.ny, 2), dtype=np.float32)
        velocity[..., 0] = 0.01
        s.momentum_velocity_lattice.from_numpy(velocity)
        volumes = s.phase_populations.to_numpy().sum(axis=-1)
        temperatures = np.broadcast_to(np.arange(s.nx)[:, None] + 20.0, volumes.shape)
        scale = s.config.rho_water * s.config.thermal.properties.specific_heat_water_j_kg_k * s.config.dx**2
        energies = scale * volumes * temperatures
        s.thermal.water_sensible_energy.from_numpy(energies)
        s._collide_phase()
        s.conservative_phase.prepare_transport()
        s.conservative_phase.begin_fluid(1)
        s._stream()
        s.conservative_phase.limit_stream()
        s.conservative_phase.finish_energy()
        s._update_streamed_macroscopic_fields()
        s.conservative_phase.repair_bounds()
        self.assertAlmostEqual(water(s), volumes.sum(), delta=2e-10)
        after = s.thermal.water_sensible_energy.to_numpy()
        self.assertLess(after[16, 10], energies[16, 10] - 1e-5)
        self.assertAlmostEqual(after.sum(), energies.sum(), delta=1e-8)

    def test_isothermal_geometry_transfers_preserve_temperature(self):
        s = IceFlow2D(config(ice_base_y_cells=9.0, gravity=(0, 0)))
        energy = s.thermal.water_sensible_energy.to_numpy().sum()
        s.conservative_phase.cache_momentum_phase()
        s.body_reference_origin[None] += np.array([0.21, -0.12], np.float32)
        s.body_angle[None] += 0.1
        s._refresh_fluid_domain(resolve_contact=True)
        volume = s.thermal.water_volume_m2.to_numpy()
        actual = s.thermal.water_temperature.to_numpy()
        np.testing.assert_allclose(actual[volume > 1e-14], 90.0, atol=2e-9)
        self.assertAlmostEqual(s.thermal.water_sensible_energy.to_numpy().sum(), energy, delta=1e-8)

    def test_partial_and_full_melting_at_different_density_ratios(self):
        for density in (500.0, 917.0, 1000.0):
            with self.subTest(density=density):
                s = IceFlow2D(config(rho_ice=density, ice_base_y_cells=9.0, ice_initial_angle=0.0))
                total = water(s) + ice(s)
                initial_water, initial_ice = water(s), ice(s)
                s.thermal._body_heat_delta.fill(0.2 * s.thermal._initial_body_cell_mass * s.thermal._latent_heat_j_kg)
                s.thermal._apply_body_heat_and_phase_change()
                s.thermal.last_thermal_substep_s = s._time_step_s
                s._refresh_eroded_body()
                self.assert_state(s, total)
                self.assertAlmostEqual(water(s) - initial_water, initial_ice - ice(s), delta=2e-10)
                self.assertGreater(ice(s), 0)
                self.assertAlmostEqual(s.conservative_phase.melt_rate.to_numpy().sum(), initial_ice - ice(s), delta=2e-10)
                s.thermal._body_heat_delta.fill(s.thermal._initial_body_cell_mass * s.thermal._latent_heat_j_kg)
                s.thermal._apply_body_heat_and_phase_change()
                s._refresh_eroded_body()
                self.assert_state(s, total)
                self.assertEqual(ice(s), 0)
                self.assertAlmostEqual(water(s), total, delta=2e-10)

    def test_air_melting_generates_cold_water_and_retains_excess_heat(self):
        s = IceFlow2D(config(ice_initial_angle=0.0))
        total = water(s) + ice(s)
        energy = s.thermal.water_sensible_energy.to_numpy().sum()
        latent = s.thermal._latent_heat_j_kg
        mass = s.thermal._initial_body_cell_mass
        s.thermal._body_heat_delta.fill(2 * mass * latent)
        s.thermal._apply_body_heat_and_phase_change()
        extra = s.thermal.melted_sensible_energy.to_numpy().sum()
        s.thermal.last_thermal_substep_s = s._time_step_s
        s._refresh_eroded_body()
        self.assert_state(s, total)
        self.assertAlmostEqual(s.thermal.water_sensible_energy.to_numpy().sum(), energy + extra, delta=1e-8)

    def test_d2q5_thermal_also_preserves_mass(self):
        cfg = config()
        cfg.thermal = replace(cfg.thermal, scheme="enthalpy_d2q5")
        s = IceFlow2D(cfg)
        total = water(s) + ice(s)
        s.step(16)
        self.assert_state(s, total)

    def test_closed_components_never_exchange_water(self):
        s = IceFlow2D(config(phase_warmup_steps=0))
        c = s.conservative_phase
        capacity = c.capacity.to_numpy()
        capacity[16, :] = 0
        c.capacity.from_numpy(capacity)
        c.previous_capacity.from_numpy(capacity)
        h = np.zeros((s.nx, s.ny, 9), dtype=np.float64)
        h[:, :20, 0] = 0.5 * capacity[:, :20]
        # Overcapacity on the left creates a local redistribution event.
        h[8, 8, 0] = 1.2
        s.phase_populations.from_numpy(h)
        c.recover(0)
        before_left, before_right = h[:16].sum(), h[17:].sum()
        c.remap()
        after = s.phase_populations.to_numpy()
        self.assertAlmostEqual(after[:16].sum(), before_left, delta=2e-11)
        self.assertAlmostEqual(after[17:].sum(), before_right, delta=2e-11)

    def test_disconnected_drops_sharing_air_never_pay_each_others_events(self):
        s = IceFlow2D(config(phase_warmup_steps=0))
        c = s.conservative_phase
        h = np.zeros((s.nx, s.ny, 9), dtype=np.float64)
        h[4:10, 5:11, 0] = 0.5
        h[21:27, 5:11, 0] = 0.5
        h[7, 8, 0] = 1.2
        s.phase_populations.from_numpy(h)
        c.recover(0)
        before = h[21:27, 5:11].copy()
        c.repair_bounds()
        np.testing.assert_array_equal(s.phase_populations.to_numpy()[21:27, 5:11], before)
        self.assertAlmostEqual(water(s), h.sum(), delta=2e-10)


if __name__ == "__main__":
    unittest.main()
