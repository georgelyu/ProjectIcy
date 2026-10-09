"""CUDA regressions for ice erosion without liquid or momentum sources."""

from dataclasses import replace
import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.config import LatticeScales, ThermalConfig
from iceflow2d_v3.simulator import MovingBodyThermal2D, ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config


def _submerged_config(**overrides):
    values = dict(
        ice_base_y_cells=8.0,
        ice_initial_angle=0.0,
        gravity=(0.0, 0.0),
        sigma=0.0,
        phase_warmup_steps=10,
        thermal=ThermalConfig(update_interval_lbm_steps=1),
    )
    values.update(overrides)
    return _falling_config(**values)


class MeltingCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_unresolved_remnant_keeps_thermal_mass_without_free_fall(self):
        simulation = IceFlow2D(_submerged_config(gravity=(0.0, -9.8)))
        thermal = simulation.thermal
        mass = np.zeros_like(thermal.body_solid_mass.to_numpy())
        mass[3, 3] = 0.2 * thermal._initial_body_cell_mass
        thermal.body_solid_mass.from_numpy(mass)
        thermal.body_sensible_energy.fill(0.0)
        simulation._refresh_eroded_body()
        self.assertFalse(np.any(simulation.solid_mask.to_numpy()))
        self.assertEqual(simulation._body_has_resolved_boundary[None], 0)
        self.assertGreater(simulation.body_mass_lattice[None], 0.0)
        self.assertGreater(thermal.world_body_solid_fraction.to_numpy().max(), 0.0)
        pose = np.asarray(simulation.body_reference_origin[None]).copy()
        mass_before = thermal.body_solid_mass.to_numpy().sum()
        simulation.body_velocity[None] = (0.03, -0.08)
        simulation.body_angular_velocity[None] = 0.02
        simulation._integrate_rigid_ice()
        np.testing.assert_array_equal(simulation.body_reference_origin[None], pose)
        np.testing.assert_array_equal(simulation.body_velocity[None], (0.0, 0.0))
        self.assertEqual(simulation.body_angular_velocity[None], 0.0)
        self.assertEqual(thermal.body_solid_mass.to_numpy().sum(), mass_before)
        # Heat still crosses fractional coverage even without a sharp ice node.
        simulation._advance_moving_thermal_slow(32)
        self.assertLess(thermal.body_solid_mass.to_numpy().sum(), mass_before)
        np.testing.assert_array_equal(simulation.body_reference_origin[None], pose)

    def test_partial_melting_changes_only_material_state(self):
        simulation = IceFlow2D(_submerged_config())
        thermal = simulation.thermal
        water = thermal.water_volume_m2.to_numpy()
        energy = thermal.water_sensible_energy.to_numpy()
        phase = simulation.water_phase.to_numpy()
        populations = simulation.momentum_populations.to_numpy()
        mass = thermal._initial_body_cell_mass
        thermal._body_heat_delta.fill(0.0)
        thermal.body_sensible_energy[0, 0] = -mass * thermal._ice_specific_heat * 10.0
        thermal._body_heat_delta[0, 0] = mass * (
            thermal._ice_specific_heat * 10.0 + 0.5 * thermal._latent_heat_j_kg
        )
        thermal._apply_body_heat_and_phase_change()
        self.assertAlmostEqual(thermal.body_solid_mass[0, 0], 0.5 * mass, delta=1e-16)
        self.assertEqual(thermal.body_temperature[0, 0], thermal._melting_temperature_c)
        np.testing.assert_array_equal(thermal.water_volume_m2.to_numpy(), water)
        np.testing.assert_array_equal(thermal.water_sensible_energy.to_numpy(), energy)
        np.testing.assert_array_equal(simulation.water_phase.to_numpy(), phase)
        np.testing.assert_array_equal(
            simulation.momentum_populations.to_numpy(), populations
        )

    def test_full_erosion_restores_the_surrounding_bath_without_material_sources(self):
        simulation = IceFlow2D(_submerged_config())
        thermal = simulation.thermal
        old_solid = simulation.solid_mask.to_numpy() != 0
        old_fluid = ~old_solid & (simulation.wall_mask.to_numpy() == 0)
        phase = simulation.water_phase.to_numpy()
        populations = simulation.momentum_populations.to_numpy()
        volume = thermal.water_volume_m2.to_numpy()
        # Sufficient prescribed material heat erases the complete body in one update.
        thermal.body_sensible_energy.fill(
            2.0 * thermal._initial_body_cell_mass * thermal._latent_heat_j_kg
        )
        simulation._advance_moving_thermal_slow(1)
        self.assertFalse(simulation.body_active[None])
        self.assertEqual(simulation.body_mass_lattice[None], 0.0)
        self.assertEqual(simulation.body_inertia_lattice[None], 0.0)
        self.assertFalse(np.any(simulation.solid_mask.to_numpy()))
        self.assertFalse(np.any(thermal.body_solid_mass.to_numpy()))
        self.assertFalse(np.any(thermal.world_body_solid_fraction.to_numpy()))
        result_phase = simulation.water_phase.to_numpy()
        np.testing.assert_allclose(result_phase[old_solid], 1.0, atol=1e-6)
        np.testing.assert_array_equal(result_phase[old_fluid], phase[old_fluid])
        np.testing.assert_array_equal(
            thermal.water_volume_m2.to_numpy()[old_fluid], volume[old_fluid]
        )
        np.testing.assert_allclose(
            thermal.water_volume_m2.to_numpy()[old_solid], simulation.config.dx**2
        )
        # The middle of the old body is four cells from existing water. It
        # must inherit the bath temperature instead of creating a cold pocket.
        self.assertGreater(thermal.water_temperature.to_numpy()[old_solid].min(), 19.0)
        np.testing.assert_array_equal(
            simulation.momentum_populations.to_numpy()[old_fluid],
            populations[old_fluid],
        )
        np.testing.assert_allclose(
            simulation.phase_populations.to_numpy()[old_solid].sum(axis=-1),
            1.0,
            atol=1e-6,
        )
        simulation.step(4)
        self.assertFalse(simulation.body_active[None])
        self.assertTrue(np.isfinite(simulation.temperature.to_numpy()).all())

    def test_erosion_in_a_quiet_bath_does_not_launch_capillary_waves(self):
        simulation = IceFlow2D(
            _submerged_config(
                water_height_fraction=46.0 / 48.0,
                phase_warmup_steps=0,
                sigma=0.072,
            )
        )
        thermal = simulation.thermal
        old_solid = simulation.solid_mask.to_numpy() != 0
        thermal.body_sensible_energy[0, 3] = (
            2.0 * thermal._initial_body_cell_mass * thermal._latent_heat_j_kg
        )
        simulation._advance_moving_thermal_slow(1)
        exposed = old_solid & (simulation.solid_mask.to_numpy() == 0)
        self.assertTrue(exposed.any())
        np.testing.assert_allclose(simulation.water_phase.to_numpy()[exposed], 1.0)
        simulation.step(32)
        speed = (
            np.linalg.norm(simulation.physical_velocity_lattice.to_numpy(), axis=-1)
            * simulation.scales.velocity_scale_m_s
        )
        self.assertLess(speed.max(), 1e-6)

    def test_forced_erosion_in_air_does_not_create_water(self):
        simulation = IceFlow2D(_falling_config(phase_warmup_steps=0))
        thermal = simulation.thermal
        phase = simulation.water_phase.to_numpy()
        old_solid = simulation.solid_mask.to_numpy() != 0
        thermal.body_sensible_energy.fill(
            2.0 * thermal._initial_body_cell_mass * thermal._latent_heat_j_kg
        )
        simulation._advance_moving_thermal_slow(1)
        self.assertFalse(simulation.body_active[None])
        np.testing.assert_array_equal(simulation.water_phase.to_numpy(), phase)
        np.testing.assert_array_equal(
            thermal.water_volume_m2.to_numpy()[old_solid], 0.0
        )

    def test_eroded_box_edge_cannot_leave_ghost_solid_cells(self):
        simulation = IceFlow2D(_submerged_config())
        simulation.body_reference_origin[None] = (16.5, 12.5)
        mass = np.zeros((8, 8), dtype=np.float64)
        mass[2:6, 2:6] = simulation.thermal._initial_body_cell_mass
        simulation.thermal.body_solid_mass.from_numpy(mass)
        simulation._refresh_eroded_body()
        coverage = simulation.thermal.world_body_solid_fraction.to_numpy()
        expected = (coverage >= simulation.config.thermal.solid_liquid_threshold) & (
            simulation.wall_mask.to_numpy() == 0
        )
        np.testing.assert_array_equal(simulation.solid_mask.to_numpy() != 0, expected)
        # This cell center lies exactly on the former rectangle's right edge.
        self.assertEqual(coverage[20, 12], 0.0)
        self.assertGreater(simulation.body_signed_distance_m[20, 12], 0.0)

    def test_every_diffusion_substep_commits_erosion_before_further_heat_exchange(self):
        simulation = IceFlow2D(_submerged_config())
        thermal = simulation.thermal
        thermal.body_sensible_energy.fill(
            2.0 * thermal._initial_body_cell_mass * thermal._latent_heat_j_kg
        )
        refresh = simulation._refresh_eroded_body
        states = []

        def refresh_and_observe():
            refresh()
            states.append(
                (
                    bool(simulation.body_active[None]),
                    simulation.solid_mask.to_numpy().copy(),
                    thermal.water_volume_m2.to_numpy().copy(),
                )
            )

        simulation._refresh_eroded_body = refresh_and_observe
        steps = int(
            np.ceil(
                2.1 * thermal.maximum_diffusion_time_step_s / simulation.scales.dt_s
            )
        )
        simulation._advance_moving_thermal_slow(steps)
        self.assertEqual(len(states), 3)
        for active, solid, volume in states:
            self.assertFalse(active)
            self.assertFalse(solid.any())
            np.testing.assert_allclose(volume[12:20, 8:16], simulation.config.dx**2)

    def test_hot_water_erodes_ice_and_multirate_steps_stay_synchronized(self):
        simulation = IceFlow2D(
            _submerged_config(thermal=ThermalConfig(update_interval_lbm_steps=4))
        )
        initial = simulation.body_mass_lattice[None]
        simulation.step(3)
        self.assertEqual(simulation.body_mass_lattice[None], initial)
        self.assertEqual(simulation.thermal.steps, 0)
        simulation.step()
        self.assertLess(simulation.body_mass_lattice[None], initial)
        self.assertAlmostEqual(simulation.thermal.time_s, simulation.physical_time_s)
        simulation.step(4)
        self.assertAlmostEqual(simulation.thermal.time_s, simulation.physical_time_s)
        temperature = simulation.temperature.to_numpy()
        self.assertTrue(np.isfinite(temperature).all())
        self.assertGreaterEqual(temperature.min(), -1e-8)
        self.assertLessEqual(temperature.max(), 90.0 + 1e-8)


class LocalThermalCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def _bath(self, config=ThermalConfig()):
        thermal = MovingBodyThermal2D(
            24,
            20,
            1,
            1,
            config,
            LatticeScales(1e-4, 1e-5, 1000.0),
            density_water_kg_m3=1000.0,
            density_ice_kg_m3=917.0,
        )
        phase = ti.field(ti.f32, shape=(24, 20))
        wall = ti.field(ti.i8, shape=(24, 20))
        solid = ti.field(ti.i8, shape=(24, 20))
        origin = ti.Vector.field(2, ti.f32, shape=())
        angle = ti.field(ti.f32, shape=())
        origin[None] = (6.5, 8.5)
        return thermal, phase, wall, solid, origin, angle

    def test_aperture_remap_uses_local_temperature_without_global_correction(self):
        thermal, phase, wall, solid, _, _ = self._bath()
        old_phase = np.zeros((24, 20), dtype=np.float32)
        old_phase[3:10, 3:13] = 0.8
        old_phase[14:21, 3:13] = 0.8
        old_phase[6, 8] = 0.0
        volume = old_phase.astype(np.float64) * 1e-8
        temperature = np.full((24, 20), 90.0)
        temperature[:12] = 5.0
        thermal.water_volume_m2.from_numpy(volume)
        thermal.water_sensible_energy.from_numpy(1000.0 * 4186.0 * volume * temperature)
        new_phase = old_phase.copy()
        new_phase[6, 8] = 0.8
        new_phase[17, 13] = 0.8
        new_phase[3:6, 3:7] = 0.0
        phase.from_numpy(new_phase)
        thermal.synchronize_water_aperture(phase, wall, solid)
        result = thermal.water_temperature.to_numpy()
        self.assertAlmostEqual(result[6, 8], 5.0, delta=1e-8)
        self.assertAlmostEqual(result[17, 13], 90.0, delta=1e-8)
        np.testing.assert_allclose(
            thermal.water_volume_m2.to_numpy(),
            new_phase.astype(np.float64) * 1e-8,
            atol=1e-22,
        )
        # Even losing every liquid aperture needs no capacity/total-balance check.
        phase.fill(0.0)
        thermal.synchronize_water_aperture(phase, wall, solid)
        self.assertFalse(np.any(thermal.water_volume_m2.to_numpy()))
        self.assertFalse(np.any(thermal.water_sensible_energy.to_numpy()))

    def test_hot_bath_completely_melts_material_without_creating_water(self):
        thermal, phase, wall, solid, origin, angle = self._bath(
            ThermalConfig(initial_ice_temperature_c=-10.0)
        )
        phase.fill(1.0)
        solid[6, 8] = 1
        thermal.initialize(phase, wall, solid)
        thermal.world_body_solid_fraction[6, 8] = 1.0
        volume = thermal.water_volume_m2.to_numpy()
        for _ in range(500):
            thermal.advance_slow(
                thermal.maximum_diffusion_time_step_s, phase, wall, solid, origin, angle
            )
            if thermal.body_solid_mass[0, 0] == 0.0:
                break
        self.assertEqual(thermal.body_solid_mass[0, 0], 0.0)
        self.assertEqual(thermal.body_sensible_energy[0, 0], 0.0)
        np.testing.assert_array_equal(thermal.water_volume_m2.to_numpy(), volume)
        wet = volume > 0.0
        temperatures = thermal.water_temperature.to_numpy()[wet]
        self.assertGreaterEqual(temperatures.min(), -1e-8)
        self.assertLess(temperatures.min(), 90.0)
        self.assertLessEqual(temperatures.max(), 90.0 + 1e-8)

    def test_insulated_dry_ice_does_not_create_liquid(self):
        thermal, phase, wall, solid, origin, angle = self._bath(
            replace(ThermalConfig(), initial_air_temperature_c=90.0)
        )
        thermal.initialize(phase, wall, solid)
        mass = thermal.body_solid_mass.to_numpy()
        thermal.advance_slow(
            thermal.maximum_diffusion_time_step_s, phase, wall, solid, origin, angle
        )
        np.testing.assert_array_equal(thermal.body_solid_mass.to_numpy(), mass)
        self.assertFalse(np.any(thermal.water_volume_m2.to_numpy()))


if __name__ == "__main__":
    unittest.main()
