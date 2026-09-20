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

    def test_full_erosion_leaves_empty_cells_and_no_fluid_sources(self):
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
        np.testing.assert_array_equal(result_phase[old_solid], 0.0)
        np.testing.assert_array_equal(result_phase[old_fluid], phase[old_fluid])
        np.testing.assert_array_equal(thermal.water_volume_m2.to_numpy(), volume)
        np.testing.assert_array_equal(
            simulation.momentum_populations.to_numpy()[old_fluid],
            populations[old_fluid],
        )
        np.testing.assert_allclose(
            simulation.phase_populations.to_numpy()[old_solid].sum(axis=-1),
            0.0,
            atol=1e-12,
        )
        simulation.step(4)
        self.assertFalse(simulation.body_active[None])
        self.assertTrue(np.isfinite(simulation.temperature.to_numpy()).all())

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
