"""Physical and coupling regressions for optional D2Q5 heat transport."""

import contextlib
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.config import (
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
)
from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as example
from iceflow2d_v3.simulator import MovingBodyThermal2D, ensure_taichi_cuda
from iceflow2d_v3.test_thermal_coupling import _submerged_config


class ThermalSchemeConfigTests(unittest.TestCase):
    def test_default_and_explicit_cli_schemes_reach_serialized_config(self):
        self.assertEqual(ThermalConfig().scheme, "enthalpy_fv")
        self.assertEqual(example.create_config().thermal.scheme, "enthalpy_fv")
        for scheme in ("enthalpy_fv", "enthalpy_d2q5"):
            config = example.create_config(
                example.parse_args(["--thermal-scheme", scheme])
            )
            self.assertEqual(config.thermal.scheme, scheme)
            self.assertEqual(config.to_dict()["thermal"]["scheme"], scheme)

    def test_unknown_scheme_is_rejected_in_config_and_cli(self):
        with self.assertRaisesRegex(ValueError, "scheme"):
            ThermalConfig(scheme="unknown")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            example.parse_args(["--thermal-scheme", "unknown"])


class D2Q5ThermalCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def _bath(self, config=None, *, shape=(24, 20), body_shape=(4, 3)):
        config = config or ThermalConfig(
            scheme="enthalpy_d2q5", initial_ice_temperature_c=-10.0
        )
        thermal = MovingBodyThermal2D(
            *shape,
            *body_shape,
            config,
            LatticeScales(1e-4, 1e-5, 1000.0),
            density_water_kg_m3=1000.0,
            density_ice_kg_m3=917.0,
            water_phase_cutoff=1e-3,
        )
        phase = ti.field(ti.f32, shape=shape)
        wall = ti.field(ti.i8, shape=shape)
        solid = ti.field(ti.i8, shape=shape)
        origin = ti.Vector.field(2, ti.f32, shape=())
        angle = ti.field(ti.f32, shape=())
        origin[None] = (6.5, 8.5)
        phase.fill(1.0)
        walls = np.zeros(shape, dtype=np.int8)
        walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = 1
        wall.from_numpy(walls)
        thermal.initialize(phase, wall, solid)
        return thermal, phase, wall, solid, origin, angle

    def _assert_moments(self, thermal):
        for populations, energy in (
            (thermal._d2q5.water_populations, thermal.water_sensible_energy),
            (thermal._d2q5.body_populations, thermal.body_sensible_energy),
        ):
            np.testing.assert_allclose(
                populations.to_numpy().sum(axis=-1),
                energy.to_numpy(),
                rtol=2e-14,
                atol=2e-14,
            )

    def test_uniform_temperature_survives_partial_apertures_and_dry_cells(self):
        thermal, phase, wall, solid, origin, angle = self._bath()
        fractions = (
            np.linspace(0.0, 1.0, thermal.nx * thermal.ny)
            .reshape(thermal.nx, thermal.ny)
            .astype(np.float32)
        )
        fractions[5:8, 6:9] = 0.0
        phase.from_numpy(fractions)
        solid[12, 8] = 1
        thermal.initialize(phase, wall, solid)
        before = thermal.water_sensible_energy.to_numpy()
        body_before = thermal.body_sensible_energy.to_numpy()
        self._assert_moments(thermal)
        thermal.advance_slow(
            2.3 * thermal.maximum_diffusion_time_step_s,
            phase,
            wall,
            solid,
            origin,
            angle,
        )
        np.testing.assert_allclose(thermal.water_sensible_energy.to_numpy(), before)
        np.testing.assert_allclose(thermal.body_sensible_energy.to_numpy(), body_before)
        wet = thermal.water_volume_m2.to_numpy() > 0
        np.testing.assert_allclose(thermal.water_temperature.to_numpy()[wet], 20.0)
        self.assertFalse(np.any(thermal._d2q5.water_populations.to_numpy()[~wet]))
        self._assert_moments(thermal)

    def test_diffusion_matches_analytic_cosine_decay_and_conserves_heat(self):
        config = ThermalConfig(
            scheme="enthalpy_d2q5",
            properties=PhaseChangeProperties(conductivity_water_w_m_k=4.0),
        )
        thermal, phase, wall, solid, origin, angle = self._bath(config, shape=(26, 6))
        volume = thermal.water_volume_m2.to_numpy()
        profile = 30.0 + 10.0 * np.cos(np.pi * (np.arange(26) - 0.5) / 24)
        energy = 1000.0 * 4186.0 * volume * profile[:, None]
        thermal.water_sensible_energy.from_numpy(energy)
        dt = thermal.maximum_diffusion_time_step_s
        for _ in range(100):
            thermal.advance_slow(dt, phase, wall, solid, origin, angle)
        alpha = 4.0 / (1000.0 * 4186.0)
        attenuation = np.exp(-alpha * (np.pi / (24e-4)) ** 2 * 100 * dt)
        expected = 30.0 + attenuation * (profile[1:-1] - 30.0)
        np.testing.assert_allclose(
            thermal.water_temperature.to_numpy()[1:-1, 2],
            expected,
            atol=0.003,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            thermal.water_sensible_energy.to_numpy().sum(), energy.sum(), delta=1e-11
        )
        self._assert_moments(thermal)

    def test_d2q5_matches_fv_for_cut_cells_ice_and_all_wall_conditions(self):
        config = ThermalConfig(
            properties=PhaseChangeProperties(melting_temperature_c=5.0),
            initial_ice_temperature_c=-10.0,
            boundaries=ThermalBoundarySet(
                left=ThermalBoundary.dirichlet(60.0),
                right=ThermalBoundary.dirichlet(40.0),
                bottom=ThermalBoundary.dirichlet(25.0),
                top=ThermalBoundary.dirichlet(10.0),
            ),
        )
        rng = np.random.default_rng(140)
        fractions = rng.uniform(0.0011, 1.0, (24, 20)).astype(np.float32)
        fractions[5:8, 6:9] = 0.0
        temperatures = rng.uniform(5.0, 55.0, (24, 20))
        mass_fractions = rng.uniform(0.1, 1.0, (4, 3))
        mass_fractions[0, 0] = 0.0
        ice_temperatures = rng.uniform(-20.0, 0.0, (4, 3))
        results = []
        for scheme in ("enthalpy_fv", "enthalpy_d2q5"):
            thermal, phase, wall, solid, origin, angle = self._bath(
                replace(config, scheme=scheme)
            )
            phase.from_numpy(fractions)
            solid[12, 8] = 1
            thermal.initialize(phase, wall, solid)
            volume = thermal.water_volume_m2.to_numpy()
            thermal.water_sensible_energy.from_numpy(
                volume * 1000.0 * 4186.0 * (temperatures - 5.0)
            )
            mass = mass_fractions * thermal._initial_body_cell_mass
            thermal.body_solid_mass.from_numpy(mass)
            thermal.body_sensible_energy.from_numpy(
                mass * 2100.0 * (ice_temperatures - 5)
            )
            with contextlib.ExitStack() as stack:
                if scheme == "enthalpy_d2q5":
                    for method in (
                        "_accumulate_body_conduction",
                        "_compute_water_conduction_flux_x",
                        "_compute_water_conduction_flux_y",
                        "_update_water_conduction",
                    ):
                        stack.enter_context(
                            patch.object(
                                thermal, method, side_effect=AssertionError("FV called")
                            )
                        )
                else:
                    self.assertIsNone(thermal._d2q5)
                count = thermal.advance_slow(
                    2.3 * thermal.maximum_diffusion_time_step_s,
                    phase,
                    wall,
                    solid,
                    origin,
                    angle,
                )
            self.assertEqual(count, 3)
            results.append(
                (
                    thermal.water_sensible_energy.to_numpy(),
                    thermal.body_sensible_energy.to_numpy(),
                    thermal.body_solid_mass.to_numpy(),
                )
            )
            if scheme == "enthalpy_d2q5":
                self._assert_moments(thermal)
        for finite_volume, distributions in zip(*results):
            np.testing.assert_allclose(
                distributions, finite_volume, rtol=2e-12, atol=1e-13
            )

    def test_wall_heat_matches_half_cell_dirichlet_flux(self):
        config = ThermalConfig(
            scheme="enthalpy_d2q5",
            properties=PhaseChangeProperties(melting_temperature_c=5.0),
            initial_water_temperature_c=20.0,
            boundaries=ThermalBoundarySet(
                left=ThermalBoundary.dirichlet(70.0),
                right=ThermalBoundary.dirichlet(40.0),
                bottom=ThermalBoundary.dirichlet(30.0),
                top=ThermalBoundary.dirichlet(10.0),
            ),
        )
        thermal, phase, wall, solid, origin, angle = self._bath(config)
        dt = thermal.maximum_diffusion_time_step_s
        before = thermal.water_sensible_energy.to_numpy()
        thermal.advance_slow(dt, phase, wall, solid, origin, angle)
        expected = np.zeros_like(before)
        for wall_slice, temperature in (
            ((1, slice(1, -1)), 70.0),
            ((-2, slice(1, -1)), 40.0),
            ((slice(1, -1), 1), 30.0),
            ((slice(1, -1), -2), 10.0),
        ):
            expected[wall_slice] += 2.0 * 0.6 * dt * (temperature - 20.0)
        np.testing.assert_allclose(
            thermal.water_sensible_energy.to_numpy() - before, expected, atol=2e-16
        )

    def test_remap_and_advection_keep_distribution_energy_synchronized(self):
        thermal, phase, wall, solid, origin, angle = self._bath()
        thermal.advance_slow(thermal.scales.dt_s, phase, wall, solid, origin, angle)
        velocity = ti.Vector.field(2, ti.f32, shape=(thermal.nx, thermal.ny))
        velocity.fill((0.05, 0.02))
        phase[5, 5] = 0.0
        phase[6, 5] = 0.2
        thermal.advance_fast(
            thermal.scales.dt_s,
            velocity,
            phase,
            wall,
            solid,
            maximum_outflow_rate_lattice=0.2,
        )
        self._assert_moments(thermal)
        np.testing.assert_array_equal(thermal._d2q5.water_populations[5, 5], 0.0)
        phase[5, 5] = 1.0
        thermal.synchronize_water_aperture(phase, wall, solid)
        self._assert_moments(thermal)
        self.assertAlmostEqual(thermal.water_temperature[5, 5], 20.0, delta=1e-12)

    def test_hot_water_completely_erodes_ice_without_adding_liquid(self):
        thermal, phase, wall, solid, origin, angle = self._bath(body_shape=(1, 1))
        solid[6, 8] = 1
        thermal.initialize(phase, wall, solid)
        thermal.world_body_solid_fraction[6, 8] = 1.0
        volume = thermal.water_volume_m2.to_numpy()
        for _ in range(500):
            thermal.advance_slow(
                thermal.maximum_diffusion_time_step_s,
                phase,
                wall,
                solid,
                origin,
                angle,
            )
            if thermal.body_solid_mass[0, 0] == 0.0:
                break
        self.assertEqual(thermal.body_solid_mass[0, 0], 0.0)
        self.assertFalse(np.any(thermal._d2q5.body_populations.to_numpy()))
        np.testing.assert_array_equal(thermal.water_volume_m2.to_numpy(), volume)
        temperature = thermal.water_temperature.to_numpy()[volume > 0.0]
        self.assertTrue(np.isfinite(temperature).all())
        self.assertGreaterEqual(temperature.min(), -1e-10)
        self.assertLessEqual(temperature.max(), 20.0 + 1e-10)
        self._assert_moments(thermal)

    def test_coupled_multirate_motion_and_full_erosion_refresh_populations(self):
        simulation = IceFlow2D(
            _submerged_config(
                thermal=ThermalConfig(
                    scheme="enthalpy_d2q5", update_interval_lbm_steps=4
                )
            )
        )
        simulation.step(3)
        self.assertEqual(simulation.thermal.steps, 0)
        simulation.step(1)
        self.assertEqual(simulation.thermal.steps, 1)
        self.assertAlmostEqual(simulation.thermal.time_s, simulation.physical_time_s)
        self._assert_moments(simulation.thermal)
        old_solid = simulation.solid_mask.to_numpy() != 0
        thermal = simulation.thermal
        thermal.body_sensible_energy.fill(
            2.0 * thermal._initial_body_cell_mass * thermal._latent_heat_j_kg
        )
        simulation.step(4)
        self.assertFalse(simulation.body_active[None])
        self.assertFalse(np.any(thermal._d2q5.body_populations.to_numpy()))
        self.assertFalse(np.any(simulation.solid_mask.to_numpy()))
        self.assertGreater(thermal.water_temperature.to_numpy()[old_solid].min(), 19.0)
        self._assert_moments(thermal)


if __name__ == "__main__":
    unittest.main()
