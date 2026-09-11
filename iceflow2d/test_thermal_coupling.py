"""CUDA conservation tests for moving-body heat, melt, and water coupling."""

from __future__ import annotations

import unittest

import numpy as np
import taichi as ti

from iceflow2d import IceFlow2D, create_iceflow_config
from iceflow2d.simulator import ensure_taichi_cuda
from iceflow2d.thermal import (
    LatticeScales,
    MovingBodyThermal2D,
    PhaseChangeProperties,
    ThermalConfig,
)


def _moving_coupled_config():
    thermal = ThermalConfig(
        properties=PhaseChangeProperties(),
        initial_water_temperature_c=90.0,
        initial_ice_temperature_c=0.0,
        initial_air_temperature_c=20.0,
        update_interval_lbm_steps=1,
        water_buoyancy_model="linear",
        buoyancy_reference_temperature_c=90.0,
        moving_body_scheme="body_ale",
    )
    return create_iceflow_config(
        resolution=(24, 32),
        dx=1.0e-4,
        reference_velocity=0.4,
        rho_water=1000.0,
        rho_air=1.25,
        rho_ice=917.0,
        viscosity_water=1.0e-6,
        viscosity_air=1.5e-5,
        sigma=0.0,
        gravity=(0.0, 0.0),
        phase_warmup_steps=10,
        well_balanced_hydrostatics=True,
        water_width_fraction=22.5 / 24.0,
        water_height_fraction=24.5 / 32.0,
        ice_width_fraction=8.5 / 24.0,
        ice_height_fraction=8.5 / 32.0,
        ice_base_y_cells=10.0,
        boundary_cells=2,
        ice_fixed=False,
        rigid_boundary_scheme="unified",
        thermal=thermal,
        air_interface_relaxation_time=0.8,
        volume_projection_tolerance=1.0e-8,
    )


class ThermalCouplingCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:  # pragma: no cover - host dependent
            raise unittest.SkipTest(str(exc)) from exc

    def test_moving_thermal_step_closes_energy_mass_and_melt_momentum(self):
        simulation = IceFlow2D(_moving_coupled_config())
        initial_body_mass = float(simulation.body_mass_lattice[None])
        initial_energy = simulation.thermal.total_enthalpy_j_m(simulation.wall)
        initial_total_mass = simulation.thermal.mass_energy_totals().total_mass_kg_m

        simulation.body_velocity[None] = (0.02, -0.01)
        simulation.body_angular_velocity[None] = 0.003
        # update_interval=1, so the final step is already thermally synchronized.
        simulation.step(8)

        self.assertLess(float(simulation.body_mass_lattice[None]), initial_body_mass)
        temperature = simulation.temperature.to_numpy()
        self.assertTrue(np.isfinite(temperature).all())
        self.assertGreaterEqual(float(temperature.min()), -1.0e-9)
        self.assertLessEqual(float(temperature.max()), 90.0 + 1.0e-8)

        final_energy = simulation.thermal.total_enthalpy_j_m(simulation.wall)
        boundary_heat = float(simulation.thermal.boundary_heat_input_j_m[None])
        self.assertAlmostEqual(
            final_energy - initial_energy, boundary_heat, delta=1.0e-8
        )
        final_total_mass = simulation.thermal.mass_energy_totals().total_mass_kg_m
        self.assertAlmostEqual(final_total_mass, initial_total_mass, delta=2.0e-11)
        melted_volume = (
            (initial_body_mass - float(simulation.body_mass_lattice[None]))
            * simulation.cfg.rho_water
            / simulation.cfg.rho_ice
        )
        added_water = float(simulation.water_volume_current[None]) - float(
            simulation.phase_change_initial_water_volume[None]
        )
        self.assertAlmostEqual(
            added_water,
            0.917 * melted_volume,
            delta=simulation.cfg.volume_projection_tolerance
            * float(simulation.water_volume_target[None]),
        )

        outgoing = np.asarray(
            simulation.cumulative_melted_momentum_lattice[None], dtype=np.float64
        )
        injected = np.asarray(
            simulation.cumulative_fluid_melt_momentum_lattice[None], dtype=np.float64
        )
        residual = np.asarray(
            simulation.melt_momentum_residual_lattice[None], dtype=np.float64
        )
        np.testing.assert_allclose(
            residual, injected - outgoing, rtol=0.0, atol=1.0e-15
        )
        self.assertLessEqual(
            float(np.linalg.norm(residual)),
            5.0e-6 * float(np.linalg.norm(outgoing)) + 1.0e-11,
        )

        outgoing_angular = float(
            simulation.cumulative_melted_angular_momentum_lattice[None]
        )
        injected_angular = float(
            simulation.cumulative_fluid_melt_angular_momentum_lattice[None]
        )
        angular_residual = float(
            simulation.melt_angular_momentum_residual_lattice[None]
        )
        self.assertAlmostEqual(
            angular_residual,
            injected_angular - outgoing_angular,
            delta=1.0e-14,
        )
        self.assertLessEqual(
            abs(angular_residual),
            5.0e-6 * abs(outgoing_angular) + 1.0e-10,
        )
        self.assertAlmostEqual(
            float(simulation.thermal.melt_injection_mass_residual_kg_m[None]),
            0.0,
            delta=1.0e-16,
        )

    def test_water_aperture_transfer_is_volume_and_energy_conservative(self):
        simulation = IceFlow2D(_moving_coupled_config())
        thermal = simulation.thermal
        phase = simulation.phi.to_numpy()
        wall = simulation.wall.to_numpy()
        solid = simulation.solid.to_numpy()
        volume_before = thermal.water_volume_m2.to_numpy()
        energy_before = thermal.water_sensible_energy.to_numpy()

        donors = np.argwhere(
            (wall == 0) & (solid == 0) & (phase >= 0.99) & (volume_before > 0.0)
        )
        cutoff = simulation.cfg.volume_projection_interface_cutoff
        receivers = np.argwhere(
            (wall == 0) & (solid == 0) & (phase <= cutoff) & (volume_before == 0.0)
        )
        self.assertGreater(len(donors), 0)
        self.assertGreater(len(receivers), 0)
        donor = tuple(int(value) for value in donors[0])
        receiver = tuple(int(value) for value in receivers[0])
        phase[donor] = 0.0
        phase[receiver] = 0.49
        simulation.phi.from_numpy(phase.astype(np.float32))

        thermal.synchronize_water_aperture(
            simulation.phi, simulation.wall, simulation.solid
        )

        volume_after = thermal.water_volume_m2.to_numpy()
        energy_after = thermal.water_sensible_energy.to_numpy()
        self.assertEqual(float(volume_after[donor]), 0.0)
        self.assertGreater(float(volume_after[receiver]), 0.0)
        self.assertAlmostEqual(
            float(volume_after.sum()), float(volume_before.sum()), delta=1.0e-18
        )
        self.assertAlmostEqual(
            float(energy_after.sum()), float(energy_before.sum()), delta=1.0e-9
        )
        self.assertAlmostEqual(
            float(thermal.phase_aperture_volume_residual_m2[None]),
            0.0,
            delta=1.0e-18,
        )
        self.assertAlmostEqual(
            float(thermal.phase_aperture_energy_residual_j_m[None]),
            0.0,
            delta=1.0e-9,
        )


class LocalThermalRemapCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def _single_cell_bath(self, config):
        nx, ny = 8, 8
        thermal = MovingBodyThermal2D(
            nx,
            ny,
            1,
            1,
            config,
            LatticeScales(1.0e-4, 1.0e-5, 1000.0),
            density_water_kg_m3=1000.0,
            density_ice_kg_m3=917.0,
        )
        phase = ti.field(ti.f32, shape=(nx, ny))
        wall = ti.field(ti.i8, shape=(nx, ny))
        solid = ti.field(ti.i8, shape=(nx, ny))
        origin = ti.Vector.field(2, ti.f32, shape=())
        angle = ti.field(ti.f32, shape=())
        mask = np.ones((nx, ny), dtype=np.int8)
        mask[1:-1, 1:-1] = 0
        wall.from_numpy(mask)
        phase.fill(0.8)  # Leave geometric capacity for the melt water.
        solid[4, 4] = 1
        thermal.initialize(phase, wall, solid)
        thermal.world_body_indicator[4, 4] = 1.0
        return thermal, phase, wall, solid, origin, angle

    def test_cold_refill_and_rising_hot_surface_keep_their_local_temperatures(self):
        # Two patches exchange equal amounts of water locally. A pooled
        # donor temperature gives both receivers 47.5 C, despite correct
        # global mass/energy. This test detects that spatial error directly.
        nx, ny = 24, 20
        thermal = MovingBodyThermal2D(
            nx,
            ny,
            1,
            1,
            ThermalConfig(),
            LatticeScales(1.0e-4, 1.0e-5, 1000.0),
            density_water_kg_m3=1000.0,
            density_ice_kg_m3=917.0,
        )
        phase = ti.field(ti.f32, shape=(nx, ny))
        wall = ti.field(ti.i8, shape=(nx, ny))
        solid = ti.field(ti.i8, shape=(nx, ny))
        old_phase = np.zeros((nx, ny), dtype=np.float32)
        old_phase[3:10, 3:13] = 0.8
        old_phase[14:21, 3:13] = 0.8
        cold_refill = (6, 8)
        hot_surface = (17, 13)
        old_phase[cold_refill] = 0.0
        volume = old_phase.astype(np.float64) * 1.0e-8
        temperature = np.full((nx, ny), 90.0)
        temperature[:12, :] = 5.0
        energy = 1000.0 * 4186.0 * volume * temperature
        thermal.water_volume_m2.from_numpy(volume)
        thermal.water_sensible_energy.from_numpy(energy)

        new_phase = old_phase.copy()
        new_phase[cold_refill] = 0.8
        new_phase[hot_surface] = 0.8
        new_phase[3, 3] = 0.0
        new_phase[14, 3] = 0.0
        phase.from_numpy(new_phase)
        thermal.synchronize_water_aperture(phase, wall, solid)
        result = thermal.water_temperature.to_numpy()
        self.assertAlmostEqual(result[cold_refill], 5.0, delta=1.0e-8)
        self.assertAlmostEqual(result[hot_surface], 90.0, delta=1.0e-8)
        self.assertAlmostEqual(
            thermal.water_volume_m2.to_numpy().sum(), volume.sum(), delta=1.0e-18
        )
        self.assertAlmostEqual(
            thermal.water_sensible_energy.to_numpy().sum(), energy.sum(), delta=1.0e-9
        )

        # A further unbalanced surface move exercises the energy correction,
        # including temperature bounds, without any prescribed heat source.
        new_phase[4, 4] = 0.0
        new_phase[18, 13] = 0.8
        phase.from_numpy(new_phase)
        thermal.synchronize_water_aperture(phase, wall, solid)
        wet = thermal.water_volume_m2.to_numpy() > 0.0
        result = thermal.water_temperature.to_numpy()[wet]
        self.assertGreaterEqual(result.min(), 5.0 - 1.0e-8)
        self.assertLessEqual(result.max(), 90.0 + 1.0e-8)
        self.assertAlmostEqual(
            thermal.water_sensible_energy.to_numpy().sum(), energy.sum(), delta=1.0e-9
        )

    def test_insulated_melt_pays_sensible_and_latent_heat_and_adds_density_scaled_water(
        self,
    ):
        # One subcooled material cell melts in a closed water bath. The
        # fixed raster isolates the actual thermal kernels from mechanics.
        # After melting, its mass-weighted mean must equal the calorimetric
        # equilibrium even before spatial mixing has finished.
        config = ThermalConfig(
            initial_water_temperature_c=90.0, initial_ice_temperature_c=-10.0
        )
        thermal, phase, wall, solid, origin, angle = self._single_cell_bath(config)
        initial = thermal.mass_energy_totals()
        initial_water_volume = thermal.water_volume_m2.to_numpy().sum()
        for _ in range(500):
            thermal.advance_slow(
                thermal.maximum_diffusion_time_step_s, phase, wall, solid, origin, angle
            )
            if thermal.body_solid_mass[0, 0] < initial.initial_body_mass_kg_m * 1.0e-10:
                break
        final = thermal.mass_energy_totals()
        self.assertLess(
            final.solid_body_mass_kg_m, initial.initial_body_mass_kg_m * 1.0e-10
        )
        self.assertEqual(float(thermal.boundary_heat_input_j_m[None]), 0.0)
        self.assertAlmostEqual(
            final.total_mass_kg_m, initial.total_mass_kg_m, delta=1.0e-14
        )
        self.assertAlmostEqual(
            final.total_energy_j_m, initial.total_energy_j_m, delta=1.0e-8
        )
        water_volume = thermal.water_volume_m2.to_numpy()
        added_volume = water_volume.sum() - initial_water_volume
        self.assertAlmostEqual(added_volume, 0.917e-8, delta=1.0e-15)
        props = config.properties
        expected_temperature = (
            initial.water_mass_kg_m * props.specific_heat_water_j_kg_k * 90.0
            - initial.initial_body_mass_kg_m
            * (props.latent_heat_j_kg + 10.0 * props.specific_heat_ice_j_kg_k)
        ) / (initial.total_mass_kg_m * props.specific_heat_water_j_kg_k)
        actual_temperature = np.sum(
            thermal.water_temperature.to_numpy() * water_volume
        ) / water_volume.sum()
        self.assertAlmostEqual(actual_temperature, expected_temperature, delta=1.0e-7)

    def test_insulated_bath_cannot_melt_more_ice_than_its_available_heat(self):
        for water_temperature in (0.0, 1.0):
            with self.subTest(water_temperature=water_temperature):
                config = ThermalConfig(
                    initial_water_temperature_c=water_temperature,
                    initial_ice_temperature_c=0.0,
                    initial_air_temperature_c=90.0,
                )
                thermal, phase, wall, solid, origin, angle = self._single_cell_bath(
                    config
                )
                initial = thermal.mass_energy_totals()
                available_heat = initial.water_mass_kg_m * 4186.0 * water_temperature
                max_melt_mass = available_heat / 334000.0
                self.assertLess(max_melt_mass, initial.initial_body_mass_kg_m)
                for _ in range(100):
                    thermal.advance_slow(
                        thermal.maximum_diffusion_time_step_s,
                        phase,
                        wall,
                        solid,
                        origin,
                        angle,
                    )
                final = thermal.mass_energy_totals()
                melted_mass = initial.solid_body_mass_kg_m - final.solid_body_mass_kg_m
                self.assertLessEqual(melted_mass, max_melt_mass + 1.0e-14)
                self.assertGreater(final.solid_body_mass_kg_m, 0.0)
                if water_temperature == 0.0:
                    self.assertEqual(melted_mass, 0.0)
                else:
                    self.assertGreater(melted_mass, 0.0)
                self.assertEqual(float(thermal.boundary_heat_input_j_m[None]), 0.0)
                self.assertAlmostEqual(
                    final.total_energy_j_m, available_heat, delta=1.0e-10
                )
                self.assertAlmostEqual(
                    final.total_mass_kg_m, initial.total_mass_kg_m, delta=1.0e-14
                )
                wet = thermal.water_volume_m2.to_numpy() > 0.0
                temperatures = thermal.water_temperature.to_numpy()[wet]
                self.assertGreaterEqual(float(temperatures.min()), -1.0e-10)
                self.assertLessEqual(
                    float(temperatures.max()), water_temperature + 1.0e-10
                )


if __name__ == "__main__":
    unittest.main()
