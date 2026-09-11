"""CUDA conservation tests for moving-body heat, melt, and water coupling."""

from __future__ import annotations

import unittest

import numpy as np

from iceflow2d import IceFlow2D, create_iceflow_config
from iceflow2d.simulator import ensure_taichi_cuda
from iceflow2d.thermal import PhaseChangeProperties, ThermalConfig


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


if __name__ == "__main__":
    unittest.main()
