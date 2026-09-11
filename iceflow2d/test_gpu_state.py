"""Regressions for derived device state, face CFL, and time-step contracts."""

from dataclasses import replace
import math
import unittest

import numpy as np
import taichi as ti

from iceflow2d.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d.test_iceflow import _falling_config, _thermal_config


@ti.kernel
def _sample_device_views(
    simulation: ti.template(), output: ti.types.ndarray(dtype=ti.f64, ndim=3)
):
    for i, j in simulation.water_phase:
        output[i, j, 0] = simulation.wall_mask[i, j]
        output[i, j, 1] = simulation.body_signed_distance_m[i, j]
        output[i, j, 2] = simulation.physical_velocity_lattice[i, j].x
        output[i, j, 3] = simulation.physical_velocity_lattice[i, j].y
        output[i, j, 4] = simulation.thermal.water_temperature[i, j]


class DerivedGpuStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_views_follow_authoritative_state_without_refresh_or_device_storage(self):
        simulation = IceFlow2D(_falling_config(phase_warmup_steps=0))
        thermal = simulation.thermal
        volume = 0.25 * simulation.config.dx**2
        thermal.water_volume_m2[8, 8] = volume
        thermal.water_sensible_energy[8, 8] = (
            volume
            * simulation.config.rho_water
            * thermal.config.properties.specific_heat_water_j_kg_k
            * 35.0
        )
        thermal.body_solid_mass[0, 0] = 0.5 * thermal._initial_body_cell_mass
        thermal.body_sensible_energy[0, 0] = (
            -7.0 * thermal._ice_specific_heat * thermal.body_solid_mass[0, 0]
        )
        simulation.momentum_velocity_lattice[8, 8] = (0.12, -0.03)
        simulation.fluid_acceleration_lattice[8, 8] = (0.04, -0.02)

        views = [
            simulation.wall_mask,
            simulation.body_signed_distance_m,
            simulation.physical_velocity_lattice,
            simulation.body_center,
            simulation.water_volume_target,
            thermal.body_temperature,
            thermal.body_solid_fraction,
            thermal.water_temperature,
        ]
        for view in views:
            self.assertNotIsInstance(view, (ti.Field, ti.MatrixField))
        with self.assertRaisesRegex(TypeError, "conserved state"):
            thermal.body_solid_fraction[0, 0] = 1.0
        self.assertAlmostEqual(float(thermal.body_solid_fraction[0, 0]), 0.5)
        self.assertAlmostEqual(float(thermal.body_temperature[0, 0]), -7.0)
        self.assertAlmostEqual(float(thermal.water_temperature[8, 8]), 35.0)
        output = np.empty((simulation.nx, simulation.ny, 5), dtype=np.float64)
        _sample_device_views(simulation, output)
        np.testing.assert_array_equal(output[..., 0], simulation.wall_mask.to_numpy())
        np.testing.assert_allclose(
            output[..., 1],
            simulation.body_signed_distance_m.to_numpy(),
            rtol=3e-6,
            atol=2e-9,
        )
        np.testing.assert_allclose(
            output[..., 2:4],
            simulation.physical_velocity_lattice.to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            output[..., 4], thermal.water_temperature.to_numpy(), rtol=1e-7, atol=1e-8
        )
        fields = simulation.sample_thermal_fields()
        self.assertAlmostEqual(fields["temperature_c"][8, 8], 35.0)

    def test_face_outflow_cfl_bounds_divergent_velocity_and_preserves_water(self):
        thermal_config = replace(_thermal_config(), max_courant_number=1.0)
        simulation = IceFlow2D(
            _falling_config(
                thermal=thermal_config,
                phase_warmup_steps=0,
                gravity=(0.0, 0.0),
                sigma=0.0,
            )
        )
        simulation.momentum_velocity_lattice.fill(0.0)
        simulation.fluid_acceleration_lattice.fill(0.0)
        for cell, velocity in (
            ((9, 8), (0.8, 0.0)),
            ((7, 8), (-0.8, 0.0)),
            ((8, 9), (0.0, 0.8)),
            ((8, 7), (0.0, -0.8)),
        ):
            simulation.momentum_velocity_lattice[cell] = velocity
        simulation._measure_water_outflow_rate()
        outflow = float(simulation.maximum_water_outflow_rate_lattice[None])
        # All cell speeds are <= 0.8, but the center sends water out through
        # four faces at 0.4 each. A cell-speed-only CFL would miss this case.
        self.assertAlmostEqual(outflow, 1.6, places=6)
        thermal = simulation.thermal
        initial_volume = thermal.water_volume_m2.to_numpy().sum()
        initial_energy = thermal.water_sensible_energy.to_numpy().sum()
        substeps = thermal.advance_fast(
            simulation.scales.dt_s,
            simulation.physical_velocity_lattice,
            simulation.water_phase,
            simulation.wall_mask,
            simulation.solid_mask,
            maximum_outflow_rate_lattice=outflow,
        )
        self.assertEqual(substeps, 2)
        volume = thermal.water_volume_m2.to_numpy()
        self.assertGreaterEqual(float(volume.min()), -1e-18)
        self.assertAlmostEqual(float(volume.sum()), float(initial_volume), delta=1e-16)
        self.assertAlmostEqual(
            float(thermal.water_sensible_energy.to_numpy().sum()),
            float(initial_energy),
            delta=1e-8,
        )

    def test_invalid_step_counts_do_not_advance_state(self):
        simulation = IceFlow2D(_falling_config(phase_warmup_steps=0))
        for count in (-1, 1.5, True, "2", math.inf, math.nan):
            with self.subTest(count=count), self.assertRaisesRegex(
                ValueError, "non-negative integer"
            ):
                simulation.step(count)
        simulation.step(0)
        self.assertEqual(simulation.steps, 0)


if __name__ == "__main__":
    unittest.main()
