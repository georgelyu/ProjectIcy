"""CUDA regression for column pressure and release without artificial viscosity."""

import unittest

import numpy as np

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example
from iceflow2d_v3.simulator import ensure_taichi_cuda


class DamBreakCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_initial_column_hydrostatic_pressure(self):
        args = example.parse_args([
            "--phase-warmup-steps", "0",
            "--floor-water-depth-m", "0", "--ice-floor-clearance-cells", "0",
        ])
        config = example.create_config(args)
        simulation = IceFlow2D(config)
        simulation._thermal_stability_check_enabled = True
        reference = simulation.hydrostatic_reference_pressure.to_numpy()
        pressure = simulation.pressure_lattice.to_numpy()
        residual = pressure - reference
        # Check the supported column before release against rho*g*h, including
        # the cell-center half height. The dry side has only air hydrostatics.
        i = config.boundary_cells + (config.water_width - config.boundary_cells) // 2
        j = 30
        expected = (
            (1.0 - config.rho_air / config.rho_water)
            * -simulation._gravity_lattice[1]
            * (config.water_height - j - 0.5)
        )
        self.assertAlmostEqual(float(residual[i, j]), expected, delta=1e-9)
        dry_start = config.water_width + 5
        dry_end = (
            int(config.bodies[0].center_cells[0] - config.bodies[0].size_cells[0] / 2)
            - 2
        )
        np.testing.assert_allclose(residual[dry_start:dry_end, 3:140], 0.0, atol=1e-10)
        np.testing.assert_allclose(
            simulation.physical_velocity_lattice.to_numpy(), 0.0, atol=1e-12
        )

    def test_initial_pool_has_water_and_hydrostatic_pressure_below_all_four_blocks(self):
        config = example.create_config(example.parse_args(["--phase-warmup-steps", "0"]))
        simulation = IceFlow2D(config)
        phase = simulation.water_phase.to_numpy()
        solid = simulation.solid_mask.to_numpy()
        pressure = simulation.pressure_lattice.to_numpy()
        reference = simulation.hydrostatic_reference_pressure.to_numpy()
        floor = config.boundary_cells
        pool_top = config.floor_water_height
        clear_column = config.water_width + 5
        np.testing.assert_allclose(phase[floor:-floor, floor:floor + 3], 1.0, atol=1e-6)
        np.testing.assert_array_equal(solid[floor:-floor, floor:floor + 3], 0)
        np.testing.assert_allclose(phase[clear_column, floor:pool_top], 1.0, atol=1e-6)
        np.testing.assert_allclose(phase[clear_column, pool_top:-floor], 0.0, atol=1e-10)
        # In the downstream pool the reference itself supplies rho*g*h,
        # including the submerged ice's underside, without a pressure deficit.
        expected = -simulation._gravity_lattice[1] * (pool_top - floor - 0.5)
        self.assertAlmostEqual(float(pressure[clear_column, floor]), expected, delta=1e-9)
        for body in config.bodies:
            i = int(body.center_cells[0])
            np.testing.assert_allclose(
                pressure[i, floor:floor + 3], pressure[clear_column, floor:floor + 3],
                atol=1e-9,
            )
            np.testing.assert_allclose(
                simulation.temperature.to_numpy()[i, floor:floor + 3],
                config.thermal.initial_water_temperature_c, atol=1e-6,
            )
        # The released column retains the additional head above the pool.
        i = floor + (config.water_width - floor) // 2
        expected_excess = (
            (1.0 - config.rho_air / config.rho_water)
            * -simulation._gravity_lattice[1] * (config.water_height - pool_top)
        )
        self.assertAlmostEqual(
            float(pressure[i, floor] - reference[i, floor]), expected_excess, delta=1e-9
        )
        np.testing.assert_allclose(simulation.physical_velocity_lattice.to_numpy(), 0.0)

    def test_default_wet_floor_survives_warmup_and_short_release(self):
        simulation = IceFlow2D(example.create_config())
        simulation._thermal_stability_check_enabled = True
        floor = simulation.config.boundary_cells
        self.assertGreater(
            float(simulation.water_phase.to_numpy()[floor:-floor, floor:floor + 3].min()),
            0.99,
        )
        initial_mass = simulation.body_mass_lattice.array().copy()
        simulation.step(1200)
        self.assertTrue(np.isfinite(simulation.physical_velocity_lattice.to_numpy()).all())
        self.assertTrue(np.all(simulation.body_mass_lattice.array() < initial_mass))
        self.assertTrue(np.all(simulation.body_mass_lattice.array() > 0.0))
        phase = simulation.water_phase.to_numpy()
        owner = simulation.solid_body_id.to_numpy()
        solid = simulation.solid_mask.to_numpy()
        for body_id in range(simulation.body_count):
            ii, jj = np.nonzero((solid == 1) & (owner == body_id))
            self.assertGreaterEqual(int(jj.min()), floor + 2)
            bottom = jj == jj.min()
            self.assertGreater(float(phase[ii[bottom], jj[bottom] - 1].min()), 0.99)

    def test_original_wide_interface_column_survives_impact_and_free_body_motion(self):
        simulation = IceFlow2D(
            example.create_config(
                example.parse_args(
                    [
                        "--interface-width",
                        "12",
                        "--floor-water-depth-m", "0",
                        "--ice-floor-clearance-cells", "0",
                        "--water-column-width-m",
                        "0.065",
                        "--ice-centers-x-m",
                        "0.085",
                        "0.103",
                        "0.121",
                        "0.139",
                    ]
                )
            )
        )
        simulation._thermal_stability_check_enabled = True
        initial_mass = simulation.body_mass_lattice.array().copy()
        simulation.step(1200)
        velocity = simulation.physical_velocity_lattice.to_numpy()
        self.assertTrue(np.isfinite(velocity).all())
        wet = simulation.water_phase.to_numpy() > 0.5
        # The water starts flowing right into the initially dry region.
        self.assertGreater(float(velocity[..., 0][wet].max()), 1e-4)
        self.assertLess(float(np.abs(velocity).sum(axis=-1).max()), 0.25)
        self.assertTrue(
            np.all(simulation.body_mass_lattice.array() <= initial_mass + 1e-8)
        )
        self.assertTrue(np.all(simulation.body_mass_lattice.array() >= 0.0))
        # Cover the original failures at 0.04/0.08 s and subsequent impacts.
        simulation.step(34800)
        velocity = simulation.physical_velocity_lattice.to_numpy()
        self.assertTrue(np.isfinite(velocity).all())
        self.assertLess(float(np.abs(velocity).sum(axis=-1).max()), 0.5)
        self.assertTrue(np.all(simulation.body_mass_lattice.array() < initial_mass))
        self.assertTrue(np.all(simulation.body_mass_lattice.array() > 0.0))
        initial_x = np.array([b.center_cells[0] for b in simulation.config.bodies])
        self.assertGreater(
            float(np.abs(simulation.body_center.to_numpy()[:, 0] - initial_x).max()),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
