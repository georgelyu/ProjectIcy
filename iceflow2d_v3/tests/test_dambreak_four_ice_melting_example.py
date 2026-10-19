"""Geometry, physical properties and output contracts for the dam-break case."""

import contextlib
import io
import math
import unittest

import numpy as np

from iceflow2d_v3.config import LatticeScales
from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as common
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example
from iceflow2d_v3.examples import dambreak_single_ice_melting_2d as single
from iceflow2d_v3.examples import four_falling_ice_melting_2d as four


class DamBreakExampleTests(unittest.TestCase):
    def test_exact_column_size_and_four_blocks_above_wet_floor(self):
        config = example.create_config()
        reference = four.create_config()
        self.assertEqual(config.resolution, reference.resolution)
        self.assertEqual(config.dx, reference.dx)
        self.assertEqual((config.water_width, config.water_height), (198, 138))
        self.assertAlmostEqual(
            (config.water_width - config.boundary_cells) * config.dx, 0.065
        )
        self.assertAlmostEqual(
            (config.water_height - config.boundary_cells) * config.dx, 0.045
        )
        self.assertFalse(config.ice_fixed)
        self.assertTrue(config.well_balanced_hydrostatics)
        self.assertEqual(config.floor_water_depth_cells, 31)
        self.assertEqual(config.floor_water_height, 34)
        self.assertAlmostEqual(config.floor_water_depth_cells * config.dx, 0.0103333333333)
        self.assertEqual(len(config.bodies), 4)
        for body, x in zip(config.bodies, (0.085, 0.103, 0.121, 0.139), strict=True):
            self.assertEqual(body.size_cells, (30, 30))
            np.testing.assert_allclose(
                np.array(body.center_cells) * config.dx, (x, 0.007)
            )
            self.assertEqual(
                body.center_cells[1] - 0.5 * body.size_cells[1], config.boundary_cells + 3
            )
            self.assertGreater(body.center_cells[0] - 15, config.water_width)
            self.assertEqual(body.velocity_lattice, (0.0, 0.0))
            self.assertEqual(body.angle_rad, 0.0)

    def test_pool_depth_tracks_ice_draft_and_requested_clearance(self):
        for argv, expected_depth in (
            (["--ice-floor-clearance-cells", "5"], 33),
            (["--density-ice", "800"], 27),
            (["--resolution-x", "900", "--resolution-y", "300"], 59),
            (["--floor-water-depth-m", "0.012"], 36),
        ):
            with self.subTest(argv=argv):
                args = example.parse_args(argv)
                config = example.create_config(args)
                self.assertEqual(config.floor_water_depth_cells, expected_depth)
                for body in config.bodies:
                    self.assertAlmostEqual(
                        body.center_cells[1] - body.size_cells[1] / 2,
                        config.boundary_cells + args.ice_floor_clearance_cells,
                    )

    def test_rotated_ice_keeps_lowest_corner_clear_of_floor(self):
        config = example.create_config(example.parse_args(["--ice-angle-degrees", "15"]))
        for body in config.bodies:
            extent = body.size_cells[0] / 2 * (
                abs(math.cos(body.angle_rad)) + abs(math.sin(body.angle_rad))
            )
            self.assertAlmostEqual(body.center_cells[1] - extent, config.boundary_cells + 3)
            self.assertGreater(config.floor_water_height, body.center_cells[1])

    def test_dry_floor_can_be_restored_and_single_ice_keeps_its_geometry(self):
        config = example.create_config(example.parse_args([
            "--floor-water-depth-m", "0", "--ice-floor-clearance-cells", "0"
        ]))
        self.assertEqual(config.floor_water_depth_cells, 0)
        np.testing.assert_allclose(config.bodies[0].center_cells, (255.0, 18.0))
        single_config = single.create_config()
        self.assertEqual(single_config.floor_water_depth_cells, 0)
        self.assertEqual(single_config.bodies, config.bodies[:1])
        self.assertEqual(four.create_config().floor_water_depth_cells, 0)

    def test_molecular_viscosities_and_thermal_model_match_falling_case(self):
        config, reference = example.create_config(), four.create_config()
        self.assertIsNone(config.viscosity_water)
        self.assertIsNone(config.viscosity_air)
        for temperature in (0.0, 20.0, 90.0):
            self.assertEqual(
                config.water_viscosity_at(temperature),
                reference.water_viscosity_at(temperature),
            )
            self.assertEqual(
                config.air_viscosity_at(temperature),
                reference.air_viscosity_at(temperature),
            )
        self.assertEqual(config.thermal, reference.thermal)
        for name in (
            "rho_water",
            "rho_air",
            "rho_ice",
            "gravity",
            "sigma",
            "momentum_collision_model",
            "phase_momentum_coupling",
        ):
            self.assertEqual(getattr(config, name), getattr(reference, name))

    def test_default_run_targets_eight_seconds_with_air_visuals_and_no_raw_arrays(self):
        args = example.parse_args([])
        config = example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)
        targets = common._snapshot_step_targets(
            config, args.end_time_s, args.output_interval_s
        )
        self.assertEqual(args.end_time_s, 8.0)
        self.assertEqual(config.interface_width, 5.0)
        self.assertEqual(config.density_gradient_source_limit, 0.05)
        self.assertEqual(config.phase_flux_acceleration_limit, 0.01)
        self.assertTrue(config.refill_moment_projection)
        self.assertTrue(config.gas_state_repair)
        self.assertEqual(args.velocity_visualization_max_m_s, 0.8)
        self.assertTrue(args.velocity_include_air)
        self.assertTrue(args.vorticity_include_air)
        self.assertEqual(args.air_velocity_visualization_max_m_s, 4.0)
        self.assertEqual(args.air_vorticity_visualization_max_s_1, 5000.0)
        self.assertFalse(args.no_plot)
        self.assertFalse(args.save_npz)
        self.assertFalse(config.save_npz)
        self.assertTrue(args.check_stability)
        self.assertEqual((targets[0], targets[-1], len(targets)), (0, 960000, 801))
        self.assertAlmostEqual(targets[-1] * scales.dt_s, 8.0)

    def test_invalid_column_pool_and_ice_are_rejected_before_cuda(self):
        for argv in (
            ["--water-column-width-m", "0.15"],
            ["--water-column-height-m", "0.05"],
            ["--water-column-width-m", "0.0251"],
            ["--ice-centers-x-m", "0.020"],
            ["--ice-centers-x-m", "0.148"],
            ["--ice-floor-clearance-cells", "-1"],
            ["--ice-floor-clearance-cells", "1.5"],
            ["--floor-water-depth-m", "-0.01"],
            ["--floor-water-depth-m", "nan"],
            ["--floor-water-depth-m", "0.0101"],
            ["--floor-water-depth-m", "0.045"],
            ["--floor-water-depth-m", "0.001"],
            ["--density-ice", "1000"],
        ):
            with (
                self.subTest(argv=argv),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                example.parse_args(argv)


if __name__ == "__main__":
    unittest.main()
