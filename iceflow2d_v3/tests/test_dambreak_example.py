"""CPU contracts for the hot dam-break geometry and default output schedule."""

import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import iceflow2d_v3

from iceflow2d_v3.config import LatticeScales
from iceflow2d_v3.examples import dambreak_four_ice_melting_2d as example


class DamBreakExampleTests(unittest.TestCase):
    def test_agreed_geometry_and_default_schedule(self):
        args = example.parse_args([])
        config = example.create_config(args)
        self.assertEqual(config.resolution, (450, 150))
        self.assertEqual((config.water_width, config.water_height), (198, 138))
        self.assertAlmostEqual((config.water_width - config.boundary_cells) * config.dx, .065)
        self.assertAlmostEqual((config.water_height - config.boundary_cells) * config.dx, .045)
        self.assertEqual(config.initial_water_geometry, "dam_break")
        self.assertEqual(config.thermal.initial_water_temperature_c, 90.)
        self.assertEqual(config.angular_damping, 1.)
        self.assertTrue(config.close_subgrid_wall_gaps)
        self.assertEqual(config.subgrid_wall_gap_cells, 1)
        self.assertTrue(config.preserve_contact_fluid_gap)
        self.assertEqual(config.phase_momentum_flux_model, "accepted_links")
        self.assertEqual(config.momentum_phase_gradient, "open_links_air")
        self.assertEqual(config.phase_momentum_water_cutoff, .001)
        self.assertEqual(config.air_viscosity_floor_lattice, .1)
        self.assertEqual(len(config.bodies), 4)
        for body, x in zip(config.bodies, (.08, .10, .12, .14)):
            self.assertEqual(body.size_cells, (30, 30))
            self.assertAlmostEqual(body.center_cells[0] * config.dx, x)
            self.assertAlmostEqual(body.center_cells[1] * config.dx, .006)
            self.assertEqual(body.temperature_c, 0.)
        self.assertFalse(args.no_plot)
        self.assertFalse(args.save_npz)
        self.assertTrue(args.check_stability)
        self.assertTrue(args.stop_when_all_ice_melted)
        self.assertEqual(args.end_time_s, 10.)
        self.assertAlmostEqual(LatticeScales.from_iceflow_config(config).dt_s, 1 / 120000)
        targets = example.common._snapshot_step_targets(config, args.end_time_s, args.output_interval_s)
        self.assertEqual((len(targets), targets[-1]), (1001, 1200000))

    def test_integer_stop_requires_zero_mass_and_respects_requested_end(self):
        stop = example.common._integer_stop_after_melt
        self.assertIsNone(stop(4.8, 1e-20, 1/120000, 1200000))
        self.assertEqual(stop(4.81, 0., 1/120000, 1200000), 600000)
        self.assertEqual(stop(5.0, 0., 1/120000, 1200000), 600000)
        self.assertEqual(stop(5.000000000000001, 0., 1/120000, 1200000), 600000)
        self.assertEqual(stop(9.95, 0., 1/120000, 1100000), 1100000)
        self.assertFalse(example.parse_args(["--no-stop-when-all-ice-melted"]).stop_when_all_ice_melted)

    def test_early_stop_finalizes_all_retained_snapshots_and_metadata(self):
        common = example.common
        for count in (1, 4):
            with self.subTest(body_count=count), TemporaryDirectory() as directory:
                argv = ["--end-time-s", "2", "--output-interval-s", ".25",
                        "--no-plot", "--save-npz", "--quiet", "--output-dir", directory]
                if count == 1:
                    argv += ["--ice-centers-x-m", ".08"]
                args = example.parse_args(argv)
                config = example.create_config(args)
                dt = LatticeScales.from_iceflow_config(config).dt_s

                class HostSimulation:
                    steps = 0

                    def __init__(self, config):
                        pass

                    def phase_change_solid_volume_cells(self):
                        return 1.

                def advance(s, target, *args, **kwargs):
                    s.steps = target

                def capture(s, **kwargs):
                    time = s.steps * dt
                    mass = .01 if time < .6 else 0.
                    return SimpleNamespace(
                        physical_time_s=time, lbm_steps=s.steps,
                        body_mass_lattice=mass, melted_fraction=float(mass == 0),
                        bodies=tuple(SimpleNamespace(body_mass_lattice=mass) for _ in range(count)) if count > 1 else (),
                    )

                with ExitStack() as stack:
                    # Replace the package attribute without loading Taichi;
                    # this checks the host runner's real schedule and finalize
                    # path, not a second copy of its stopping implementation.
                    stack.enter_context(patch.dict(vars(iceflow2d_v3), {"IceFlow2D": HostSimulation}))
                    stack.enter_context(patch.object(common, "_capture_snapshot", side_effect=capture))
                    stack.enter_context(patch.object(common, "_advance_simulation_to_target", side_effect=advance))
                    stack.enter_context(patch.object(common, "_history_columns", return_value=["time_s"]))
                    stack.enter_context(patch.object(common, "_history_rows", side_effect=lambda s: [[s.physical_time_s]]))
                    archive = stack.enter_context(patch.object(common, "write_fields_npz"))
                    metadata = stack.enter_context(patch.object(common, "write_metadata"))
                    targets = common._snapshot_step_targets(config, 2., .25)
                    result = common._run_case(args, config, targets, stop_when_all_ice_melted=True)
                self.assertEqual(result["physical_time_s"], 1.)
                self.assertEqual(result["snapshots"], 5)
                archive.assert_called_once()
                self.assertEqual([s.physical_time_s for s in archive.call_args.args[2]], [0., .25, .5, .75, 1.])
                metadata.assert_called_once()
                self.assertEqual(metadata.call_args.kwargs["targets"], [0,30000,60000,90000,120000])
                self.assertEqual(len((Path(directory)/"history.csv").read_text().splitlines()), 6)

    def test_column_outside_domain_is_rejected(self):
        args = example.build_parser().parse_args(["--water-column-height-m", ".05"])
        with self.assertRaisesRegex(ValueError, "air cap"):
            example.create_config(args)

    def test_ice_inside_column_is_rejected(self):
        args = example.build_parser().parse_args(["--ice-centers-x-m", ".07", ".10", ".12", ".14"])
        with self.assertRaisesRegex(ValueError, "downstream"):
            example.create_config(args)

    def test_overlapping_ice_is_rejected(self):
        args = example.build_parser().parse_args(["--ice-centers-x-m", ".08", ".085", ".12", ".14"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            example.create_config(args)


if __name__ == "__main__":
    unittest.main()
