"""Pure-CPU contract tests for the falling-and-melting ice example."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from iceflow2d_v3.examples import coupled_falling_ice_melting_2d as example
from iceflow2d_v3.config import LatticeScales


class _FakeField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _key):
        return self.value

    def to_numpy(self):
        return np.asarray(self.value)


class _FakeMovingSimulation:
    def __init__(self, config):
        self.config = config
        shape = (config.nx, config.ny)
        vector_shape = (*shape, 2)
        scalar_float = _FakeField(np.zeros(shape, dtype=np.float32))
        self.temperature = _FakeField(np.zeros(shape, dtype=np.float64))
        self.liquid_fraction = scalar_float
        self.thermal_enthalpy = _FakeField(np.zeros(shape, dtype=np.float64))
        self.water_phase = scalar_float
        self.solid_mask = _FakeField(np.zeros(shape, dtype=np.int8))
        self.wall_mask = _FakeField(np.zeros(shape, dtype=np.int8))
        self.body_signed_distance_m = scalar_float
        self.phase_change_material = _FakeField(np.zeros(shape, dtype=np.int8))
        self.momentum_velocity_lattice = _FakeField(
            np.zeros(vector_shape, dtype=np.float32)
        )
        self.fluid_acceleration_lattice = _FakeField(
            np.zeros(vector_shape, dtype=np.float32)
        )
        self.scales = LatticeScales.from_iceflow_config(config)
        self.body_center = _FakeField(config.ice_initial_center)
        self.body_velocity = _FakeField(config.ice_initial_velocity)
        self.body_local_center_of_mass = _FakeField((0.0, 0.0))
        self.body_mass_lattice = _FakeField(config.ice_mass_lattice)
        self.body_inertia_lattice = _FakeField(config.ice_inertia_lattice)
        self.body_active = _FakeField(1)
        self.body_angle = _FakeField(config.ice_initial_angle)
        self.body_angular_velocity = _FakeField(config.ice_initial_angular_velocity)
        self.cumulative_melted_mass_lattice = _FakeField(0.0)
        self.thermal = SimpleNamespace(time_s=0.0, steps=0)
        self.steps = 0
        self.physical_time_s = 0.0

    def phase_change_solid_volume_cells(self) -> float:
        return float(self.config.ice_width * self.config.ice_height)

    def phase_change_geometry_volume_cells(self) -> float:
        return self.phase_change_solid_volume_cells()


class CoupledFallingMeltingExampleTests(unittest.TestCase):
    def test_module_can_be_loaded_while_cuda_simulator_import_is_blocked(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "iceflow2d_v3"
            / "examples"
            / "coupled_falling_ice_melting_2d.py"
        )
        spec = importlib.util.spec_from_file_location(
            "iceflow2d_falling_melting_cpu_contract", path
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        previous = sys.modules.get("iceflow2d_v3.simulator", ...)
        sys.modules["iceflow2d_v3.simulator"] = None
        try:
            spec.loader.exec_module(module)
            config = module.create_config(module.parse_args([]))
        finally:
            sys.modules.pop(spec.name, None)
            if previous is ...:
                sys.modules.pop("iceflow2d_v3.simulator", None)
            else:
                sys.modules["iceflow2d_v3.simulator"] = previous

        self.assertFalse(config.ice_fixed)
        self.assertEqual(config.thermal.moving_body_scheme, "body_ale")

    def test_default_geometry_is_centimetre_scale_and_starts_in_air(self):
        args = example.parse_args([])
        geometry = example.derive_geometry(args)

        self.assertEqual((args.domain_width_m, args.domain_height_m), (0.025, 0.050))
        self.assertEqual((args.resolution_x, args.resolution_y), (100, 200))
        self.assertEqual((geometry["water_width"], geometry["water_height"]), (97, 120))
        self.assertEqual((geometry["ice_width"], geometry["ice_height"]), (32, 32))
        self.assertAlmostEqual(geometry["dx_m"], 2.5e-4)
        self.assertAlmostEqual(geometry["drop_height_cells"], 6.0)
        self.assertAlmostEqual(geometry["ice_lowest_y_cells"], 126.0)
        self.assertGreater(geometry["ice_lowest_y_cells"], geometry["water_height"])
        self.assertLess(
            geometry["ice_highest_y_cells"],
            args.resolution_y - args.boundary_cells,
        )

        extent_y = 16.0 * (
            abs(math.sin(math.radians(5.0))) + abs(math.cos(math.radians(5.0)))
        )
        self.assertAlmostEqual(
            geometry["ice_center_y_cells"] - geometry["ice_lowest_y_cells"],
            extent_y,
        )

    def test_default_config_selects_the_full_moving_thermal_coupling(self):
        args = example.parse_args([])
        config = example.create_config(args)

        self.assertTrue(args.progress)
        self.assertFalse(example.parse_args(["--no-progress"]).progress)
        self.assertEqual((config.nx, config.ny), (100, 200))
        self.assertEqual((config.water_width, config.water_height), (97, 120))
        self.assertEqual(config.water_width, config.nx - config.boundary_cells)
        self.assertEqual((config.ice_width, config.ice_height), (32, 32))
        self.assertFalse(config.ice_fixed)
        self.assertEqual(config.thermal.moving_body_scheme, "body_ale")
        self.assertEqual(config.thermal.update_interval_lbm_steps, 8)
        self.assertTrue(config.thermal.advection_enabled)
        self.assertTrue(config.thermal.water_air_interface_adiabatic)
        self.assertEqual(config.thermal.water_buoyancy_model, "linear")
        self.assertEqual(config.thermal.initial_water_temperature_c, 90.0)
        self.assertEqual(config.thermal.buoyancy_reference_temperature_c, 90.0)
        for side in ("left", "right", "bottom", "top"):
            boundary = getattr(config.thermal.boundaries, side)
            self.assertEqual(boundary.kind, "adiabatic")
            self.assertEqual(boundary.value, 0.0)
        self.assertEqual(config.gravity, (0.0, -9.8))
        self.assertEqual(config.sigma, 0.0)
        self.assertEqual(config.reference_velocity, 4.0)
        self.assertIsNone(config.viscosity_water)
        self.assertIsNone(config.viscosity_air)
        self.assertEqual(config.momentum_collision_model, "cm_mrt")
        self.assertEqual(config.phase_normal_epsilon, 1e-3)
        self.assertEqual(config.cm_acoustic_relaxation_rate, 0.25)
        self.assertEqual(args.end_time_s, 5.0)
        self.assertEqual(args.output_interval_s, 0.01)

    def test_cli_selects_momentum_model_and_its_acoustic_default(self):
        defaults = example.create_config().to_dict()
        defaults.pop("momentum_collision_model")
        defaults.pop("cm_acoustic_relaxation_rate")
        for model in ("rm_mrt", "cm_mrt"):
            with self.subTest(model=model):
                args = example.parse_args(["--momentum-collision-model", model])
                selected = example.create_config(args).to_dict()
                self.assertEqual(selected.pop("momentum_collision_model"), model)
                self.assertEqual(
                    selected.pop("cm_acoustic_relaxation_rate"),
                    0.25 if model == "cm_mrt" else 1.0,
                )
                self.assertEqual(selected, defaults)

    def test_cli_selects_stability_controls_without_changing_physics(self):
        defaults = example.create_config(example.parse_args([])).to_dict()
        for epsilon, rate in ((0.0, 0.25), (1e-3, 1.0), (1e-3, 0.25)):
            selected = example.create_config(
                example.parse_args(
                    [
                        "--momentum-collision-model",
                        "cm_mrt",
                        "--phase-normal-epsilon",
                        str(epsilon),
                        "--cm-acoustic-relaxation-rate",
                        str(rate),
                    ]
                )
            ).to_dict()
            expected = dict(
                defaults,
                momentum_collision_model="cm_mrt",
                phase_normal_epsilon=epsilon,
                cm_acoustic_relaxation_rate=rate,
            )
            self.assertEqual(selected, expected)
        with self.assertRaisesRegex(
            ValueError, "requires momentum_collision_model='cm_mrt'"
        ):
            example.create_config(
                example.parse_args(
                    [
                        "--momentum-collision-model",
                        "rm_mrt",
                        "--cm-acoustic-relaxation-rate",
                        "0.25",
                    ]
                )
            )

    def test_initial_rigid_velocity_is_not_artificially_capped(self):
        args = example.parse_args(
            [
                "--initial-horizontal-speed-lattice",
                "0.12",
                "--initial-vertical-speed-lattice",
                "0.09",
            ]
        )
        config = example.create_config(args)
        self.assertEqual(config.ice_initial_velocity, (0.12, 0.09))

    def test_physical_time_scheduler_is_thermal_step_aligned(self):
        args = example.parse_args([])
        config = example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)
        targets = example._snapshot_step_targets(
            config, args.end_time_s, args.output_interval_s
        )

        self.assertAlmostEqual(scales.dt_s, 6.25e-6)
        self.assertEqual(targets[0], 0)
        self.assertEqual(targets[1], 1600)
        self.assertEqual(targets[-1], 800000)
        self.assertEqual(len(targets), 501)
        self.assertTrue(
            all(
                target % config.thermal.update_interval_lbm_steps == 0
                for target in targets
            )
        )
        self.assertGreaterEqual(targets[-1] * scales.dt_s, args.end_time_s)
        self.assertEqual(example._progress_chunk_steps(targets[-1]), 400)

    def test_chunked_advance_updates_step_progress_before_a_field_frame(self):
        class FakeSimulation:
            def __init__(self):
                self.steps = 3
                self.calls: list[int] = []

            def step(self, count):
                self.calls.append(count)
                self.steps += count

        class FakeProgress:
            def __init__(self):
                self.increments: list[int] = []

            def advance(self, count):
                self.increments.append(count)

        simulation = FakeSimulation()
        progress = FakeProgress()
        example._advance_simulation_to_target(
            simulation,
            12,
            progress,
            chunk_steps=4,
        )

        self.assertEqual(simulation.calls, [4, 4, 1])
        self.assertEqual(progress.increments, [4, 4, 1])
        self.assertEqual(simulation.steps, 12)

    def test_parser_rejects_invalid_geometry_and_update_interval(self):
        invalid_cases = (
            ["--thermal-update-interval", "0"],
            ["--drop-height-m", "0.0001"],
            ["--domain-height-m", "0.051"],
        )
        for argv in invalid_cases:
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        example.parse_args(argv)

    def test_output_contract_contains_motion_and_temperature(self):
        args = example.parse_args(
            [
                "--save-npz",
                "--momentum-collision-model",
                "cm_mrt",
                "--phase-normal-epsilon",
                "0.001",
                "--cm-acoustic-relaxation-rate",
                "0.25",
            ]
        )
        config = example.create_config(args)
        simulation = _FakeMovingSimulation(config)
        phase = np.zeros((config.nx, config.ny), dtype=np.float32)
        phase[5:10, 5:10] = 1.0
        simulation.water_phase = _FakeField(phase)
        simulation.temperature = _FakeField(np.where(phase > 0.0, 35.0, 20.0))
        snapshot = example._capture_snapshot(
            simulation,
            initial_solid_volume_cells=simulation.phase_change_solid_volume_cells(),
        )
        self.assertAlmostEqual(snapshot.water_mean_temperature_c, 35.0)
        self.assertEqual(snapshot.water_volume_current_cells, 25.0)
        self.assertEqual(
            len(example._history_row(snapshot)), len(example.HISTORY_COLUMNS)
        )
        self.assertEqual(
            len(example.HISTORY_COLUMNS), len(set(example.HISTORY_COLUMNS))
        )
        removed = {
            "generated_water_volume_cells",
            "water_volume_target_cells",
            "water_volume_residual_cells",
            "energy_residual_j_m",
            "melt_injection_mass_residual_kg_m",
            "thermal_total_mass_residual_kg_m",
            "cumulative_fluid_melt_momentum_lattice",
        }
        self.assertTrue(removed.isdisjoint(example.HISTORY_COLUMNS))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            example.write_history_csv(output / "history.csv", [snapshot])
            lines = (output / "history.csv").read_text().splitlines()
            self.assertEqual(tuple(lines[0].split(",")), example.HISTORY_COLUMNS)
            self.assertEqual(len(lines[1].split(",")), len(example.HISTORY_COLUMNS))
            example.write_fields_npz(output / "fields.npz", config, [snapshot])
            with np.load(output / "fields.npz") as archive:
                self.assertTrue(removed.isdisjoint(archive.files))
                self.assertEqual(archive["water_mean_temperature_c"][0], 35.0)
                np.testing.assert_array_equal(archive["water_phase"][0], phase.T)
            example.write_metadata(
                output / "metadata.json",
                args=args,
                config=config,
                scales=simulation.scales,
                targets=[0],
                initial=snapshot,
                final=snapshot,
                snapshot_count=1,
                velocity_sequence=None,
                vorticity_sequence=None,
                temperature_sequence=None,
            )
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["config"]["phase_normal_epsilon"], 1e-3)
            self.assertEqual(metadata["config"]["cm_acoustic_relaxation_rate"], 0.25)
            self.assertTrue(removed.isdisjoint(metadata["results"]))
            self.assertNotIn("closed_bath_reference", metadata)
            self.assertIn("disappears", metadata["phase_change"])
            self.assertEqual(metadata["results"]["water_mean_temperature_c"], 35.0)
            self.assertEqual(
                metadata["viscosity"]["water_model"], "iapws_local_temperature"
            )
            self.assertEqual(
                metadata["viscosity"]["air_model"], "sutherland_prescribed_temperature"
            )
            self.assertTrue(metadata["field_output"]["raw_fields_retained"])

    def test_common_sequence_writers_accept_a_falling_melting_title(self):
        import inspect

        for writer in (
            example.reporting.VelocitySequenceStream,
            example.reporting.VorticitySequenceStream,
            example.reporting.TemperatureSequenceStream,
        ):
            with self.subTest(writer=writer.__name__):
                self.assertIn("scenario_label", inspect.signature(writer).parameters)


if __name__ == "__main__":
    unittest.main()
