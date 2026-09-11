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

from iceflow2d.examples import coupled_falling_ice_melting_2d as example
from iceflow2d.thermal import LatticeScales


class _FakeField:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _key):
        return self.value

    def to_numpy(self):
        return np.asarray(self.value)


class _FakeMovingThermal:
    def __init__(self, total_mass_kg_m: float):
        self.time_s = 0.0
        self.steps = 0
        self.total_mass_kg_m = total_mass_kg_m
        self.boundary_heat_input_j_m = _FakeField(0.0)
        self.melt_injection_mass_residual_kg_m = _FakeField(4.0e-15)

    def total_enthalpy_j_m(self, _wall) -> float:
        return 0.0

    def mass_energy_totals(self):
        return SimpleNamespace(total_mass_kg_m=self.total_mass_kg_m)


class _FakeMovingSimulation:
    def __init__(self, config):
        self.cfg = config
        shape = (config.nx, config.ny)
        vector_shape = (*shape, 2)
        scalar_float = _FakeField(np.zeros(shape, dtype=np.float32))
        self.temperature = _FakeField(np.zeros(shape, dtype=np.float64))
        self.liquid_fraction = scalar_float
        self.thermal_enthalpy = _FakeField(np.zeros(shape, dtype=np.float64))
        self.phi = scalar_float
        self.solid = _FakeField(np.zeros(shape, dtype=np.int8))
        self.wall = _FakeField(np.zeros(shape, dtype=np.int8))
        self.sdf = scalar_float
        self.phase_change_material = _FakeField(np.zeros(shape, dtype=np.int8))
        self.u = _FakeField(np.zeros(vector_shape, dtype=np.float32))
        self.fluid_force = _FakeField(np.zeros(vector_shape, dtype=np.float32))
        water_volume = float(config.water_width * config.water_height)
        self.water_volume_target = _FakeField(water_volume)
        self.water_volume_current = _FakeField(water_volume)
        self.phase_aperture_water_residual_cells = _FakeField(1.0e-12)
        self.phase_aperture_energy_residual_j_m = _FakeField(2.0e-12)
        self.phase_aperture_capacity_margin_cells = _FakeField(3.0)
        self.thermal = _FakeMovingThermal(total_mass_kg_m=0.75)
        self.steps = 0
        self.physical_time_s = 0.0

    def phase_change_solid_volume_cells(self) -> float:
        return float(self.cfg.ice_width * self.cfg.ice_height)

    def phase_change_geometry_volume_cells(self) -> float:
        return self.phase_change_solid_volume_cells()


class CoupledFallingMeltingExampleTests(unittest.TestCase):
    def test_module_can_be_loaded_while_cuda_simulator_import_is_blocked(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "iceflow2d"
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
        previous = sys.modules.get("iceflow2d.simulator", ...)
        sys.modules["iceflow2d.simulator"] = None
        try:
            spec.loader.exec_module(module)
            config = module.create_config(module.parse_args([]))
        finally:
            sys.modules.pop(spec.name, None)
            if previous is ...:
                sys.modules.pop("iceflow2d.simulator", None)
            else:
                sys.modules["iceflow2d.simulator"] = previous

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
        self.assertEqual(config.thermal.boundaries.left.value, 90.0)
        self.assertEqual(config.thermal.boundaries.right.value, 90.0)
        self.assertEqual(config.thermal.boundaries.bottom.value, 90.0)
        self.assertEqual(config.thermal.boundaries.top.kind, "adiabatic")
        self.assertEqual(config.gravity, (0.0, -9.8))
        self.assertEqual(config.sigma, 0.072)
        self.assertEqual(config.reference_velocity, 4.0)
        self.assertEqual(config.air_interface_relaxation_time, 0.8)
        self.assertEqual(args.end_time_s, 2.0)
        self.assertEqual(args.output_interval_s, 0.01)

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
        self.assertEqual(targets[-1], 320000)
        self.assertEqual(len(targets), 201)
        self.assertTrue(
            all(
                target % config.thermal.update_interval_lbm_steps == 0
                for target in targets
            )
        )
        self.assertGreaterEqual(targets[-1] * scales.dt_s, args.end_time_s)
        self.assertEqual(example._progress_chunk_steps(targets[-1]), 160)

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

    def test_parser_rejects_nonconservative_moving_body_controls(self):
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

    def test_history_contract_contains_motion_phase_and_ale_diagnostics(self):
        required = {
            "physical_time_s",
            "melted_fraction",
            "body_center_x_m",
            "body_center_y_m",
            "body_velocity_x_m_s",
            "body_velocity_y_m_s",
            "body_mass_lattice",
            "body_inertia_lattice",
            "cumulative_melted_mass_lattice",
            "cumulative_fluid_melt_momentum_x_lattice",
            "cumulative_fluid_melt_momentum_y_lattice",
            "melt_momentum_residual_x_lattice",
            "melt_momentum_residual_y_lattice",
            "cumulative_melted_angular_momentum_lattice",
            "cumulative_fluid_melt_angular_momentum_lattice",
            "melt_angular_momentum_residual_lattice",
            "ale_water_residual_cells",
            "ale_energy_residual_j_m",
            "phase_aperture_water_residual_cells",
            "phase_aperture_energy_residual_j_m",
            "phase_aperture_capacity_margin_cells",
            "melt_injection_mass_residual_kg_m",
            "thermal_initial_total_mass_kg_m",
            "thermal_total_mass_kg_m",
            "thermal_total_mass_residual_kg_m",
        }
        self.assertTrue(required.issubset(example.HISTORY_COLUMNS))
        self.assertEqual(
            len(example.HISTORY_COLUMNS), len(set(example.HISTORY_COLUMNS))
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            example.write_history_csv(path, [])
            header = path.read_text(encoding="utf-8").strip().split(",")
        self.assertEqual(tuple(header), example.HISTORY_COLUMNS)

    def test_conservative_diagnostics_reach_npz_and_metadata(self):
        args = example.parse_args(
            ["--resolution-x", "100", "--resolution-y", "200", "--save-npz"]
        )
        config = example.create_config(args)
        simulation = _FakeMovingSimulation(config)
        initial_solid = simulation.phase_change_solid_volume_cells()
        initial = example._capture_snapshot(
            simulation,
            initial_total_enthalpy_j_m=0.0,
            initial_solid_volume_cells=initial_solid,
        )
        simulation.thermal.total_mass_kg_m -= 2.5e-9
        final = example._capture_snapshot(
            simulation,
            initial_total_enthalpy_j_m=0.0,
            initial_solid_volume_cells=initial_solid,
            initial_total_mass_kg_m=initial.thermal_total_mass_kg_m,
        )

        self.assertEqual(initial.thermal_total_mass_kg_m, 0.75)
        self.assertAlmostEqual(final.thermal_total_mass_residual_kg_m, -2.5e-9)
        self.assertEqual(final.phase_aperture_capacity_margin_cells, 3.0)
        self.assertEqual(final.melt_injection_mass_residual_kg_m, 4.0e-15)
        self.assertEqual(len(example._history_row(final)), len(example.HISTORY_COLUMNS))

        diagnostic_names = {
            "phase_aperture_water_residual_cells",
            "phase_aperture_energy_residual_j_m",
            "phase_aperture_capacity_margin_cells",
            "melt_injection_mass_residual_kg_m",
            "thermal_initial_total_mass_kg_m",
            "thermal_total_mass_kg_m",
            "thermal_total_mass_residual_kg_m",
            "body_solid_center_local_cells",
            "water_volume_residual_cells",
            "cumulative_melted_angular_momentum_lattice",
            "cumulative_fluid_melt_angular_momentum_lattice",
            "melt_angular_momentum_residual_lattice",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            npz_path = output / "fields.npz"
            example.write_fields_npz(npz_path, config, [initial, final])
            with np.load(npz_path) as archive:
                self.assertTrue(diagnostic_names.issubset(archive.files))
                self.assertAlmostEqual(
                    archive["thermal_total_mass_residual_kg_m"][-1], -2.5e-9
                )

            metadata_path = output / "metadata.json"
            example.write_metadata(
                metadata_path,
                args=args,
                config=config,
                scales=LatticeScales.from_iceflow_config(config),
                targets=[0],
                initial=initial,
                final=final,
                snapshot_count=2,
                velocity_sequence=None,
                vorticity_sequence=None,
                temperature_sequence=None,
            )
            results = json.loads(metadata_path.read_text(encoding="utf-8"))["results"]
            self.assertTrue(diagnostic_names.issubset(results))
            self.assertAlmostEqual(results["thermal_total_mass_residual_kg_m"], -2.5e-9)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["requested"]["progress_enabled"])
            self.assertEqual(
                metadata["field_output"]["write_mode"],
                "streaming_png_gif",
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
