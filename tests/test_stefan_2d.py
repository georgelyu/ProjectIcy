from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from iceflow2d.examples import stefan_melting_2d
from iceflow2d.stefan2d import (
    Stefan2DConfig,
    Stefan2DSolver,
    d4_symmetry_error,
    recover_temperature_and_liquid_fraction,
)


class Stefan2DConfigTests(unittest.TestCase):
    def test_default_scale_resolves_visible_melting(self):
        config = Stefan2DConfig()
        self.assertAlmostEqual(config.domain_width_m, 0.030, delta=1.0e-14)
        self.assertAlmostEqual(config.domain_height_m, 0.030, delta=1.0e-14)
        self.assertEqual((config.cells_x, config.cells_y), (120, 120))
        self.assertAlmostEqual(config.dx_m, 0.00025, delta=1.0e-14)
        self.assertAlmostEqual(config.dy_m, 0.00025, delta=1.0e-14)
        self.assertAlmostEqual(config.ice_width_m, 0.012, delta=1.0e-14)
        self.assertAlmostEqual(config.ice_height_m, 0.012, delta=1.0e-14)
        self.assertAlmostEqual(config.end_time_s, 180.0, delta=1.0e-14)
        self.assertAlmostEqual(
            config.actual_time_step_s,
            0.008948863636363635,
            delta=1.0e-15,
        )

    def test_invalid_geometry_or_unstable_controls_are_rejected(self):
        for overrides in (
            {"cells_x": 3},
            {"domain_width_m": 0.0},
            {"end_time_s": -1.0},
            {"fourier_number": 0.17},
            {"bath_temperature_c": 0.0},
            {"cells_x": 121},
            {"ice_width_m": 0.01225},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    Stefan2DConfig(**overrides)

        baseline = Stefan2DConfig()
        with self.assertRaisesRegex(ValueError, "stability limit"):
            Stefan2DConfig(time_step_s=baseline.maximum_time_step_s * 1.01)


class Stefan2DThermodynamicsTests(unittest.TestCase):
    def test_enthalpy_inversion_covers_all_phase_branches_in_two_dimensions(self):
        config = Stefan2DConfig()
        latent_volume = config.density_kg_m3 * config.latent_heat_j_kg
        enthalpy = np.asarray(
            [
                [
                    -config.density_kg_m3 * config.specific_heat_ice_j_kg_k * 7.0,
                    0.0,
                    0.125 * latent_volume,
                ],
                [
                    0.5 * latent_volume,
                    latent_volume,
                    latent_volume
                    + config.density_kg_m3 * config.specific_heat_water_j_kg_k * 13.0,
                ],
            ],
            dtype=np.float64,
        )
        temperature, liquid_fraction = recover_temperature_and_liquid_fraction(
            enthalpy, config
        )
        np.testing.assert_allclose(
            temperature,
            ((-7.0, 0.0, 0.0), (0.0, 0.0, 13.0)),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            liquid_fraction,
            ((0.0, 0.0, 0.125), (0.5, 1.0, 1.0)),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_one_rectangular_grid_step_matches_four_wall_heat_input(self):
        config = Stefan2DConfig(
            domain_width_m=0.008,
            domain_height_m=0.008,
            cells_x=8,
            cells_y=4,
            ice_width_m=0.004,
            ice_height_m=0.004,
            end_time_s=0.001,
            output_interval_s=0.001,
        )
        solver = Stefan2DSolver(config)
        temperature = np.linspace(5.0, 15.0, 32, dtype=np.float64).reshape(4, 8)
        latent_volume = config.density_kg_m3 * config.latent_heat_j_kg
        solver.temperature_c = temperature.copy()
        solver.liquid_fraction = np.ones_like(temperature)
        solver.enthalpy_j_m3 = latent_volume + config.density_kg_m3 * (
            config.specific_heat_water_j_kg_k
            * (temperature - config.melting_temperature_c)
        )
        solver.boundary_heat_input_j_m = 0.0
        solver._initial_total_enthalpy_j_m = solver.total_enthalpy_j_m

        hot = config.bath_temperature_c
        conductivity = config.conductivity_water_w_m_k
        expected_power = 2.0 * conductivity * config.dy_m / config.dx_m * float(
            np.sum(hot - temperature[:, 0]) + np.sum(hot - temperature[:, -1])
        ) + 2.0 * conductivity * config.dx_m / config.dy_m * float(
            np.sum(hot - temperature[0, :]) + np.sum(hot - temperature[-1, :])
        )
        time_step = 1.0e-4
        initial_energy = solver.total_enthalpy_j_m
        solver.step(time_step)
        expected_heat = time_step * expected_power
        stored_change = solver.total_enthalpy_j_m - initial_energy
        tolerance = max(1.0e-12, abs(expected_heat) * 1.0e-10)
        self.assertAlmostEqual(
            solver.boundary_heat_input_j_m, expected_heat, delta=tolerance
        )
        self.assertAlmostEqual(stored_change, expected_heat, delta=tolerance)
        self.assertAlmostEqual(solver.energy_residual_j_m, 0.0, delta=tolerance)

    def test_rectangular_equivalent_erosion_depth_uses_both_dimensions(self):
        config = Stefan2DConfig(
            domain_width_m=0.008,
            domain_height_m=0.006,
            cells_x=8,
            cells_y=6,
            ice_width_m=0.004,
            ice_height_m=0.002,
            end_time_s=0.001,
            output_interval_s=0.001,
        )
        solver = Stefan2DSolver(config)
        solver.liquid_fraction.fill(1.0)
        solver.liquid_fraction.flat[:3] = 0.0
        self.assertAlmostEqual(solver.ice_area_m2, 3.0e-6, delta=1.0e-15)
        self.assertAlmostEqual(
            solver.equivalent_uniform_melt_depth_m,
            0.0005,
            delta=1.0e-15,
        )

    def test_d4_error_detects_a_non_symmetric_field(self):
        symmetric = np.asarray([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0], [1.0, 2.0, 1.0]])
        self.assertEqual(d4_symmetry_error(symmetric), 0.0)
        asymmetric = symmetric.copy()
        asymmetric[0, 0] = 4.0
        self.assertGreater(d4_symmetry_error(asymmetric), 0.0)
        with self.assertRaises(ValueError):
            d4_symmetry_error(np.zeros((2, 3)))


class Stefan2DNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Stefan2DConfig()
        cls.solver = Stefan2DSolver(cls.config)
        cls.snapshots = cls.solver.run()

    def test_ice_area_decreases_and_default_melting_is_clearly_resolved(self):
        ice_area = np.asarray([item.ice_area_m2 for item in self.snapshots])
        self.assertTrue(np.all(np.diff(ice_area) < 0.0))
        self.assertAlmostEqual(ice_area[0], 144.0e-6, delta=1.0e-15)
        final = self.snapshots[-1]
        self.assertGreater(final.melted_fraction, 0.42)
        self.assertLess(final.melted_fraction, 0.46)
        self.assertGreater(
            final.equivalent_uniform_melt_depth_m,
            5.0 * min(self.config.dx_m, self.config.dy_m),
        )
        self.assertEqual(
            final.liquid_fraction[self.config.cells_y // 2, self.config.cells_x // 2],
            0.0,
        )

    def test_integral_geometry_measures_are_consistent(self):
        final = self.snapshots[-1]
        cell_area = self.config.dx_m * self.config.dy_m
        measured_area = float(
            np.sum(1.0 - final.liquid_fraction, dtype=np.float64) * cell_area
        )
        self.assertAlmostEqual(final.ice_area_m2, measured_area, delta=1.0e-15)
        self.assertAlmostEqual(
            final.equivalent_square_side_m**2, final.ice_area_m2, delta=1.0e-15
        )
        self.assertAlmostEqual(
            final.equivalent_square_half_width_m,
            0.5 * final.equivalent_square_side_m,
            delta=1.0e-15,
        )
        self.assertAlmostEqual(
            final.equivalent_uniform_melt_depth_m,
            0.5 * (self.config.ice_width_m - final.equivalent_square_side_m),
            delta=1.0e-15,
        )

    def test_enthalpy_update_is_conservative(self):
        final = self.snapshots[-1]
        relative_residual = abs(final.energy_residual_j_m) / max(
            1.0, abs(final.boundary_heat_input_j_m)
        )
        self.assertLess(relative_residual, 1.0e-10)

    def test_fields_remain_finite_bounded_and_d4_symmetric(self):
        final = self.snapshots[-1]
        for field in (
            final.enthalpy_j_m3,
            final.temperature_c,
            final.liquid_fraction,
        ):
            self.assertTrue(np.isfinite(field).all())
            scale = max(1.0, float(np.max(np.abs(field))))
            self.assertLess(d4_symmetry_error(field) / scale, 1.0e-12)
        self.assertGreaterEqual(float(np.min(final.liquid_fraction)), 0.0)
        self.assertLessEqual(float(np.max(final.liquid_fraction)), 1.0)
        self.assertGreaterEqual(
            float(np.min(final.temperature_c)),
            self.config.melting_temperature_c - 1.0e-12,
        )
        self.assertLessEqual(
            float(np.max(final.temperature_c)),
            self.config.bath_temperature_c + 1.0e-12,
        )
        self.assertLess(final.normalized_symmetry_error, 1.0e-12)

    def test_equivalent_melt_depth_converges_under_grid_refinement(self):
        depths = []
        for cells in (80, 120, 160):
            config = Stefan2DConfig(
                cells_x=cells,
                cells_y=cells,
                end_time_s=60.0,
                output_interval_s=60.0,
            )
            depths.append(
                Stefan2DSolver(config).run()[-1].equivalent_uniform_melt_depth_m
            )
        coarse_difference = abs(depths[0] - depths[1])
        fine_difference = abs(depths[1] - depths[2])
        self.assertGreater(depths[0], depths[1])
        self.assertGreater(depths[1], depths[2])
        self.assertGreater(depths[2], 0.78e-3)
        self.assertLess(depths[2], 0.84e-3)
        self.assertLess(fine_difference, coarse_difference)
        self.assertLess(fine_difference, 0.05 * (0.030 / 160.0))


class Stefan2DExampleTests(unittest.TestCase):
    def test_cpu_cli_path_writes_two_dimensional_outputs(self):
        with tempfile.TemporaryDirectory(prefix="stefan-2d-test-") as directory:
            output = Path(directory)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                stefan_melting_2d.main(
                    [
                        "--domain-width-mm",
                        "6",
                        "--domain-height-mm",
                        "4",
                        "--cells-x",
                        "24",
                        "--cells-y",
                        "16",
                        "--ice-width-mm",
                        "3",
                        "--ice-height-mm",
                        "2",
                        "--end-time-s",
                        "0.1",
                        "--output-interval-s",
                        "0.05",
                        "--output-dir",
                        str(output),
                        "--quiet",
                    ]
                )
            self.assertEqual(stdout.getvalue(), "")

            history_path = output / "history.csv"
            metadata_path = output / "metadata.json"
            fields_path = output / "fields.npz"
            self.assertGreater(history_path.stat().st_size, 0)
            self.assertGreater(metadata_path.stat().st_size, 0)
            self.assertGreater(fields_path.stat().st_size, 0)
            self.assertGreater((output / "stefan_melting_2d.png").stat().st_size, 0)

            with history_path.open(encoding="utf-8", newline="") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(len(history), 3)
            self.assertEqual(
                [float(row["time_s"]) for row in history], [0.0, 0.05, 0.1]
            )
            self.assertTrue(
                all(
                    np.isfinite(float(value))
                    for row in history
                    for value in row.values()
                )
            )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["backend"], "numpy-cpu")
            self.assertEqual(metadata["mechanics"], "fixed ice; no fluid motion")
            self.assertAlmostEqual(metadata["config"]["domain_width_m"], 0.006)
            self.assertAlmostEqual(metadata["config"]["domain_height_m"], 0.004)
            self.assertEqual(metadata["config"]["cells_x"], 24)
            self.assertEqual(metadata["config"]["cells_y"], 16)
            self.assertNotIn("analytic", json.dumps(metadata).lower())

            with np.load(fields_path) as fields:
                self.assertEqual(fields["x_m"].shape, (24,))
                self.assertEqual(fields["y_m"].shape, (16,))
                self.assertEqual(fields["temperature_c"].shape, (3, 16, 24))
                self.assertEqual(fields["liquid_fraction"].shape, (3, 16, 24))
                self.assertEqual(fields["enthalpy_j_m3"].shape, (3, 16, 24))
                self.assertAlmostEqual(float(fields["time_s"][-1]), 0.1)
                self.assertTrue(np.isfinite(fields["enthalpy_j_m3"]).all())
                diagnostics = (
                    "ice_area_m2",
                    "equivalent_square_side_m",
                    "equivalent_square_half_width_m",
                    "equivalent_uniform_melt_depth_m",
                    "melted_fraction",
                    "total_enthalpy_j_m",
                    "boundary_heat_input_j_m",
                    "energy_residual_j_m",
                    "normalized_symmetry_error",
                )
                for name in diagnostics:
                    self.assertEqual(fields[name].shape, (3,))
                    self.assertTrue(np.isfinite(fields[name]).all())

                cross_file_fields = (
                    "ice_area_m2",
                    "equivalent_square_side_m",
                    "equivalent_square_half_width_m",
                    "equivalent_uniform_melt_depth_m",
                    "melted_fraction",
                    "total_enthalpy_j_m",
                    "boundary_heat_input_j_m",
                    "energy_residual_j_m",
                    "normalized_symmetry_error",
                )
                for name in cross_file_fields:
                    expected = float(fields[name][-1])
                    self.assertAlmostEqual(float(history[-1][name]), expected)
                    self.assertAlmostEqual(metadata["results"][name], expected)


if __name__ == "__main__":
    unittest.main()
