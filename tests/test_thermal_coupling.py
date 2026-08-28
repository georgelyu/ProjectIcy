"""Pure-CPU contract tests for the optional thermal/LBM coupling.

This module deliberately imports neither :mod:`taichi` nor
``iceflow2d.simulator``.  It exercises the configuration, dimensional scaling,
and thermodynamic reference functions used by both the CPU benchmark and the
CUDA implementation.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from iceflow2d.examples import coupled_fixed_ice_melting_2d as coupled_example
from iceflow2d.thermal import (
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
    phase_change_active_water_target_cells,
    phase_change_enthalpy_numpy,
    phase_change_water_target_cells,
    recover_temperature_and_liquid_fraction_numpy,
    water_density_anomaly_ratio_to_reference,
    water_density_ratio_to_reference,
)


class ThermalConfigTests(unittest.TestCase):
    def test_warm_bath_configuration_is_explicit_and_validated(self):
        hot_wall = ThermalBoundary.dirichlet(20.0)
        boundaries = ThermalBoundarySet(
            left=hot_wall,
            right=hot_wall,
            bottom=hot_wall,
            top=hot_wall,
        )
        config = ThermalConfig(
            boundaries=boundaries,
            initial_water_temperature_c=20.0,
            initial_ice_temperature_c=0.0,
            initial_air_temperature_c=20.0,
            advection_enabled=True,
        )

        self.assertEqual(config.scheme, "enthalpy_fv")
        self.assertTrue(config.advection_enabled)
        self.assertTrue(config.water_air_interface_adiabatic)
        self.assertEqual(config.update_interval_lbm_steps, 1)
        self.assertAlmostEqual(config.solid_liquid_threshold, 0.5)
        self.assertAlmostEqual(config.max_fourier_number, 0.15)
        self.assertAlmostEqual(config.max_courant_number, 0.5)
        self.assertEqual(config.buoyancy_reference_temperature_c, 20.0)
        self.assertEqual(
            tuple(
                getattr(config.boundaries, side)
                for side in ("left", "right", "bottom", "top")
            ),
            (hot_wall,) * 4,
        )

    def test_invalid_thermal_controls_are_rejected_without_gpu_runtime(self):
        for overrides in (
            {"initial_water_temperature_c": -1.0},
            {"initial_ice_temperature_c": 1.0},
            {"solid_liquid_threshold": 0.0},
            {"solid_liquid_threshold": 1.0},
            {"max_fourier_number": 0.17},
            {"max_courant_number": 1.01},
            {"update_interval_lbm_steps": 0},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    ThermalConfig(**overrides)

        with self.assertRaisesRegex(ValueError, "adiabatic"):
            ThermalBoundary("adiabatic", 1.0)
        with self.assertRaisesRegex(ValueError, "kind"):
            ThermalBoundary("radiative", 0.0)  # type: ignore[arg-type]


class FreshwaterBuoyancyTests(unittest.TestCase):
    @staticmethod
    def config_for_bath(temperature_c: float) -> ThermalConfig:
        return ThermalConfig(
            initial_water_temperature_c=temperature_c,
            initial_air_temperature_c=temperature_c,
            water_buoyancy_model="freshwater_quadratic",
            buoyancy_reference_temperature_c=temperature_c,
        )

    def test_quadratic_density_is_maximal_at_four_degrees_and_symmetric(self):
        config = self.config_for_bath(8.0)
        ratios = water_density_ratio_to_reference(np.asarray([0.0, 4.0, 8.0]), config)

        self.assertGreater(ratios[1], ratios[0])
        self.assertGreater(ratios[1], ratios[2])
        self.assertEqual(ratios[0], ratios[2])
        self.assertEqual(ratios[2], 1.0)

    def test_far_field_anomaly_is_exactly_zero_for_each_paper_regime(self):
        for bath_temperature in (4.0, 5.6, 8.0):
            with self.subTest(bath_temperature=bath_temperature):
                config = self.config_for_bath(bath_temperature)
                self.assertEqual(
                    water_density_anomaly_ratio_to_reference(bath_temperature, config),
                    0.0,
                )

    def test_quadratic_buoyancy_changes_direction_across_density_maximum(self):
        gravity_y = -9.8

        def acceleration_y(bath_temperature: float, local_temperature: float) -> float:
            anomaly = water_density_anomaly_ratio_to_reference(
                local_temperature, self.config_for_bath(bath_temperature)
            )
            return gravity_y * anomaly

        self.assertGreater(acceleration_y(4.0, 0.0), 0.0)
        self.assertGreater(acceleration_y(5.6, 0.0), 0.0)
        self.assertLess(acceleration_y(5.6, 4.0), 0.0)
        self.assertLess(acceleration_y(8.0, 4.0), 0.0)
        self.assertLess(acceleration_y(8.0, 6.0), 0.0)
        # The idealized EOS is exactly symmetric about 4 degC.
        self.assertEqual(acceleration_y(8.0, 0.0), 0.0)

    def test_invalid_quadratic_eos_controls_are_rejected(self):
        invalid_overrides = (
            {"water_buoyancy_model": "cubic"},
            {"freshwater_density_quadratic_coefficient_1_k2": -1.0e-6},
            {"freshwater_density_quadratic_coefficient_1_k2": math.nan},
            {"freshwater_density_max_temperature_c": math.nan},
            {"buoyancy_reference_temperature_c": math.nan},
            {
                "initial_water_temperature_c": 6.0,
                "water_buoyancy_model": "freshwater_quadratic",
                "buoyancy_reference_temperature_c": 6.0,
                "freshwater_density_max_temperature_c": 4.0,
                "freshwater_density_quadratic_coefficient_1_k2": 0.25,
            },
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    ThermalConfig(**overrides)


class UnequalDensityThermodynamicsTests(unittest.TestCase):
    def test_real_density_enthalpy_round_trip_covers_solid_mushy_and_liquid(self):
        properties = PhaseChangeProperties()
        density_ice = 917.0
        density_water = 1000.0
        temperature = np.asarray(
            [[-10.0, 0.0, 0.0], [0.0, 0.0, 20.0]], dtype=np.float64
        )
        liquid_fraction = np.asarray(
            [[0.0, 0.0, 0.25], [0.50, 1.0, 1.0]], dtype=np.float64
        )

        enthalpy = phase_change_enthalpy_numpy(
            temperature,
            liquid_fraction,
            properties,
            density_ice_kg_m3=density_ice,
            density_water_kg_m3=density_water,
        )
        recovered_temperature, recovered_fraction = (
            recover_temperature_and_liquid_fraction_numpy(
                enthalpy,
                properties,
                density_ice_kg_m3=density_ice,
                density_water_kg_m3=density_water,
            )
        )

        latent_volume = density_ice * properties.latent_heat_j_kg
        self.assertAlmostEqual(enthalpy[0, 0], -10.0 * density_ice * 2100.0)
        self.assertAlmostEqual(enthalpy[0, 2], 0.25 * latent_volume)
        self.assertAlmostEqual(enthalpy[1, 1], latent_volume)
        self.assertAlmostEqual(
            enthalpy[1, 2],
            latent_volume
            + density_water * properties.specific_heat_water_j_kg_k * 20.0,
        )
        np.testing.assert_allclose(
            recovered_temperature, temperature, rtol=0.0, atol=1.0e-13
        )
        np.testing.assert_allclose(
            recovered_fraction, liquid_fraction, rtol=0.0, atol=1.0e-15
        )

    def test_water_target_uses_ice_to_water_density_ratio_and_is_reversible(self):
        initial_water = 400.0
        initial_solid = 100.0
        density_ice = 917.0
        density_water = 1000.0

        partly_melted = phase_change_water_target_cells(
            initial_water,
            initial_solid,
            60.0,
            density_ice_kg_m3=density_ice,
            density_water_kg_m3=density_water,
        )
        refrozen = phase_change_water_target_cells(
            initial_water,
            initial_solid,
            initial_solid,
            density_ice_kg_m3=density_ice,
            density_water_kg_m3=density_water,
        )
        equal_density = phase_change_water_target_cells(
            initial_water,
            initial_solid,
            60.0,
            density_ice_kg_m3=density_water,
            density_water_kg_m3=density_water,
        )

        self.assertAlmostEqual(partly_melted, 400.0 + 40.0 * 0.917)
        self.assertAlmostEqual(refrozen, initial_water)
        self.assertAlmostEqual(equal_density, 440.0)
        self.assertLess(partly_melted, equal_density)

    def test_active_water_target_compensates_sharp_node_transitions(self):
        arguments = {
            "initial_water_volume_cells": 400.0,
            "initial_solid_volume_cells": 100.0,
            "density_ice_kg_m3": 917.0,
            "density_water_kg_m3": 1000.0,
        }
        before_mask_release = phase_change_active_water_target_cells(
            current_solid_volume_cells=60.0,
            initial_sharp_geometry_volume_cells=100.0,
            current_sharp_geometry_volume_cells=100.0,
            **arguments,
        )
        after_mask_release = phase_change_active_water_target_cells(
            current_solid_volume_cells=60.0,
            initial_sharp_geometry_volume_cells=100.0,
            current_sharp_geometry_volume_cells=60.0,
            **arguments,
        )
        total_liquid = phase_change_water_target_cells(
            400.0,
            100.0,
            60.0,
            density_ice_kg_m3=917.0,
            density_water_kg_m3=1000.0,
        )

        self.assertAlmostEqual(before_mask_release, 400.0 - 40.0 * 0.083)
        self.assertAlmostEqual(after_mask_release, total_liquid)
        self.assertAlmostEqual(after_mask_release - before_mask_release, 40.0)


class LatticeScalesTests(unittest.TestCase):
    def test_fixed_reference_velocity_sets_time_and_diffusivity_scales(self):
        reference_velocity = 1.0e-3
        scales = LatticeScales.from_reference_velocity(
            dx_m=0.25e-3,
            reference_velocity_m_s=reference_velocity,
            reference_density_kg_m3=1000.0,
        )
        expected_dt = 0.1 * 0.25e-3 / reference_velocity

        self.assertAlmostEqual(scales.dx_m, 0.25e-3)
        self.assertAlmostEqual(scales.dt_s, expected_dt)
        self.assertAlmostEqual(scales.reference_velocity_m_s, reference_velocity)
        self.assertAlmostEqual(scales.velocity_to_physical(0.1), reference_velocity)
        self.assertAlmostEqual(scales.velocity_to_lattice(reference_velocity), 0.1)
        self.assertAlmostEqual(
            scales.velocity_to_physical(scales.velocity_to_lattice(0.02)), 0.02
        )
        diffusivity = 2.2 / (917.0 * 2100.0)
        self.assertAlmostEqual(
            scales.diffusivity_to_lattice(diffusivity),
            diffusivity * expected_dt / (0.25e-3) ** 2,
        )

    def test_config_scaling_is_independent_of_gravity(self):
        common = {
            "dx": 1.25e-4,
            "reference_velocity": 1.0e-3,
            "rho_water": 1000.0,
        }
        zero_gravity = SimpleNamespace(**common, gravity=(0.0, 0.0))
        physical_gravity = SimpleNamespace(**common, gravity=(0.0, -9.8))
        zero_scales = LatticeScales.from_iceflow_config(zero_gravity)
        gravity_scales = LatticeScales.from_iceflow_config(physical_gravity)

        self.assertEqual(zero_scales, gravity_scales)
        self.assertAlmostEqual(zero_scales.dt_s, 0.0125)


class CoupledExampleConfigTests(unittest.TestCase):
    def test_default_example_is_a_real_density_centimetre_case(self):
        args = coupled_example.parse_args([])
        config = coupled_example.create_config()
        thermal = config.thermal
        self.assertIsNotNone(thermal)
        assert thermal is not None

        self.assertEqual(config.resolution, (200, 280))
        self.assertAlmostEqual(config.nx * config.dx, 0.025)
        self.assertAlmostEqual(config.ny * config.dx, 0.035)
        self.assertAlmostEqual(config.dx, 1.25e-4)
        self.assertAlmostEqual(config.reference_velocity, 1.0e-3)
        self.assertEqual(config.boundary_cells, 3)
        self.assertEqual((config.water_width, config.water_height), (197, 220))
        self.assertEqual((config.ice_width, config.ice_height), (64, 64))
        self.assertEqual(config.ice_initial_center, (100.0, 120.0))
        self.assertAlmostEqual(config.ice_width * config.dx, 0.008)
        self.assertAlmostEqual(config.water_height * config.dx, 0.0275)
        self.assertAlmostEqual(config.ice_initial_center[1] * config.dx, 0.015)
        self.assertTrue(config.ice_fixed)
        self.assertEqual(config.ice_initial_velocity, (0.0, 0.0))
        self.assertEqual(config.ice_initial_angular_velocity, 0.0)
        self.assertEqual(config.gravity, (0.0, 0.0))
        self.assertEqual(config.sigma, 0.0)
        self.assertFalse(config.well_balanced_hydrostatics)
        self.assertEqual(config.volume_projection_tolerance, 1.0e-8)

        self.assertEqual(
            (config.rho_ice, config.rho_water, config.rho_air),
            (917.0, 1000.0, 1.25),
        )
        self.assertEqual(
            (
                thermal.initial_ice_temperature_c,
                thermal.initial_water_temperature_c,
                thermal.initial_air_temperature_c,
            ),
            (0.0, 60.0, 40.0),
        )
        self.assertEqual(thermal.properties.melting_temperature_c, 0.0)
        self.assertEqual(thermal.update_interval_lbm_steps, 10)
        self.assertTrue(thermal.advection_enabled)
        self.assertTrue(thermal.water_air_interface_adiabatic)
        self.assertEqual(thermal.water_buoyancy_model, "freshwater_quadratic")
        self.assertEqual(thermal.buoyancy_reference_temperature_c, 60.0)
        self.assertEqual(
            tuple(
                getattr(thermal.boundaries, side).kind
                for side in ("left", "right", "bottom", "top")
            ),
            ("dirichlet", "dirichlet", "dirichlet", "adiabatic"),
        )
        self.assertEqual(
            tuple(
                getattr(thermal.boundaries, side).value
                for side in ("left", "right", "bottom", "top")
            ),
            (60.0, 60.0, 60.0, 0.0),
        )

        scales = LatticeScales.from_iceflow_config(config)
        targets = coupled_example._snapshot_step_targets(
            config, args.end_time_s, args.output_interval_s
        )
        self.assertAlmostEqual(args.end_time_s, 2.0)
        self.assertAlmostEqual(args.output_interval_s, 0.01)
        self.assertFalse(args.save_npz)
        self.assertAlmostEqual(
            args.velocity_visualization_max_m_s,
            coupled_example.DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S,
        )
        self.assertAlmostEqual(
            args.vorticity_visualization_max_s_1,
            coupled_example.DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
        )
        self.assertEqual(targets, list(range(0, 161, 10)))
        self.assertAlmostEqual(scales.dt_s, 0.0125)
        self.assertAlmostEqual(scales.velocity_scale_m_s, 0.01)
        self.assertAlmostEqual(scales.reference_velocity_m_s, 1.0e-3)
        self.assertAlmostEqual(scales.velocity_to_physical(0.1), 1.0e-3)
        self.assertAlmostEqual(scales.velocity_to_lattice(1.0e-3), 0.1)
        self.assertAlmostEqual(
            0.5 + 3.0 * config.viscosity_water * scales.dt_s / config.dx**2,
            2.9,
        )
        self.assertAlmostEqual(
            0.5 + 3.0 * config.viscosity_air * scales.dt_s / config.dx**2,
            36.5,
        )
        self.assertGreaterEqual(targets[-1] * scales.dt_s, args.end_time_s)
        self.assertLess(
            targets[-1] * scales.dt_s - args.end_time_s,
            thermal.update_interval_lbm_steps * scales.dt_s,
        )

    def test_field_archive_is_explicitly_opt_in(self):
        self.assertFalse(coupled_example.parse_args([]).save_npz)
        self.assertTrue(coupled_example.parse_args(["--save-npz"]).save_npz)
        self.assertFalse(
            coupled_example.parse_args(["--save-npz", "--no-save-npz"]).save_npz
        )
        # Keep the previous spelling as a hidden compatibility alias.
        self.assertFalse(
            coupled_example.parse_args(["--save-npz", "--no-npz"]).save_npz
        )

    def test_nonzero_gravity_temperature_cases_use_quadratic_eos_and_wb(self):
        for bath_temperature in (4.0, 5.6, 8.0):
            with self.subTest(bath_temperature=bath_temperature):
                args = coupled_example.parse_args(
                    [
                        "--water-temperature-c",
                        str(bath_temperature),
                        "--air-temperature-c",
                        str(bath_temperature),
                        "--gravity-m-s2",
                        "9.8",
                        "--reference-velocity-m-s",
                        "0.4",
                    ]
                )
                config = coupled_example.create_config(args)
                thermal = config.thermal
                assert thermal is not None

                self.assertEqual(config.gravity, (0.0, -9.8))
                self.assertTrue(config.well_balanced_hydrostatics)
                self.assertEqual(thermal.water_buoyancy_model, "freshwater_quadratic")
                self.assertEqual(
                    thermal.buoyancy_reference_temperature_c, bath_temperature
                )
                self.assertEqual(
                    tuple(
                        getattr(thermal.boundaries, side).value
                        for side in ("left", "right", "bottom")
                    ),
                    (bath_temperature,) * 3,
                )

        disabled_args = coupled_example.parse_args(
            [
                "--water-temperature-c",
                "5.6",
                "--gravity-m-s2",
                "9.8",
                "--reference-velocity-m-s",
                "0.4",
                "--no-well-balanced-hydrostatics",
            ]
        )
        self.assertFalse(
            coupled_example.create_config(disabled_args).well_balanced_hydrostatics
        )

    def test_nonzero_gravity_accepts_the_low_default_velocity_scale(self):
        args = coupled_example.parse_args(["--gravity-m-s2", "9.8"])
        config = coupled_example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)

        self.assertEqual(config.gravity, (0.0, -9.8))
        self.assertTrue(config.well_balanced_hydrostatics)
        self.assertAlmostEqual(config.reference_velocity, 1.0e-3)
        self.assertAlmostEqual(scales.dt_s, 0.0125)
        self.assertAlmostEqual(
            abs(config.gravity[1]) * scales.dt_s**2 / config.dx,
            12.25,
        )

    def test_temperature_sweep_uses_distinct_stable_subdirectory_names(self):
        args = coupled_example.parse_args(["--water-temperatures-c", "4", "5.6", "8"])
        self.assertEqual(args.water_temperatures_c, [4.0, 5.6, 8.0])
        self.assertEqual(
            [
                coupled_example._temperature_directory_name(value)
                for value in args.water_temperatures_c
            ],
            ["T_4C", "T_5p6C", "T_8C"],
        )

    def test_visualization_limits_are_shared_across_temperature_cases(self):
        args = coupled_example.parse_args(["--water-temperatures-c", "8", "4", "5.6"])
        limits = coupled_example._shared_visualization_limits(
            args, args.water_temperatures_c
        )
        self.assertEqual(
            limits.velocity_max_m_s,
            coupled_example.DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S,
        )
        self.assertEqual(
            limits.vorticity_abs_max_s_1,
            coupled_example.DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
        )
        self.assertEqual(limits.temperature_min_c, 0.0)
        self.assertEqual(limits.temperature_max_c, 8.0)

        custom_args = coupled_example.parse_args(
            [
                "--water-temperature-c",
                "5.6",
                "--velocity-visualization-max-m-s",
                "7e-6",
                "--vorticity-visualization-max-s-1",
                "0.35",
            ]
        )
        custom_limits = coupled_example._shared_visualization_limits(
            custom_args, [custom_args.water_temperature_c]
        )
        self.assertEqual(custom_limits.velocity_max_m_s, 7.0e-6)
        self.assertEqual(custom_limits.vorticity_abs_max_s_1, 0.35)
        self.assertEqual(custom_limits.temperature_max_c, 5.6)

        for invalid in ("0", "-1e-6", "nan", "inf"):
            with self.subTest(invalid=invalid):
                invalid_args = coupled_example.parse_args(
                    [f"--velocity-visualization-max-m-s={invalid}"]
                )
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    coupled_example._shared_visualization_limits(
                        invalid_args, [invalid_args.water_temperature_c]
                    )

        for invalid in ("0", "-0.2", "nan", "inf"):
            with self.subTest(vorticity_invalid=invalid):
                invalid_args = coupled_example.parse_args(
                    [f"--vorticity-visualization-max-s-1={invalid}"]
                )
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    coupled_example._shared_visualization_limits(
                        invalid_args, [invalid_args.water_temperature_c]
                    )

    def test_sweep_main_passes_one_visualization_scale_to_every_case(self):
        recorded_limits = []

        def fake_run_case(
            case_args,
            config,
            targets,
            *,
            visualization_limits,
        ):
            recorded_limits.append(visualization_limits)
            return {"water_temperature_c": case_args.water_temperature_c}

        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch.object(
                coupled_example, "_run_case", side_effect=fake_run_case
            ):
                coupled_example.main(
                    [
                        "--water-temperatures-c",
                        "8",
                        "4",
                        "5.6",
                        "--output-dir",
                        temporary_directory,
                        "--no-plot",
                        "--quiet",
                    ]
                )

            self.assertEqual(len(recorded_limits), 3)
            self.assertTrue(all(item == recorded_limits[0] for item in recorded_limits))
            self.assertEqual(
                recorded_limits[0].velocity_max_m_s,
                coupled_example.DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S,
            )
            self.assertEqual(
                recorded_limits[0].vorticity_abs_max_s_1,
                coupled_example.DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
            )
            self.assertEqual(recorded_limits[0].temperature_max_c, 8.0)

            summary_path = Path(temporary_directory) / "temperature_sweep.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["visualization_scales"]["velocity_magnitude_range_m_s"],
                [0.0, coupled_example.DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S],
            )
            self.assertEqual(
                summary["visualization_scales"]["velocity_quiver_reference_speed_m_s"],
                coupled_example.DEFAULT_VELOCITY_VISUALIZATION_MAX_M_S,
            )
            self.assertEqual(
                summary["visualization_scales"]["vorticity_range_s_1"],
                [
                    -coupled_example.DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
                    coupled_example.DEFAULT_VORTICITY_VISUALIZATION_MAX_S_1,
                ],
            )
            self.assertEqual(
                summary["visualization_scales"]["temperature_range_c"],
                [0.0, 8.0],
            )

    def test_metadata_records_streamed_sequences_and_shared_scales(self):
        args = coupled_example.parse_args(
            [
                "--resolution-x",
                "20",
                "--resolution-y",
                "28",
                "--ice-size-m",
                "0.01",
            ]
        )
        config = coupled_example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)
        limits = coupled_example._shared_visualization_limits(
            args, [args.water_temperature_c]
        )
        snapshot = SimpleNamespace(
            lbm_steps=0,
            thermal_substeps=0,
            physical_time_s=0.0,
            thermal_time_s=0.0,
            solid_volume_cells=64.0,
            sharp_geometry_volume_cells=64.0,
            melted_fraction=0.0,
            solid_area_m2=64.0 * config.dx**2,
            equivalent_square_side_m=8.0 * config.dx,
            equivalent_uniform_melt_depth_m=0.0,
            generated_water_volume_cells=0.0,
            phase_change_contraction_cells=0.0,
            water_volume_target_cells=100.0,
            water_volume_current_cells=100.0,
            water_volume_residual_cells=0.0,
            total_enthalpy_j_m=0.0,
            boundary_heat_input_j_m=0.0,
            energy_residual_j_m=0.0,
        )
        velocity_summary = coupled_example.VelocitySequenceSummary(
            frame_directory="velocity_frames",
            frame_pattern="frame_*.png",
            frame_count=2,
            animation="velocity_field.gif",
            observed_max_m_s=4.0e-6,
            color_max_m_s=limits.velocity_max_m_s,
            quiver_stride_cells=2,
        )
        vorticity_summary = coupled_example.VorticitySequenceSummary(
            frame_directory="vorticity_frames",
            frame_pattern="frame_*.png",
            frame_count=2,
            animation="vorticity_field.gif",
            observed_abs_max_s_1=0.17,
            color_abs_max_s_1=limits.vorticity_abs_max_s_1,
            quiver_stride_cells=2,
        )
        temperature_summary = coupled_example.TemperatureSequenceSummary(
            frame_directory="temperature_frames",
            frame_pattern="frame_*.png",
            frame_count=2,
            animation="temperature_field.gif",
            observed_min_c=0.0,
            observed_max_c=5.4,
            color_min_c=limits.temperature_min_c,
            color_max_c=limits.temperature_max_c,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path = Path(temporary_directory) / "metadata.json"
            coupled_example.write_metadata(
                metadata_path,
                args=args,
                config=config,
                scales=scales,
                targets=[0],
                initial_snapshot=snapshot,
                final_snapshot=snapshot,
                snapshot_count=1,
                velocity_sequence=velocity_summary,
                vorticity_sequence=vorticity_summary,
                temperature_sequence=temperature_summary,
                visualization_limits=limits,
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(metadata["geometry"]["domain_width_m"], 0.025)
        self.assertEqual(metadata["geometry"]["domain_height_m"], 0.035)
        self.assertEqual(metadata["geometry"]["water_level_m"], 0.0275)
        self.assertEqual(metadata["geometry"]["ice_width_m"], 0.01)
        lattice_scaling = metadata["lattice_scaling"]
        self.assertEqual(lattice_scaling["dx_m"], 0.00125)
        self.assertEqual(lattice_scaling["dt_s"], 0.125)
        self.assertEqual(lattice_scaling["velocity_scale_m_s"], 0.01)
        self.assertEqual(lattice_scaling["reference_lattice_velocity"], 0.1)
        self.assertEqual(lattice_scaling["reference_velocity_m_s"], 0.001)

        vorticity = metadata["vorticity_visualization"]
        self.assertEqual(
            vorticity["field_definition"],
            "omega_z = d(v)/dx - d(u)/dy",
        )
        self.assertEqual(vorticity["source_field"], "physical_velocity_lattice")
        self.assertEqual(vorticity["units"], "s^-1")
        self.assertEqual(
            vorticity["range_s_1"],
            [-limits.vorticity_abs_max_s_1, limits.vorticity_abs_max_s_1],
        )
        self.assertEqual(vorticity["observed_abs_max_s_1"], 0.17)
        self.assertEqual(
            vorticity["quiver_reference_speed_m_s"],
            limits.velocity_max_m_s,
        )
        self.assertEqual(
            metadata["outputs"]["vorticity_frames"],
            {
                "directory": "vorticity_frames",
                "pattern": "frame_*.png",
                "count": 2,
            },
        )
        self.assertEqual(
            metadata["outputs"]["vorticity_animation"],
            "vorticity_field.gif",
        )
        temperature = metadata["temperature_visualization"]
        self.assertEqual(temperature["source_field"], "temperature_c")
        self.assertEqual(temperature["units"], "degC")
        self.assertEqual(
            temperature["range_c"],
            [limits.temperature_min_c, limits.temperature_max_c],
        )
        self.assertEqual(temperature["observed_range_c"], [0.0, 5.4])
        self.assertEqual(
            metadata["outputs"]["temperature_frames"],
            {
                "directory": "temperature_frames",
                "pattern": "frame_*.png",
                "count": 2,
            },
        )
        self.assertEqual(
            metadata["outputs"]["temperature_animation"],
            "temperature_field.gif",
        )
        self.assertFalse(metadata["requested"]["save_npz"])
        self.assertIsNone(metadata["outputs"]["fields"])
        self.assertEqual(metadata["results"]["snapshots"], 1)

    def test_run_case_streams_each_snapshot_without_default_field_archive(self):
        import iceflow2d

        class FakeThermal:
            steps = 0
            time_s = 0.0

            @staticmethod
            def total_enthalpy_j_m(_wall):
                return 0.0

        class FakeSimulation:
            def __init__(self, config):
                self.cfg = config
                self.thermal = FakeThermal()
                self.wall = object()
                self.steps = 0

            @staticmethod
            def phase_change_solid_volume_cells():
                return 64.0

            def step(self, count):
                self.steps += count

        class SequenceSpy:
            def __init__(self, summary):
                self.summary = summary
                self.appended_steps = []
                self.closed = False

            def append(self, snapshot):
                self.appended_steps.append(snapshot.lbm_steps)

            def close(self):
                self.closed = True
                return self.summary

        def capture_snapshot(simulation, **_):
            step = simulation.steps
            return SimpleNamespace(
                physical_time_s=float(step),
                thermal_time_s=float(step),
                lbm_steps=step,
                thermal_substeps=step // 10,
                solid_volume_cells=64.0,
                sharp_geometry_volume_cells=64.0,
                solid_area_m2=1.0,
                melted_fraction=0.0,
                equivalent_square_side_m=1.0,
                equivalent_uniform_melt_depth_m=0.0,
                generated_water_volume_cells=0.0,
                phase_change_contraction_cells=0.0,
                water_volume_target_cells=100.0,
                water_volume_current_cells=100.0,
                water_volume_residual_cells=0.0,
                total_enthalpy_j_m=0.0,
                boundary_heat_input_j_m=0.0,
                energy_residual_j_m=0.0,
            )

        velocity_summary = coupled_example.VelocitySequenceSummary(
            frame_directory="velocity_frames",
            frame_pattern="frame_*.png",
            frame_count=3,
            animation="velocity_field.gif",
            observed_max_m_s=0.0,
            color_max_m_s=5.5e-6,
            quiver_stride_cells=1,
        )
        vorticity_summary = coupled_example.VorticitySequenceSummary(
            frame_directory="vorticity_frames",
            frame_pattern="frame_*.png",
            frame_count=3,
            animation="vorticity_field.gif",
            observed_abs_max_s_1=0.0,
            color_abs_max_s_1=0.2,
            quiver_stride_cells=1,
        )
        temperature_summary = coupled_example.TemperatureSequenceSummary(
            frame_directory="temperature_frames",
            frame_pattern="frame_*.png",
            frame_count=3,
            animation="temperature_field.gif",
            observed_min_c=0.0,
            observed_max_c=60.0,
            color_min_c=0.0,
            color_max_c=60.0,
        )
        velocity_stream = SequenceSpy(velocity_summary)
        vorticity_stream = SequenceSpy(vorticity_summary)
        temperature_stream = SequenceSpy(temperature_summary)

        with tempfile.TemporaryDirectory() as temporary_directory:
            args = coupled_example.parse_args(
                [
                    "--resolution-x",
                    "20",
                    "--resolution-y",
                    "28",
                    "--ice-size-m",
                    "0.01",
                    "--output-dir",
                    temporary_directory,
                    "--quiet",
                ]
            )
            config = coupled_example.create_config(args)
            limits = coupled_example._shared_visualization_limits(
                args, [args.water_temperature_c]
            )
            with (
                mock.patch.dict(
                    iceflow2d.__dict__,
                    {"IceFlow2D": FakeSimulation},
                ),
                mock.patch.object(
                    coupled_example,
                    "_capture_snapshot",
                    side_effect=capture_snapshot,
                ),
                mock.patch.object(
                    coupled_example,
                    "VelocitySequenceStream",
                    return_value=velocity_stream,
                ),
                mock.patch.object(
                    coupled_example,
                    "VorticitySequenceStream",
                    return_value=vorticity_stream,
                ),
                mock.patch.object(
                    coupled_example,
                    "TemperatureSequenceStream",
                    return_value=temperature_stream,
                ) as temperature_stream_factory,
                mock.patch.object(coupled_example, "write_fields_npz") as write_npz,
                mock.patch.object(coupled_example, "write_final_plot"),
                mock.patch.object(coupled_example, "write_metadata") as metadata,
            ):
                summary = coupled_example._run_case(
                    args,
                    config,
                    [0, 10, 20],
                    visualization_limits=limits,
                )

            history_lines = (
                (Path(temporary_directory) / "history.csv")
                .read_text(encoding="utf-8")
                .splitlines()
            )

        write_npz.assert_not_called()
        self.assertEqual(len(history_lines), 4)
        self.assertEqual(velocity_stream.appended_steps, [0, 10, 20])
        self.assertEqual(vorticity_stream.appended_steps, [0, 10, 20])
        self.assertEqual(temperature_stream.appended_steps, [0, 10, 20])
        self.assertTrue(velocity_stream.closed)
        self.assertTrue(vorticity_stream.closed)
        self.assertTrue(temperature_stream.closed)
        temperature_stream_factory.assert_called_once_with(
            Path(temporary_directory),
            config,
            temperature_min_c=limits.temperature_min_c,
            temperature_max_c=limits.temperature_max_c,
        )
        self.assertEqual(summary["snapshots"], 3)
        metadata.assert_called_once()
        metadata_kwargs = metadata.call_args.kwargs
        self.assertNotIn("snapshots", metadata_kwargs)
        self.assertEqual(metadata_kwargs["snapshot_count"], 3)
        self.assertEqual(metadata_kwargs["initial_snapshot"].lbm_steps, 0)
        self.assertEqual(metadata_kwargs["final_snapshot"].lbm_steps, 20)
        self.assertEqual(
            metadata_kwargs["temperature_sequence"],
            temperature_summary,
        )

    def test_first_coupling_rejects_unsupported_geometry_and_air_heat_modes(self):
        config = coupled_example.create_config()
        assert config.thermal is not None

        with self.assertRaisesRegex(ValueError, "ice_fixed=True"):
            replace(config, ice_fixed=False)
        with self.assertRaisesRegex(ValueError, "free surface"):
            replace(
                config,
                water_height_fraction=(config.ny - config.boundary_cells + 0.5)
                / config.ny,
            )
        with self.assertRaisesRegex(ValueError, "adiabatic"):
            replace(
                config,
                thermal=replace(
                    config.thermal,
                    water_air_interface_adiabatic=False,
                ),
            )

    def test_velocity_sequence_uses_one_water_only_physical_speed_limit(self):
        args = coupled_example.parse_args(
            [
                "--resolution-x",
                "20",
                "--resolution-y",
                "28",
                "--ice-size-m",
                "0.01",
            ]
        )
        config = coupled_example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)
        scalar_shape = (config.ny, config.nx)
        vector_shape = (*scalar_shape, 2)

        water_phase = np.ones(scalar_shape, dtype=np.float32)
        solid = np.zeros(scalar_shape, dtype=np.int8)
        first_velocity = np.zeros(vector_shape, dtype=np.float32)
        second_velocity = np.zeros(vector_shape, dtype=np.float32)
        first_velocity[5, 5] = (3.0, 4.0)
        second_velocity[7, 7] = (6.0, 8.0)

        # Large values in air and solid must not flatten the water color scale.
        first_velocity[9, 9] = (100.0, 0.0)
        water_phase[9, 9] = 0.0
        second_velocity[11, 11] = (200.0, 0.0)
        solid[11, 11] = 1
        snapshots = [
            SimpleNamespace(
                physical_velocity_lattice=first_velocity,
                water_phase=water_phase,
                solid=solid,
                physical_time_s=0.0,
                melted_fraction=0.0,
            ),
            SimpleNamespace(
                physical_velocity_lattice=second_velocity,
                water_phase=water_phase,
                solid=solid,
                physical_time_s=0.05,
                melted_fraction=0.1,
            ),
        ]

        observed, color_max = coupled_example.velocity_sequence_limits_m_s(
            config, scales, snapshots
        )
        expected = 10.0 * scales.velocity_scale_m_s
        self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(color_max, expected)
        fixed_observed, fixed_color_max = coupled_example.velocity_sequence_limits_m_s(
            config,
            scales,
            snapshots,
            display_max_m_s=5.5e-6,
        )
        self.assertAlmostEqual(fixed_observed, expected)
        self.assertEqual(fixed_color_max, 5.5e-6)

        zero_snapshot = SimpleNamespace(
            physical_velocity_lattice=np.zeros(vector_shape, dtype=np.float32),
            water_phase=np.ones(scalar_shape, dtype=np.float32),
            solid=np.zeros(scalar_shape, dtype=np.int8),
        )
        observed_zero, color_max_zero = coupled_example.velocity_sequence_limits_m_s(
            config, scales, [zero_snapshot]
        )
        self.assertEqual(observed_zero, 0.0)
        self.assertGreater(color_max_zero, 0.0)
        fixed_zero_observed, fixed_zero_color_max = (
            coupled_example.velocity_sequence_limits_m_s(
                config,
                scales,
                [zero_snapshot],
                display_max_m_s=5.5e-6,
            )
        )
        self.assertEqual(fixed_zero_observed, 0.0)
        self.assertEqual(fixed_zero_color_max, 5.5e-6)

    def test_quiver_components_use_the_shared_physical_speed_scale(self):
        velocity_m_s = np.zeros((3, 4, 2), dtype=np.float64)
        velocity_m_s[1, 2] = (2.75e-6, -1.375e-6)
        arrow_u, arrow_v = coupled_example._scaled_quiver_components(
            velocity_m_s,
            np.asarray([2]),
            np.asarray([1]),
            display_max_m_s=5.5e-6,
            reference_arrow_length_m=0.0012,
        )
        self.assertAlmostEqual(float(arrow_u[0, 0]), 0.0006)
        self.assertAlmostEqual(float(arrow_v[0, 0]), -0.0003)

    def test_water_vorticity_matches_solid_body_rotation_without_crossing_masks(self):
        spacing_m = 2.5e-4
        rows, columns = np.indices((9, 11), dtype=np.float64)
        x_m = (columns + 0.5) * spacing_m
        y_m = (rows + 0.5) * spacing_m
        angular_speed_s_1 = 0.075
        velocity_m_s = np.empty((9, 11, 2), dtype=np.float64)
        velocity_m_s[..., 0] = -angular_speed_s_1 * y_m
        velocity_m_s[..., 1] = angular_speed_s_1 * x_m
        visible_water = np.ones((9, 11), dtype=bool)

        vorticity_s_1 = coupled_example._water_vorticity_s_1(
            velocity_m_s,
            visible_water,
            spacing_m=spacing_m,
        )
        finite = np.isfinite(vorticity_s_1)
        self.assertTrue(np.all(finite[1:-1, 1:-1]))
        self.assertTrue(np.all(~finite[[0, -1], :]))
        self.assertTrue(np.all(~finite[:, [0, -1]]))
        np.testing.assert_allclose(
            vorticity_s_1[finite],
            2.0 * angular_speed_s_1,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        visible_water[4, 5] = False
        contaminated_velocity = velocity_m_s.copy()
        contaminated_velocity[4, 5] = (1.0e6, -1.0e6)
        masked_vorticity = coupled_example._water_vorticity_s_1(
            contaminated_velocity,
            visible_water,
            spacing_m=spacing_m,
        )
        for row, column in ((4, 5), (3, 5), (5, 5), (4, 4), (4, 6)):
            self.assertFalse(np.isfinite(masked_vorticity[row, column]))
        masked_finite = np.isfinite(masked_vorticity)
        np.testing.assert_allclose(
            masked_vorticity[masked_finite],
            2.0 * angular_speed_s_1,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

        reverse_vorticity = coupled_example._water_vorticity_s_1(
            -velocity_m_s,
            np.ones((9, 11), dtype=bool),
            spacing_m=spacing_m,
        )
        np.testing.assert_allclose(
            reverse_vorticity[np.isfinite(reverse_vorticity)],
            -2.0 * angular_speed_s_1,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_streaming_gif_is_readable_and_grows_after_each_append(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            first_png = output_dir / "first.png"
            second_png = output_dir / "second.png"
            Image.new("RGB", (8, 8), "red").save(first_png)
            Image.new("RGB", (8, 8), "blue").save(second_png)
            animation_path = output_dir / "stream.gif"
            stream = coupled_example._StreamingGif(animation_path)

            try:
                stream.append_png(first_png)
                first_size = animation_path.stat().st_size
                self.assertGreater(first_size, 0)
                with Image.open(animation_path) as animation:
                    self.assertEqual(animation.n_frames, 1)
                    self.assertEqual(animation.info["duration"], 800)
                    self.assertEqual(animation.info["loop"], 0)

                stream.append_png(second_png)
                second_size = animation_path.stat().st_size
                self.assertGreater(second_size, first_size)
                with Image.open(animation_path) as animation:
                    self.assertEqual(animation.n_frames, 2)
                    self.assertEqual(animation.info["duration"], 800)
            finally:
                stream.close()

            closed_size = animation_path.stat().st_size
            self.assertEqual(animation_path.read_bytes()[-1:], b";")
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 2)
            stream.close()
            self.assertEqual(animation_path.stat().st_size, closed_size)
            with self.assertRaisesRegex(RuntimeError, "closed GIF"):
                stream.append_png(first_png)

    def test_temperature_sequence_writes_ordered_frames_with_one_fixed_scale(self):
        from matplotlib.axes import Axes

        args = coupled_example.parse_args(
            [
                "--resolution-x",
                "20",
                "--resolution-y",
                "28",
                "--ice-size-m",
                "0.01",
                "--water-temperature-c",
                "8",
                "--air-temperature-c",
                "8",
            ]
        )
        config = coupled_example.create_config(args)
        scalar_shape = (config.ny, config.nx)
        boundary = config.boundary_cells

        water_phase = np.zeros(scalar_shape, dtype=np.float32)
        water_phase[
            boundary : config.water_height,
            boundary : config.nx - boundary,
        ] = 1.0
        phase_change_material = np.zeros(scalar_shape, dtype=np.int8)
        phase_change_material[10:14, 8:12] = 1
        water_phase[phase_change_material != 0] = 0.0
        solid = phase_change_material.copy()
        liquid_fraction = np.ones(scalar_shape, dtype=np.float32)
        liquid_fraction[phase_change_material != 0] = 0.0
        active = (water_phase >= 0.5) | (phase_change_material != 0)

        first_temperature = np.full(scalar_shape, 90.0, dtype=np.float64)
        first_temperature[active] = 2.0
        first_temperature[phase_change_material != 0] = 0.0
        first_temperature[:boundary, :] = -50.0
        second_temperature = np.full(scalar_shape, 95.0, dtype=np.float64)
        second_temperature[active] = 7.25
        second_temperature[phase_change_material != 0] = 0.5
        second_temperature[:, -boundary:] = -60.0
        snapshots = [
            SimpleNamespace(
                temperature_c=temperature,
                water_phase=water_phase,
                phase_change_material=phase_change_material,
                solid=solid,
                liquid_fraction=liquid_fraction,
                physical_time_s=0.05 * index,
                melted_fraction=0.1 * index,
            )
            for index, temperature in enumerate((first_temperature, second_temperature))
        ]

        imshow_scales = []
        imshow_colormaps = []
        original_imshow = Axes.imshow

        def recording_imshow(axis, *plot_args, **plot_kwargs):
            normalization = plot_kwargs.get("norm")
            if normalization is None:
                imshow_scales.append((plot_kwargs.get("vmin"), plot_kwargs.get("vmax")))
            else:
                imshow_scales.append((normalization.vmin, normalization.vmax))
            colormap = plot_kwargs.get("cmap")
            imshow_colormaps.append(getattr(colormap, "name", colormap))
            return original_imshow(axis, *plot_args, **plot_kwargs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            stale_frame = output_dir / "temperature_frames" / "frame_99999.png"
            stale_frame.parent.mkdir()
            stale_frame.write_bytes(b"stale")
            stream = coupled_example.TemperatureSequenceStream(
                output_dir,
                config,
                temperature_min_c=0.0,
                temperature_max_c=8.0,
            )
            self.assertFalse(stale_frame.exists())

            with mock.patch.object(Axes, "imshow", new=recording_imshow):
                first_frame = stream.append(snapshots[0])
                animation_path = output_dir / "temperature_field.gif"
                first_animation_size = animation_path.stat().st_size
                self.assertTrue(first_frame.is_file())
                self.assertGreater(first_frame.stat().st_size, 0)
                self.assertGreater(first_animation_size, 0)
                with Image.open(animation_path) as animation:
                    self.assertEqual(animation.n_frames, 1)

                second_frame = stream.append(snapshots[1])
                self.assertTrue(second_frame.is_file())
                self.assertGreater(animation_path.stat().st_size, first_animation_size)
                with Image.open(animation_path) as animation:
                    self.assertEqual(animation.n_frames, 2)

            summary = stream.close()
            self.assertEqual(stream.close(), summary)
            with self.assertRaisesRegex(RuntimeError, "closed temperature"):
                stream.append(snapshots[0])

            self.assertEqual(summary.frame_directory, "temperature_frames")
            self.assertEqual(summary.frame_pattern, "frame_*.png")
            self.assertEqual(summary.animation, "temperature_field.gif")
            frames = sorted((output_dir / summary.frame_directory).glob("frame_*.png"))
            self.assertEqual(
                [frame.name for frame in frames],
                ["frame_00000.png", "frame_00001.png"],
            )
            self.assertTrue(all(frame.stat().st_size > 0 for frame in frames))
            self.assertEqual(summary.frame_count, 2)
            self.assertEqual(summary.observed_min_c, 0.0)
            self.assertEqual(summary.observed_max_c, 7.25)
            self.assertEqual(summary.color_min_c, 0.0)
            self.assertEqual(summary.color_max_c, 8.0)
            with Image.open(output_dir / summary.animation) as animation:
                self.assertEqual(animation.n_frames, 2)
                self.assertEqual(animation.info["duration"], 800)

        self.assertEqual(imshow_scales, [(0.0, 8.0), (0.0, 8.0)])
        self.assertEqual(imshow_colormaps, ["inferno", "inferno"])

    def test_vorticity_sequence_writes_ordered_frames_and_animation(self):
        args = coupled_example.parse_args(
            [
                "--resolution-x",
                "20",
                "--resolution-y",
                "28",
                "--ice-size-m",
                "0.01",
            ]
        )
        config = coupled_example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)
        scalar_shape = (config.ny, config.nx)
        rows, columns = np.indices(scalar_shape, dtype=np.float64)
        x_m = (columns + 0.5) * config.dx
        y_m = (rows + 0.5) * config.dx
        x_center_m = 0.5 * config.nx * config.dx
        y_center_m = 0.5 * config.ny * config.dx
        water_phase = np.ones(scalar_shape, dtype=np.float32)
        solid = np.zeros(scalar_shape, dtype=np.int8)
        snapshots = []
        angular_speed_s_1 = 0.05
        for index, signed_angular_speed in enumerate(
            (angular_speed_s_1, -angular_speed_s_1)
        ):
            physical_velocity = np.empty((*scalar_shape, 2), dtype=np.float64)
            physical_velocity[..., 0] = -signed_angular_speed * (y_m - y_center_m)
            physical_velocity[..., 1] = signed_angular_speed * (x_m - x_center_m)
            snapshots.append(
                SimpleNamespace(
                    physical_velocity_lattice=(
                        physical_velocity / scales.velocity_scale_m_s
                    ),
                    water_phase=water_phase,
                    solid=solid,
                    physical_time_s=0.05 * index,
                    melted_fraction=0.1 * index,
                )
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            stale_frame = output_dir / "vorticity_frames" / "frame_99999.png"
            stale_frame.parent.mkdir()
            stale_frame.write_bytes(b"stale")
            stream = coupled_example.VorticitySequenceStream(
                output_dir,
                config,
                scales,
                display_abs_max_s_1=0.2,
                velocity_display_max_m_s=2.0e-4,
            )
            self.assertFalse(stale_frame.exists())

            first_frame = stream.append(snapshots[0])
            animation_path = output_dir / "vorticity_field.gif"
            first_animation_size = animation_path.stat().st_size
            self.assertTrue(first_frame.is_file())
            self.assertGreater(first_frame.stat().st_size, 0)
            self.assertGreater(first_animation_size, 0)
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 1)

            second_frame = stream.append(snapshots[1])
            self.assertTrue(second_frame.is_file())
            self.assertGreater(animation_path.stat().st_size, first_animation_size)
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 2)
            summary = stream.close()
            self.assertEqual(stream.close(), summary)
            with self.assertRaisesRegex(RuntimeError, "closed vorticity"):
                stream.append(snapshots[0])

            self.assertEqual(summary.frame_directory, "vorticity_frames")
            self.assertEqual(summary.frame_pattern, "frame_*.png")
            self.assertEqual(summary.animation, "vorticity_field.gif")
            frames = sorted((output_dir / summary.frame_directory).glob("frame_*.png"))
            self.assertEqual(
                [frame.name for frame in frames],
                ["frame_00000.png", "frame_00001.png"],
            )
            self.assertTrue(all(frame.stat().st_size > 0 for frame in frames))
            self.assertEqual(summary.frame_count, 2)
            self.assertAlmostEqual(summary.observed_abs_max_s_1, 0.1)
            self.assertEqual(summary.color_abs_max_s_1, 0.2)
            animation_path = output_dir / summary.animation
            self.assertTrue(animation_path.is_file())
            self.assertGreater(animation_path.stat().st_size, 0)
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 2)
                self.assertEqual(animation.info["duration"], 800)

    def test_velocity_sequence_writes_ordered_frames_and_animation(self):
        args = coupled_example.parse_args(
            [
                "--resolution-x",
                "20",
                "--resolution-y",
                "28",
                "--ice-size-m",
                "0.01",
            ]
        )
        config = coupled_example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)
        scalar_shape = (config.ny, config.nx)
        vector_shape = (*scalar_shape, 2)
        water_phase = np.ones(scalar_shape, dtype=np.float32)
        solid = np.zeros(scalar_shape, dtype=np.int8)
        snapshots = []
        # The first frame is commonly at rest even when later frames move.
        physical_speed_max = 2.75e-6
        lattice_speed_max = physical_speed_max / scales.velocity_scale_m_s
        for index, speed in enumerate((0.0, lattice_speed_max)):
            velocity = np.zeros(vector_shape, dtype=np.float32)
            velocity[5, 5, 0] = speed
            snapshots.append(
                SimpleNamespace(
                    physical_velocity_lattice=velocity,
                    water_phase=water_phase,
                    solid=solid,
                    physical_time_s=0.05 * index,
                    melted_fraction=0.1 * index,
                )
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            stale_frame = output_dir / "velocity_frames" / "frame_99999.png"
            stale_frame.parent.mkdir()
            stale_frame.write_bytes(b"stale")
            stream = coupled_example.VelocitySequenceStream(
                output_dir,
                config,
                scales,
                display_max_m_s=5.5e-6,
            )
            self.assertFalse(stale_frame.exists())

            first_frame = stream.append(snapshots[0])
            animation_path = output_dir / "velocity_field.gif"
            first_animation_size = animation_path.stat().st_size
            self.assertTrue(first_frame.is_file())
            self.assertGreater(first_frame.stat().st_size, 0)
            self.assertGreater(first_animation_size, 0)
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 1)

            second_frame = stream.append(snapshots[1])
            self.assertTrue(second_frame.is_file())
            self.assertGreater(animation_path.stat().st_size, first_animation_size)
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 2)
            summary = stream.close()
            self.assertEqual(stream.close(), summary)
            with self.assertRaisesRegex(RuntimeError, "closed velocity"):
                stream.append(snapshots[0])

            self.assertEqual(summary.frame_directory, "velocity_frames")
            self.assertEqual(summary.frame_pattern, "frame_*.png")
            self.assertEqual(summary.animation, "velocity_field.gif")
            frames = sorted((output_dir / summary.frame_directory).glob("frame_*.png"))
            self.assertEqual(
                [frame.name for frame in frames],
                ["frame_00000.png", "frame_00001.png"],
            )
            self.assertTrue(all(frame.stat().st_size > 0 for frame in frames))
            self.assertEqual(summary.frame_count, 2)
            self.assertAlmostEqual(
                summary.observed_max_m_s,
                physical_speed_max,
            )
            self.assertEqual(summary.color_max_m_s, 5.5e-6)
            animation_path = output_dir / summary.animation
            self.assertTrue(animation_path.is_file())
            self.assertGreater(animation_path.stat().st_size, 0)
            with Image.open(animation_path) as animation:
                self.assertEqual(animation.n_frames, 2)
                self.assertEqual(animation.info["duration"], 800)

            zero_snapshots = [
                SimpleNamespace(
                    physical_velocity_lattice=np.zeros(vector_shape, dtype=np.float32),
                    water_phase=water_phase,
                    solid=solid,
                    physical_time_s=0.05 * index,
                    melted_fraction=0.1 * index,
                )
                for index in range(2)
            ]
            zero_output_dir = output_dir / "all_zero"
            zero_summary = coupled_example.write_velocity_sequence(
                zero_output_dir,
                config,
                scales,
                zero_snapshots,
                display_max_m_s=5.5e-6,
            )
            self.assertEqual(zero_summary.observed_max_m_s, 0.0)
            self.assertEqual(zero_summary.color_max_m_s, 5.5e-6)
            self.assertEqual(
                len(list((zero_output_dir / "velocity_frames").glob("frame_*.png"))),
                2,
            )
            self.assertGreater(
                (zero_output_dir / "velocity_field.gif").stat().st_size,
                0,
            )

    def test_final_temperature_plot_accepts_the_sweep_wide_maximum(self):
        from matplotlib.axes import Axes

        args = coupled_example.parse_args(
            [
                "--resolution-x",
                "20",
                "--resolution-y",
                "28",
                "--ice-size-m",
                "0.01",
                "--water-temperature-c",
                "4",
            ]
        )
        config = coupled_example.create_config(args)
        scalar_shape = (config.ny, config.nx)
        row, column = np.indices(scalar_shape)
        liquid_fraction = (row + column) / float(config.nx + config.ny - 2)
        water_phase = np.zeros(scalar_shape, dtype=np.float32)
        water_phase[: config.water_height] = 1.0
        snapshot = SimpleNamespace(
            temperature_c=np.full(scalar_shape, 4.0, dtype=np.float64),
            phase_change_material=np.ones(scalar_shape, dtype=np.int8),
            liquid_fraction=liquid_fraction,
            water_phase=water_phase,
            physical_time_s=0.1,
            melted_fraction=0.05,
            water_volume_residual_cells=0.0,
        )

        imshow_limits = []
        original_imshow = Axes.imshow

        def recording_imshow(axis, *plot_args, **plot_kwargs):
            imshow_limits.append((plot_kwargs.get("vmin"), plot_kwargs.get("vmax")))
            return original_imshow(axis, *plot_args, **plot_kwargs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            plot_path = Path(temporary_directory) / "temperature.png"
            with mock.patch.object(Axes, "imshow", new=recording_imshow):
                coupled_example.write_final_plot(
                    plot_path,
                    config,
                    snapshot,
                    temperature_max_c=8.0,
                )
            self.assertTrue(plot_path.is_file())
            self.assertGreater(plot_path.stat().st_size, 0)

        self.assertEqual(imshow_limits[0], (0.0, 8.0))
        self.assertEqual(imshow_limits[1], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
