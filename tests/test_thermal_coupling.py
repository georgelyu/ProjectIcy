"""Pure-CPU contract tests for the optional thermal/LBM coupling.

This module deliberately imports neither :mod:`taichi` nor
``iceflow2d.simulator``.  It exercises the configuration, dimensional scaling,
and thermodynamic reference functions used by both the CPU benchmark and the
CUDA implementation.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

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
    def test_millimetre_grid_has_consistent_velocity_time_and_diffusivity_scales(self):
        scales = LatticeScales.from_hydrodynamic_reference(
            dx_m=0.25e-3,
            reference_length_cells=120,
            gravity_m_s2=9.8,
            reference_density_kg_m3=1000.0,
        )
        reference_velocity = math.sqrt(4.0 * 9.8 * 120 * 0.25e-3)
        expected_dt = 0.1 * 0.25e-3 / reference_velocity

        self.assertAlmostEqual(scales.dx_m, 0.25e-3)
        self.assertAlmostEqual(scales.dt_s, expected_dt)
        self.assertAlmostEqual(scales.reference_velocity_m_s, reference_velocity)
        self.assertAlmostEqual(
            scales.velocity_to_physical(scales.velocity_to_lattice(0.02)), 0.02
        )
        diffusivity = 2.2 / (917.0 * 2100.0)
        self.assertAlmostEqual(
            scales.diffusivity_to_lattice(diffusivity),
            diffusivity * expected_dt / (0.25e-3) ** 2,
        )


class CoupledExampleConfigTests(unittest.TestCase):
    def test_default_example_is_a_short_real_density_millimetre_case(self):
        args = coupled_example.parse_args([])
        config = coupled_example.create_config()
        thermal = config.thermal
        self.assertIsNotNone(thermal)
        assert thermal is not None

        self.assertEqual(config.resolution, (80, 112))
        self.assertAlmostEqual(config.nx * config.dx, 2.5e-3)
        self.assertAlmostEqual(config.ny * config.dx, 3.5e-3)
        self.assertAlmostEqual(config.dx, 31.25e-6)
        self.assertEqual(config.boundary_cells, 3)
        self.assertEqual((config.water_width, config.water_height), (77, 88))
        self.assertEqual((config.ice_width, config.ice_height), (32, 32))
        self.assertEqual(config.ice_initial_center, (40.0, 48.0))
        self.assertAlmostEqual(config.ice_width * config.dx, 1.0e-3)
        self.assertAlmostEqual(config.water_height * config.dx, 2.75e-3)
        self.assertAlmostEqual(config.ice_initial_center[1] * config.dx, 1.5e-3)
        self.assertTrue(config.ice_fixed)
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
        self.assertAlmostEqual(args.end_time_s, 0.30)
        self.assertAlmostEqual(args.output_interval_s, 0.05)
        self.assertEqual(targets, [0, 5930, 11860, 17780, 23710, 29640, 35560])
        self.assertAlmostEqual(scales.dt_s, 8.436706986813107e-6)
        self.assertGreaterEqual(targets[-1] * scales.dt_s, args.end_time_s)
        self.assertLess(
            targets[-1] * scales.dt_s - args.end_time_s,
            thermal.update_interval_lbm_steps * scales.dt_s,
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


if __name__ == "__main__":
    unittest.main()
