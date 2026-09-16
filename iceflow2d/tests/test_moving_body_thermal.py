"""CPU-only contracts for the retained moving-body thermal configuration."""

from __future__ import annotations

import math
import unittest
from unittest.mock import Mock

from iceflow2d.examples import coupled_falling_ice_melting_2d as example
from iceflow2d.simulator import IceFlow2D
from iceflow2d.config import (
    LatticeScales,
    PhaseChangeProperties,
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
)


class _ScalarFieldStub:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, _key):
        return self.value


class _ThermalAdvanceStub:
    def __init__(self):
        self.calls = []

    def advance_fast(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FastThermalSimulatorStub:
    def __init__(self, stability_check_enabled):
        self._thermal_stability_check_enabled = stability_check_enabled
        self._measure_water_outflow_rate = Mock()
        self._check_thermal_advection_stability = Mock()
        self.maximum_water_outflow_rate_lattice = _ScalarFieldStub(0.25)
        self.physical_velocity_lattice = object()
        self.water_phase = object()
        self.wall_mask = object()
        self.solid_mask = object()
        self._time_step_s = 0.1
        self.thermal = _ThermalAdvanceStub()


class MovingBodyThermalConfigTests(unittest.TestCase):
    def test_default_scheme_is_body_ale_and_supports_multirate_updates(self):
        config = ThermalConfig(update_interval_lbm_steps=8)

        self.assertEqual(config.moving_body_scheme, "body_ale")
        self.assertEqual(config.update_interval_lbm_steps, 8)
        self.assertTrue(config.water_air_interface_adiabatic)
        self.assertEqual(config.water_buoyancy_model, "linear")

    def test_removed_fixed_eulerian_scheme_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "body_ale"):
            ThermalConfig(moving_body_scheme="fixed_eulerian")  # type: ignore[arg-type]

    def test_multirate_and_interface_guards_are_validated(self):
        for value in (0, -1, False, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "update_interval"):
                    ThermalConfig(update_interval_lbm_steps=value)  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "adiabatic"):
            ThermalConfig(water_air_interface_adiabatic=False)

    def test_falling_example_uses_one_consistent_physical_scale(self):
        args = example.parse_args(["--resolution-x", "100", "--resolution-y", "200"])
        config = example.create_config(args)
        scales = LatticeScales.from_iceflow_config(config)

        self.assertAlmostEqual(config.dx, 2.5e-4)
        self.assertAlmostEqual(scales.dt_s, 6.25e-6)
        self.assertAlmostEqual(scales.reference_velocity_m_s, 4.0)
        self.assertAlmostEqual(scales.velocity_scale_m_s, 40.0)
        self.assertEqual(config.thermal.update_interval_lbm_steps, 8)


class ThermalStabilitySwitchTests(unittest.TestCase):
    def test_fast_path_skips_optional_stability_scan_by_default(self):
        simulation = _FastThermalSimulatorStub(stability_check_enabled=False)

        IceFlow2D._advance_moving_thermal_fast(simulation, target_step=7)

        simulation._measure_water_outflow_rate.assert_called_once_with()
        simulation._check_thermal_advection_stability.assert_not_called()
        self.assertEqual(len(simulation.thermal.calls), 1)
        args, kwargs = simulation.thermal.calls[0]
        self.assertEqual(args[0], 0.1)
        self.assertIs(args[1], simulation.physical_velocity_lattice)
        self.assertEqual(kwargs["maximum_outflow_rate_lattice"], 0.25)

    def test_fast_path_runs_stability_scan_when_self_flag_is_enabled(self):
        simulation = _FastThermalSimulatorStub(stability_check_enabled=True)

        IceFlow2D._advance_moving_thermal_fast(simulation, target_step=9)

        simulation._check_thermal_advection_stability.assert_called_once_with(
            target_step=9
        )


class ThermalMaterialAndBoundaryTests(unittest.TestCase):
    def test_phase_change_properties_are_positive_and_finite(self):
        properties = PhaseChangeProperties()
        for name in (
            "specific_heat_water_j_kg_k",
            "specific_heat_ice_j_kg_k",
            "specific_heat_air_j_kg_k",
            "conductivity_water_w_m_k",
            "conductivity_ice_w_m_k",
            "conductivity_air_w_m_k",
            "latent_heat_j_kg",
        ):
            value = getattr(properties, name)
            self.assertGreater(value, 0.0)
            self.assertTrue(math.isfinite(value))

        with self.assertRaisesRegex(ValueError, "latent_heat"):
            PhaseChangeProperties(latent_heat_j_kg=0.0)

    def test_boundary_set_keeps_hot_bath_walls_and_adiabatic_top_explicit(self):
        hot = ThermalBoundary.dirichlet(90.0)
        boundaries = ThermalBoundarySet(
            left=hot,
            right=hot,
            bottom=hot,
            top=ThermalBoundary.adiabatic(),
        )

        self.assertEqual(boundaries.left.value, 90.0)
        self.assertEqual(boundaries.right.kind, "dirichlet")
        self.assertEqual(boundaries.bottom.kind, "dirichlet")
        self.assertEqual(boundaries.top.kind, "adiabatic")
        with self.assertRaisesRegex(ValueError, "adiabatic"):
            ThermalBoundary("adiabatic", 1.0)


if __name__ == "__main__":
    unittest.main()
