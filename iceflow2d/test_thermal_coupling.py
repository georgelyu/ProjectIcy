"""CUDA regression tests for fixed-ice thermal/LBM phase change."""

from __future__ import annotations

import unittest

import numpy as np

from iceflow2d.config import create_iceflow_config
from iceflow2d.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d.thermal import (
    ThermalBoundary,
    ThermalBoundarySet,
    ThermalConfig,
    phase_change_active_water_target_cells,
    phase_change_water_target_cells,
)


def _coupled_config(*, water_temperature_c: float, wall_temperature_c: float):
    wall = ThermalBoundary.dirichlet(wall_temperature_c)
    thermal = ThermalConfig(
        boundaries=ThermalBoundarySet(left=wall, right=wall, bottom=wall),
        initial_water_temperature_c=water_temperature_c,
        initial_ice_temperature_c=0.0,
        initial_air_temperature_c=max(0.0, water_temperature_c),
        update_interval_lbm_steps=4,
    )
    return create_iceflow_config(
        resolution=(24, 32),
        dx=1.0e-4,
        reference_length_cells=32,
        rho_water=1000.0,
        rho_air=1.25,
        rho_ice=917.0,
        viscosity_water=1.0e-6,
        viscosity_air=1.5e-5,
        sigma=0.0,
        gravity=(0.0, 0.0),
        phase_warmup_steps=10,
        water_width_fraction=22.5 / 24.0,
        water_height_fraction=24.5 / 32.0,
        ice_width_fraction=8.5 / 24.0,
        ice_height_fraction=8.5 / 32.0,
        ice_base_y_cells=10.0,
        boundary_cells=2,
        ice_fixed=True,
        thermal=thermal,
        volume_projection_tolerance=1.0e-8,
    )


class ThermalCouplingCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:  # pragma: no cover - host dependent
            raise unittest.SkipTest(str(exc)) from exc

    def assert_coupled_state(self, simulation: IceFlow2D) -> None:
        temperature = simulation.temperature.to_numpy()
        liquid_fraction = simulation.liquid_fraction.to_numpy()
        phase = simulation.phi.to_numpy()
        self.assertTrue(np.isfinite(temperature).all())
        self.assertTrue(np.isfinite(liquid_fraction).all())
        self.assertTrue(np.isfinite(phase).all())
        self.assertGreaterEqual(float(liquid_fraction.min()), -1.0e-7)
        self.assertLessEqual(float(liquid_fraction.max()), 1.0 + 1.0e-7)

        initial_water = float(simulation.phase_change_initial_water_volume[None])
        initial_solid = float(simulation.phase_change_initial_solid_volume[None])
        current_solid = simulation.phase_change_solid_volume_cells()
        initial_geometry = float(simulation.phase_change_initial_geometry_volume[None])
        current_geometry = simulation.phase_change_geometry_volume_cells()
        expected_active = phase_change_active_water_target_cells(
            initial_water,
            initial_solid,
            current_solid,
            initial_geometry,
            current_geometry,
            density_ice_kg_m3=simulation.cfg.rho_ice,
            density_water_kg_m3=simulation.cfg.rho_water,
        )
        expected_total = phase_change_water_target_cells(
            initial_water,
            initial_solid,
            current_solid,
            density_ice_kg_m3=simulation.cfg.rho_ice,
            density_water_kg_m3=simulation.cfg.rho_water,
        )
        self.assertAlmostEqual(
            float(simulation.water_volume_target[None]),
            expected_active,
            delta=1.0e-6,
        )
        self.assertGreaterEqual(expected_total, 0.0)
        tolerance = simulation.cfg.volume_projection_tolerance * max(
            1.0, expected_active
        )
        self.assertLessEqual(
            abs(
                float(simulation.water_volume_current[None])
                - float(simulation.water_volume_target[None])
            ),
            tolerance,
        )
        self.assertAlmostEqual(
            simulation.physical_time_s, simulation.thermal.time_s, places=13
        )

    def test_hot_water_melts_nodes_and_rebuilds_active_water(self):
        simulation = IceFlow2D(
            _coupled_config(water_temperature_c=90.0, wall_temperature_c=90.0)
        )
        initial_solid = simulation.phase_change_solid_volume_cells()
        initial_geometry = simulation.phase_change_geometry_volume_cells()

        simulation.step(600)
        simulation.synchronize_thermal()

        self.assertLess(simulation.phase_change_solid_volume_cells(), initial_solid)
        self.assertLess(
            simulation.phase_change_geometry_volume_cells(), initial_geometry
        )
        self.assertGreater(simulation.phase_change_melted_fraction, 0.10)
        self.assert_coupled_state(simulation)

    def test_cold_walls_freeze_water_and_expand_sharp_solid(self):
        simulation = IceFlow2D(
            _coupled_config(water_temperature_c=0.0, wall_temperature_c=-60.0)
        )
        initial_solid = simulation.phase_change_solid_volume_cells()
        initial_geometry = simulation.phase_change_geometry_volume_cells()

        simulation.step(700)
        simulation.synchronize_thermal()

        self.assertGreater(simulation.phase_change_solid_volume_cells(), initial_solid)
        self.assertGreater(
            simulation.phase_change_geometry_volume_cells(), initial_geometry
        )
        self.assertLess(simulation.phase_change_melted_fraction, 0.0)
        self.assert_coupled_state(simulation)


if __name__ == "__main__":
    unittest.main()
