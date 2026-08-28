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
    water_density_anomaly_ratio_to_reference,
)


def _coupled_config(
    *,
    water_temperature_c: float,
    wall_temperature_c: float,
    gravity_m_s2: float = 0.0,
    water_buoyancy_model: str = "linear",
    freshwater_density_beta_1_k2: float = 8.0e-6,
    reference_velocity_m_s: float = 0.35417509793885854,
):
    wall = ThermalBoundary.dirichlet(wall_temperature_c)
    thermal = ThermalConfig(
        boundaries=ThermalBoundarySet(left=wall, right=wall, bottom=wall),
        initial_water_temperature_c=water_temperature_c,
        initial_ice_temperature_c=0.0,
        initial_air_temperature_c=max(0.0, water_temperature_c),
        update_interval_lbm_steps=4,
        water_buoyancy_model=water_buoyancy_model,
        buoyancy_reference_temperature_c=water_temperature_c,
        freshwater_density_quadratic_coefficient_1_k2=(
            freshwater_density_beta_1_k2
        ),
    )
    return create_iceflow_config(
        resolution=(24, 32),
        dx=1.0e-4,
        reference_velocity=reference_velocity_m_s,
        rho_water=1000.0,
        rho_air=1.25,
        rho_ice=917.0,
        viscosity_water=1.0e-6,
        viscosity_air=1.5e-5,
        sigma=0.0,
        gravity=(0.0, -gravity_m_s2),
        phase_warmup_steps=10,
        well_balanced_hydrostatics=gravity_m_s2 > 0.0,
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
        for name in ("f", "h", "p", "u", "fluid_force"):
            with self.subTest(field=name):
                self.assertTrue(
                    np.isfinite(getattr(simulation, name).to_numpy()).all()
                )

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

    def test_low_fixed_reference_velocity_zero_gravity_stays_finite(self):
        simulation = IceFlow2D(
            _coupled_config(
                water_temperature_c=20.0,
                wall_temperature_c=20.0,
                reference_velocity_m_s=1.0e-3,
            )
        )
        simulation.step(4)
        self.assert_coupled_state(simulation)

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

    def test_quadratic_eos_device_force_has_expected_direction(self):
        probe_cases = (
            (4.0, 0.0, 1),
            (5.6, 0.0, 1),
            (5.6, 4.0, -1),
            (8.0, 4.0, -1),
            (8.0, 6.0, -1),
            (8.0, 0.0, 0),
        )
        simulations: dict[float, tuple[IceFlow2D, tuple[int, int]]] = {}
        density_beta = 8.0e-4

        for bath_temperature, local_temperature, expected_sign in probe_cases:
            if bath_temperature not in simulations:
                simulation = IceFlow2D(
                    _coupled_config(
                        water_temperature_c=bath_temperature,
                        wall_temperature_c=bath_temperature,
                        gravity_m_s2=9.8,
                        water_buoyancy_model="freshwater_quadratic",
                        freshwater_density_beta_1_k2=density_beta,
                    )
                )
                phase = simulation.phi.to_numpy()
                wall = simulation.wall.to_numpy()
                solid = simulation.solid.to_numpy()
                probe = None
                for i in range(3, simulation.nx - 3):
                    for j in range(3, simulation.ny - 3):
                        if (
                            wall[i, j] == 0
                            and solid[i, j] == 0
                            and np.min(phase[i - 1 : i + 2, j - 1 : j + 2])
                            > 0.999
                        ):
                            probe = (i, j)
                            break
                    if probe is not None:
                        break
                self.assertIsNotNone(probe)
                assert probe is not None

                simulation._collide_velocity()
                baseline_y = float(simulation.fluid_force.to_numpy()[probe][1])
                self.assertAlmostEqual(baseline_y, 0.0, delta=1.0e-10)
                simulations[bath_temperature] = (simulation, probe)

            simulation, probe = simulations[bath_temperature]
            temperature = simulation.temperature.to_numpy()
            temperature[probe] = local_temperature
            simulation.temperature.from_numpy(temperature)
            simulation._collide_velocity()
            force_y = float(simulation.fluid_force.to_numpy()[probe][1])
            anomaly = water_density_anomaly_ratio_to_reference(
                local_temperature, simulation.cfg.thermal
            )
            expected_force_y = float(simulation._gravity_l[1]) * anomaly

            with self.subTest(
                bath_temperature=bath_temperature,
                local_temperature=local_temperature,
            ):
                self.assertAlmostEqual(force_y, expected_force_y, delta=2.0e-9)
                if expected_sign > 0:
                    self.assertGreater(force_y, 0.0)
                elif expected_sign < 0:
                    self.assertLess(force_y, 0.0)
                else:
                    self.assertAlmostEqual(force_y, 0.0, delta=1.0e-10)

            temperature[probe] = bath_temperature
            simulation.temperature.from_numpy(temperature)

    def test_well_balanced_phase_change_refill_carries_only_dynamic_pressure(self):
        simulation = IceFlow2D(
            _coupled_config(
                water_temperature_c=8.0,
                wall_temperature_c=8.0,
                gravity_m_s2=9.8,
                water_buoyancy_model="freshwater_quadratic",
            )
        )
        solid_before = simulation.solid.to_numpy()
        wall = simulation.wall.to_numpy()
        release_node = None
        for i, j in np.argwhere(solid_before == 1):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (
                        (di != 0 or dj != 0)
                        and wall[i + di, j + dj] == 0
                        and solid_before[i + di, j + dj] == 0
                    ):
                        release_node = (int(i), int(j))
                        break
                if release_node is not None:
                    break
            if release_node is not None:
                break
        self.assertIsNotNone(release_node)
        assert release_node is not None

        hydrostatic_pressure = simulation.hydrostatic_reference_pressure.to_numpy()
        prescribed_dynamic_pressure = 3.0e-4
        pressure = simulation.p.to_numpy()
        active_before = (wall == 0) & (solid_before == 0)
        pressure[active_before] = (
            hydrostatic_pressure[active_before] + prescribed_dynamic_pressure
        )
        simulation.p.from_numpy(pressure)
        simulation.solid_prev.from_numpy(solid_before)
        solid_after = solid_before.copy()
        solid_after[release_node] = 0
        simulation.solid.from_numpy(solid_after)

        simulation._refill_phase_change_nodes()

        pressure_after = simulation.p.to_numpy()
        distributions = simulation.f.to_numpy()
        dynamic_after = (
            pressure_after[release_node] - hydrostatic_pressure[release_node]
        )
        self.assertAlmostEqual(
            float(dynamic_after), prescribed_dynamic_pressure, delta=2.0e-7
        )
        # q=1 is an axis population with w=1/9, hence f_eq=3*w*p_dyn.
        self.assertAlmostEqual(
            float(distributions[release_node][1]),
            prescribed_dynamic_pressure / 3.0,
            delta=2.0e-7,
        )
        momentum = np.zeros(2, dtype=np.float64)
        directions = np.asarray(
            (
                (0, 0),
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, 1),
                (-1, -1),
                (1, -1),
            ),
            dtype=np.float64,
        )
        momentum = distributions[release_node] @ directions
        np.testing.assert_allclose(momentum, 0.0, rtol=0.0, atol=1.0e-9)


if __name__ == "__main__":
    unittest.main()
