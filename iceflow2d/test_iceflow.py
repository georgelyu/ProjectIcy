"""Focused contracts for the retained falling-and-melting solver path."""

import math
import unittest

import numpy as np
import taichi as ti

from iceflow2d import IceFlow2D, create_iceflow_config
from iceflow2d.simulator import ensure_taichi_cuda
from iceflow2d.config import ThermalBoundary, ThermalBoundarySet, ThermalConfig


@ti.kernel
def _prepare_test_cut_links(simulation: ti.template()):
    for i, j in simulation.water_phase:
        if simulation._is_fluid_cell(i, j):
            raw = ti.Vector.zero(ti.f32, 9)
            for direction in range(9):
                raw[direction] = simulation.momentum_stream_buffer[i, j, direction]
            prepared = simulation._prepare_momentum_cut_links(i, j, raw)
            for direction in range(9):
                simulation.momentum_stream_buffer[i, j, direction] = prepared[direction]


def _thermal_config(*, update_interval: int = 2) -> ThermalConfig:
    hot = ThermalBoundary.dirichlet(90.0)
    return ThermalConfig(
        boundaries=ThermalBoundarySet(
            left=hot,
            right=hot,
            bottom=hot,
            top=ThermalBoundary.adiabatic(),
        ),
        initial_water_temperature_c=90.0,
        initial_ice_temperature_c=0.0,
        initial_air_temperature_c=20.0,
        update_interval_lbm_steps=update_interval,
        water_buoyancy_model="linear",
        buoyancy_reference_temperature_c=90.0,
        moving_body_scheme="body_ale",
    )


def _falling_config(**overrides):
    nx, ny = 32, 48
    boundary = 2
    values = dict(
        resolution=(nx, ny),
        dx=5.0e-4,
        reference_velocity=4.0,
        rho_water=1000.0,
        rho_air=1.25,
        rho_ice=917.0,
        viscosity_water=1.0e-6,
        viscosity_air=1.5e-5,
        sigma=0.072,
        gravity=(0.0, -9.8),
        phase_warmup_steps=2,
        well_balanced_hydrostatics=True,
        water_width_fraction=(nx - boundary) / nx,
        water_height_fraction=24.0 / ny,
        ice_width_fraction=8.0 / nx,
        ice_height_fraction=8.0 / ny,
        ice_base_y_cells=30.0,
        boundary_cells=boundary,
        ice_initial_angle=math.radians(5.0),
        ice_fixed=False,
        rigid_boundary_scheme="unified",
        air_interface_relaxation_time=0.8,
        thermal=_thermal_config(),
    )
    values.update(overrides)
    return create_iceflow_config(**values)


def _assert_body_inside(test: unittest.TestCase, simulation: IceFlow2D) -> None:
    center = np.asarray(simulation.body_center[None], dtype=np.float64)
    angle = float(simulation.body_angle[None])
    cosine = abs(math.cos(angle))
    sine = abs(math.sin(angle))
    extent_x = (
        cosine * simulation._body_half_width + sine * simulation._body_half_height
    )
    extent_y = (
        sine * simulation._body_half_width + cosine * simulation._body_half_height
    )
    boundary = simulation.config.boundary_cells
    tolerance = 2.0e-5
    test.assertGreaterEqual(center[0] - extent_x, boundary - tolerance)
    test.assertLessEqual(center[0] + extent_x, simulation.nx - boundary + tolerance)
    test.assertGreaterEqual(center[1] - extent_y, boundary - tolerance)
    test.assertLessEqual(center[1] + extent_y, simulation.ny - boundary + tolerance)


class IceFlowConfigTests(unittest.TestCase):
    def test_supported_configuration_is_explicit(self):
        config = _falling_config()

        self.assertFalse(config.ice_fixed)
        self.assertEqual(config.rigid_boundary_scheme, "unified")
        self.assertTrue(config.well_balanced_hydrostatics)
        self.assertEqual(config.thermal.moving_body_scheme, "body_ale")
        self.assertEqual(config.thermal.water_buoyancy_model, "linear")
        self.assertEqual((config.water_width, config.water_height), (30, 24))
        self.assertEqual((config.ice_width, config.ice_height), (8, 8))
        self.assertGreater(config.ice_mass_lattice, 0.0)
        self.assertGreater(config.ice_inertia_lattice, 0.0)

    def test_unknown_and_removed_solver_modes_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unknown IceFlowConfig option"):
            _falling_config(unrelated_solver_option=True)

        invalid = (
            ({"thermal": None}, "thermal"),
            ({"ice_fixed": True}, "ice_fixed"),
            ({"rigid_boundary_scheme": "halfway"}, "unified"),
            ({"well_balanced_hydrostatics": False}, "well_balanced"),
        )
        for overrides, message in invalid:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    _falling_config(**overrides)

    def test_numerical_guards_remain_bounded(self):
        for value in (0.5, 2.01, float("nan")):
            with self.subTest(air_interface_relaxation_time=value):
                with self.assertRaisesRegex(
                    ValueError, "air_interface_relaxation_time"
                ):
                    _falling_config(air_interface_relaxation_time=value)

    def test_initial_geometry_respects_the_physical_walls(self):
        with self.assertRaisesRegex(ValueError, "side wall"):
            _falling_config(
                ice_width_fraction=30.0 / 32.0,
                ice_initial_angle=0.0,
            )
        with self.assertRaisesRegex(ValueError, "bottom wall"):
            _falling_config(ice_base_y_cells=1.5, ice_initial_angle=0.0)


class IceFlowCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:  # pragma: no cover - depends on test host
            raise unittest.SkipTest(str(exc)) from exc

    def test_short_coupled_run_stays_finite_and_conservative(self):
        simulation = IceFlow2D(_falling_config())
        initial_center = np.asarray(simulation.body_center[None], dtype=np.float64)
        initial_mass = simulation.thermal.mass_energy_totals().total_mass_kg_m

        # Finish on a slow-thermal boundary so all public fields share one time.
        simulation.step(2)

        self.assertEqual(simulation.steps, 2)
        self.assertAlmostEqual(
            simulation.physical_time_s, simulation.thermal.time_s, places=13
        )
        for name in (
            "momentum_populations",
            "phase_populations",
            "water_phase",
            "momentum_velocity_lattice",
            "fluid_acceleration_lattice",
            "pressure_lattice",
            "temperature",
            "liquid_fraction",
            "thermal_enthalpy",
        ):
            with self.subTest(field=name):
                self.assertTrue(np.isfinite(getattr(simulation, name).to_numpy()).all())

        target = float(simulation.water_volume_target[None])
        current = float(simulation.water_volume_current[None])
        tolerance = simulation.config.volume_projection_tolerance * max(1.0, target)
        self.assertLessEqual(abs(current - target), tolerance)
        final_mass = simulation.thermal.mass_energy_totals().total_mass_kg_m
        self.assertAlmostEqual(final_mass, initial_mass, delta=2.0e-11)
        self.assertLess(float(simulation.body_velocity[None][1]), 0.0)
        self.assertLess(float(simulation.body_center[None][1]), initial_center[1])
        _assert_body_inside(self, simulation)

    def test_single_node_reflection_applies_to_every_cut_link(self):
        simulation = IceFlow2D(
            _falling_config(
                gravity=(0.0, 0.0),
                phase_warmup_steps=0,
                ice_initial_angle=math.radians(5.0),
            )
        )
        directions = (
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        )
        opposites = (0, 3, 4, 1, 2, 7, 8, 5, 6)
        solid = simulation.solid_mask.to_numpy()
        wall = simulation.wall_mask.to_numpy()
        active = (solid == 0) & (wall == 0)
        cut_link = None
        for i in range(simulation.nx):
            for j in range(simulation.ny):
                if not active[i, j]:
                    continue
                for q in range(1, len(directions)):
                    di, dj = directions[q]
                    ni, nj = i + di, j + dj
                    back_i, back_j = i - di, j - dj
                    if not (
                        0 <= ni < simulation.nx
                        and 0 <= nj < simulation.ny
                        and 0 <= back_i < simulation.nx
                        and 0 <= back_j < simulation.ny
                    ):
                        continue
                    if solid[ni, nj] == 1 and active[back_i, back_j]:
                        cut_link = (i, j, q, back_i, back_j)
                        break
                if cut_link is not None:
                    break
            if cut_link is not None:
                break
        self.assertIsNotNone(cut_link)
        i, j, q, back_i, back_j = cut_link
        opposite = opposites[q]

        boundary_velocity = np.asarray([0.012, -0.008], dtype=np.float64)
        simulation.body_velocity[None] = tuple(boundary_velocity)
        simulation.body_angular_velocity[None] = 0.0
        simulation.momentum_velocity_lattice[i, j] = (0.0, 0.0)
        simulation.fluid_acceleration_lattice[i, j] = (0.0, 0.0)
        simulation.pressure_lattice[i, j] = simulation.hydrostatic_reference_pressure[
            i, j
        ]
        simulation.water_phase[i, j] = 1.0
        simulation.water_phase[back_i, back_j] = 1.0
        outgoing = 0.91
        incoming_post = 0.17
        outgoing_nonequilibrium = 0.31
        outgoing_phase = 0.43
        simulation.momentum_stream_buffer[i, j, q] = outgoing
        simulation.momentum_stream_buffer[i, j, opposite] = incoming_post
        # Use a sentinel to prove that Tao's one-point formula does not read
        # the back-node population.
        simulation.momentum_stream_buffer[back_i, back_j, q] = -3.7
        simulation.momentum_populations[i, j, q] = outgoing_nonequilibrium
        simulation.phase_stream_buffer[i, j, q] = outgoing_phase

        sdf_fluid = max(float(simulation.body_signed_distance_m[i, j]), 1.0e-6)
        eta = min(
            0.95,
            max(
                0.05,
                sdf_fluid
                / (
                    sdf_fluid
                    - float(
                        simulation.body_signed_distance_m[
                            i + directions[q][0], j + directions[q][1]
                        ]
                    )
                    + 1.0e-12
                ),
            ),
        )
        weight = 1.0 / 9.0 if q <= 4 else 1.0 / 36.0
        incoming_direction = np.asarray(directions[opposite], dtype=np.float64)
        incoming_wall_speed = float(incoming_direction.dot(boundary_velocity))
        wall_equilibrium = (
            weight
            * simulation._water_density_lattice
            * (
                3.0 * incoming_wall_speed
                + 4.5 * incoming_wall_speed * incoming_wall_speed
                - 1.5 * float(boundary_velocity.dot(boundary_velocity))
            )
        )
        expected_single_node = (
            wall_equilibrium + outgoing_nonequilibrium + eta * incoming_post
        ) / (1.0 + eta)
        outgoing_wall_speed = float(
            np.asarray(directions[q], dtype=np.float64).dot(boundary_velocity)
        )
        moving_correction = 6.0 * weight * 1.0 * outgoing_wall_speed
        _prepare_test_cut_links(simulation)
        simulation._stream()
        self.assertAlmostEqual(
            float(simulation.momentum_populations[i, j, opposite]),
            expected_single_node,
            places=6,
        )
        self.assertAlmostEqual(
            float(simulation.phase_populations[i, j, opposite]),
            outgoing_phase - moving_correction,
            places=6,
        )

        # The contact band and an inactive back node must use the same
        # single-node reconstruction.  Neither is an input to Tao's formula.
        simulation.water_phase[i, j] = 0.5
        simulation.solid_mask[back_i, back_j] = 1
        simulation.momentum_stream_buffer[i, j, q] = outgoing
        simulation.momentum_stream_buffer[i, j, opposite] = incoming_post
        simulation.momentum_populations[i, j, q] = outgoing_nonequilibrium
        simulation.phase_stream_buffer[i, j, q] = outgoing_phase
        _prepare_test_cut_links(simulation)
        simulation._stream()
        contact_density = 0.5 * (
            simulation._water_density_lattice + simulation._air_density_lattice
        )
        contact_wall_equilibrium = (
            weight
            * contact_density
            * (
                3.0 * incoming_wall_speed
                + 4.5 * incoming_wall_speed * incoming_wall_speed
                - 1.5 * float(boundary_velocity.dot(boundary_velocity))
            )
        )
        expected_contact_single_node = (
            contact_wall_equilibrium + outgoing_nonequilibrium + eta * incoming_post
        ) / (1.0 + eta)
        self.assertAlmostEqual(
            float(simulation.momentum_populations[i, j, opposite]),
            expected_contact_single_node,
            places=6,
        )
        expected_contact_phase = (
            outgoing_phase - 6.0 * weight * 0.5 * outgoing_wall_speed
        )
        self.assertAlmostEqual(
            float(simulation.phase_populations[i, j, opposite]),
            expected_contact_phase,
            places=6,
        )

    def test_rigid_integration_uses_physical_mass_and_inertia(self):
        simulation = IceFlow2D(
            _falling_config(
                gravity=(0.0, 0.0),
                phase_warmup_steps=0,
                linear_damping=1.0,
                angular_damping=1.0,
            )
        )
        mass = float(simulation.body_mass_lattice[None])
        inertia = float(simulation.body_inertia_lattice[None])
        # Use large increments to verify that the integrator retains the
        # physical impulse update without artificial translational or angular
        # clipping.
        delta_velocity = np.asarray([0.2, -0.1], dtype=np.float64)
        delta_omega = 1.0e-2
        velocity_before = np.asarray(simulation.body_velocity[None], dtype=np.float64)
        omega_before = float(simulation.body_angular_velocity[None])

        simulation.hydrodynamic_impulse[None] = tuple(mass * delta_velocity)
        simulation.hydrodynamic_torque[None] = inertia * delta_omega
        simulation._integrate_rigid_ice()

        np.testing.assert_allclose(
            np.asarray(simulation.body_velocity[None], dtype=np.float64)
            - velocity_before,
            delta_velocity,
            rtol=0.0,
            atol=2.0e-8,
        )
        self.assertAlmostEqual(
            float(simulation.body_angular_velocity[None]) - omega_before,
            delta_omega,
            delta=2.0e-9,
        )

    def test_contact_support_matches_unclipped_sharp_raster(self):
        simulation = IceFlow2D(
            _falling_config(
                phase_warmup_steps=0,
                ice_width_fraction=16.0 / 32.0,
                ice_height_fraction=16.0 / 48.0,
                ice_base_y_cells=26.0,
            )
        )
        simulation.body_angle[None] = math.radians(31.0)
        simulation.body_center[None] = (16.0, 30.0)
        simulation.body_reference_origin[None] = (16.0, 30.0)
        threshold = simulation.config.thermal.solid_liquid_threshold

        # This above-threshold material cell is too small to produce any
        # thresholded node after the same bilinear world sampling used by the
        # sharp solver.  It must not create a several-cell "ghost" contact.
        isolated = np.zeros((16, 16), dtype=np.float32)
        isolated[0, 8] = 0.51
        simulation.thermal.body_solid_mass.from_numpy(
            isolated.astype(np.float64) * simulation.thermal._initial_body_cell_mass
        )
        simulation._solve_contact_and_rasterize(resolve_contact=False)
        sharp = simulation.thermal.world_body_solid_fraction.to_numpy() >= threshold
        self.assertFalse(bool(np.any(sharp)))
        simulation._prepare_moving_body_contact_support()
        self.assertEqual(int(simulation._body_contact_geometry_active[None]), 0)

        # With a resolved core present, adding the same unresolved cell leaves
        # the contact bounds equal to the actual thresholded sharp-node bounds.
        material = isolated.copy()
        material[6:10, 6:10] = 1.0
        simulation.thermal.body_solid_mass.from_numpy(
            material.astype(np.float64) * simulation.thermal._initial_body_cell_mass
        )
        simulation._solve_contact_and_rasterize(resolve_contact=False)
        sharp = simulation.thermal.world_body_solid_fraction.to_numpy() >= threshold
        sharp_indices = np.argwhere(sharp)
        self.assertGreater(sharp_indices.shape[0], 0)
        expected_world_extrema = np.asarray(
            [
                sharp_indices[:, 0].min(),
                sharp_indices[:, 0].max() + 1,
                sharp_indices[:, 1].min(),
                sharp_indices[:, 1].max() + 1,
            ],
            dtype=np.float64,
        )
        simulation._prepare_moving_body_contact_support()
        origin = np.asarray(simulation.body_reference_origin[None], dtype=np.float64)
        support = simulation._body_contact_support_extrema.to_numpy().astype(np.float64)
        support_world_extrema = support + np.asarray(
            [origin[0], origin[0], origin[1], origin[1]], dtype=np.float64
        )
        np.testing.assert_allclose(
            support_world_extrema, expected_world_extrema, rtol=0.0, atol=2.0e-6
        )

    def test_exact_left_wall_contact_rejects_leftward_velocity(self):
        simulation = IceFlow2D(
            _falling_config(
                gravity=(0.0, 0.0),
                phase_warmup_steps=0,
                angular_damping=1.0,
            )
        )
        simulation.body_angle[None] = math.radians(30.0)
        simulation.body_center[None] = (16.0, 30.0)
        simulation.body_reference_origin[None] = (16.0, 30.0)
        simulation._prepare_moving_body_contact_support()
        origin = np.asarray(simulation.body_reference_origin[None], dtype=np.float64)
        center = np.asarray(simulation.body_center[None], dtype=np.float64)
        minimum_x = float(simulation._body_contact_support_extrema[0])
        shift = simulation.config.boundary_cells - (origin[0] + minimum_x)
        origin[0] += shift
        center[0] += shift
        simulation.body_reference_origin[None] = tuple(origin)
        simulation.body_center[None] = tuple(center)
        simulation.body_velocity[None] = (-1.0e-3, 2.0e-4)
        simulation.body_angular_velocity[None] = 4.0e-4
        simulation._prepare_moving_body_contact_support()

        center_before = np.asarray(simulation.body_center[None], dtype=np.float64)
        omega_before = float(simulation.body_angular_velocity[None])
        simulation._project_moving_body_inside_container()

        np.testing.assert_allclose(
            np.asarray(simulation.body_center[None], dtype=np.float64),
            center_before,
            rtol=0.0,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            simulation.body_velocity.to_numpy(),
            np.asarray([0.0, 2.0e-4], dtype=np.float32),
            rtol=0.0,
            atol=1.0e-10,
        )
        self.assertEqual(float(simulation.body_angular_velocity[None]), omega_before)

    def test_wall_projection_clamps_only_outward_velocity(self):
        simulation = IceFlow2D(
            _falling_config(
                gravity=(0.0, 0.0),
                phase_warmup_steps=0,
                angular_damping=1.0,
            )
        )
        # The contact support follows thresholded world cells, so crossing one
        # complete lattice spacing guarantees a resolved support penetration.
        penetration = 1.02
        boundary = simulation.config.boundary_cells
        wall_cases = (
            ("left", 0, 0, boundary, -1.0, 1.0, (-1.0e-3, 2.0e-4)),
            (
                "right",
                0,
                1,
                simulation.nx - boundary,
                1.0,
                -1.0,
                (1.0e-3, 2.0e-4),
            ),
            (
                "bottom",
                1,
                2,
                boundary,
                -1.0,
                1.0,
                (2.0e-4, -1.0e-3),
            ),
            (
                "top",
                1,
                3,
                simulation.ny - boundary,
                1.0,
                -1.0,
                (2.0e-4, 1.0e-3),
            ),
        )
        for (
            name,
            axis,
            extrema_index,
            plane,
            crossing_sign,
            inward_sign,
            velocity,
        ) in wall_cases:
            with self.subTest(wall=name):
                simulation.body_angle[None] = math.radians(30.0)
                simulation.body_center[None] = (16.0, 30.0)
                simulation.body_reference_origin[None] = (16.0, 30.0)
                simulation._prepare_moving_body_contact_support()
                origin = np.asarray(
                    simulation.body_reference_origin[None], dtype=np.float64
                )
                center = np.asarray(simulation.body_center[None], dtype=np.float64)
                extrema = simulation._body_contact_support_extrema.to_numpy()
                support_world = origin[axis] + extrema[extrema_index]
                shift = plane - support_world + crossing_sign * penetration
                center[axis] += shift
                origin[axis] += shift
                simulation.body_center[None] = tuple(center)
                simulation.body_reference_origin[None] = tuple(origin)
                simulation.body_velocity[None] = velocity
                simulation.body_angular_velocity[None] = 4.0e-4
                simulation._prepare_moving_body_contact_support()

                center_before = np.asarray(
                    simulation.body_center[None], dtype=np.float64
                )
                velocity_before = np.asarray(
                    simulation.body_velocity[None], dtype=np.float64
                )
                omega_before = float(simulation.body_angular_velocity[None])
                simulation._project_moving_body_inside_container()

                center_after = np.asarray(
                    simulation.body_center[None], dtype=np.float64
                )
                velocity_after = np.asarray(
                    simulation.body_velocity[None], dtype=np.float64
                )
                omega_after = float(simulation.body_angular_velocity[None])
                self.assertGreater(
                    inward_sign * (center_after[axis] - center_before[axis]), 0.0
                )
                self.assertEqual(velocity_after[axis], 0.0)
                self.assertEqual(velocity_after[1 - axis], velocity_before[1 - axis])
                self.assertEqual(omega_after, omega_before)
                simulation._prepare_moving_body_contact_support()
                origin_after = np.asarray(
                    simulation.body_reference_origin[None], dtype=np.float64
                )
                support_after = (
                    simulation._body_contact_support_extrema.to_numpy().astype(
                        np.float64
                    )
                )
                world_extrema = support_after + np.asarray(
                    [
                        origin_after[0],
                        origin_after[0],
                        origin_after[1],
                        origin_after[1],
                    ]
                )
                self.assertGreaterEqual(world_extrema[0], boundary - 2.0e-5)
                self.assertLessEqual(
                    world_extrema[1], simulation.nx - boundary + 2.0e-5
                )
                self.assertGreaterEqual(world_extrema[2], boundary - 2.0e-5)
                self.assertLessEqual(
                    world_extrema[3], simulation.ny - boundary + 2.0e-5
                )


if __name__ == "__main__":
    unittest.main()
