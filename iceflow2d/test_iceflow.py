"""Focused regression tests for the retained IceFlow2D simulation API."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from iceflow2d import IceFlow2D, create_iceflow_config
from iceflow2d.simulator import ensure_taichi_cuda

REGRESSION_REFERENCE_VELOCITY_M_S = 10.844353369380768


def _full_active_fraction(size: int, boundary: int = 3) -> float:
    """Return a fraction whose integer extent ends at the far wall."""

    return (size - boundary + 0.5) / size


def _active_mask(simulation: IceFlow2D) -> np.ndarray:
    return (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)


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
    boundary = simulation.cfg.boundary_cells
    tolerance = 2.0e-5
    test.assertGreaterEqual(center[0] - extent_x, boundary - tolerance)
    test.assertLessEqual(center[0] + extent_x, simulation.nx - boundary + tolerance)
    test.assertGreaterEqual(center[1] - extent_y, boundary - tolerance)
    test.assertLessEqual(center[1] + extent_y, simulation.ny - boundary + tolerance)


class IceFlowConfigTests(unittest.TestCase):
    def test_default_geometry_and_rigid_properties(self):
        config = create_iceflow_config()
        self.assertEqual(config.resolution, (600, 300))
        self.assertEqual((config.water_width, config.water_height), (150, 200))
        self.assertEqual((config.ice_width, config.ice_height), (90, 90))
        self.assertEqual(config.ice_initial_center, (300.0, 48.0))
        self.assertAlmostEqual(config.ice_mass_lattice, 7427.7)
        self.assertGreater(config.ice_inertia_lattice, 0.0)

    def test_unknown_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unknown IceFlowConfig option"):
            create_iceflow_config(unrelated_solver_option=True)
        with self.assertRaisesRegex(TypeError, "reference_length_cells"):
            create_iceflow_config(reference_length_cells=300)

    def test_reference_velocity_is_fixed_independently_of_resolution(self):
        config = create_iceflow_config(resolution=(120, 60))
        self.assertEqual(config.reference_velocity, 1.0e-3)
        explicit = create_iceflow_config(
            resolution=(120, 60), reference_velocity=2.0e-3
        )
        self.assertEqual(explicit.reference_velocity, 2.0e-3)

    def test_reference_velocity_must_be_positive_and_finite(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "reference_velocity"):
                    create_iceflow_config(reference_velocity=value)

    def test_fixed_ice_requires_zero_initial_motion(self):
        self.assertTrue(create_iceflow_config(ice_fixed=True).ice_fixed)
        with self.assertRaisesRegex(ValueError, "fixed ice must have zero"):
            create_iceflow_config(ice_fixed=True, ice_initial_velocity=(0.01, 0.0))
        with self.assertRaisesRegex(ValueError, "fixed ice must have zero"):
            create_iceflow_config(ice_fixed=True, ice_initial_angular_velocity=1.0e-4)

    def test_contact_coefficients_are_bounded(self):
        for name in ("wall_restitution", "wall_friction", "bottom_wall_friction"):
            for value in (-0.01, 1.01):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        create_iceflow_config(**{name: value})

    def test_well_balanced_hydrostatic_geometry_is_validated(self):
        with self.assertRaisesRegex(ValueError, "full-width horizontal pool"):
            create_iceflow_config(well_balanced_hydrostatics=True)

        size = 64
        config = create_iceflow_config(
            resolution=(size, size),
            water_width_fraction=_full_active_fraction(size),
            water_height_fraction=0.5,
            well_balanced_hydrostatics=True,
        )
        self.assertTrue(config.well_balanced_hydrostatics)
        self.assertNotIn("hydrostatic_initialization", config.to_dict())
        self.assertNotIn("pressure_completed_gimem", config.to_dict())

        with self.assertRaisesRegex(ValueError, "requires vertical gravity"):
            create_iceflow_config(
                resolution=(size, size),
                water_width_fraction=_full_active_fraction(size),
                water_height_fraction=0.5,
                gravity=(1.0, -9.8),
                well_balanced_hydrostatics=True,
            )

    def test_removed_hydrostatic_options_are_rejected(self):
        for option in ("hydrostatic_initialization", "pressure_completed_gimem"):
            for value in (False, True):
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(
                        TypeError, f"Unknown IceFlowConfig option: {option}"
                    ):
                        create_iceflow_config(**{option: value})

        config = create_iceflow_config()
        self.assertFalse(hasattr(config, "hydrostatic_initialization"))
        self.assertFalse(hasattr(config, "pressure_completed_gimem"))

    def test_well_balanced_hydrostatics_requires_boolean(self):
        for value in (None, 0, 1, "true"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "well_balanced_hydrostatics must be a boolean"
                ):
                    create_iceflow_config(well_balanced_hydrostatics=value)


class IceFlowCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:  # pragma: no cover - depends on test host
            raise unittest.SkipTest(str(exc)) from exc

    def test_short_run_stays_finite_conservative_and_writes_state(self):
        with tempfile.TemporaryDirectory(prefix="iceflow2d-test-") as directory:
            output = Path(directory)
            config = create_iceflow_config(
                resolution=(64, 32),
                reference_velocity=REGRESSION_REFERENCE_VELOCITY_M_S,
                phase_warmup_steps=2,
                output_dir=str(output),
            )
            simulation = IceFlow2D(config)

            self.assertIsNotNone(simulation.hydrodynamic_impulse)
            self.assertIsNotNone(simulation.hydrodynamic_torque)
            self.assertIsNotNone(simulation.solid_prev)
            self.assertIsNone(simulation.hydrostatic_reference_pressure)
            self.assertIsNone(simulation.hydrostatic_reference_density)

            simulation.step(3)
            self.assertEqual(simulation.steps, 3)
            for name in ("f", "h", "phi", "u", "fluid_force", "p"):
                self.assertTrue(
                    np.isfinite(getattr(simulation, name).to_numpy()).all(), name
                )
            self.assertTrue(
                np.isfinite(np.asarray(simulation.hydrodynamic_impulse[None])).all()
            )
            self.assertTrue(math.isfinite(float(simulation.hydrodynamic_torque[None])))

            active = _active_mask(simulation)
            phase = simulation.phi.to_numpy()
            self.assertGreater(np.count_nonzero(active), 0)
            self.assertGreaterEqual(float(np.min(phase[active])), 0.0)
            self.assertLessEqual(float(np.max(phase[active])), 1.0)
            target = float(simulation.water_volume_target[None])
            current = float(simulation.water_volume_current[None])
            host_volume = float(np.sum(phase[active], dtype=np.float64))
            tolerance = max(5.0e-5, 5.0e-8 * max(1.0, abs(target)))
            self.assertAlmostEqual(current, host_volume, delta=1.0e-8)
            self.assertAlmostEqual(current, target, delta=tolerance)
            _assert_body_inside(self, simulation)

            frame_path = output / "frame.png"
            state_path = output / "state.npz"
            simulation.save_frame(frame_path)
            simulation.save_state_npz(state_path)
            self.assertGreater(frame_path.stat().st_size, 0)
            with np.load(state_path) as state:
                self.assertEqual(state["phi"].shape, config.resolution)
                self.assertEqual(state["u"].shape, config.resolution + (2,))
                self.assertEqual(state["solid"].shape, config.resolution)
                self.assertNotIn("hydrostatic_reference_pressure", state.files)

    def test_fixed_ice_keeps_pose_and_avoids_moving_body_allocations(self):
        size = 48
        config = create_iceflow_config(
            resolution=(size, size),
            reference_velocity=REGRESSION_REFERENCE_VELOCITY_M_S,
            phase_warmup_steps=0,
            water_width_fraction=_full_active_fraction(size),
            water_height_fraction=_full_active_fraction(size),
            ice_width_fraction=8.0 / size,
            ice_height_fraction=8.0 / size,
            ice_base_y_cells=20.0,
            ice_fixed=True,
            well_balanced_hydrostatics=True,
        )
        simulation = IceFlow2D(config)
        center_before = np.asarray(simulation.body_center[None], dtype=np.float64)
        angle_before = float(simulation.body_angle[None])
        solid_before = simulation.solid.to_numpy()

        self.assertIsNone(simulation.hydrodynamic_impulse)
        self.assertIsNone(simulation.hydrodynamic_torque)
        self.assertIsNone(simulation.solid_prev)
        self.assertIsNotNone(simulation.hydrostatic_reference_pressure)
        self.assertIsNotNone(simulation.hydrostatic_reference_density)

        simulation.step(2)
        np.testing.assert_array_equal(simulation.solid.to_numpy(), solid_before)
        np.testing.assert_allclose(
            np.asarray(simulation.body_center[None]), center_before, rtol=0.0, atol=0.0
        )
        self.assertEqual(float(simulation.body_angle[None]), angle_before)
        np.testing.assert_array_equal(
            np.asarray(simulation.body_velocity[None]), np.zeros(2, dtype=np.float32)
        )
        self.assertEqual(float(simulation.body_angular_velocity[None]), 0.0)
        self.assertTrue(np.isfinite(simulation.p.to_numpy()).all())
        _assert_body_inside(self, simulation)

    def test_unforced_moving_body_advances_under_gravity(self):
        size = 64
        config = create_iceflow_config(
            resolution=(size, size),
            reference_velocity=REGRESSION_REFERENCE_VELOCITY_M_S,
            phase_warmup_steps=0,
            ice_width_fraction=8.0 / size,
            ice_height_fraction=8.0 / size,
            ice_base_y_cells=35.0,
            linear_damping=1.0,
            angular_damping=1.0,
        )
        simulation = IceFlow2D(config)
        center_before = np.asarray(simulation.body_center[None], dtype=np.float64)
        simulation.hydrodynamic_impulse[None] = (0.0, 0.0)
        simulation.hydrodynamic_torque[None] = 0.0

        for _ in range(8):
            simulation._integrate_rigid_ice()

        center_after = np.asarray(simulation.body_center[None], dtype=np.float64)
        velocity = np.asarray(simulation.body_velocity[None], dtype=np.float64)
        self.assertTrue(np.isfinite(center_after).all())
        self.assertTrue(np.isfinite(velocity).all())
        self.assertLess(center_after[1], center_before[1])
        self.assertLess(velocity[1], 0.0)
        self.assertAlmostEqual(velocity[0], 0.0, delta=1.0e-8)
        self.assertAlmostEqual(float(simulation.body_angular_velocity[None]), 0.0)
        _assert_body_inside(self, simulation)

    def test_cut_link_load_is_f64_and_cleared_after_rigid_integration(self):
        size = 64
        config = create_iceflow_config(
            resolution=(size, size),
            reference_velocity=REGRESSION_REFERENCE_VELOCITY_M_S,
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            ice_width_fraction=8.0 / size,
            ice_height_fraction=8.0 / size,
            ice_base_y_cells=35.0,
            linear_damping=1.0,
            angular_damping=1.0,
        )
        simulation = IceFlow2D(config)
        impulse = np.asarray([0.01, -0.02]) * simulation._body_mass
        torque = 1.0e-4 * simulation._body_inertia
        simulation.hydrodynamic_impulse[None] = impulse
        simulation.hydrodynamic_torque[None] = torque

        self.assertEqual(simulation.hydrodynamic_impulse.to_numpy().dtype, np.float64)
        self.assertEqual(simulation.hydrodynamic_torque.to_numpy().dtype, np.float64)
        simulation._integrate_rigid_ice()

        np.testing.assert_allclose(
            np.asarray(simulation.body_velocity[None]),
            np.asarray([0.01, -0.02], dtype=np.float32),
            rtol=0.0,
            atol=2.0e-7,
        )
        self.assertAlmostEqual(
            float(simulation.body_angular_velocity[None]), 1.0e-4, delta=1.0e-7
        )
        np.testing.assert_array_equal(
            simulation.hydrodynamic_impulse.to_numpy(),
            np.zeros(2, dtype=np.float64),
        )
        self.assertEqual(float(simulation.hydrodynamic_torque[None]), 0.0)

    def test_bottom_contact_prevents_penetration_and_applies_friction(self):
        nx, ny = 80, 48
        initial_velocity = (0.01, -0.01)
        config = create_iceflow_config(
            resolution=(nx, ny),
            reference_velocity=REGRESSION_REFERENCE_VELOCITY_M_S,
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            ice_width_fraction=8.0 / nx,
            ice_height_fraction=6.0 / ny,
            ice_base_y_cells=3.0,
            ice_initial_velocity=initial_velocity,
            linear_damping=1.0,
            angular_damping=1.0,
            wall_restitution=0.2,
            bottom_wall_friction=0.5,
        )
        simulation = IceFlow2D(config)
        simulation.hydrodynamic_impulse[None] = (0.0, 0.0)
        simulation.hydrodynamic_torque[None] = 0.0
        simulation._integrate_rigid_ice()

        velocity = np.asarray(simulation.body_velocity[None], dtype=np.float64)
        self.assertTrue(np.isfinite(velocity).all())
        self.assertGreaterEqual(velocity[1], -1.0e-6)
        self.assertLess(abs(velocity[0]), abs(initial_velocity[0]))
        _assert_body_inside(self, simulation)


if __name__ == "__main__":
    unittest.main()
