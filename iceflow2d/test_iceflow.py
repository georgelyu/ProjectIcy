"""CUDA regression tests kept inside the independent iceflow2d package."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from iceflow2d import IceFlow2D, create_iceflow_config


class IceFlowConfigTests(unittest.TestCase):
    def test_default_matches_legacy_coupled_ice_geometry(self):
        config = create_iceflow_config()
        self.assertEqual(config.resolution, (600, 300))
        self.assertEqual((config.water_width, config.water_height), (150, 200))
        self.assertEqual((config.ice_width, config.ice_height), (90, 90))
        self.assertEqual(config.ice_initial_center, (300.0, 48.0))
        self.assertAlmostEqual(config.ice_mass_lattice, 7427.7)
        self.assertEqual(config.linear_damping, 1.0)
        self.assertEqual(config.bottom_wall_friction, 0.03)

    def test_unknown_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Unknown IceFlowConfig option"):
            create_iceflow_config(temperature=273.15)

    def test_bottom_friction_range_is_validated(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "bottom_wall_friction must be in"):
                    create_iceflow_config(bottom_wall_friction=value)


class IceFlowCudaTests(unittest.TestCase):
    def test_coupling_stays_finite_and_writes_png(self):
        output = Path(tempfile.mkdtemp(prefix="iceflow2d-test-"))
        config = create_iceflow_config(
            resolution=(64, 32),
            reference_length_cells=300,
            phase_warmup_steps=5,
            output_dir=str(output),
        )
        simulation = IceFlow2D(config)
        initial = simulation.diagnostics()
        self.assertEqual(initial["solid_cells"], config.ice_width * config.ice_height)
        simulation.step(200)
        result = simulation.diagnostics()
        self.assertTrue(result["fluid_finite"])
        self.assertTrue(result["rigid_finite"])
        self.assertTrue(result["body_inside"])
        self.assertGreater(result["cut_links"], 0)
        relative_water_error = abs(result["water_volume_error"]) / result["water_volume_target"]
        self.assertLess(relative_water_error, 1.0e-4)
        self.assertLess(result["max_fluid_speed"], 0.1)
        for field in (simulation.f, simulation.h, simulation.phi, simulation.u, simulation.p, simulation.sdf):
            self.assertTrue(np.isfinite(field.to_numpy()).all())

        frame_path = output / "frame_00000.png"
        simulation.save_frame(frame_path)
        self.assertTrue(frame_path.is_file())
        try:
            from PIL import Image

            with Image.open(frame_path) as image:
                self.assertEqual(image.size, (64, 32))
        except ImportError:
                self.assertGreater(frame_path.stat().st_size, 0)

    def test_fully_submerged_body_displaces_its_area(self):
        config = create_iceflow_config(
            resolution=(120, 60),
            reference_length_cells=300,
            phase_warmup_steps=0,
            water_width_fraction=0.95,
            water_height_fraction=0.80,
            ice_base_y_cells=20.0,
            gravity=(0.0, 0.0),
        )
        simulation = IceFlow2D(config)
        simulation._compute_hydrostatic_buoyancy()
        result = simulation.diagnostics()
        expected_area = config.ice_width * config.ice_height
        self.assertEqual(result["solid_cells"], expected_area)
        self.assertAlmostEqual(result["displaced_water_volume"], expected_area, delta=1.0e-5 * expected_area)

    def test_horizontal_waterline_gives_geometric_partial_displacement(self):
        config = create_iceflow_config(
            resolution=(120, 60),
            reference_length_cells=300,
            phase_warmup_steps=0,
            water_width_fraction=0.95,
            water_height_fraction=0.50,
            ice_base_y_cells=20.0,
            gravity=(0.0, 0.0),
        )
        simulation = IceFlow2D(config)
        simulation._compute_hydrostatic_buoyancy()
        # The 18-cell-high body spans y=20..38 while water occupies cell
        # centers below y=30: ten immersed rows out of eighteen.
        expected = config.ice_width * 10.0
        self.assertAlmostEqual(
            simulation.diagnostics()["displaced_water_volume"], expected, delta=1.0e-5 * expected
        )

    def test_asymmetric_displacement_applies_force_without_artificial_torque(self):
        config = create_iceflow_config(
            resolution=(120, 60),
            reference_length_cells=300,
            phase_warmup_steps=0,
            water_width_fraction=0.50,
            water_height_fraction=0.80,
            ice_base_y_cells=20.0,
        )
        simulation = IceFlow2D(config)
        simulation._compute_hydrostatic_buoyancy()
        result = simulation.diagnostics()
        self.assertGreater(result["buoyancy_impulse_y"], 0.0)
        self.assertEqual(result["buoyancy_torque"], 0.0)

    def test_bottom_contact_impulse_updates_translation_and_rotation(self):
        config = create_iceflow_config(
            resolution=(120, 60),
            reference_length_cells=300,
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            hydrodynamic_force_scale=0.0,
            hydrostatic_buoyancy_scale=0.0,
            linear_damping=1.0,
            angular_damping=1.0,
            max_ice_angular_speed=0.01,
            bottom_wall_friction=0.0,
        )
        simulation = IceFlow2D(config)
        angle = 0.15
        half_width = 0.5 * config.ice_width
        half_height = 0.5 * config.ice_height
        extent_y = abs(np.sin(angle)) * half_width + abs(np.cos(angle)) * half_height
        simulation.body_angle[None] = angle
        simulation.body_center[None] = (0.5 * config.nx, config.boundary_cells + extent_y + 0.001)
        simulation.body_velocity[None] = (0.0, -0.02)
        simulation.body_angular_velocity[None] = 0.0

        simulation._advance_rigid_ice()
        result = simulation.diagnostics()
        self.assertGreater(result["bottom_contact_impulse"], 0.0)
        self.assertAlmostEqual(result["body_velocity_y"], -0.009303354, delta=2.0e-5)
        self.assertAlmostEqual(result["body_angular_velocity"], -0.001496341, delta=2.0e-6)
        self.assertAlmostEqual(result["bottom_contact_impulse"], 3.178059, delta=2.0e-4)
        self.assertLess(abs(result["bottom_position_correction"]), 1.0e-5)

    def test_corner_position_projection_does_not_inject_velocity(self):
        config = create_iceflow_config(
            resolution=(120, 60),
            reference_length_cells=300,
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            hydrodynamic_force_scale=0.0,
            hydrostatic_buoyancy_scale=0.0,
            linear_damping=1.0,
            angular_damping=1.0,
            max_ice_angular_speed=0.01,
            bottom_wall_friction=0.0,
        )
        simulation = IceFlow2D(config)
        angle = 0.15
        half_width = 0.5 * config.ice_width
        half_height = 0.5 * config.ice_height
        extent_y = abs(np.sin(angle)) * half_width + abs(np.cos(angle)) * half_height
        center_y = config.boundary_cells + extent_y - 0.01
        simulation.body_angle[None] = angle
        simulation.body_center[None] = (0.5 * config.nx, center_y)
        simulation.body_velocity[None] = (0.0, 0.0)
        simulation.body_angular_velocity[None] = 0.0

        simulation._advance_rigid_ice()
        result = simulation.diagnostics()
        corrected_angle = result["body_angle"]
        corrected_center_y = result["body_center_y"]
        corners = []
        cosine = np.cos(corrected_angle)
        sine = np.sin(corrected_angle)
        for local_x in (-half_width, half_width):
            for local_y in (-half_height, half_height):
                rotated_y = sine * local_x + cosine * local_y
                corners.append(corrected_center_y + rotated_y)
        self.assertGreaterEqual(min(corners), config.boundary_cells - 2.0e-5)
        self.assertEqual(result["body_velocity_x"], 0.0)
        self.assertEqual(result["body_velocity_y"], 0.0)
        self.assertEqual(result["body_angular_velocity"], 0.0)
        self.assertGreater(result["bottom_position_correction"], 0.0)
        self.assertLess(result["bottom_position_correction"], 0.01)
        self.assertLess(corrected_angle, angle)

    def test_bottom_coulomb_friction_reduces_sliding_with_bounded_impulse(self):
        config = create_iceflow_config(
            resolution=(120, 60),
            reference_length_cells=300,
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            hydrodynamic_force_scale=0.0,
            hydrostatic_buoyancy_scale=0.0,
            linear_damping=1.0,
            angular_damping=1.0,
            max_ice_angular_speed=0.01,
            wall_restitution=0.0,
            bottom_wall_friction=0.03,
        )
        simulation = IceFlow2D(config)
        initial_vx = 0.01
        simulation.body_velocity[None] = (initial_vx, -0.01)
        simulation.body_angular_velocity[None] = 0.0

        simulation._advance_rigid_ice()
        result = simulation.diagnostics()
        normal_impulse = result["bottom_contact_impulse"]
        tangent_impulse = result["bottom_friction_impulse"]
        friction_torque = result["bottom_friction_torque_impulse"]
        self.assertGreater(normal_impulse, 0.0)
        self.assertLess(tangent_impulse, 0.0)
        self.assertLess(result["body_velocity_x"], initial_vx)
        self.assertLessEqual(
            result["bottom_friction_abs_impulse"],
            config.bottom_wall_friction * normal_impulse + 2.0e-5,
        )
        self.assertAlmostEqual(
            config.ice_mass_lattice * (result["body_velocity_x"] - initial_vx),
            tangent_impulse,
            delta=2.0e-5,
        )
        self.assertAlmostEqual(
            config.ice_inertia_lattice * result["body_angular_velocity"],
            result["bottom_contact_torque_impulse"] + friction_torque,
            delta=2.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
