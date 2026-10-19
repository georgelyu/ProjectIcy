"""Accepted contact displacement and the fluid wall share one time interval."""

import math
import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceBodyConfig, IceFlow2D
from iceflow2d_v3.simulator import ensure_taichi_cuda
from iceflow2d_v3.test_multibody import _two_config


@ti.kernel
def wall_samples(s: ti.template(), values: ti.template()):
    for b in range(s.body_count):
        values[b] = s._body_velocity_at(s.body_center[b] + ti.Vector([1.0, 2.0]), b)


class BodyMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_taichi_cuda()

    def assert_admissible(self, s):
        fractions = s._body_fractions.to_numpy()
        self.assertFalse(np.any((fractions >= 0.5).sum(axis=0) > 1))
        # Committed fractions are clipped at the walls; also check the raw
        # candidate against them so clipping cannot hide penetration.
        self.assertEqual(s.body_motion.check(), 0)

    def test_subcell_pair_contact_does_not_push_bodies_half_a_cell(self):
        s = IceFlow2D(_two_config())
        s.body_center[0], s.body_center[1] = (9.4996, 14), (13.5004, 14)
        s._refresh_fluid_domain(resolve_contact=False)
        s.body_velocity[0], s.body_velocity[1] = (0.001, 0), (-0.001, 0)
        old = s.body_center.to_numpy()
        momentum = (
            s.body_velocity.to_numpy() * s.body_mass_lattice.array()[:, None]
        ).sum(axis=0)
        s._prepare_body_motion()
        np.testing.assert_array_equal(s.body_center.to_numpy(), old)
        self.assertTrue(np.all(s.body_motion.affected.to_numpy() == 1))
        np.testing.assert_allclose(
            (s.body_velocity.to_numpy() * s.body_mass_lattice.array()[:, None]).sum(
                axis=0
            ),
            momentum,
            atol=1e-9,
        )
        s._advance_body()
        self.assertLess(np.abs(s.body_center.to_numpy() - old).max(), 0.002)
        self.assert_admissible(s)

    def test_wall_contact_preserves_tangential_motion(self):
        s = IceFlow2D(_two_config())
        s.body_center[0] = (9, 4.0004)
        s._refresh_fluid_domain(resolve_contact=False)
        s.body_velocity[0] = (0.002, -0.001)
        old = s.body_center.to_numpy()
        s._prepare_body_motion()
        accepted = s.body_motion.wall_velocity.to_numpy().copy()
        self.assertAlmostEqual(accepted[0, 0], 0.002, delta=1e-6)
        # A corner cut by the raster may also rotate/lift the body; the
        # frictionless wall must preserve tangential translation, not pin it.
        self.assertGreaterEqual(accepted[0, 1], -1e-6)
        self.assertLess(accepted[0, 1], 0.001)
        s._advance_body()
        np.testing.assert_allclose(s.body_center.to_numpy() - old, accepted, atol=2e-6)
        self.assert_admissible(s)

    def test_rotation_and_translation_used_by_wall_equal_accepted_motion(self):
        s = IceFlow2D(_two_config())
        s.body_velocity.from_numpy(
            np.array([[0.0023, -0.0017], [-0.0008, 0.0012]], np.float32)
        )
        s.body_angular_velocity.from_numpy(np.array([0.0003, -0.0002], np.float32))
        old = s.body_center.to_numpy()
        old_angles = s.body_angle.to_numpy()
        s._prepare_body_motion()
        accepted = s.body_motion.wall_velocity.to_numpy().copy()
        omega = s.body_motion.wall_omega.to_numpy().copy()
        samples = ti.Vector.field(2, ti.f32, shape=s.body_count)
        wall_samples(s, samples)
        np.testing.assert_allclose(
            samples.to_numpy(), accepted + omega[:, None] * [-2, 1], atol=1e-9
        )
        np.testing.assert_array_equal(s.body_center.to_numpy(), old)
        np.testing.assert_array_equal(s.body_angle.to_numpy(), old_angles)
        s._advance_body()
        np.testing.assert_allclose(s.body_center.to_numpy() - old, accepted, atol=2e-6)
        np.testing.assert_allclose(
            s.body_angle.to_numpy() - old_angles, omega, atol=1e-9
        )
        self.assertEqual(s.body_motion.active[None], 0)
        self.assert_admissible(s)

    def test_bounded_retry_leaves_an_uninvolved_body_moving(self):
        s = IceFlow2D(
            _two_config(
                ice_bodies=(
                    IceBodyConfig((4, 4), (9, 14)),
                    IceBodyConfig((4, 4), (14, 14)),
                    IceBodyConfig((4, 4), (24, 4.0001)),
                )
            )
        )
        s.body_center[1] = (12.5001, 14)
        s._refresh_fluid_domain(resolve_contact=False)
        s.body_velocity[1] = (-0.001, 0)
        # This body has an independent floor contact. It must not inherit
        # the retry fraction of the other two bodies' mutual collision.
        s.body_velocity[2] = (0.003, -0.001)
        old = s.body_center.to_numpy()
        s._prepare_body_motion()
        self.assertTrue(np.any(s.body_motion.fraction.to_numpy()[:2] < 1))
        self.assertEqual(s.body_motion.fraction[2], 1)
        s._advance_body()
        np.testing.assert_allclose(
            s.body_center.to_numpy()[2] - old[2], [0.003, 0.0], atol=2e-6
        )
        self.assertLess(np.abs(s.body_center.to_numpy()[:2] - old[:2]).max(), 0.002)
        self.assert_admissible(s)

    def test_still_bodies_keep_pose_and_fluid_at_rest(self):
        s = IceFlow2D(_two_config())
        old = s.body_reference_origin.to_numpy().copy()
        s.step(4)
        np.testing.assert_array_equal(s.body_reference_origin.to_numpy(), old)
        self.assertLess(np.abs(s.physical_velocity_lattice.to_numpy()).max(), 1e-6)
        self.assertTrue(np.isfinite(s.phase_populations.to_numpy()).all())

    def test_cached_raster_matches_direct_sampling_and_retains_melt_remnant(self):
        s = IceFlow2D(
            _two_config(
                ice_bodies=(
                    IceBodyConfig((4, 6), (9, 14), angle_rad=0.2),
                    IceBodyConfig((4, 4), (23, 32), angle_rad=-0.4),
                )
            )
        )
        mass = s.thermal.body_solid_mass.to_numpy()
        mass[s.thermal.material_body_id.to_numpy() == 1] *= 0.3
        s.thermal.body_solid_mass.from_numpy(mass)
        s._refresh_eroded_body()
        self.assertEqual(s._body_has_resolved_boundary[1], 0)
        s.body_velocity[0] = (0.002, -0.001)
        s.body_angular_velocity[0] = 0.0003
        s._prepare_body_motion()
        s._advance_body()
        cached = s._body_fractions.to_numpy()
        self.assertGreater(cached[1].max(), 0)
        self.assertLess(cached[1].max(), 0.5)
        s._calculate_moving_body_fraction()
        s._commit_moving_body_raster()
        direct = s._body_fractions.to_numpy()
        np.testing.assert_allclose(cached, direct, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(cached >= 0.5, direct >= 0.5)

    def test_body_slides_along_floor_without_raster_locking(self):
        angle = 0.12
        extent = 2 * (math.cos(angle) + math.sin(angle))
        s = IceFlow2D(
            _two_config(
                ice_bodies=(
                    IceBodyConfig((4, 4), (9, 2 + extent + 0.0001), angle_rad=angle),
                    IceBodyConfig((4, 4), (24, 32)),
                )
            )
        )
        s.body_velocity[0] = (0.002, -0.002)
        initial_x = float(s.body_center[0][0])
        for _ in range(200):
            s._prepare_body_motion()
            s._advance_body()
        self.assertGreater(float(s.body_center[0][0]) - initial_x, 0.38)
        self.assertLess(int(s.body_motion.held_steps[0]), 4)
        self.assert_admissible(s)

    def test_contacting_pair_keeps_common_translation_along_floor(self):
        s = IceFlow2D(_two_config())
        s.body_center[0], s.body_center[1] = (9, 4.0001), (13.0004, 4.0001)
        s._refresh_fluid_domain(resolve_contact=False)
        s.body_velocity[0], s.body_velocity[1] = (0.003, 0), (0.001, 0)
        old = s.body_center.to_numpy()
        for _ in range(200):
            s._prepare_body_motion()
            s._advance_body()
        moved = s.body_center.to_numpy() - old
        self.assertGreater(moved[:, 0].min(), 0.38)
        self.assertLess(int(s.body_motion.held_steps.to_numpy().max()), 4)
        np.testing.assert_allclose(s.body_velocity.to_numpy()[:, 0], 0.002, atol=1e-6)
        self.assert_admissible(s)


if __name__ == "__main__":
    unittest.main()
