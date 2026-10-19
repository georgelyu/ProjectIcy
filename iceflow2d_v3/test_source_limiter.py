"""Moment and conservation contracts for the local numerical-source guards."""

import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3.config import IceFlowConfig
from iceflow2d_v3.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config
from iceflow2d_v3.test_momentum_collision import (
    _C,
    _CollisionProbe,
    _equilibrium,
    _reference_collision,
)
from iceflow2d_v3.test_stability_options import flux_force


@ti.kernel
def measure_scale(
    s: ti.template(), u: ti.types.vector(2, ti.f32), grad: ti.types.vector(2, ti.f32)
) -> ti.f32:
    return s._density_source_limiter(10, 10, u, grad)


@ti.kernel
def gas_outlier(s: ti.template(), u: ti.types.vector(2, ti.f32)) -> ti.types.vector(4, ti.f32):
    return s._gas_velocity_outlier(10, 10, u)


@ti.data_oriented
class GuardedCollisionProbe(_CollisionProbe):
    def __init__(self):
        IceFlow2D.__init__(
            self,
            _falling_config(
                phase_warmup_steps=0,
                phase_gradient_scheme="connected_ls",
                density_gradient_source_limit=0.05,
                gravity=(0, 0),
                sigma=0,
            ),
        )
        self.probe_force_data = ti.Vector.field(4, ti.f32, shape=(self.nx, self.ny))


class SourceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_taichi_cuda()
        cls.s = IceFlow2D(
            _falling_config(
                phase_warmup_steps=0,
                phase_gradient_scheme="connected_ls",
                phase_momentum_coupling="consistent_flux",
                density_gradient_source_limit=0.2,
                phase_flux_acceleration_limit=0.05,
                refill_moment_projection=True,
                gas_state_repair=True,
                sigma=0,
            )
        )

    def setUp(self):
        s = self.s
        s.solid_mask.fill(0)
        s.previous_solid_mask.fill(0)
        s.solid_body_id.fill(-1)
        s.previous_body_id.fill(-1)
        s.water_phase.fill(0)
        s.momentum_velocity_lattice.fill(0)
        s.fluid_acceleration_lattice.fill(0)
        s.pressure_lattice.fill(0)
        s.reference_pressure_by_row.fill(0)
        s.momentum_populations.fill(0)
        s.refill_valid.fill(0)
        s.gas_repair_mask.fill(0)
        s.gas_repair_count.fill(0)
        s.gas_repair_peak_count.fill(0)
        s.stability_pressure.fill(0)
        s.stability_limit_parameters[None] = (0.2, 0.05)
        s._prepare_phase_geometry()

    def test_uniform_fast_air_and_small_source_are_unchanged(self):
        s = self.s
        self.assertEqual(measure_scale(s, [0.7, -0.4], [0, 0]), 1)
        self.assertEqual(measure_scale(s, [0.1, 0.1], [1e-5, 0]), 1)
        s.water_phase[10, 10] = 0.5
        self.assertEqual(measure_scale(s, [0.5, 0], [0.2, 0]), 1)
        s.water_phase[10, 10] = 0
        alpha = measure_scale(s, [0.02, 0], [-0.25, 0])
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 0.1)
        s.stability_limit_parameters[None] = (0, 0)
        self.assertEqual(measure_scale(s, [0.02, 0], [-0.25, 0]), 1)

    def test_face_guard_conserves_momentum_and_bounds_acceleration(self):
        s = self.s
        s.water_phase.fill(0)
        rng = np.random.default_rng(125)
        shape = (s.nx, s.ny, 2)
        s.stability_velocity.from_numpy(
            rng.uniform(-0.2, 0.2, shape).astype(np.float32)
        )
        s.phase_mass_flux.from_numpy(rng.uniform(-0.5, 0.5, shape).astype(np.float32))
        force = np.empty(shape, np.float32)
        flux_force(s, force)
        active = s.wall_mask.to_numpy() == 0
        np.testing.assert_allclose(force[active].sum(axis=0), 0, atol=1e-8)
        acceleration = np.abs(force[active]).sum(axis=1) / s._air_density_lattice
        self.assertLessEqual(float(acceleration.max()), 0.050001)

    def test_refill_projection_matches_macros_and_preserves_other_moments(self):
        s = self.s
        phase = np.zeros((s.nx, s.ny), np.float32)
        phase[:10, :] = 1
        phase[10, :] = 0.2
        s.water_phase.from_numpy(phase)
        s._prepare_phase_gradients()
        phi, pressure = 0.2, 0.0007
        velocity = np.array([0.027, -0.018])
        rho = phi * s._water_density_lattice + (1 - phi) * s._air_density_lattice
        pre = _equilibrium(rho, pressure, velocity)
        pre += np.array([0.0001, 0.0002, -0.0001, 0, 0, 0, 0, 0, 0])
        for q in range(9):
            s.momentum_populations[10, 10, q] = pre[q]
        s.previous_solid_mask[10, 10] = 1
        s.refill_valid[10, 10] = 1
        s.refill_state[10, 10] = [phi, pressure, *velocity]
        s._project_refill_moments()
        post = s.momentum_populations.to_numpy()[10, 10].astype(float)
        acceleration = np.array(s.fluid_acceleration_lattice[10, 10])
        np.testing.assert_allclose(
            post @ _C / rho + 0.5 * acceleration, velocity, atol=2e-7
        )
        self.assertAlmostEqual(float(post.sum()), float(pre.sum()), delta=3e-9)
        deviatoric_basis = np.array(
            [_C[:, 0] ** 2 - _C[:, 1] ** 2, _C[:, 0] * _C[:, 1]]
        )
        np.testing.assert_allclose(deviatoric_basis @ (post - pre), 0, atol=3e-9)
        np.testing.assert_array_equal(s.water_phase.to_numpy(), phase)
        s._update_pressure()
        self.assertAlmostEqual(float(s.pressure_lattice[10, 10]), pressure, delta=2e-9)

    def test_invalid_source_limits_are_rejected(self):
        for name in ("density_gradient_source_limit", "phase_flux_acceleration_limit"):
            for value in (-0.1, True, None, float("nan"), float("inf")):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    IceFlowConfig(**{name: value})

    def test_gas_repair_requires_both_temporal_and_spatial_outlier(self):
        s = self.s
        s.stability_velocity.fill((0.7, 0))
        self.assertEqual(gas_outlier(s, [0.9, 0])[2], 0)
        s.stability_velocity[10, 10] = (0.05, 0)
        self.assertEqual(gas_outlier(s, [0.7, 0.05])[2], 0)
        s.stability_velocity.fill((0.02, -0.01))
        np.testing.assert_allclose(gas_outlier(s, [0.35, 0.1]), [0.02, -0.01, 1, 0], atol=1e-8)
        s.water_phase[10, 10] = 0.5
        self.assertEqual(gas_outlier(s, [0.35, 0.1])[2], 0)
        self.assertEqual(gas_outlier(s, [2.0, 0.0])[2], 1)

    def test_gas_repair_reconstructs_phase_pressure_and_forced_momentum(self):
        s = self.s
        phi = np.full((s.nx, s.ny), 0.01, np.float32)
        phi[:10] = 0.8
        weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        s.phase_populations.from_numpy((phi[..., None] * weights).astype(np.float32))
        s.water_phase.from_numpy(phi)
        s.stability_velocity.fill((0.02, -0.01))
        s.fluid_acceleration_lattice.fill((0.0001, -0.0002))
        pressure = 0.0007
        s.pressure_lattice.fill(pressure)
        s.stability_pressure.fill(pressure)
        rho = 0.01 * s._water_density_lattice + 0.99 * s._air_density_lattice
        bad = _equilibrium(rho, pressure, np.array([0.35, 0.1]))
        for q in range(9):
            s.momentum_populations[10, 10, q] = bad[q]
        s._update_streamed_macroscopic_fields()
        s._prepare_phase_gradients()
        s._update_pressure()
        self.assertEqual(int(s.gas_repair_count[None]), 1)
        self.assertEqual(int(s.gas_repair_peak_count[None]), 1)
        active = s.wall_mask.to_numpy() == 0
        np.testing.assert_allclose(s.water_phase.to_numpy()[active], phi[active], atol=1e-7)
        post = s.momentum_populations.to_numpy()[10, 10].astype(float)
        force = np.array(s.fluid_acceleration_lattice[10, 10])
        np.testing.assert_allclose(post @ _C / rho + 0.5 * force, [0.02, -0.01], atol=2e-7)
        self.assertAlmostEqual(float(s.phase_populations.to_numpy()[10, 10].sum()), 0.01, delta=1e-8)
        self.assertAlmostEqual(float(s.pressure_lattice[10, 10]), pressure, delta=2e-8)
        s._update_pressure()
        self.assertAlmostEqual(float(s.pressure_lattice[10, 10]), pressure, delta=2e-8)

    def test_limited_collision_matches_independent_moment_transform(self):
        s = GuardedCollisionProbe()
        s.solid_mask.fill(0)
        s.water_phase.fill(0)
        s.reference_pressure_by_row.fill(0)
        s._prepare_phase_geometry()
        velocity = np.array([0.035, -0.017])
        gradient = np.array([-0.25, 0.03])
        force = np.array([0.00002, -0.00001])
        pressure = 0.0002
        rho = s._air_density_lattice
        pre = _equilibrium(rho, pressure, velocity)
        pre += np.array([0, 2e-6, -1e-6, -2e-6, 1e-6, 0, 0, 0, 0])
        s.momentum_velocity_lattice[10, 10] = velocity - 0.5 * force
        s.pressure_lattice[10, 10] = pressure
        s.probe_force_data[10, 10] = [*force, *gradient]
        for q in range(9):
            s.momentum_populations[10, 10, q] = pre[q]
        s._collide_momentum_kernel()
        alpha = float(s.density_source_scale[10, 10])
        self.assertLess(alpha, 0.1)
        expected = _reference_collision(
            pre,
            rho,
            pressure,
            velocity,
            force,
            gradient * alpha,
            s._water_density_lattice - rho,
            s._air_viscosity_lattice,
            "cm_mrt",
            s.config.cm_acoustic_relaxation_rate,
        )
        np.testing.assert_allclose(
            s.momentum_stream_buffer.to_numpy()[10, 10], expected, atol=3e-10
        )


if __name__ == "__main__":
    unittest.main()
