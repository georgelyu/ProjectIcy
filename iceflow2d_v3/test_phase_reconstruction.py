"""Manufactured boundary fields, rank loss, and staged connected refill."""

import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.phase_reconstruction import PhaseGradientStatus
from iceflow2d_v3.simulator import ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config


@ti.kernel
def visible_link(s: ti.template(), out: ti.types.ndarray()):
    out[0] = ti.cast(s._phase_visible_diagonal(10, 10, 11, 11), ti.i32)


@ti.kernel
def capillary_probe(s: ti.template(), i: ti.i32, j: ti.i32, out: ti.types.ndarray()):
    force = s._capillary_force_density(i, j)
    out[0], out[1] = force.x, force.y


class PhaseReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_taichi_cuda()
        cls.s = IceFlow2D(
            _falling_config(
                resolution=(32, 64),
                phase_warmup_steps=0,
                phase_gradient_scheme="connected_ls",
                gravity=(0, 0),
                sigma=0,
            )
        )
        cls.original_mass = cls.s.thermal.body_solid_mass.to_numpy().copy()
        cls.original_origin = cls.s.body_reference_origin.to_numpy().copy()
        cls.original_angle = cls.s.body_angle.to_numpy().copy()
        cls.original_fractions = cls.s._body_fractions.to_numpy().copy()

    def setUp(self):
        s = self.s
        s.thermal.body_solid_mass.from_numpy(self.original_mass)
        s.body_reference_origin.from_numpy(self.original_origin)
        s.body_angle.from_numpy(self.original_angle)
        s._body_fractions.from_numpy(self.original_fractions)
        s.steps = 0
        s.refill_pending_since.fill(0)
        s.refill_pending_events.fill(0)
        s.refill_pending_activations.fill(0)
        s.refill_pending_max_age_steps.fill(0)
        s._pending_refill_count = s._pending_refill_peak = 0
        s.solid_mask.fill(0)
        s.previous_solid_mask.fill(0)
        s.solid_body_id.fill(-1)
        s.previous_body_id.fill(-1)
        s.water_phase.fill(0.3)
        s.momentum_velocity_lattice.fill(0)
        s.fluid_acceleration_lattice.fill(0)
        s.pressure_lattice.fill(0)
        s.reference_pressure_by_row.fill(0)

    def test_wall_preserves_affine_tangential_gradient(self):
        s = self.s
        phi = (
            np.broadcast_to(0.25 + 0.005 * np.arange(s.nx)[:, None], (s.nx, s.ny))
            .astype(np.float32)
            .copy()
        )
        s.water_phase.from_numpy(phi)
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        np.testing.assert_allclose(
            s.phase_gradient_lattice[10, 3], [0.005, 0], atol=2e-8
        )
        np.testing.assert_allclose(
            s.source_gradient_lattice[10, 3], [0.005, 0], atol=2e-8
        )

    def test_single_layer_retains_tangent_and_rejects_cross_wall_donor(self):
        s = self.s
        solid = np.zeros((s.nx, s.ny), np.int8)
        solid[5:20, 4:9] = 1
        # An intervening solid node separates a different-valued fluid cell.
        solid[9, 3] = 1
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        phi = (
            np.broadcast_to(0.25 + 0.005 * np.arange(s.nx)[:, None], solid.shape)
            .astype(np.float32)
            .copy()
        )
        phi[8, 3] = 0.9
        s.water_phase.from_numpy(phi)
        s._prepare_phase_geometry()
        # Candidate bits contain only the immediate 3x3 patch. Left neighbor
        # (9,3) is solid, and the fluid at (8,3) cannot become a donor.
        self.assertEqual(int(s.phase_donor_mask[10, 3]) & (1 << 1), 0)
        self.assertEqual(int(s.phase_donor_mask[10, 3]) & ~511, 0)
        s._prepare_phase_gradients()
        np.testing.assert_allclose(
            s.phase_gradient_lattice[10, 3], [0.005, 0], atol=2e-8
        )

    def test_isolated_node_has_finite_constant_closure(self):
        s = self.s
        solid = np.ones((s.nx, s.ny), np.int8)
        solid[10, 10] = 0
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        np.testing.assert_array_equal(s.phase_gradient_lattice[10, 10], [0, 0])
        self.assertEqual(
            int(s.phase_gradient_status[10, 10]),
            PhaseGradientStatus.ISOLATED_CONSTANT_CLOSURE,
        )

    def test_single_row_retains_observed_gradient_without_wall_normal_fit(self):
        s = self.s
        solid = np.ones((s.nx, s.ny), np.int8)
        solid[5:20, 10] = 0
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        phase = np.full(solid.shape, 0.3, np.float32)
        phase[:, 10] += 0.005 * (np.arange(s.nx) - 10)
        s.water_phase.from_numpy(phase)
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        np.testing.assert_allclose(
            s.phase_gradient_lattice[10, 10], [0.005, 0], atol=2e-8
        )
        self.assertEqual(int(s.phase_gradient_spatial_rank[10, 10]), 1)
        self.assertIs(s.phase_gradient_lattice, s.source_gradient_lattice)

    def test_boundary_linear_fit_matches_independent_numpy_lstsq(self):
        s = self.s
        solid = np.ones((s.nx, s.ny), np.int8)
        offsets = [(-1, 0), (1, 0), (0, 1), (1, 1)]
        solid[10, 10] = 0
        phase = np.full(solid.shape, 0.2, np.float32)
        values = [0.1, 0.45, 0.28, 0.41]
        for (dx, dy), value in zip(offsets, values):
            solid[10 + dx, 10 + dy] = 0
            phase[10 + dx, 10 + dy] = value
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        s.water_phase.from_numpy(phase)
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        a, b = [], []
        for dx, dy in offsets:
            weight = np.sqrt(1 / 36 if dx and dy else 1 / 9)
            a.append(np.array([dx, dy]) * weight)
            b.append(float(phase[10 + dx, 10 + dy] - phase[10, 10]) * weight)
        expected = np.linalg.lstsq(a, b, rcond=None)[0]
        np.testing.assert_allclose(
            s.phase_gradient_lattice[10, 10], expected, atol=5e-8
        )
        self.assertEqual(int(s.phase_gradient_data_rank[10, 10]), 2)

    def test_corner_preserves_affine_field(self):
        s = self.s
        boundary = s.config.boundary_cells
        x, y = np.meshgrid(
            np.arange(s.nx) - boundary, np.arange(s.ny) - boundary, indexing="ij"
        )
        s.water_phase.from_numpy((0.25 + 0.001 * x + 0.002 * y).astype(np.float32))
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        np.testing.assert_allclose(
            s.phase_gradient_lattice[boundary, boundary], [0.001, 0.002], atol=5e-8
        )
        self.assertEqual(
            int(s.phase_gradient_status[boundary, boundary]),
            PhaseGradientStatus.LINEAR_FIT,
        )

    def test_second_ring_cannot_change_gradient(self):
        s = self.s
        phase = np.full((s.nx, s.ny), 0.3, np.float32)
        phase[9, 10], phase[11, 10] = 0.2, 0.4
        s.water_phase.from_numpy(phase)
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        before = np.array(s.phase_gradient_lattice[10, 10])
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if max(abs(dx), abs(dy)) == 2:
                    phase[10 + dx, 10 + dy] = (dx + dy) % 2
        s.water_phase.from_numpy(phase)
        s._prepare_phase_gradients()
        np.testing.assert_array_equal(s.phase_gradient_lattice[10, 10], before)

    def test_pure_phases_and_uniform_mixture_have_zero_boundary_gradient(self):
        s = self.s
        for value in (0.0, 0.37, 1.0):
            s.water_phase.fill(value)
            s._prepare_phase_geometry()
            s._prepare_phase_gradients()
            np.testing.assert_array_equal(s.phase_gradient_lattice.to_numpy(), 0)

    def test_one_sided_curved_field_accepts_first_layer_approximation(self):
        s = self.s
        boundary = s.config.boundary_cells
        x, y = np.meshgrid(
            np.arange(s.nx) - boundary, np.arange(s.ny) - boundary, indexing="ij"
        )
        phase = (0.001 * x * (x + 1) + 0.002 * y * (y + 1)).astype(np.float32)
        s.water_phase.from_numpy(phase)
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        # The deliberately linear approximation uses forward slopes, without
        # quadratic exactness or solid-side polynomial scaling.
        np.testing.assert_allclose(
            s.phase_gradient_lattice[boundary, boundary], [0.002, 0.004], atol=1e-8
        )

    def test_multilayer_refill_uses_completed_previous_pass(self):
        s = self.s
        old = np.zeros((s.nx, s.ny), np.int8)
        old[8:15, 3 : s.ny - 3] = 1
        s.previous_solid_mask.from_numpy(old)
        s.previous_body_id.from_numpy(np.where(old, 0, -1).astype(np.int32))
        phase = np.full(old.shape, 0.4, np.float32)
        phase[old != 0] = 0.0
        s.water_phase.from_numpy(phase)
        s._prepare_phase_geometry()
        s._refill_connected_nodes()
        np.testing.assert_allclose(s.water_phase.to_numpy()[old != 0], 0.4, atol=2e-7)
        self.assertEqual(int(s.refill_remaining[None]), 0)

    def test_unresolved_refill_is_inactive_without_changing_phase_or_ice_mass(self):
        s = self.s
        solid = np.ones((s.nx, s.ny), np.int8)
        solid[10, 10] = 0
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        s.previous_solid_mask.fill(1)
        s.previous_body_id.fill(0)
        s.water_phase.fill(0.7)
        s.phase_populations.fill(0.123)
        s.momentum_populations.fill(0.456)
        mass = s.body_mass_lattice.to_numpy().copy()
        geometry_volume = s.phase_change_geometry_volume_cells()
        s._prepare_phase_geometry()
        s._refill_connected_nodes()
        s._prepare_phase_gradients()
        self.assertEqual(int(s.solid_mask[10, 10]), 2)
        self.assertEqual(int(s.solid_body_id[10, 10]), -1)
        self.assertEqual(s._pending_refill_count, 1)
        self.assertFalse(s._refill_had_openings)
        self.assertAlmostEqual(float(s.water_phase[10, 10]), 0.7)
        np.testing.assert_allclose(s.phase_populations.to_numpy()[10, 10], 0.123)
        np.testing.assert_allclose(s.momentum_populations.to_numpy()[10, 10], 0.456)
        self.assertEqual(int(s.phase_donor_mask[10, 10]), 0)
        self.assertEqual(int(s.phase_gradient_status[10, 10]), PhaseGradientStatus.INACTIVE)
        self.assertEqual(s.phase_change_geometry_volume_cells(), geometry_volume)
        np.testing.assert_array_equal(s.body_mass_lattice.to_numpy(), mass)
        s.thermal._initialize_water_state(s.water_phase, s.wall_mask, s.solid_mask)
        self.assertEqual(float(s.thermal.water_volume_m2[10, 10]), 0)

    def test_pending_cluster_retries_and_activates_only_after_connection(self):
        s = self.s
        solid = np.ones((s.nx, s.ny), np.int8)
        solid[5, 10] = 0  # persistent donor, separated by physical ice
        solid[10:13, 10] = 0
        previous = np.ones_like(solid)
        previous[5, 10] = 0
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        s.previous_solid_mask.from_numpy(previous)
        s.previous_body_id.from_numpy(np.where(previous, 0, -1).astype(np.int32))
        s.water_phase.fill(0.9)
        s.water_phase[5, 10] = 0.25
        s._prepare_phase_geometry()
        s._refill_connected_nodes()
        np.testing.assert_array_equal(s.solid_mask.to_numpy()[10:13, 10], 2)
        self.assertEqual(int(s.refill_pending_events[None]), 3)
        # The geometry pass clears mask 2, but it must not make these cells donors.
        s._body_fractions.from_numpy(solid[None].astype(np.float32))
        s.steps = 4
        s._update_solid_mask()
        s._prepare_phase_geometry()
        s._refill_connected_nodes()
        self.assertEqual(s._pending_refill_count, 3)
        self.assertEqual(int(s.refill_pending_events[None]), 3)
        self.assertEqual(int(s.refill_pending_max_age_steps[None]), 4)
        # Opening a bridge must refill the whole old pending cluster too,
        # even though its previous/current body IDs are both -1.
        solid[6:10, 10] = 0
        s._body_fractions.from_numpy(solid[None].astype(np.float32))
        s.steps = 7
        s._update_solid_mask()
        s._prepare_phase_geometry()
        s._refill_connected_nodes()
        self.assertEqual(s._pending_refill_count, 0)
        np.testing.assert_array_equal(s.solid_mask.to_numpy()[5:13, 10], 0)
        np.testing.assert_allclose(s.water_phase.to_numpy()[5:13, 10], 0.25)
        np.testing.assert_array_equal(s.refill_pending_since.to_numpy()[10:13, 10], 0)
        self.assertEqual(int(s.refill_pending_activations[None]), 3)
        self.assertEqual(int(s.refill_pending_max_age_steps[None]), 7)

    def test_pending_does_not_block_refill_of_a_separate_reachable_region(self):
        s = self.s
        solid = np.ones((s.nx, s.ny), np.int8)
        solid[5:16, 10] = 0
        solid[24:27, 20] = 0
        previous = np.ones_like(solid)
        previous[5, 10] = 0
        s.solid_mask.from_numpy(solid)
        s.solid_body_id.from_numpy(np.where(solid, 0, -1).astype(np.int32))
        s.previous_solid_mask.from_numpy(previous)
        s.previous_body_id.from_numpy(np.where(previous, 0, -1).astype(np.int32))
        s.water_phase.fill(0.8)
        s.water_phase[5, 10] = 0.4
        s._prepare_phase_geometry()
        s._refill_connected_nodes()
        np.testing.assert_allclose(s.water_phase.to_numpy()[5:16, 10], 0.4)
        np.testing.assert_array_equal(s.refill_valid.to_numpy()[5:16, 10], 1)
        np.testing.assert_array_equal(s.solid_mask.to_numpy()[24:27, 20], 2)
        self.assertEqual(s._pending_refill_count, 3)
        self.assertTrue(s._refill_had_openings)

    def test_pending_reflects_both_populations_and_cannot_stream_stale_state(self):
        s = self.s
        s.solid_mask[10, 10] = 2
        s.phase_populations.fill(0)
        s.momentum_populations.fill(0)
        s.phase_stream_buffer.fill(0)
        s.momentum_stream_buffer.fill(0)
        s.phase_stream_buffer[9, 10, 1] = 0.3
        s.momentum_stream_buffer[9, 10, 1] = 0.2
        for q in range(9):
            s.phase_stream_buffer[10, 10, q] = 99
            s.momentum_stream_buffer[10, 10, q] = 99
        s._stream()
        np.testing.assert_array_equal(s.phase_populations.to_numpy()[10, 10], 0)
        np.testing.assert_array_equal(s.momentum_populations.to_numpy()[10, 10], 0)
        self.assertAlmostEqual(float(s.phase_populations[9, 10, 3]), 0.3)
        self.assertAlmostEqual(float(s.momentum_populations[9, 10, 3]), 0.2)
        self.assertLess(float(s.phase_populations.to_numpy().max()), 1)
        self.assertLess(float(s.momentum_populations.to_numpy().max()), 1)

    def test_diagonal_visibility_checks_interior_bilinear_maximum(self):
        s = self.s
        size = s.thermal.material_size.to_numpy()[0]
        ix, iy = int(size[0] // 2), int(size[1] // 2)
        origin = np.array(
            [[10.5 - (ix + 0.5 - size[0] / 2), 10.5 - (iy + 0.5 - size[1] / 2)]],
            np.float32,
        )
        s.body_reference_origin.from_numpy(origin)
        s.body_angle.fill(0)
        values = np.zeros_like(self.original_mass)
        out = np.zeros(1, np.int32)
        for endpoint, expected in ((0.2, 1), (0.49, 0)):
            values[ix, iy] = endpoint
            values[ix + 1, iy + 1] = endpoint
            values[ix + 1, iy] = values[ix, iy + 1] = 0.52
            s.thermal.body_solid_mass.from_numpy(
                values * s.thermal._initial_body_cell_mass
            )
            visible_link(s, out)
            self.assertEqual(int(out[0]), expected)

    def test_enabled_capillarity_retains_flat_tangential_interface_balance(self):
        s = IceFlow2D(
            _falling_config(
                resolution=(32, 64),
                phase_warmup_steps=0,
                phase_gradient_scheme="connected_ls",
                gravity=(0, 0),
                sigma=0.01,
            )
        )
        s.solid_mask.fill(0)
        s.solid_body_id.fill(-1)
        phase = (
            np.broadcast_to(0.25 + 0.005 * np.arange(s.nx)[:, None], (s.nx, s.ny))
            .astype(np.float32)
            .copy()
        )
        s.water_phase.from_numpy(phase)
        s._prepare_phase_geometry()
        s._prepare_phase_gradients()
        s._prepare_capillary_stress()
        force = np.zeros(2, np.float32)
        capillary_probe(s, 10, s.config.boundary_cells, force)
        np.testing.assert_allclose(force, 0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
