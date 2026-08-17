"""Regression tests for conservative, interface-only water-volume projection.

The CUDA tests deliberately use small lattices and call the projection and
moving-mask stages directly.  This keeps them much cheaper than reproducing
the frame-60 dam-break failure while retaining the invariants that prevent
that failure: exact global mass, bounds, and untouched bulk phases.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import fields

import numpy as np

from iceflow2d import IceFlow2D, IceFlowConfig, create_iceflow_config
from iceflow2d.lattice import C, W


def _phase_equilibrium(phi: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """NumPy equivalent of ``_heq`` for checking equilibrium lifting."""

    directions = np.asarray(C, dtype=np.float64)
    weights = np.asarray(W, dtype=np.float64)
    velocity64 = np.asarray(velocity, dtype=np.float64)
    phase64 = np.asarray(phi, dtype=np.float64)
    cu = np.einsum("...d,qd->...q", velocity64, directions)
    speed_squared = np.einsum("...d,...d->...", velocity64, velocity64)
    factor = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared[..., None]
    return phase64[..., None] * weights * factor


def _assert_strict_mass(test: unittest.TestCase, simulation: IceFlow2D) -> None:
    simulation._measure_water_volume()
    result = simulation.diagnostics()
    target = result["water_volume_target"]
    # State is stored in f32, so a f64 scalar solve cannot generally realize
    # a residual below a few ulps per interfacial cell.  This is still a much
    # stricter condition than the old per-step fractional correction.
    tolerance = max(5.0e-5, 5.0e-8 * max(1.0, abs(target)))
    test.assertLessEqual(abs(result["water_volume_error"]), tolerance)
    test.assertAlmostEqual(result["water_volume_current"], target, delta=tolerance)


class IceFlowVolumeProjectionConfigTests(unittest.TestCase):
    def test_fractional_volume_rate_has_been_removed(self):
        names = {item.name for item in fields(IceFlowConfig)}
        self.assertNotIn("volume_correction_rate", names)
        with self.assertRaisesRegex(TypeError, "Unknown IceFlowConfig option"):
            create_iceflow_config(volume_correction_rate=0.2)

    def test_projection_controls_are_positive_and_finite(self):
        config = create_iceflow_config()
        self.assertGreater(config.volume_projection_tolerance, 0.0)
        self.assertTrue(math.isfinite(config.volume_projection_tolerance))
        self.assertGreater(config.volume_projection_interface_cutoff, 0.0)
        self.assertLess(config.volume_projection_interface_cutoff, 0.5)
        self.assertGreater(config.volume_projection_max_shift, 0.0)
        self.assertTrue(math.isfinite(config.volume_projection_max_shift))
        self.assertGreaterEqual(config.volume_projection_max_iterations, 1)

        for name in ("volume_projection_tolerance", "volume_projection_max_shift"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        create_iceflow_config(**{name: value})
        for value in (0.0, 0.5, -1.0, float("nan"), float("inf")):
            with self.subTest(name="volume_projection_interface_cutoff", value=value):
                with self.assertRaisesRegex(ValueError, "volume_projection_interface_cutoff"):
                    create_iceflow_config(volume_projection_interface_cutoff=value)
        for value in (False, 0, -1, 1.5):
            with self.subTest(name="volume_projection_max_iterations", value=value):
                with self.assertRaisesRegex(ValueError, "volume_projection_max_iterations"):
                    create_iceflow_config(volume_projection_max_iterations=value)


class IceFlowVolumeProjectionCudaTests(unittest.TestCase):
    @staticmethod
    def _simulation() -> IceFlow2D:
        config = create_iceflow_config(
            resolution=(72, 40),
            reference_length_cells=300,
            phase_warmup_steps=16,
            water_width_fraction=0.75,
            water_height_fraction=0.65,
            gravity=(0.0, 0.0),
            hydrodynamic_force_scale=0.0,
            hydrostatic_buoyancy_scale=0.0,
            linear_damping=1.0,
            angular_damping=1.0,
        )
        return IceFlow2D(config)

    def test_projection_is_conservative_bounded_and_bulk_invariant(self):
        simulation = self._simulation()
        wall = simulation.wall.to_numpy()
        solid = simulation.solid.to_numpy()
        active = (wall == 0) & (solid == 0)
        phase_before = simulation.phi.to_numpy()
        h_before = simulation.h.to_numpy()
        velocity_before = simulation.u.to_numpy()

        interface = active & (phase_before > 0.02) & (phase_before < 0.98)
        pure_air = active & (phase_before == 0.0)
        pure_water = active & (phase_before == 1.0)
        self.assertGreater(np.count_nonzero(interface), 10)
        self.assertGreater(np.count_nonzero(pure_air), 10)
        self.assertGreater(np.count_nonzero(pure_water), 10)

        target = float(np.sum(phase_before[active], dtype=np.float64))
        perturbation = np.zeros_like(phase_before)
        perturbation[interface] = -0.03 * phase_before[interface] * (1.0 - phase_before[interface])
        phase_perturbed = phase_before + perturbation
        h_perturbed = h_before + perturbation[..., None] * np.asarray(W, dtype=np.float32)
        simulation.phi.from_numpy(phase_perturbed.astype(np.float32))
        simulation.h.from_numpy(h_perturbed.astype(np.float32))
        simulation.water_volume_target[None] = target

        nonequilibrium_before = h_perturbed.astype(np.float64) - _phase_equilibrium(
            phase_perturbed, velocity_before
        )
        simulation._correct_water_volume()
        phase_after = simulation.phi.to_numpy()
        h_after = simulation.h.to_numpy()
        velocity_after = simulation.u.to_numpy()
        result = simulation.diagnostics()

        _assert_strict_mass(self, simulation)
        self.assertTrue(result["water_projection_feasible"])
        self.assertGreater(result["water_projection_interface_cells"], 0)
        self.assertGreater(result["water_projection_iterations"], 0)
        self.assertGreaterEqual(float(np.min(phase_after[active])), 0.0)
        self.assertLessEqual(float(np.max(phase_after[active])), 1.0)
        np.testing.assert_array_equal(phase_after[pure_air], phase_before[pure_air])
        np.testing.assert_array_equal(phase_after[pure_water], phase_before[pure_water])

        nonequilibrium_after = h_after.astype(np.float64) - _phase_equilibrium(
            phase_after, velocity_after
        )
        changed = active & (np.abs(phase_after - phase_perturbed) > 1.0e-7)
        self.assertGreater(np.count_nonzero(changed), 0)
        np.testing.assert_allclose(
            nonequilibrium_after[changed],
            nonequilibrium_before[changed],
            rtol=0.0,
            atol=3.0e-7,
        )

    def test_projection_clips_predictor_overshoot_without_bulk_seeding(self):
        simulation = self._simulation()
        active = (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)
        phase = simulation.phi.to_numpy()
        h = simulation.h.to_numpy()
        target = float(np.sum(phase[active], dtype=np.float64))
        exact_air = active & (phase == 0.0)
        exact_water = active & (phase == 1.0)

        interface_indices = np.argwhere(active & (phase > 0.1) & (phase < 0.9))
        self.assertGreaterEqual(len(interface_indices), 2)
        low_index = tuple(interface_indices[0])
        high_index = tuple(interface_indices[-1])
        for index, value in ((low_index, -0.02), (high_index, 1.02)):
            delta = np.float32(value - phase[index])
            phase[index] = np.float32(value)
            h[index] += delta * np.asarray(W, dtype=np.float32)

        # Sub-resolution minority-phase tails are the seeds of the original
        # snowflake artifact.  The bulk canonicalization must remove them on
        # both sides, with their small mass change paid only by the interface.
        air_index = tuple(np.argwhere(exact_air)[0])
        water_index = tuple(np.argwhere(exact_water)[0])
        cutoff = simulation.cfg.volume_projection_interface_cutoff
        for index, value in ((air_index, 0.5 * cutoff), (water_index, 1.0 - 0.5 * cutoff)):
            delta = np.float32(value - phase[index])
            phase[index] = np.float32(value)
            h[index] += delta * np.asarray(W, dtype=np.float32)

        simulation.phi.from_numpy(phase)
        simulation.h.from_numpy(h)
        simulation.water_volume_target[None] = target
        simulation._correct_water_volume()
        corrected = simulation.phi.to_numpy()

        _assert_strict_mass(self, simulation)
        self.assertGreaterEqual(float(np.min(corrected[active])), 0.0)
        self.assertLessEqual(float(np.max(corrected[active])), 1.0)
        np.testing.assert_array_equal(
            corrected[exact_air],
            np.zeros(np.count_nonzero(exact_air), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            corrected[exact_water],
            np.ones(np.count_nonzero(exact_water), dtype=np.float32),
        )

        # An inconsistent target with no interface is deliberately
        # infeasible.  The safe response is to expose that status, not create
        # a low-amplitude water phase throughout otherwise pure air.
        all_air = np.zeros_like(corrected)
        simulation.phi.from_numpy(all_air)
        simulation.h.from_numpy(np.zeros_like(simulation.h.to_numpy()))
        simulation.water_volume_target[None] = 1.0
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            simulation._correct_water_volume()
        infeasible = simulation.diagnostics()
        self.assertFalse(infeasible["water_projection_feasible"])
        np.testing.assert_array_equal(
            simulation.phi.to_numpy()[active],
            np.zeros(np.count_nonzero(active), dtype=np.float32),
        )

    def test_moving_mask_refill_then_projection_preserves_volume(self):
        simulation = self._simulation()
        config = simulation.cfg
        target = float(simulation.water_volume_target[None])
        initial_center = np.asarray(simulation.body_center[None], dtype=np.float64)
        initial_solid = simulation.solid.to_numpy()
        initial_count = int(np.count_nonzero(initial_solid))
        initial_phase = simulation.phi.to_numpy()
        active_before = (simulation.wall.to_numpy() == 0) & (initial_solid == 0)
        y_indices = np.indices(initial_phase.shape)[1]
        remote_air = active_before & (initial_phase == 0.0) & (y_indices >= config.water_height + 6)
        self.assertGreater(np.count_nonzero(remote_air), 10)

        simulation._measure_geometry_water_before_remap()
        simulation._save_previous_geometry()
        selected = False
        for angle in (0.04, 0.08, 0.12, 0.16, 0.20):
            for offset_x in (0.2, 0.45, 0.7, 0.95):
                simulation.body_center[None] = (
                    float(initial_center[0] + offset_x),
                    float(initial_center[1] + 1.5),
                )
                simulation.body_angle[None] = angle
                simulation._rasterize_ice()
                candidate = simulation.solid.to_numpy()
                changed = np.count_nonzero(candidate != initial_solid)
                if changed > 0 and np.count_nonzero(candidate) != initial_count:
                    selected = True
                    break
            if selected:
                break
        self.assertTrue(selected, "test poses did not change the rasterized solid-cell count")

        simulation._update_fractional_geometry()
        simulation._measure_geometry_swept_water()
        simulation._refill_changed_nodes()
        simulation._measure_geometry_water_after_remap()
        simulation._update_fluid_macro()
        before_projection = simulation.diagnostics()
        self.assertGreater(before_projection["fresh_cells"], 0)
        self.assertGreater(before_projection["covered_cells"], 0)
        remote_air_before_projection = simulation.phi.to_numpy()[remote_air]

        simulation._correct_water_volume()
        simulation._measure_water_volume()
        result = simulation.diagnostics()
        phase_after = simulation.phi.to_numpy()
        active_after = (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)

        self.assertEqual(float(simulation.water_volume_target[None]), target)
        _assert_strict_mass(self, simulation)
        self.assertTrue(result["water_projection_feasible"])
        self.assertLessEqual(abs(result["water_projection_shift"]), config.volume_projection_max_shift)
        self.assertLess(abs(result["geometric_solid_area_error"]), 1.0e-6)
        self.assertLess(abs(result["geometric_gcl_residual"]), 1.0e-6)
        self.assertTrue(math.isfinite(result["geometric_water_remap_error"]))
        self.assertTrue(math.isfinite(result["binary_remap_error"]))
        self.assertGreaterEqual(float(np.min(phase_after[active_after])), 0.0)
        self.assertLessEqual(float(np.max(phase_after[active_after])), 1.0)
        unchanged_remote_air = remote_air & active_after
        self.assertLessEqual(
            float(np.max(np.abs(phase_after[unchanged_remote_air]))),
            max(1.0e-12, float(np.max(np.abs(remote_air_before_projection))) + 1.0e-12),
        )


if __name__ == "__main__":
    unittest.main()
