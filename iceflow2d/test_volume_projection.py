"""Regression tests for the retained conservative water-volume projection."""

from __future__ import annotations

import math
import unittest

import numpy as np

from iceflow2d import IceFlow2D, create_iceflow_config
from iceflow2d.simulator import ensure_taichi_cuda

DIRECTIONS = np.asarray(
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
WEIGHTS = np.asarray(
    (4.0 / 9.0,) + (1.0 / 9.0,) * 4 + (1.0 / 36.0,) * 4,
    dtype=np.float64,
)


def _phase_equilibrium(phi: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    velocity64 = np.asarray(velocity, dtype=np.float64)
    phase64 = np.asarray(phi, dtype=np.float64)
    cu = np.einsum("...d,qd->...q", velocity64, DIRECTIONS)
    speed_squared = np.einsum("...d,...d->...", velocity64, velocity64)
    factor = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared[..., None]
    return phase64[..., None] * WEIGHTS * factor


def _active_mask(simulation: IceFlow2D) -> np.ndarray:
    return (simulation.wall.to_numpy() == 0) & (simulation.solid.to_numpy() == 0)


def _assert_strict_mass(test: unittest.TestCase, simulation: IceFlow2D) -> None:
    active = _active_mask(simulation)
    target = float(simulation.water_volume_target[None])
    current = float(simulation.water_volume_current[None])
    host_volume = float(np.sum(simulation.phi.to_numpy()[active], dtype=np.float64))
    tolerance = max(5.0e-5, 5.0e-8 * max(1.0, abs(target)))
    test.assertAlmostEqual(current, host_volume, delta=1.0e-8)
    test.assertAlmostEqual(current, target, delta=tolerance)


class IceFlowVolumeProjectionConfigTests(unittest.TestCase):
    def test_projection_controls_are_positive_and_finite(self):
        config = create_iceflow_config()
        self.assertGreater(config.volume_projection_tolerance, 0.0)
        self.assertTrue(math.isfinite(config.volume_projection_tolerance))
        self.assertGreater(config.volume_projection_interface_cutoff, 0.0)
        self.assertLess(config.volume_projection_interface_cutoff, 0.5)
        self.assertGreater(config.volume_projection_max_shift, 0.0)
        self.assertTrue(math.isfinite(config.volume_projection_max_shift))
        self.assertGreaterEqual(config.volume_projection_max_iterations, 1)

    def test_invalid_projection_controls_are_rejected(self):
        for name in ("volume_projection_tolerance", "volume_projection_max_shift"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        create_iceflow_config(**{name: value})
        for value in (0.0, 0.5, -1.0, float("nan"), float("inf")):
            with self.subTest(name="volume_projection_interface_cutoff", value=value):
                with self.assertRaisesRegex(
                    ValueError, "volume_projection_interface_cutoff"
                ):
                    create_iceflow_config(volume_projection_interface_cutoff=value)
        for value in (False, 0, -1, 1.5):
            with self.subTest(name="volume_projection_max_iterations", value=value):
                with self.assertRaisesRegex(
                    ValueError, "volume_projection_max_iterations"
                ):
                    create_iceflow_config(volume_projection_max_iterations=value)


class IceFlowVolumeProjectionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:  # pragma: no cover - depends on test host
            raise unittest.SkipTest(str(exc)) from exc

    @staticmethod
    def _simulation() -> IceFlow2D:
        config = create_iceflow_config(
            resolution=(72, 40),
            reference_length_cells=300,
            phase_warmup_steps=16,
            water_width_fraction=0.75,
            water_height_fraction=0.65,
            gravity=(0.0, 0.0),
            linear_damping=1.0,
            angular_damping=1.0,
        )
        return IceFlow2D(config)

    def test_projection_is_conservative_bounded_and_bulk_invariant(self):
        simulation = self._simulation()
        active = _active_mask(simulation)
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
        perturbation[interface] = (
            -0.03 * phase_before[interface] * (1.0 - phase_before[interface])
        )
        phase_perturbed = phase_before + perturbation
        h_perturbed = h_before + perturbation[..., None] * WEIGHTS.astype(np.float32)
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

        _assert_strict_mass(self, simulation)
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

    def test_projection_clips_overshoot_without_seeding_bulk(self):
        simulation = self._simulation()
        active = _active_mask(simulation)
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
            h[index] += delta * WEIGHTS.astype(np.float32)

        air_index = tuple(np.argwhere(exact_air)[0])
        water_index = tuple(np.argwhere(exact_water)[0])
        cutoff = simulation.cfg.volume_projection_interface_cutoff
        for index, value in (
            (air_index, 0.5 * cutoff),
            (water_index, 1.0 - 0.5 * cutoff),
        ):
            delta = np.float32(value - phase[index])
            phase[index] = np.float32(value)
            h[index] += delta * WEIGHTS.astype(np.float32)

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

        simulation.phi.from_numpy(np.zeros_like(corrected))
        simulation.h.from_numpy(np.zeros_like(simulation.h.to_numpy()))
        simulation.water_volume_target[None] = 1.0
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            simulation._correct_water_volume()
        np.testing.assert_array_equal(
            simulation.phi.to_numpy()[active],
            np.zeros(np.count_nonzero(active), dtype=np.float32),
        )

    def test_moving_mask_refill_then_projection_preserves_state(self):
        simulation = self._simulation()
        target = float(simulation.water_volume_target[None])
        center_before = np.asarray(simulation.body_center[None], dtype=np.float64)
        solid_before = simulation.solid.to_numpy()
        phase_before = simulation.phi.to_numpy()
        wall = simulation.wall.to_numpy()
        y_indices = np.indices(phase_before.shape)[1]
        remote_air = (
            (wall == 0)
            & (solid_before == 0)
            & (phase_before == 0.0)
            & (y_indices >= simulation.cfg.water_height + 6)
        )
        self.assertGreater(np.count_nonzero(remote_air), 10)

        simulation.body_center[None] = (
            float(center_before[0] + 1.0),
            float(center_before[1] + 1.0),
        )
        simulation._update_ice_geometry()
        solid_after_move = simulation.solid.to_numpy()
        fresh = (solid_before == 1) & (solid_after_move == 0) & (wall == 0)
        covered = (solid_before == 0) & (solid_after_move == 1) & (wall == 0)
        self.assertGreater(np.count_nonzero(fresh), 0)
        self.assertGreater(np.count_nonzero(covered), 0)
        self.assertEqual(np.count_nonzero(fresh), np.count_nonzero(covered))

        phase_before_refill = simulation.phi.to_numpy()
        simulation._refill_changed_nodes()
        simulation._correct_water_volume()
        phase_after = simulation.phi.to_numpy()
        active_after = _active_mask(simulation)

        self.assertEqual(float(simulation.water_volume_target[None]), target)
        _assert_strict_mass(self, simulation)
        self.assertTrue(np.isfinite(phase_after).all())
        self.assertTrue(np.isfinite(simulation.f.to_numpy()).all())
        self.assertTrue(np.isfinite(simulation.h.to_numpy()).all())
        self.assertTrue(np.isfinite(simulation.u.to_numpy()).all())
        self.assertTrue(np.isfinite(simulation.p.to_numpy()).all())
        self.assertGreaterEqual(float(np.min(phase_after[active_after])), 0.0)
        self.assertLessEqual(float(np.max(phase_after[active_after])), 1.0)
        # Covered cells retain their bounded refill reservoir directly in
        # phi; the active mask excludes it from physical water volume.
        np.testing.assert_array_equal(
            phase_after[covered], np.clip(phase_before_refill[covered], 0.0, 1.0)
        )
        unchanged_remote_air = remote_air & active_after
        np.testing.assert_array_equal(
            phase_after[unchanged_remote_air],
            phase_before_refill[unchanged_remote_air],
        )

    def test_full_width_hydrostatic_surface_reuses_phase_projection_scratch(self):
        nx, ny = 72, 96
        boundary = 3
        config = create_iceflow_config(
            resolution=(nx, ny),
            reference_length_cells=ny,
            phase_warmup_steps=32,
            water_width_fraction=(nx - boundary + 0.5) / nx,
            water_height_fraction=0.5,
            ice_width_fraction=8.0 / nx,
            ice_height_fraction=6.0 / ny,
            ice_base_y_cells=70.0,
            ice_initial_angle=math.radians(5.0),
            well_balanced_hydrostatics=True,
        )
        simulation = IceFlow2D(config)
        simulation.step(3)
        _assert_strict_mass(self, simulation)

        postcollision_before = simulation.h_post.to_numpy()
        simulation._seed_water_projection_interface()
        postcollision_seeded = simulation.h_post.to_numpy()
        seed_state = postcollision_seeded[..., :2]
        seed_count = int(np.count_nonzero(seed_state[..., 0]))
        self.assertGreater(seed_count, 0)
        self.assertEqual(int(np.count_nonzero(seed_state[..., 1])), 0)
        np.testing.assert_array_equal(
            postcollision_seeded[..., 2:], postcollision_before[..., 2:]
        )
        self.assertEqual(simulation._volume_projection_band_radius % 2, 0)
        for pass_index in range(simulation._volume_projection_band_radius):
            simulation._dilate_water_projection_interface(
                1 if pass_index % 2 == 0 else 0
            )
        postcollision_dilated = simulation.h_post.to_numpy()
        mask_state = postcollision_dilated[..., :2]
        self.assertEqual(mask_state.dtype, np.dtype(np.float32))
        self.assertTrue(np.all((mask_state == 0.0) | (mask_state == 1.0)))
        primary_count = int(np.count_nonzero(mask_state[..., 0]))
        secondary_count = int(np.count_nonzero(mask_state[..., 1]))
        self.assertGreaterEqual(primary_count, secondary_count)
        self.assertGreaterEqual(primary_count, seed_count)
        np.testing.assert_array_equal(
            postcollision_dilated[..., 2:], postcollision_before[..., 2:]
        )
        self.assertFalse(hasattr(simulation, "water_projection_mask"))
        self.assertFalse(hasattr(simulation, "water_projection_mask_next"))

        # The next collision must overwrite both scratch planes before phase
        # streaming reads h_post again.
        simulation.step()
        _assert_strict_mass(self, simulation)


if __name__ == "__main__":
    unittest.main()
