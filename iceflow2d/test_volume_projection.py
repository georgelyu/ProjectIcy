"""Regression tests for the retained conservative water-volume projection."""

from __future__ import annotations

import math
import unittest

import numpy as np

from iceflow2d import IceFlow2D, create_iceflow_config
from iceflow2d.simulator import ensure_taichi_cuda
from iceflow2d.config import ThermalConfig

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


def _projection_config(**overrides):
    nx, ny = 48, 64
    boundary = 2
    values = dict(
        resolution=(nx, ny),
        dx=2.5e-4,
        reference_velocity=4.0,
        rho_water=1000.0,
        rho_air=1.25,
        rho_ice=917.0,
        viscosity_water=1.0e-6,
        viscosity_air=1.5e-5,
        gravity=(0.0, 0.0),
        phase_warmup_steps=8,
        well_balanced_hydrostatics=True,
        water_width_fraction=(nx - boundary) / nx,
        water_height_fraction=32.0 / ny,
        ice_width_fraction=10.0 / nx,
        ice_height_fraction=10.0 / ny,
        ice_base_y_cells=42.0,
        ice_initial_angle=math.radians(5.0),
        boundary_cells=boundary,
        ice_fixed=False,
        rigid_boundary_scheme="unified",
        air_interface_relaxation_time=0.8,
        thermal=ThermalConfig(
            initial_water_temperature_c=90.0,
            initial_ice_temperature_c=0.0,
            initial_air_temperature_c=20.0,
            update_interval_lbm_steps=4,
            water_buoyancy_model="linear",
            buoyancy_reference_temperature_c=90.0,
            moving_body_scheme="body_ale",
        ),
        volume_projection_tolerance=1.0e-8,
    )
    values.update(overrides)
    return create_iceflow_config(**values)


def _phase_equilibrium(water_phase: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    velocity64 = np.asarray(velocity, dtype=np.float64)
    phase64 = np.asarray(water_phase, dtype=np.float64)
    cu = np.einsum("...d,qd->...q", velocity64, DIRECTIONS)
    speed_squared = np.einsum("...d,...d->...", velocity64, velocity64)
    factor = 1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * speed_squared[..., None]
    return phase64[..., None] * WEIGHTS * factor


def _active_mask(simulation: IceFlow2D) -> np.ndarray:
    return (simulation.wall_mask.to_numpy() == 0) & (
        simulation.solid_mask.to_numpy() == 0
    )


def _assert_strict_mass(test: unittest.TestCase, simulation: IceFlow2D) -> None:
    active = _active_mask(simulation)
    phase = simulation.water_phase.to_numpy()
    target = float(simulation.water_volume_target[None])
    current = float(simulation.water_volume_current[None])
    host_volume = float(np.sum(phase[active], dtype=np.float64))
    tolerance = max(5.0e-5, 5.0e-8 * max(1.0, abs(target)))
    test.assertAlmostEqual(current, host_volume, delta=1.0e-8)
    test.assertAlmostEqual(current, target, delta=tolerance)
    cutoff = float(simulation.config.volume_projection_interface_cutoff)
    unresolved_air = active & (phase > 0.0) & (phase <= cutoff)
    unresolved_water = active & (phase >= 1.0 - cutoff) & (phase < 1.0)
    test.assertEqual(int(np.count_nonzero(unresolved_air)), 0)
    test.assertEqual(int(np.count_nonzero(unresolved_water)), 0)


class IceFlowVolumeProjectionConfigTests(unittest.TestCase):
    def test_projection_controls_are_positive_and_finite(self):
        config = _projection_config()
        self.assertGreater(config.volume_projection_tolerance, 0.0)
        self.assertTrue(math.isfinite(config.volume_projection_tolerance))
        self.assertGreater(config.volume_projection_interface_cutoff, 0.0)
        self.assertLess(config.volume_projection_interface_cutoff, 0.5)
        self.assertGreater(config.volume_projection_max_shift, 0.0)
        self.assertGreaterEqual(config.volume_projection_max_iterations, 1)

    def test_invalid_projection_controls_are_rejected(self):
        for name in ("volume_projection_tolerance", "volume_projection_max_shift"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        _projection_config(**{name: value})
        for value in (0.0, 0.5, -1.0, float("nan"), float("inf")):
            with self.subTest(volume_projection_interface_cutoff=value):
                with self.assertRaisesRegex(
                    ValueError, "volume_projection_interface_cutoff"
                ):
                    _projection_config(volume_projection_interface_cutoff=value)
        for value in (False, 0, -1, 1.5):
            with self.subTest(volume_projection_max_iterations=value):
                with self.assertRaisesRegex(
                    ValueError, "volume_projection_max_iterations"
                ):
                    _projection_config(volume_projection_max_iterations=value)


class IceFlowVolumeProjectionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:  # pragma: no cover - depends on test host
            raise unittest.SkipTest(str(exc)) from exc

    def test_projection_is_conservative_bounded_and_bulk_invariant(self):
        simulation = IceFlow2D(_projection_config())
        active = _active_mask(simulation)
        phase_before = simulation.water_phase.to_numpy()
        h_before = simulation.phase_populations.to_numpy()
        velocity_before = simulation.momentum_velocity_lattice.to_numpy()

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
        simulation.water_phase.from_numpy(phase_perturbed.astype(np.float32))
        simulation.phase_populations.from_numpy(h_perturbed.astype(np.float32))
        simulation.water_volume_target[None] = target

        nonequilibrium_before = h_perturbed.astype(np.float64) - _phase_equilibrium(
            phase_perturbed, velocity_before
        )
        simulation._correct_water_volume()
        phase_after = simulation.water_phase.to_numpy()
        h_after = simulation.phase_populations.to_numpy()
        velocity_after = simulation.momentum_velocity_lattice.to_numpy()

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
        simulation = IceFlow2D(_projection_config())
        active = _active_mask(simulation)
        phase = simulation.water_phase.to_numpy()
        phase_populations = simulation.phase_populations.to_numpy()
        target = float(np.sum(phase[active], dtype=np.float64))
        exact_air = active & (phase == 0.0)
        exact_water = active & (phase == 1.0)
        interface_indices = np.argwhere(active & (phase > 0.1) & (phase < 0.9))
        self.assertGreaterEqual(len(interface_indices), 2)

        for index, value in (
            (tuple(interface_indices[0]), -0.02),
            (tuple(interface_indices[-1]), 1.02),
        ):
            delta = np.float32(value - phase[index])
            phase[index] = np.float32(value)
            phase_populations[index] += delta * WEIGHTS.astype(np.float32)

        simulation.water_phase.from_numpy(phase)
        simulation.phase_populations.from_numpy(phase_populations)
        simulation.water_volume_target[None] = target
        simulation._correct_water_volume()
        corrected = simulation.water_phase.to_numpy()

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


if __name__ == "__main__":
    unittest.main()
