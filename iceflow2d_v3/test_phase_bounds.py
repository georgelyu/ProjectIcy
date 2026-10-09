"""Strict out-of-range phase repair without altering valid distributions."""

import unittest

import numpy as np

from iceflow2d_v3 import IceFlow2D
from iceflow2d_v3.simulator import ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config


class PhaseCutoffTests(unittest.TestCase):
    def test_invalid_cutoffs_are_rejected(self):
        for cutoff in (0.0, 0.5, -1.0, float("nan"), float("inf")):
            with (
                self.subTest(cutoff=cutoff),
                self.assertRaisesRegex(ValueError, "water_phase_cutoff"),
            ):
                _falling_config(water_phase_cutoff=cutoff)

    def test_removed_volume_projection_controls_are_rejected(self):
        for name in (
            "volume_projection_tolerance",
            "volume_projection_max_shift",
            "volume_projection_max_iterations",
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(TypeError, "Unknown IceFlowConfig option"),
            ):
                _falling_config(**{name: 1})


class PhaseBoundsCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_clipping_changes_only_out_of_bounds_fluid_cells(self):
        simulation = IceFlow2D(_falling_config(phase_warmup_steps=10))
        active = (simulation.wall_mask.to_numpy() == 0) & (
            simulation.solid_mask.to_numpy() == 0
        )
        phase = simulation.water_phase.to_numpy()
        populations = simulation.phase_populations.to_numpy()
        values = (
            -0.02,
            1.02,
            -1.0e-7,
            np.nextafter(np.float32(1.0), np.float32(2.0)),
            0.0,
            0.0001,
            0.37,
            0.999688,
            1.0,
        )
        cells = [tuple(cell) for cell in np.argwhere(active)[: len(values)]]
        for cell, value in zip(cells, values):
            # Keep the supplied zeroth moment consistent before the bound operation.
            populations[cell][0] += value - float(phase[cell])
            phase[cell] = value
        # The function must not repair covered ice or wall reservoir values.
        inactive_cells = [tuple(cell) for cell in np.argwhere(~active)[:2]]
        for cell, value in zip(inactive_cells, (-0.25, 1.25)):
            populations[cell][0] += value - float(phase[cell])
            phase[cell] = value
        simulation.water_phase.from_numpy(phase)
        simulation.phase_populations.from_numpy(populations)
        momentum_before = simulation.momentum_populations.to_numpy()
        velocity_before = simulation.physical_velocity_lattice.to_numpy()
        pressure_before = simulation.pressure_lattice.to_numpy()
        simulation._bound_water_phase()
        result = simulation.water_phase.to_numpy()
        result_populations = simulation.phase_populations.to_numpy()
        np.testing.assert_allclose(
            [result[cell] for cell in cells], np.clip(values, 0.0, 1.0), atol=1e-7
        )
        unchanged = ~active | ((phase >= 0.0) & (phase <= 1.0))
        np.testing.assert_array_equal(result[unchanged], phase[unchanged])
        np.testing.assert_array_equal(
            result_populations[unchanged], populations[unchanged]
        )
        np.testing.assert_allclose(
            result_populations.sum(axis=-1)[active],
            result[active],
            atol=2e-7,
        )
        np.testing.assert_array_equal(
            simulation.momentum_populations.to_numpy(), momentum_before
        )
        np.testing.assert_array_equal(
            simulation.physical_velocity_lattice.to_numpy(), velocity_before
        )
        np.testing.assert_array_equal(
            simulation.pressure_lattice.to_numpy(), pressure_before
        )
        # Once repaired, a second call must be a complete no-op.
        simulation._bound_water_phase()
        np.testing.assert_array_equal(simulation.water_phase.to_numpy(), result)
        np.testing.assert_array_equal(
            simulation.phase_populations.to_numpy(), result_populations
        )
        self.assertFalse(hasattr(simulation, "water_volume_target"))

    def test_valid_endpoints_do_not_trigger_population_regularization(self):
        simulation = IceFlow2D(_falling_config())
        active = (simulation.wall_mask.to_numpy() == 0) & (
            simulation.solid_mask.to_numpy() == 0
        )
        phase = simulation.water_phase.to_numpy()
        populations = simulation.phase_populations.to_numpy()
        cells = [tuple(cell) for cell in np.argwhere(active)[:2]]
        for cell, value in zip(cells, (0.0, 1.0)):
            phase[cell] = value
            # Large cancelling populations used to trigger unrelated L1
            # regularization despite the phase being exactly in range.
            populations[cell] = (value, 4.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        simulation.water_phase.from_numpy(phase)
        simulation.phase_populations.from_numpy(populations)

        simulation._bound_water_phase()

        np.testing.assert_array_equal(simulation.water_phase.to_numpy(), phase)
        np.testing.assert_array_equal(
            simulation.phase_populations.to_numpy(), populations
        )


if __name__ == "__main__":
    unittest.main()
