"""Local phase bounds and population consistency without a volume target."""

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

    def test_clipping_changes_only_unbounded_cells_and_updates_populations(self):
        simulation = IceFlow2D(_falling_config(phase_warmup_steps=10))
        active = (simulation.wall_mask.to_numpy() == 0) & (
            simulation.solid_mask.to_numpy() == 0
        )
        phase = simulation.water_phase.to_numpy()
        populations = simulation.phase_populations.to_numpy()
        cells = [tuple(cell) for cell in np.argwhere(active)[:4]]
        for cell, value in zip(cells, (-0.02, 1.02, 0.0001, 0.37)):
            # Keep the supplied zeroth moment consistent before the bound operation.
            populations[cell][0] += value - float(phase[cell])
            phase[cell] = value
        simulation.water_phase.from_numpy(phase)
        simulation.phase_populations.from_numpy(populations)
        simulation._bound_water_phase()
        result = simulation.water_phase.to_numpy()
        np.testing.assert_allclose(
            [result[cell] for cell in cells], (0.0, 1.0, 0.0, 0.37), atol=1e-7
        )
        unchanged = active.copy()
        for cell in cells[:3]:
            unchanged[cell] = False
        np.testing.assert_array_equal(result[unchanged], phase[unchanged])
        np.testing.assert_allclose(
            simulation.phase_populations.to_numpy().sum(axis=-1)[active],
            result[active],
            atol=2e-7,
        )
        self.assertFalse(hasattr(simulation, "water_volume_target"))


if __name__ == "__main__":
    unittest.main()
