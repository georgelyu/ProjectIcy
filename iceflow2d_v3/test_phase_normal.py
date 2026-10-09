"""Bulk-noise suppression and interface preservation by phase-normal regularization."""

import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3.lattice import lattice_direction
from iceflow2d_v3.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config


@ti.data_oriented
class _PeriodicPhaseProbe(IceFlow2D):
    @ti.func
    def _is_fluid_cell(self, i, j):
        return True

    @ti.func
    def _phase_neighbor(self, i, j, sx, sy, distance):
        return self.water_phase[
            (i + sx * distance + 2 * self.nx) % self.nx,
            (j + sy * distance + 2 * self.ny) % self.ny,
        ]

    @ti.kernel
    def stream_periodic_phase(self):
        for i, j, q in self.phase_populations:
            c = lattice_direction(q)
            self.phase_populations[i, j, q] = self.phase_stream_buffer[
                (i - c.x + self.nx) % self.nx,
                (j - c.y + self.ny) % self.ny,
                q,
            ]

    def set_phase(self, phi):
        weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=np.float32)
        phi = np.broadcast_to(phi, (self.nx, self.ny)).astype(np.float32).copy()
        self.water_phase.from_numpy(phi)
        self.phase_populations.from_numpy(phi[..., None] * weights)
        self.momentum_velocity_lattice.fill(0.0)
        self.fluid_acceleration_lattice.fill(0.0)

    def advance_phase(self, steps):
        for _ in range(steps):
            self._collide_phase()
            self.stream_periodic_phase()
            self._update_streamed_phase()
            self._bound_water_phase()
        return self.water_phase.to_numpy()


def _phase_probe(epsilon):
    return _PeriodicPhaseProbe(
        _falling_config(
            phase_warmup_steps=0,
            gravity=(0.0, 0.0),
            sigma=0.0,
            phase_normal_epsilon=epsilon,
        )
    )


class PhaseNormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_uniform_pure_phases_remain_finite_and_unchanged(self):
        # A subnormal epsilon must not introduce division by zero at grad(phi)=0.
        for epsilon in (0.0, 1e-3, 1e-30):
            s = _phase_probe(epsilon)
            for phase in (0.0, 1.0):
                with self.subTest(epsilon=epsilon, phase=phase):
                    s.set_phase(phase)
                    result = s.advance_phase(32)
                    self.assertTrue(np.isfinite(s.phase_populations.to_numpy()).all())
                    np.testing.assert_allclose(result, phase, rtol=0, atol=2e-7)

    def test_regularization_reduces_persistent_bulk_phase_spots(self):
        deficits = {}
        for epsilon in (0.0, 1e-3):
            s = _phase_probe(epsilon)
            x = np.arange(s.nx)[:, None] - s.nx // 2
            y = np.arange(s.ny)[None, :] - s.ny // 2
            s.set_phase(1.0 - 1e-5 * np.exp(-(x * x + y * y) / 8))
            result = s.advance_phase(1024)
            deficits[epsilon] = 1.0 - float(result.min())
        self.assertGreater(deficits[0.0], 1e-5)
        self.assertLess(deficits[1e-3], 0.4 * deficits[0.0])

    def test_uniform_mixture_has_no_accumulating_collision_mass_source(self):
        s = _phase_probe(1e-3)
        # No gradient, advection or clipping is present. Even non-endpoint
        # values must stay fixed through collision/reconstruction and recovery.
        for phase in (0.13, 0.37, 0.72):
            with self.subTest(phase=phase):
                s.set_phase(phase)
                result = s.advance_phase(2048)
                np.testing.assert_allclose(result, np.float32(phase), rtol=0, atol=1e-7)

    def test_flat_interface_width_and_phase_amount_are_preserved(self):
        widths = {}
        for epsilon in (0.0, 1e-3):
            s = _phase_probe(epsilon)
            y = np.arange(s.ny) + 0.5
            width = s.config.interface_width
            phi = 0.5 * (
                np.tanh(2 * (y - s.ny * 0.25) / width)
                - np.tanh(2 * (y - s.ny * 0.75) / width)
            )
            s.set_phase(phi)
            initial_amount = s.water_phase.to_numpy().sum(dtype=np.float64)
            result = s.advance_phase(2048)
            profile = result.mean(axis=0)[: s.ny // 2]
            crossings = np.interp([0.1, 0.9], profile, y[: s.ny // 2])
            widths[epsilon] = crossings[1] - crossings[0]
            with self.subTest(epsilon=epsilon):
                self.assertLess(
                    abs(result.sum(dtype=np.float64) / initial_amount - 1), 2e-4
                )
        self.assertAlmostEqual(widths[0.0], widths[1e-3], delta=0.01)


if __name__ == "__main__":
    unittest.main()
