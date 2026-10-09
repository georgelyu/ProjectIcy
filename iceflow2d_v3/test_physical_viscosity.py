"""Measure viscosity from the production collision's shear-stress relaxation."""

import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d_v3.test_iceflow import _falling_config


@ti.data_oriented
class _CollisionOnlyIceFlow(IceFlow2D):
    @ti.func
    def _prepare_momentum_cut_links(self, i, j, raw_post):
        # Inspect the collision before boundary reflection modifies its output.
        # Retain the real ice mask, temperature, phase and velocity gradients.
        return raw_post


class PhysicalViscosityCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_actual_collision_uses_only_molecular_viscosity_in_every_region(self):
        for model, acoustic_rate in (
            ("rm_mrt", 1.0),
            ("cm_mrt", 1.0),
            ("cm_mrt", 0.25),
        ):
            for temperature_dependent in (False, True):
                with self.subTest(
                    model=model,
                    acoustic_rate=acoustic_rate,
                    temperature_dependent=temperature_dependent,
                ):
                    self._check_collision(temperature_dependent, model, acoustic_rate)

    def _check_collision(self, temperature_dependent, model, acoustic_rate):
        config = _falling_config(
            gravity=(0.0, 0.0),
            sigma=0.0,
            phase_warmup_steps=0,
            viscosity_water=None if temperature_dependent else 1e-6,
            viscosity_air=None if temperature_dependent else 1.5e-5,
            momentum_collision_model=model,
            cm_acoustic_relaxation_rate=acoustic_rate,
        )
        s = _CollisionOnlyIceFlow(config)
        solid = s.solid_mask.to_numpy() != 0
        near = np.zeros_like(solid)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                near |= np.roll(solid, (di, dj), axis=(0, 1))
        adjacent = np.argwhere(near & ~solid & (s.wall_mask.to_numpy() == 0))
        ice_cells = [tuple(cell) for cell in adjacent[[0, len(adjacent) // 2, -1]]]
        probes = [
            ("air", (5, 5), 0.0, 20.0),
            ("bulk_hot_water", (12, 6), 1.0, 90.0),
            ("bulk_cold_water", (22, 6), 1.0, 0.0),
            ("interface", (20, 10), 0.5, 45.0),
            ("ice_hot_water", ice_cells[0], 1.0, 90.0),
            ("ice_cold_water", ice_cells[1], 1.0, 0.0),
            ("ice_interface", ice_cells[2], 0.5, 45.0),
        ]
        for _, cell, _, _ in probes[:4]:
            self.assertFalse(near[cell])
        self.assertEqual(len({cell for _, cell, _, _ in probes}), len(probes))

        # Nonzero surrounding strain would have activated the removed SGS term.
        velocity = np.zeros((s.nx, s.ny, 2), dtype=np.float32)
        velocity[..., 0] = 0.002 * np.arange(s.ny)[None, :]
        populations = np.zeros((s.nx, s.ny, 9), dtype=np.float32)
        directions = np.array(
            [
                (0, 0),
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, 1),
                (-1, -1),
                (1, -1),
            ]
        )
        shear_basis = directions[:, 0] * directions[:, 1]
        stress = 1e-6
        for _, cell, phi, temperature in probes:
            velocity[cell] = 0.0
            populations[cell] = 0.25 * stress * shear_basis
            s.water_phase[cell] = phi
            volume = phi * config.dx**2
            s.thermal.water_volume_m2[cell] = volume
            s.thermal.water_sensible_energy[cell] = (
                volume
                * config.rho_water
                * config.thermal.properties.specific_heat_water_j_kg_k
                * temperature
            )
        s.momentum_velocity_lattice.from_numpy(velocity)
        s.momentum_populations.from_numpy(populations)
        s.pressure_lattice.fill(0.0)
        s._collide_momentum()
        post = s.momentum_stream_buffer.to_numpy().astype(np.float64)
        for name, cell, phi, temperature in probes:
            with self.subTest(region=name):
                before_stress = np.dot(populations[cell].astype(float), shear_basis)
                after_stress = np.dot(post[cell], shear_basis)
                # Pi_xy' = (1 - 1/tau) Pi_xy, with zero local u, p and force.
                measured_tau = 1.0 / (1.0 - after_stress / before_stress)
                measured_nu = (measured_tau - 0.5) / 3.0 * config.dx**2 / s.scales.dt_s
                expected_nu = phi * config.water_viscosity_at(temperature) + (
                    1.0 - phi
                ) * config.air_viscosity_at(20.0)
                # Molecular tau is close to 0.5; recovering nu amplifies f32 roundoff.
                self.assertAlmostEqual(measured_nu / expected_nu, 1.0, delta=0.003)


if __name__ == "__main__":
    unittest.main()
