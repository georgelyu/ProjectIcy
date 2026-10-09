"""Independent macroscopic contracts for the optional stability operators."""

import unittest

import numpy as np
import taichi as ti

from iceflow2d_v3.benchmarks.check_interface_translation import TranslationProbe
from iceflow2d_v3.benchmarks.check_regularized_momentum import (
    C,
    amplification,
    basis,
    equilibrium,
)
from iceflow2d_v3.config import IceFlowConfig
from iceflow2d_v3.examples.four_falling_ice_melting_2d import create_config, parse_args
from iceflow2d_v3.simulator import IceFlow2D, ensure_taichi_cuda
from iceflow2d_v3.stability_options import pressure_preserving_stress_population
from iceflow2d_v3.test_iceflow import _falling_config
from iceflow2d_v3.test_momentum_collision import _CollisionProbe


@ti.kernel
def lift_stress(out: ti.types.ndarray(), xx: ti.f32, xy: ti.f32, yy: ti.f32):
    for q in range(9):
        out[q] = pressure_preserving_stress_population(
            q, ti.Vector([xx, xy, yy]), ti.Vector([0.13, -0.09])
        )


@ti.kernel
def flux_force(s: ti.template(), out: ti.types.ndarray()):
    for i, j in s.water_phase:
        force = s._phase_flux_force_density(i, j)
        out[i, j, 0], out[i, j, 1] = force.x, force.y


class StabilityConfigurationTests(unittest.TestCase):
    def test_options_default_off_and_are_independent(self):
        baseline = create_config(parse_args([])).to_dict()
        for flag, value, field in (
            ("--momentum-collision-model", "cm_hrr", "momentum_collision_model"),
            ("--phase-momentum-coupling", "consistent_flux", "phase_momentum_coupling"),
            ("--exposed-node-refill", "non_equilibrium", "exposed_node_refill"),
        ):
            selected = create_config(parse_args([flag, value])).to_dict()
            self.assertEqual(selected[field], value)
            self.assertEqual(
                {k for k in baseline if baseline[k] != selected[k]}, {field}
            )
        self.assertEqual(baseline["interface_width"], 5)
        self.assertEqual(baseline["phase_momentum_coupling"], "legacy")
        self.assertEqual(baseline["exposed_node_refill"], "equilibrium")

    def test_invalid_options_fail_at_configuration(self):
        for field, values in (
            ("hrr_kinetic_blend", (-0.1, 1.1, True, None, float("nan"))),
            ("phase_momentum_coupling", (None, True, "unknown")),
            ("exposed_node_refill", (None, True, "unknown")),
        ):
            for value in values:
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    IceFlowConfig(**{field: value})

    def test_regularized_long_wave_limit_has_configured_viscosity(self):
        # At finite wavelengths HRR adds truncation damping, which the benchmark
        # reports separately. Check the continuum limit rather than mislabelling
        # that damping as a change to the configured molecular viscosity.
        nu = 0.001
        k = 2 * np.pi / 1024
        eig, vectors = np.linalg.eig(
            amplification(nu, np.zeros(2), np.array([0, k]), "cm_hrr")
        )
        mode = np.argmax(abs(C[:, 0] @ vectors))
        measured = -np.log(abs(eig[mode])) / k**2
        self.assertAlmostEqual(measured / nu, 1, delta=0.003)


class StabilityOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_taichi_cuda()
        except RuntimeError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def test_stress_lift_preserves_pressure_mass_and_momentum(self):
        pop = np.empty(9, dtype=np.float32)
        lift_stress(pop, 0.003, -0.002, 0.005)
        self.assertAlmostEqual(float(pop.sum()), 0, delta=2e-9)
        np.testing.assert_allclose(pop @ C, 0, atol=2e-9)
        self.assertAlmostEqual(float(pop[1:].sum()), 0, delta=2e-9)
        np.testing.assert_allclose(
            C.T @ (pop[:, None] * C), [[0.003, -0.002], [-0.002, 0.005]], atol=2e-9
        )

    def test_hrr_preserves_force_impulse_and_pressure_mode_with_density_source(self):
        s = _CollisionProbe("cm_hrr", 0.25)
        phi = 0.3
        rho = phi + (1 - phi) * s._air_density_lattice
        u = np.array([0.03, -0.02])
        force = np.array([2e-5, -1e-5])
        gradient = np.array([0.002, -0.001])
        pressure = 0.0001
        # Separate rho scaling from the pressure contribution.
        pop = rho * equilibrium(u) + equilibrium(np.zeros(2), pressure)
        pop += np.linalg.solve(
            basis(np.zeros(2)),
            np.array([0, 0, 0, 0.0002, 0.0001, -0.00003, 0, 0, 0.0002]),
        )
        shape = (s.nx, s.ny)
        s.water_phase.fill(phi)
        s.fluid_acceleration_lattice.fill(0)
        s.momentum_velocity_lattice.from_numpy(
            np.broadcast_to(u, (*shape, 2)).astype(np.float32).copy()
        )
        s.momentum_populations.from_numpy(
            np.broadcast_to(pop, (*shape, 9)).astype(np.float32).copy()
        )
        s.pressure_lattice.fill(pressure)
        s.reference_pressure_by_row.fill(0)
        s.probe_force_data.from_numpy(
            np.broadcast_to(np.r_[force, gradient], (*shape, 4))
            .astype(np.float32)
            .copy()
        )
        s._collide_momentum()
        post = s.momentum_stream_buffer.to_numpy()[10, 10].astype(float)
        np.testing.assert_allclose(post @ C, rho * (u + force), atol=2e-8)
        ue = u + 0.5 * force
        eq = rho * equilibrium(ue) + equilibrium(np.zeros(2), pressure)
        pre_p, eq_p = -pop[1:].sum(), -eq[1:].sum()
        source_p = -(1 - s._air_density_lattice) * (ue @ gradient)
        expected = pre_p - 0.25 * (pre_p - eq_p) + 0.875 * source_p
        self.assertAlmostEqual(-float(post[1:].sum()), expected, delta=2e-8)

    def test_options_can_be_combined_without_changing_other_configuration(self):
        s = IceFlow2D(
            _falling_config(
                momentum_collision_model="cm_hrr",
                phase_momentum_coupling="consistent_flux",
                exposed_node_refill="non_equilibrium",
            )
        )
        s.step(4)
        self.assertTrue(np.isfinite(s.momentum_populations.to_numpy()).all())
        self.assertTrue(np.isfinite(s.water_phase.to_numpy()).all())

    def test_full_mass_flux_force_is_pairwise_conservative(self):
        s = TranslationProbe(
            _falling_config(
                phase_warmup_steps=0, phase_momentum_coupling="consistent_flux"
            )
        )
        x, y = np.meshgrid(np.arange(s.nx), np.arange(s.ny), indexing="ij")
        u = np.stack(
            [
                0.04 + 0.01 * np.sin(2 * np.pi * y / s.ny),
                0.02 * np.cos(2 * np.pi * x / s.nx),
            ],
            axis=-1,
        ).astype(np.float32)
        j = np.stack(
            [
                0.002 * np.cos(2 * np.pi * y / s.ny),
                0.003 * np.sin(2 * np.pi * x / s.nx),
            ],
            axis=-1,
        ).astype(np.float32)
        s.stability_velocity.from_numpy(u)
        s.phase_mass_flux.from_numpy(j)
        actual = np.empty_like(u)
        flux_force(s, actual)
        expected = np.zeros_like(u)
        for axis in range(2):
            face = (
                0.25
                * (u + np.roll(u, -1, axis=axis))
                * (j[..., axis] + np.roll(j[..., axis], -1, axis=axis))[..., None]
            )
            expected -= face - np.roll(face, 1, axis=axis)
        np.testing.assert_allclose(actual, expected, atol=3e-11)
        np.testing.assert_allclose(actual.sum(axis=(0, 1)), 0, atol=1e-8)
        # Here div(J)=0, while J.grad(u) != 0. A scalar u*div(J)
        # compensator would incorrectly produce zero and fail this test.
        self.assertGreater(float(np.max(abs(actual))), 1e-6)

    def test_phase_flux_is_zero_for_equal_densities(self):
        s = TranslationProbe(
            _falling_config(
                phase_warmup_steps=0,
                rho_air=1000,
                phase_momentum_coupling="consistent_flux",
            )
        )
        s._cache_stability_velocity()
        s._collide_phase()
        s._measure_phase_mass_flux()
        np.testing.assert_array_equal(s.phase_mass_flux.to_numpy(), 0)

    def test_refill_recovers_stress_without_changing_imposed_macros(self):
        u = np.array([0.013, -0.009])
        pressure = 0.001
        stress = np.array([[0.0003, -0.0002], [-0.0002, 0.0005]])
        raw = np.array(
            [
                0,
                0,
                0,
                np.trace(stress),
                stress[0, 0] - stress[1, 1],
                stress[0, 1],
                u[1] * stress[0, 0] + 2 * u[0] * stress[0, 1],
                u[0] * stress[1, 1] + 2 * u[1] * stress[0, 1],
                np.trace(stress),
            ]
        )
        residual = np.linalg.solve(basis(np.zeros(2)), raw)
        for option in ("equilibrium", "non_equilibrium"):
            s = IceFlow2D(
                _falling_config(
                    phase_warmup_steps=0,
                    exposed_node_refill=option,
                    gravity=(0, 0),
                    sigma=0,
                )
            )
            s.water_phase.fill(1)
            s.fluid_acceleration_lattice.fill(0)
            s.reference_pressure_by_row.fill(0)
            s.pressure_lattice.fill(pressure)
            s.momentum_velocity_lattice.from_numpy(
                np.broadcast_to(u, (s.nx, s.ny, 2)).astype(np.float32).copy()
            )
            pop = equilibrium(u, pressure) + residual
            s.momentum_populations.from_numpy(
                np.broadcast_to(pop, (s.nx, s.ny, 9)).astype(np.float32).copy()
            )
            s.previous_solid_mask.from_numpy(s.solid_mask.to_numpy())
            s.previous_body_id.from_numpy(s.solid_body_id.to_numpy())
            cell = (16, 16)
            s.previous_solid_mask[cell] = 1
            s.previous_body_id[cell] = 0
            s._refill_changed_nodes()
            actual = s.momentum_populations.to_numpy()[cell].astype(float)
            self.assertAlmostEqual(float(actual.sum()), 0, delta=1e-8)
            np.testing.assert_allclose(actual @ C, u, atol=1e-8)
            recovered_p = 0.6 * (actual[1:].sum() - 2 / 3 * (u @ u))
            self.assertAlmostEqual(recovered_p, pressure, delta=1e-8)
            neq = actual - equilibrium(u, pressure)
            expected = stress if option == "non_equilibrium" else np.zeros((2, 2))
            np.testing.assert_allclose(C.T @ (neq[:, None] * C), expected, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
